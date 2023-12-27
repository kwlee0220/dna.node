# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

from collections import defaultdict 
import logging

import numpy as np
import numpy.typing as npt
from numpy.linalg import det
import cv2

from dna import Box, color, Point, sub_logger
from dna.camera import Image, Frame
from dna.detect import Detection
from dna.support import plot_utils
from dna.track import utils
from dna.track.dna_track_params import DistanceIoUThreshold
from .matcher import Matcher, MatchingSession, chain, matches_str, match_str, \
                                IoUDistanceCostMatcher, MetricCostMatcher, HungarianMatcher, ReciprocalCostMatcher, \
                                INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE
from .matcher.cost_matrices import build_dist_cost, build_iou_cost, \
                                                build_metric_cost, gate_metric_cost
from .kalman_filter import KalmanFilter
from .dna_track_params import DNATrackParams
from .dna_track import DNATrack
from dna.node.node_track import NodeTrack

_EMPTY_FEATURE = np.zeros(1024)


def is_close_to_border(image:Image, det:Detection):
    h, w, _ = image.shape
    x1, y1, x2, y2 = tuple(det.bbox.tlbr)
    return x1 <= 5 or x2 >= w - 5 or y1 <= 5 or y2 >= h - 10


class Tracker:
    def __init__(self, params:DNATrackParams, logger:logging.Logger):
        self.params = params
        self.kf = KalmanFilter()
        self.tracks:list[DNATrack] = []
        self._next_id = 1
        self.logger = logger

    def track(self, frame:Frame, detections: list[Detection]) -> tuple[MatchingSession, list[DNATrack], list[NodeTrack]]:
        # 추적 중인 모든 track에 대해 다음 frame에서의 위치를 Kalman filter를 사용하여 예측한다.
        for track in self.tracks:
            track.predict(self.kf, frame.index, frame.ts)

        # 추적 중인 track과 새 frame에서 검출된 detection들 사이에 matching을 실시한다.
        session = self.match(detections)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'{session}')

        # if dna.DEBUG_SHOW_IMAGE:
        #     # display matching track and detection pairs
        #     self.draw_matched_detections("detections", frame.image.copy(), session.matches, detections)

        # 새 frame에서 검출된 detection과 match된 track의 경우, 새 위치를 추정하여 갱신시킨다.
        for track, det in session.associations:
            track.update(self.kf, frame, det)
            
            # 'drop_border_detection'이 enable된 경우, detection-box가 border와 근접한 경우 무시한다.
            if self.params.drop_border_detections and is_close_to_border(frame.image, det):
                track.mark_deleted()
        
        # unmatch된 track들 중에서 해당 box의 크기가 일정 범위가 넘으면 delete시킴.
        # 그렇지 않은 track은 temporarily lost된 것으로 간주함.
        # 만일 unmatch된 track이 exit zone에 포함된 경우에는 바로 delete 시킨다.
        for tidx in session.unmatched_track_idxes:
            track = self.tracks[tidx]
            if not track.is_deleted():
                # track의 크기가 너무 큰 경우에는 delete된 것으로 간주한다.
                if self.params.detection_max_size and track.location.size() > self.params.detection_max_size:
                    track.mark_deleted()
                elif track.exit_zone >= 0:
                    # 바로 이전 frame에서 'exit-zone'에 있던 detection과 match되었던 경우는 바로 delete시킴.
                    track.mark_deleted()
                else:
                     track.mark_missed(frame)
                
        for didx in session.unmatched_strong_det_idxes:
            det = detections[didx]

            # # Exit 영역에 포함되는 detection들은 무시한다
            # if det.exit_zone >= 0:
            #     continue
            
            # 'drop_border_detection'이 enable된 경우, detection-box가 border와 근접한 경우 무시한다.
            if is_close_to_border(frame.image, det):
                continue
            
            # create a new (tentative) track for this unmatched detection
            self._initiate_track(det, frame)
            
        merged_tracks = set()
        track_event_list = []
        if self.params.stable_zones:
            merged_tracks = self.merge_fragment(session, frame, track_event_list)
                     
        for track in self.tracks:
            if self.params.find_exit_zone(track.location) >= 0:
                if track.is_tentative():
                    track.time_to_promote += 1
                elif track.is_confirmed() or track.is_temporarily_lost():
                    track.mark_deleted()
            
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        for track in self.tracks:
            if track not in merged_tracks:
                track_event_list.append(track.to_event())
        for track in deleted_tracks:
            track_event_list.append(track.to_event())

        return (session, deleted_tracks, track_event_list)

    def match(self, detections:list[Detection]) -> MatchingSession:
        session = MatchingSession(self.tracks, detections, self.params)
        if len(detections) == 0 and len(self.tracks) == 0:
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return session

        # 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        dist_cost = build_dist_cost(self.kf, self.tracks, detections)
        iou_cost = build_iou_cost(self.tracks, detections)
        
        dist_iou_matcher = IoUDistanceCostMatcher(self.tracks, detections, self.params,
                                                  dist_cost, iou_cost, sub_logger(self.logger, 'matcher'))
        self.match_by_motion(session, dist_iou_matcher)
        
        ###########################################################################################################
        ###  지금까지 match되지 못한 strong detection들 중에서 이미 matching된 detection들과 많이 겹치는 경우,
        ###  이미 matching된 detection과 score를 기준으로 비교하여 더 높은 score를 갖는 detection으로 재 matching 시킴.
        ###########################################################################################################
        if session.unmatched_strong_det_idxes:
            def select_overlaps(box:Box, candidates:list[int]):
                return [idx for idx in candidates if box.iou(d_boxes[idx]) >= self.params.match_overlap_score]

            d_boxes = [det.bbox for det in detections]
            for match  in session.matches:
                matched_det_box = d_boxes[match[1]]
                overlap_det_idxes = select_overlaps(matched_det_box, session.unmatched_strong_det_idxes)
                if overlap_det_idxes:
                    candidates = [d_idx for d_idx in overlap_det_idxes] + [match[1]]
                    ranks = sorted(candidates, key=lambda i: detections[i].score, reverse=True)
                    new_match = (match[0], ranks[0])
                    session.pull_out(match)
                    session.update([new_match])
                    session.remove_det_idxes(ranks[1:])
                    if self.logger.isEnabledFor(logging.DEBUG) and match[1] != ranks[0]:
                        self.logger.debug(f'rematch: {match_str(self.tracks, match)} -> {match_str(self.tracks, new_match)}')
                    if len(session.unmatched_strong_det_idxes) == 0:
                        break
                
        ###########################################################################################################
        ### 이 단계까지 오면 지난 frame까지 active하게 추적되던 track들 (hot_track_idxes, tentative_track_idxes)에
        ### 대한 motion 정보만을 통해 matching이 완료됨.
        ### 남은 track들의 경우에는 이전 몇 frame동안 추적되지 못한 track들이어서 motion 정보만으로 matching하기
        ### 어려운 것들만 존재함. 이 track들에 대한 matching을 위해서는 appearance를 사용한 matching을 시도한다.
        ### Appearance를 사용하는 경우는 추적의 안정성을 위해 다음의 조건을 만족하는 detection에 대해서만 matching을 시도함.
        ###     - strong (high-scored) detection
        ###     - Detection box의 크기가 일정 이상이어서 추출된 metric 값이 신뢰할 수 있는 detection
        ###     - Exit-zone에 존재하지 않는 detection
        ###########################################################################################################
        self.match_by_metric(session, detections, dist_cost)

        # 아직 match되지 못한 track이 존재하면, strong detection들과 distance & IoU에 기반한
        # Hungarian 방식으로 최종 matching을 시도함.
        self.match_by_hungarian(session, iou_cost, dist_cost)
                        
        self.revise_matches(dist_iou_matcher.matcher, session, detections)

        return session

    def _initiate_track(self, detection: Detection, frame:Frame) -> None:
        mean, covariance = self.kf.initiate(detection.bbox.xyah)
        track = DNATrack(mean, covariance, self._next_id, frame.index, frame.ts,
                         self.params, detection, logger=self.logger)
        self.tracks.append(track)
        self._next_id += 1

    def match_by_motion(self, session:MatchingSession, dist_iou_matcher:IoUDistanceCostMatcher) -> None:
        matches0 = dist_iou_matcher.match(session.unmatched_track_idxes, session.unmatched_det_idxes)
        if matches0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"(motion) hot+tent, all: {matches_str(self.tracks, matches0)}")
            session.update(matches0)

    def match_by_metric(self, session:MatchingSession, detections:list[Detection], dist_cost:np.array) -> None:
        unmatched_track_idxes = session.unmatched_track_idxes
        unmatched_metric_det_idxes = session.unmatched_metric_det_idxes
        if unmatched_track_idxes and unmatched_metric_det_idxes:
            metric_cost = build_metric_cost(self.tracks, detections, unmatched_track_idxes, unmatched_metric_det_idxes)
            gated_metric_cost = gate_metric_cost(metric_cost, dist_cost, self.params.metric_gate_distance)
            metric_matcher = MetricCostMatcher(gated_metric_cost, self.params.metric_threshold, self.logger)
            matches0 = metric_matcher.match(unmatched_track_idxes, unmatched_metric_det_idxes)
            if matches0:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"(metric) all, metric: {matches_str(self.tracks, matches0)}")     
                session.update(matches0)

    def match_by_hungarian(self, session:MatchingSession, iou_cost:np.array, dist_cost:np.array) -> None:
        unmatched_track_idxes = session.unmatched_track_idxes
        unmatched_strong_det_idxes = session.unmatched_strong_det_idxes
        if unmatched_track_idxes and unmatched_strong_det_idxes:
            iou_last_matcher = HungarianMatcher(iou_cost, self.params.iou_dist_threshold_loose.iou, INVALID_IOU_DISTANCE)
            dist_last_matcher = HungarianMatcher(dist_cost, self.params.iou_dist_threshold_loose.distance, INVALID_DIST_DISTANCE)
            last_resort_matcher = chain(iou_last_matcher, dist_last_matcher)

            if unmatched_strong_det_idxes:
                matches0 = last_resort_matcher.match(unmatched_track_idxes, unmatched_strong_det_idxes)
                if matches0:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"5. (motion) all, strong, last_resort[{last_resort_matcher}]: "
                                    f"{matches_str(self.tracks, matches0)}")
                    session.update(matches0)


    def revise_matches(self, matcher:Matcher, session:MatchingSession, detection: Detection) -> None:
        # tentative track과 strong detection 사이의 match들을 검색한다.
        tent_matched_strong_det_idxes = [m[1] for m in session.matches \
                                                    if self.tracks[m[0]].is_tentative() \
                                                        and self.params.is_strong_detection(detection[m[1]])]
        strong_det_idxes = session.unmatched_strong_det_idxes + tent_matched_strong_det_idxes
        if not strong_det_idxes:
            return
        
        # weak detection들과 match된 non-tentative track들을 검색함
        track_idxes = [m[0] for m in session.matches \
                                    if not self.tracks[m[0]].is_tentative() \
                                        and not self.params.is_strong_detection(detection[m[1]])]
        # Matching되지 못했던 confirmed track들을 추가한다
        track_idxes += session.unmatched_confirmed_track_idxes
        if not track_idxes:
            return
        
        matches0 = matcher.match(track_idxes, strong_det_idxes)
        if matches0:
            deprived_track_idxes = []
            for match in matcher.match(track_idxes, strong_det_idxes):
                old_weak_match = session.find_match_by_track(match[0])
                old_strong_match = session.find_match_by_det(match[1])
                if old_weak_match:
                    session.pull_out(old_weak_match)
                    deprived_track_idxes.append(old_weak_match[0])
                if old_strong_match:
                    session.pull_out(old_strong_match)
                    deprived_track_idxes.append(old_strong_match[0])
                    
                session.update([match])
                deprived_track_idxes.remove(match[0])
                if self.logger.isEnabledFor(logging.DEBUG):
                    old_weak_str = [match_str(self.tracks, old_weak_match)] if old_weak_match else []
                    old_strong_str = [match_str(self.tracks, old_strong_match)] if old_strong_match else []
                    old_str = ', '.join(old_weak_str+old_strong_str)
                    self.logger.debug(f'rematch: {old_str} -> {match_str(self.tracks, match)}')
                          
            matches1 = matcher.match(deprived_track_idxes, session.unmatched_det_idxes)
            if matches1:
                session.update(matches1)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"rematch yielding tracks, {matcher}: {matches_str(self.tracks, matches1)}")
    
    def merge_fragment(self, session:MatchingSession, frame:Frame, track_events:list[NodeTrack]) -> set[DNATrack]:
        merged_tracks = set()
        
        # Stable zone에서 시작된 track들을 검색한다.
        if not (stable_zone_birth_tracks := self._collect_stable_zone_birth_tracks()):
            # Stable zone에서 시작된 track이 없는 경우 merge할 대상이 없기 때문에 바로 반환함.
            return merged_tracks
        
        # 모든 stable zone에 대해서, merge할 track이 있는가 확인하여 merge를 시도한다.
        for zid, _ in enumerate(self.params.stable_zones):
            # 각 stable zone에 temporarily lost된 track들을 검색한다.
            tl_tracks = self._find_tlost_stable_tracks(zid, session)
            # 해당 stable zone에서 시작된 track들을 검색한다.
            sh_tracks = stable_zone_birth_tracks[zid]
            
            # 여기서 tl_tracks은 'zid' 식별자를 갖는 stable zone에 temporarily lost된 track들이고,
            # sh_tracks은 'zid' 식별자를 갖는 stable zone에서 시작된 track들이다.
            # tl_tracks들과 sh_tracks들 사이의 feature를 기준으로한 cost matrix를 생성하여
            # feature간 distance를 활용하여 matching 시킨다.
            
            if tl_tracks and sh_tracks:
                metric_cost = self.build_metric_cost(tl_tracks, sh_tracks)

                matcher = ReciprocalCostMatcher(metric_cost, self.params.metric_threshold, name='take_over')
                matches = matcher.match(utils.all_indices(tl_tracks), utils.all_indices(sh_tracks))
                for t_idx, d_idx in matches:
                    stable_home_track = stable_zone_birth_tracks[zid][d_idx]
                    # match된 stable zone track을 제거하고, 대신 해당 track 정보를
                    # match된 temporarily lost된 track에 붙인다
                    tl_tracks[t_idx].take_over(stable_home_track, self.kf, frame, track_events)
                    merged_tracks.add(tl_tracks[t_idx])
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'track-take-over: {stable_home_track.id} -> {tl_tracks[t_idx].id}: '
                                            f'count={len(stable_home_track.detections)} stable_zone[{zid}]')
        return merged_tracks
        
    def _find_tlost_stable_tracks(self, zid:int, session:MatchingSession) -> list[DNATrack]:
        return [track for track in utils.get_items(self.tracks, session.unmatched_tlost_track_idxes) \
                            if track.archived_state.stable_zone == zid]

    def _collect_stable_zone_birth_tracks(self) -> dict[int,list[DNATrack]]:
        # Stable zone에서 시작된 track들을 검색한다.
        # 검색된 track들은 해당 stable zone에서 lost 중인 track 때문에 생성된 것일 수 있음.
        stable_zone_birth_tracks = defaultdict(list)
        for t_idx, track in enumerate(self.tracks):
            if ( (zid:=track.home_zone) >= 0                        # track의 처음 생성될 때 특정 stable zone 내에 위치하였고,
                and (track.is_confirmed() or track.is_tentative())  # 현재 상태가 Confirmed 또는 tentative 상태이고
                and track.stable_zone == zid ):                     # 현 track의 현재 위치가 아직 자신이 생성된 stable zone내인 경우
                    stable_zone_birth_tracks[zid].append(track)
        return stable_zone_birth_tracks

    def build_metric_cost(self, tl_tracks:list[DNATrack], sh_tracks:list[DNATrack]) -> np.ndarray:
        cost_matrix = np.ones((len(tl_tracks), len(sh_tracks)))
        for i, tl_track in enumerate(tl_tracks):
            start_index = tl_track.archived_state.frame_index
            if tl_track.features:
                for j, sh_track in enumerate(sh_tracks):
                    if sh_track.last_detection.feature is not None \
                        and sh_track.first_frame_index > start_index:
                        features = np.array([sh_track.last_detection.feature])
                        distances = utils.cosine_distance(tl_track.features, features)
                        cost_matrix[i, j] = distances.min(axis=0)
        return cost_matrix

    def draw_matched_detections(self, title:str, convas:Image, matches:list[tuple[int,int]], detections:list[Detection]):
        # for zone in self.tracker.params.blind_zones:
        #     convas = zone.draw(convas, list(zone.exterior.coords), color.YELLOW, 1)
        for zone in self.params.exit_zones:
            convas = zone.draw(convas, color.RED, 1)
        for zone in self.params.stable_zones:
            convas = zone.draw(convas, color.BLUE, 1)

        if matches:
            for t_idx, d_idx in matches:
                label = f'{d_idx}({self.tracks[t_idx].id})'
                det = detections[d_idx]
                if self.params.is_strong_detection(det):
                    convas = plot_utils.draw_label(convas, label, Point(det.bbox.br.astype(int)), color.WHITE, color.BLUE, 1)
                    convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
                else:
                    convas = plot_utils.draw_label(convas, label, Point(det.bbox.br.astype(int)), color.WHITE, color.RED, 1)
                    convas = det.bbox.draw(convas, color.RED, line_thickness=1)
        cv2.imshow(title, convas)
        cv2.waitKey(1)