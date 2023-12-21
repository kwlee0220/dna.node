from __future__ import annotations

from typing import Optional, DefaultDict
import logging
from collections import defaultdict
import itertools

import numpy as np

from dna import utils, Size2d, TrackId
from dna.support import iterables
from dna.camera import Frame, FrameReader, ImageProcessor, ImageCapture
from dna.event import NodeTrack, TrackFeature, EventProcessor
from dna.track import TrackState
from dna.track.feature_extractor import MetricExtractor


class PublishReIDFeatures(FrameReader,EventProcessor):
    MAX_FRAME_BUFFER_LENGTH = 80
    REPRESENTATIVE_COUNT = 5
    
    def __init__(self, extractor:MetricExtractor,
                 distinct_distance:float, min_crop_size:Size2d, max_iou:float,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        FrameReader.__init__(self)
        EventProcessor.__init__(self)
        
        self.extractor = extractor
        self.distinct_distance = distinct_distance
        self.min_crop_size = min_crop_size
        self.max_iou = max_iou
        
        # ImageProcessor에 의해 전달되는 Frame 객체를 임시로 저장할 field.
        # ImageProcessor가 제공하는 Frame과 event processing에서 제공되는 해당 event들이
        # 동기화되서 도착되는 것이 아니기 때문에 일정 기간동안 frame을 저장할 필요가 있음.
        self.frame_buffer:list[Frame] = []
        
        self.pending_reid_tracks:list[tuple[int, list[NodeTrack]]] = []
        self.representives:DefaultDict[TrackId,list[np.ndarray]] = defaultdict(list)
        self.logger = logger
    
    def handle_event(self, candidate_tracks:list[NodeTrack]) -> None:
        # 입력된 node-track들 중에서 선택해서 feature를 생성하고 이를 kafka로 publish한다.
        # 선정 방법은
        #   - Track의 상태가 Confirm이거나 Tentative인 상태. 즉 track에 해당하는 물체의 bbox가
        #       의미가 있는 값인 경우.
        #   - Bounding-box의 크기가 일정 크기 이상인 경우. 즉 검출된 물체의 크기가 너무 작은 bbox에서
        #       추출한 feature는 noise가 될 수 있기 때문임.
        # Frame_buffer에 해당 frame 존재 여부와 무관하게 'delete'된 track 처리를 수행함.
        groups = iterables.groupby(candidate_tracks, key_func=lambda g: g.state, init_dict=defaultdict(list))
        
        # 검출된 물체의 bbox가 너무 많이 겹치는 경우, 해당 영역에서 feature를 검출하는 경우
        # noise가 많을 수 있기 때문에 feature 추출에서 제외시킨다.
        # candidates:list[NodeTrack] = self.remove_overlaps(confirmed_tracks + tentative_tracks)
        candidate_tracks = self.remove_overlaps(groups[TrackState.Confirmed] + groups[TrackState.Tentative])
        if len(candidate_tracks) > 0:
            reid_tracks:list[NodeTrack] = []
            for track in candidate_tracks:
                # ReID feature의 안정성 향상을 위해 detection box의 크기가
                # 일정 크기 이상인 것만 reid feature를 추출한다.
                if track.detection_box.size() >= self.min_crop_size:
                    reid_tracks.append(track)
                else:
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"drop a small boxed feature: track={track.track_id}, {track.detection_box.size()}")
                
            frame_index = candidate_tracks[0].frame_index  # group에 속한 모든 track들은 동일 frame에서 검출된 것이기 때문에 첫번째 track의 frame 번호를 사용한다.
            self.pending_reid_tracks.append((frame_index, reid_tracks))
            while self.pending_reid_tracks:
                frame_index, reid_tracks = self.pending_reid_tracks.pop(0)
                frame = self.pop_frame(frame_index)
                if frame and reid_tracks:
                    self._publish_reid_features(reid_tracks, frame)
                
        # 따로 뽑아둔 deleted track을 publish 시킨다.
        for track in groups[TrackState.Deleted]:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"publish closed feature: track={track.track_id}")
            self._publish_event(TrackFeature(node_id=track.node_id, track_id=track.track_id,
                                             feature=None, frame_index=track.frame_index, ts=track.ts))
        
    def remove_overlaps(self, group:list[NodeTrack]):
        if len(group) >= 2:
            overlappeds:set[NodeTrack] = set()
            for t1, t2 in itertools.combinations(enumerate(group), 2):
                iou = t1[1].detection_box.iou(t2[1].detection_box)
                if iou > self.max_iou:
                    if self.logger and self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(f"drop an overlapped features: track1={t1[1].track_id}, track2={t2[1].track_id}, iou={iou:.2f}")
                    overlappeds.add(t1[1].track_id)
                    overlappeds.add(t2[1].track_id)
            iterables.remove_if(group, cond=lambda t: t.track_id in overlappeds)
        return group
            
    def _publish_reid_features(self, reid_tracks:list[NodeTrack], frame:Frame) -> None:
        def to_feature(track:NodeTrack, feature:np.ndarray) -> TrackFeature:
            return TrackFeature(node_id=track.node_id, track_id=track.track_id, feature=feature,
                                frame_index=track.frame_index, ts=track.ts)
            
        boxes = [track.detection_box for track in reid_tracks]
        for track, feature in zip(reid_tracks, self.extractor.extract_boxes(frame.image, boxes)):
            done, dist = self._register_to_representives(track.track_id, feature)
            if done:
                if self.logger and self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"publish a feature: track={track.track_id}, dist={dist:.2f}, size={track.detection_box.size()}")
                self._publish_event(to_feature(track, feature))
            else:
                # 이전 feature와 similarity가 높은 경우는 feature 생성을 skip한다.
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"skip a feature due to a similar one: track={track.track_id}, dist={dist:.2f}")
        
    def _register_to_representives(self, track_id:TrackId, feature:np.ndarray) -> tuple[bool, float]:
        min_dist = 1
        repr = self.representives[track_id]
        for feat in repr:
            dist = self.extractor.distance(feature, feat)
            # print(f'track={track_id}: dist={dist:.3f}')
            if dist < self.distinct_distance:
                return False, dist
            elif dist < min_dist:
                min_dist = dist
        
        repr.append(feature)
        if len(repr) > PublishReIDFeatures.REPRESENTATIVE_COUNT:
            repr.pop(0)
        return True, min_dist
              
    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None: pass
    def close(self) -> None: pass

    def read(self, frame:Frame) -> None:
        self.frame_buffer.append(frame)
        
        # 너무 많은 수의 frame이 buffering되어 일정 길이 이상이 되는 경우, 오래된 frame부터 삭제시킨다.
        if len(self.frame_buffer) > PublishReIDFeatures.MAX_FRAME_BUFFER_LENGTH:
            self.frame_buffer.pop(0)
        
    def pop_frame(self, frame_index:int) -> Optional[Frame]:
        if not self.frame_buffer or frame_index < self.frame_buffer[0].index:
            # frame이 계속 생성되는 시점에서 물체가 오랫동안 검출되지 않는 경우
            # 이벤트가 생성되지 않는 상태에서 frame이 계속 쌓이게 되어 buffer overflow로 다수의 frame들이
            # victim으로 선택되어 버려지는 경우 발생됨. 결론적으로 큰 문제는 아님.
            self.logger.warn(f"Target frame has been purged already: frame={frame_index}, {self}")
            return None
        
        idx, frame = iterables.find_first(self.frame_buffer, lambda f: frame_index <= f.index)
        if idx > 1:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'throw away unused frames: count={idx-1}, '
                                  f'frame_indice=[{self.frame_buffer[0].index}..{self.frame_buffer[idx-1].index}]')
        if frame.index == frame_index:
            self.frame_buffer = self.frame_buffer[idx+1:]
            return frame
        else:
            self.logger.warn(f"Cannot find an appropriate frame, somthing wrong!: index={frame_index}")
            self.frame_buffer = self.frame_buffer[idx:]
            return None
        
    def __repr__(self) -> str:
        frames_str = f', frame_buffer[{self.frame_buffer[0].index}:{self.frame_buffer[-1].index}]' if self.frame_buffer else ""
        return f'{self.__class__.__name__}[min_det_size={self.min_crop_size}{frames_str}, tracks={list(self.representives.keys())}]'