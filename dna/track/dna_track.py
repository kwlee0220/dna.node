from __future__ import annotations
from typing import Union

import logging
import numpy as np
import cv2

import dna
from dna import Box, Size2d, Image, BGR, Point
from dna.camera import Frame
from dna.detect import Detection
from dna.support import plot_utils
from dna.track import TrackState
from dna.track.types import ObjectTrack
from .kalman_filter import KalmanFilter
from .dna_track_params import DNATrackParams
from dna.event.node_track import NodeTrack


def to_tlbr(xyah:np.ndarray) -> Box:
    ret = xyah.copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    ret[2:] = ret[:2] + ret[2:]

    return ret

def to_box(tlbr:np.ndarray) -> Box:
    box = Box(tlbr)
    if not box.is_valid():
        tl = Point(box.tl)
        br = tl + Size2d([0.00001, 0.00001])
        box = Box.from_points(tl, br)
    return box


from dataclasses import dataclass, field
@dataclass(frozen=True, eq=True)    # slots=True
class DNATrackState:
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    time_since_update: int
    stable_zone: int
    detections: list[Detection]
    features: list[np.ndarray]
    frame_index: int
    timestamp: float


_UNKNOWN_ZONE_ID = -2

class DNATrack(ObjectTrack):
    def __init__(self, mean, covariance, track_id:str, frame_index:int, ts:float,
                    params:DNATrackParams, detection:Detection, *, logger:logging.Logger) -> None:
        super().__init__(id=track_id, state=TrackState.Tentative, location=to_box(to_tlbr(mean[:4])),
                        frame_index=frame_index, timestamp=ts)

        self.mean = mean
        self.covariance = covariance
        self.hits = 1
        self.time_since_update = 0
        self.time_to_promote = params.n_init - 1
        self.params = params

        self.archived_state = None

        self.detections = [detection]
        self.features = []
        if detection.feature is not None:
            self.features.append(detection.feature)
        self.home_zone = params.find_stable_zone(detection.bbox)

        self.n_init = params.n_init
        self.max_age = params.max_age
        self.__exit_zone = _UNKNOWN_ZONE_ID
        self.__stable_zone = self.home_zone
        self.logger = logger

    @property
    def age(self) -> int:
        return len(self.detections)
    
    @property
    def last_frame_index(self) -> int:
        return self.first_frame_index + len(self.detections) - 1

    @property
    def last_detection(self) -> Detection:
        return self.detections[-1]

    @property
    def exit_zone(self) -> int:
        if self.__exit_zone == _UNKNOWN_ZONE_ID:
            self.__exit_zone = self.params.find_exit_zone(self.location)
        return self.__exit_zone

    @property
    def stable_zone(self) -> int:
        if self.__stable_zone == _UNKNOWN_ZONE_ID:
            self.__stable_zone = self.params.find_stable_zone(self.location)
        return self.__stable_zone
        
    def predict(self, kf:KalmanFilter, frame_index:int, ts:float) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.location = to_box(to_tlbr(self.mean[:4]))
        self.__exit_zone = _UNKNOWN_ZONE_ID
        self.__stable_zone = _UNKNOWN_ZONE_ID
        self.time_since_update += 1
        self.frame_index = frame_index
        self.timestamp = ts

    def update(self, kf:KalmanFilter, frame:Frame, det:Detection) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, det.bbox.xyah)
        self.location = to_box(to_tlbr(self.mean[:4]))
        self.__exit_zone = _UNKNOWN_ZONE_ID
        self.__stable_zone = _UNKNOWN_ZONE_ID
        
        self.detections.append(det)
        if det and self.params.is_metric_detection_for_registry(det):
            self.features.append(det.feature)
            if len(self.features) > self.params.max_feature_count:
                self.features = self.features[-self.params.max_feature_count:]
        self.hits += 1
        self.time_since_update = 0
        self.archived_state = None

        if self.state == TrackState.Tentative:
            if det is None:
                self.mark_deleted()
            elif self.params.is_strong_detection(det):
                self.time_to_promote -= 1
                if self.time_to_promote <= 0:
                    self.state = TrackState.Confirmed
            else:
                if self.hits - (self.n_init-self.time_to_promote) > 2:
                    self.mark_deleted()
        elif self.state == TrackState.TemporarilyLost:
            self.state = TrackState.Confirmed
                
    def mark_missed(self, frame:Frame) -> None:
        self.frame_index = frame.index
        self.timestamp = frame.ts
        self.detections.append(None)
        
        if self.state == TrackState.Tentative:
            self.mark_deleted()
        elif self.exit_zone >= 0:
            # track의 위치가 exit-zone에 위치한 경우 바로 delete시킨다.
            self.mark_deleted()
        else:   # Confirmed, TemporarilyLost, Deleted
            if self.state != TrackState.TemporarilyLost:
                self.state = TrackState.TemporarilyLost
                self.archived_state = DNATrackState(self.mean, self.covariance, self.hits, self.time_since_update,
                                                    self.stable_zone, self.detections.copy(), self.features.copy(),
                                                    self.frame_index, self.timestamp)
            max_lost_age = self.max_age
            if self.archived_state.stable_zone >= 0:
                max_lost_age *= 4
            if self.time_since_update > max_lost_age:
                self.mark_deleted()

    def mark_deleted(self) -> None:
        self.state = TrackState.Deleted

    def to_event(self) -> NodeTrack:
        d_box = d.bbox if (d := self.detections[-1]) else None
        return NodeTrack(node_id=None, track_id=str(self.id), state=self.state, bbox=self.location,
                         first_ts=self.first_timestamp, frame_index=self.frame_index, ts=self.timestamp, detection_box=d_box)
        
    @property
    def state_str(self) -> str:
        if self.state == TrackState.Confirmed:
            return f'{self.id}(C)'
        elif self.state == TrackState.TemporarilyLost:
            return f'{self.id}({self.time_since_update})'
        elif self.state == TrackState.Tentative:
            return f'{self.id}(-{self.time_to_promote})'
        elif self.state == TrackState.Deleted:
            return f'{self.id}(D)'
        else:
            raise ValueError("Shold not be here")
        
    def __repr__(self) -> str:
        interval_str = ""
        if len(self.detections):
            interval_str = f':{self.first_frame_index}-{self.last_frame_index}'

        return (f'{self.id}({self.state.abbr})[{len(self.detections)}{interval_str}]({self.time_since_update}), '
                f'nfeats={len(self.features)}, frame={self.frame_index}')

    def take_over(self, victim_track:DNATrack, kf:KalmanFilter, frame:Frame, track_events:list[NodeTrack]) -> None:
        archived_state = self.archived_state
        
        if self.logger.isEnabledFor(logging.INFO):
            last_frame_index = victim_track.last_frame_index
            self.logger.info(f'taking over other track: track={self.id}[{archived_state.frame_index+1}-{last_frame_index}], '
                             f'victim={victim_track.id}[{victim_track.first_frame_index}-{victim_track.last_frame_index}]')

        self.mean = archived_state.mean
        self.covariance = archived_state.covariance
        self.time_since_update = archived_state.time_since_update
        self.hits = archived_state.hits
        self.detections = archived_state.detections
        self.archived_state = None

        # Take-over할 track의 첫 detection 전까지는 추정된 위치를 사용한다.
        for i in range(archived_state.frame_index+1, victim_track.first_frame_index):
            self.predict(kf, frame_index=i, ts=-1)
            self.detections.append(None)

        # Take-over할 track에게 할당된 detection으로 본 track의 위치를 재조정한다.
        replay_frame = Frame(frame.image, victim_track.first_frame_index, victim_track.first_timestamp)
        for det in victim_track.detections:
            self.predict(kf, replay_frame.index, replay_frame.ts)
            if det:
                self.update(kf, replay_frame, det)
            else:
                self.frame_index = replay_frame.index
                self.timestamp = replay_frame.ts
                self.detections.append(None)
            track_events.append(self.to_event())
            replay_frame = Frame(frame.image, replay_frame.index+1, victim_track.first_timestamp)

        victim_track.mark_deleted()