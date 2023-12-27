from __future__ import annotations

from typing import Optional, Union, Iterable, Callable, Any

import sys

from omegaconf.dictconfig import DictConfig
import numpy as np

from dna import Point
from dna.event import EventNodeImpl, KafkaEvent
from .types import SilentFrame
from .node_track import NodeTrack


ALPHA = 1.0   # smoothing hyper parameter
            # ALPHA가 강할 수록 smoothing을 더 강하게 진행 (기존 location 정보를 잃을 수도 있음)

def smoothing_track(track, alpha=ALPHA):
    """

    Parameters
    ----------
    track: 입력 받은 track 정보 (location 정보)
    alpha: smoothing hyper parameter

    Returns: stabilization을 완료한 track location 정보를 반환
    -------

    """
    l = len(track)
    ll = l * 3 - 3  # l + (l - 1) + (l - 2) matrix size
    A = np.zeros((ll, l))
    A[:l, :] = np.eye(l)
    A[l:l * 2 - 1, :] = alpha * (np.eye(l) - np.eye(l, k=1))[:l - 1, :]  # l Plot1
    A[l * 2 - 1:, :] = alpha * (2 * np.eye(l) - np.eye(l, k=1) - np.eye(l, k=-1))[1:l - 1, :]  # l - 2

    b = np.zeros((1, ll))
    b[:, :l] = track

    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b.T)
    X = np.dot(np.linalg.inv(ATA), ATb)
    return X


def stabilization_location(location, frame=5, alpha=ALPHA) -> list[float]:
    """
    Parameters
    ----------
    location: trajectory의 위치 정보
    frame: 앞 뒤로 몇 프레임까지 볼 것인지.

    Returns: 안정화된 위치 정보
    -------
    """
    stab_location = []
    coord_length = len(location)
    for idx, coord in enumerate(location):
        if idx < frame and idx + frame < coord_length:
            # len(prev information) < frame
            # 과거 정보가 부족한 경우
            prev_locations = location[:idx + 1]  # prev location + current location
            frame_ = len(prev_locations)
            next_locations = location[idx + 1:idx + 1 + frame_]
            smoothing_coord = smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=alpha)[idx][0]
        elif idx < coord_length - frame and idx - frame >= 0:
            # len(next information) >= frame and len(prev information) >= frame
            prev_locations = location[idx - frame:idx + 1]  # prev location + current location
            next_locations = location[idx + 1:idx + 1 + frame]
            smoothing_coord = smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=1)[frame][0]
            # 과거 정보, 미래 정보 모두 있는 경우
        elif idx - frame >= 0:
            # len(next information) < frame
            # 미래 정보가 부족한 경우
            next_locations = location[idx + 1:]
            frame_ = len(next_locations)
            prev_locations = location[idx - frame_:idx + 1]  # prev location + current location
            if len(np.concatenate([prev_locations, next_locations])) == 1:
                smoothing_coord = location[idx]
            else:
                smoothing_coord = \
                smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=alpha)[-(frame_ + 1)][0]
        else:
            # Short frame
            # parameter로 받은 location 정보 자체가 짧은 경우
            if len(location[:idx + 1]) == 1:
                smoothing_coord = location[idx]
            else:
                smoothing_coord = smoothing_track(location[:idx + 1], alpha=alpha)[-1][0]
        stab_location.append(smoothing_coord)
    return stab_location


_MAX_FRAME_INDEX = sys.maxsize

class SmoothingSession:
    def __init__(self, look_ahead:int, smoothing_factor:float) -> None:
        self.stabilizer = PointStabilizer(look_ahead=look_ahead, smoothing_factor=smoothing_factor)
        self.pending_events:list[NodeTrack] = []
        
    def close(self) -> list[NodeTrack]:
        return [ev.updated(location=smoothed_pt)
                    for ev, smoothed_pt in zip(self.pending_events, self.stabilizer.flush())]
    
    def smooth(self, ev:NodeTrack) -> Optional[NodeTrack]:
        self.pending_events.append(ev)
        assert ev.location
        smoothed_pt = self.stabilizer.perform(ev.location)
        if smoothed_pt is not None:
            ev = self.pending_events.pop(0)
            return ev.updated(location=smoothed_pt)
        else:
            return None
        
    def min_frame_index(self) -> int:
        return self.pending_events[0].frame_index if self.pending_events else _MAX_FRAME_INDEX
    
    def __repr__(self) -> str:
        return repr(self.stabilizer)


_DEFAULT_SMOOTHING_FACTOR = 1
class TrackletSmoothProcessor(EventNodeImpl):
    def __init__(self, conf:DictConfig) -> None:
        super().__init__()
        
        self.look_ahead = conf.look_ahead
        self.smoothing_factor = conf.get("smoothing_factor", _DEFAULT_SMOOTHING_FACTOR)
        self.sessions:dict[str,SmoothingSession] = dict()
        self.last_frame_index = -1

    def on_completed(self) -> None:
        import itertools
        for ev in itertools.chain(session.close() for session in self.sessions.values()):
            self.publish_event(ev)
            
        super().on_completed()
        
    def min_frame_index(self) -> int:
        min_index = min((session.min_frame_index() for session in self.sessions.values()), default=_MAX_FRAME_INDEX)
        return min_index if min_index < _MAX_FRAME_INDEX else self.last_frame_index
    
    def handle_event(self, ev:NodeTrack|SilentFrame) -> None:
        self.last_frame_index = ev.frame_index
        if isinstance(ev, NodeTrack):
            session = self.sessions.get(ev.track_id, None)
            if session is None:
                session = SmoothingSession(self.look_ahead, self.smoothing_factor)
                self.sessions[ev.track_id] = session
                
            if ev.is_deleted():
                del self.sessions[ev.track_id]
                for ev in session.close():
                    self.publish_event(ev)
            else:
                smoothed = session.smooth(ev)
                if smoothed:
                    self.publish_event(smoothed)
        elif isinstance(ev, SilentFrame):
            assert len(self.sessions) == 0
            self.publish_event(ev)
        else:
            raise ValueError(f'unexpected event: {ev}')
    
    def __repr__(self) -> str:
        return f"Stabilizer(look_ahead={self.look_ahead}, smoothing_factor={self.smoothing_factor})"

def stabilize(points:Iterable[Point],
              *,
              look_ahead:int=5,
              alpha:float=ALPHA
              ) -> list[Point]:
    xs:list[float] = []
    ys:list[float] = []
    for pt in points:
        xs.append(pt.x)
        ys.append(pt.y)
    xs = stabilization_location(xs, look_ahead, alpha=alpha)
    ys = stabilization_location(ys, look_ahead, alpha=alpha)
    return [Point((x,y)) for x,y in zip(xs, ys)]


class PointStabilizer:
    def __init__(self,
                 look_ahead:int,
                 smoothing_factor:float=_DEFAULT_SMOOTHING_FACTOR) -> None:
        super().__init__()
        
        self.look_ahead = look_ahead
        self.alpha = smoothing_factor
        
        self.current, self.end = 0, 0
        self.pendings_x: list[float] = []
        self.pendings_y: list[float] = []

    def flush(self) -> list[Point]:
        xs = stabilization_location(self.pendings_x, self.look_ahead)
        ys = stabilization_location(self.pendings_y, self.look_ahead)
        
        return [Point((x, y)) for x, y in zip(xs[self.current:], ys[self.current:])]
        
    def perform(self, pt:Point) -> Optional[Point]:
        x, y = tuple(pt)
        self.pendings_x.append(x)
        self.pendings_y.append(y)

        if len(self.pendings_x) - self.current > self.look_ahead:
            xs = stabilization_location(self.pendings_x, self.look_ahead)
            ys = stabilization_location(self.pendings_y, self.look_ahead)
            smoothed = Point((xs[self.current], ys[self.current]))

            if self.current >= self.look_ahead:
                self.pendings_x = self.pendings_x[1:]
                self.pendings_y = self.pendings_y[1:]
            else:
                self.current += 1
                
            return smoothed
        else:
            return None
    
    def __repr__(self) -> str:
        return (f'index={self.current}:{len(self.pendings_x)}, '
                f'look_ahead={self.look_ahead}, factor={self.alpha}')

