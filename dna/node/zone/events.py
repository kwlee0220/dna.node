from __future__ import annotations

from typing import Union, Optional
from dataclasses import dataclass, replace, field

import shapely.geometry as geometry

from dna import Point, NodeId, TrackId, TrackletId
from dna.event import NodeTrack


@dataclass(frozen=True)
class LineTrack:
    node_id: NodeId
    track_id: TrackId
    line: geometry.LineString
    frame_index: int
    ts: float
    source: NodeTrack
    
    def is_point_track(self) -> bool:
        return self.line.coords[0] == self.line.coords[1]
    
    @property
    def begin_point(self) -> tuple[float,float]:
        return self.line.coords[0]
    
    @property
    def end_point(self) -> tuple[float,float]:
        return self.line.coords[1]
        
    @staticmethod
    def from_events(t0:NodeTrack, t1:NodeTrack):
        def to_line_string(pt0:Point, pt1:Point) -> geometry.LineString:
            return geometry.LineString([list(pt0.xy), list(pt1.xy)])

        p0 = t0.bbox.center()
        p1 = t1.bbox.center()
        return LineTrack(node_id=t1.node_id, track_id=t1.track_id, line=to_line_string(p0, p1),
                         frame_index=t1.frame_index, ts=t1.ts, source=t1)
    
    def __repr__(self) -> str:
        def to_line_end_points(ls:geometry.LineString) -> tuple[Point,Point]:
            return tuple(Point(xy) for xy in ls.coords[:2])
        
        if self.line:
            start, end = to_line_end_points(self.line)
            return f'{self.track_id}: line={start}-{end}, frame={self.frame_index}]'
        else:
            return f'{self.track_id}: frame={self.frame_index}]'


from datetime import timedelta
@dataclass
class ZoneVisit:
    zone_id: str
    enter_frame_index: int
    enter_ts: int
    leave_frame_index: int
    leave_ts: int
    
    @staticmethod
    def open(ev:NodeTrack) -> ZoneVisit:
        return ZoneVisit(zone_id=ev.zone_expr.zone_id, enter_frame_index=ev.frame_index, enter_ts=ev.ts,
                          leave_frame_index=-1, leave_ts=-1)
        
    def is_open(self) -> bool:
        return self.leave_ts <= 0
        
    def is_closed(self) -> bool:
        return self.leave_ts > 0
    
    def close_at_event(self, zev:NodeTrack) -> None:
        self.leave_frame_index = zev.frame_index
        self.leave_ts = zev.ts
    
    def close(self, frame_index:int, ts:float) -> None:
        self.leave_frame_index = frame_index
        self.leave_ts = ts
        
    def duplicate(self) -> ZoneVisit:
        return replace(self)
    
    def duration(self) -> timedelta:
        return timedelta(milliseconds=self.leave_ts - self.enter_ts) if self.is_closed() else None
    
    def __repr__(self) -> str:
        dur = self.duration()
        stay_str = f'{dur.seconds:.1f}s' if dur is not None else '?'
        leave_idx_str = self.leave_frame_index if self.leave_frame_index > 0 else '?'
        return f'{self.zone_id}[{self.enter_frame_index}-{leave_idx_str}:{stay_str}]'
    

class ZoneSequence:
    __slots__ = ( '_node_id', '_track_id', '_visits', '_frame_index', '_ts', '_closed' )

    def __init__(self, node_id:NodeId, track_id:TrackId, visits:list[ZoneVisit],
                 frame_index:int, ts:int, closed:bool=False) -> None:
        self._node_id = node_id
        self._track_id = track_id
        self._visits = visits
        self._frame_index = frame_index
        self._ts = ts
        self._closed = closed

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def tracklet_id(self) -> int:
        return TrackletId(self._node_id, self._track_id)

    @property
    def first_frame_index(self) -> int:
        if len(self._visits) > 0:
            return self._visits[0].enter_frame_index
        else:
            return -1

    @property
    def first_ts(self) -> int:
        if len(self._visits) > 0:
            return self._visits[0].enter_ts
        else:
            return -1

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def ts(self) -> int:
        return self._ts

    def is_closed(self) -> bool:
        return self._closed
        
    def update(self, frame_index:int, ts:int, closed:bool=False) -> None:
        self._frame_index = frame_index
        self._ts = ts
        self._closed = closed
    
    def __getitem__(self, index) -> ZoneVisit|ZoneSequence:
        if isinstance(index, int):
            return self._visits[index]
        else:
            range = self._visits[index]
            return ZoneSequence(node_id=self._node_id, track_id=self._track_id, visits=range[:],
                                frame_index=self._frame_index, ts=self._ts, closed=self._closed)
    
    def __len__(self) -> int:
        return len(self._visits)
    
    def __delitem__(self, idx) -> None:
        del self._visits[idx]
        
    def __iter__(self):
        return iter(self._visits)
    
    def append(self, visit:ZoneVisit) -> None:
        self._visits.append(visit)
        
    def remove(self, idx:int) -> None:
        self._visits.remove(idx)
        
    def duplicate(self) -> ZoneSequence:
        return ZoneSequence(node_id=self._node_id, track_id=self._track_id, visits=self._visits[:],
                            frame_index=self._frame_index, ts=self._ts, closed=self._closed)
    
    def sequence_str(self) -> str:
        seq_str = ''.join([visit.zone_id for visit in self._visits])
        return f'[{seq_str}]' if len(self._visits) == 0 or self[-1].is_closed() else f'[{seq_str})'

    def __repr__(self) -> str:
        closed_str = ", closed" if self.is_closed() else ''
        return f'{self._track_id}:{self.sequence_str()}, frame={self.frame_index}{closed_str}'
    
    
