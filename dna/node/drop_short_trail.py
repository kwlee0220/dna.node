from __future__ import annotations

from typing import Union, Optional
import sys
import logging
from collections import defaultdict

from dna import TrackId
from dna.event import EventNodeImpl
from .types import SilentFrame
from .node_track import NodeTrack
from dna.track import TrackState


class DropShortTrail(EventNodeImpl):
    __slots__ = 'min_trail_length', 'long_trails', 'pending_dict', 'logger'

    def __init__(self, min_trail_length:int, *, logger:Optional[logging.Logger]=None) -> None:
        EventNodeImpl.__init__(self)

        self.min_trail_length = min_trail_length
        self.long_trails: set[TrackId] = set()  # 'long trail' 여부
        self.pending_dict: dict[TrackId, list[NodeTrack]] = defaultdict(list)
        self.max_frame_index = -1
        self.__min_frame_index = -1
        self.logger = logger

    def on_completed(self) -> None:
        super().on_completed()
        self.pending_dict.clear()
        self.long_trails.clear()

    def handle_event(self, ev:NodeTrack|SilentFrame) -> None:
        if isinstance(ev, NodeTrack):
            self.handle_track_event(ev)
        elif isinstance(ev, SilentFrame):
            assert len(self.pending_dict) == 0
            self.publish_event(ev)
        else:
            raise AssertionError(f"unexpected event: {ev}")

    def handle_track_event(self, ev:NodeTrack) -> None:
        self.max_frame_index = max(self.max_frame_index, ev.frame_index)
        
        is_long_trail = ev.track_id in self.long_trails
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            if is_long_trail:
                self.long_trails.discard(ev.track_id)
                self.publish_event(ev)
            else:
                pendings = self.pending_dict.pop(ev.track_id, [])
                self.invalidate_min_frame_index(pendings)
                if pendings and self.logger and self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"drop short track events: track_id={ev.track_id}, length={len(pendings)}")
        elif is_long_trail:
            self.publish_event(ev)
        else:
            pendings = self.pending_dict[ev.track_id]
            pendings.append(ev)

            # pending된 event의 수가 threshold (min_trail_length) 이상이면 long-trail으로 설정하고,
            # 더 이상 pending하지 않고, 바로 publish 시킨다.
            if len(pendings) >= self.min_trail_length:
                # 'pending_dict'에서 track을 제거하기 전에 pending event를 publish 해야 한다.
                self.__publish_pendings(pendings)
                self.long_trails.add(ev.track_id)
                self.pending_dict.pop(ev.track_id, None)
                self.invalidate_min_frame_index(pendings)

    def __publish_pendings(self, pendings:list[NodeTrack]) -> None:
        for pev in pendings:
            self.publish_event(pev)
            
    def invalidate_min_frame_index(self, pendings:list[NodeTrack]) -> None:
        if pendings[0].frame_index <= self.__min_frame_index:
            self.__min_frame_index = -1
        
    def min_frame_index(self) -> int:
        if self.__min_frame_index < 0:
            self.__min_frame_index = min((tracks[0].frame_index for tracks in self.pending_dict.values()),
                                         default=self.max_frame_index)
        return self.__min_frame_index
    
    def __repr__(self) -> str:
        return f"DropShortTrail(min_trail_length={self.min_trail_length})"