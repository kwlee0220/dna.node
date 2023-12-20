from __future__ import annotations

from typing import Union, Optional
import sys
import logging
from collections import defaultdict

from dna import TrackId
from dna.event import TimeElapsed, SilentFrame, NodeTrack, EventProcessor
from dna.track import TrackState


class DropShortTrail(EventProcessor):
    __slots__ = 'min_trail_length', 'long_trails', 'pending_dict', 'logger'

    def __init__(self, min_trail_length:int, *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)

        self.min_trail_length = min_trail_length
        self.long_trails: set[TrackId] = set()  # 'long trail' 여부
        self.pending_dict: dict[TrackId, list[NodeTrack]] = defaultdict(list)
        self.logger = logger

    def close(self) -> None:
        super().close()
        self.pending_dict.clear()
        self.long_trails.clear()
        
    def min_frame_index(self) -> int:
        return min(ev_list[0].frame_index for ev_list in self.pending_dict.values()) if self.pending_dict else None

    def handle_event(self, ev:Union[NodeTrack,TimeElapsed,SilentFrame]) -> None:
        if isinstance(ev, NodeTrack):
            self.handle_track_event(ev)
        elif isinstance(ev, SilentFrame):
            assert len(self.pending_dict) == 0
            self._publish_event(ev)
        else:
            self._publish_event(ev)

    def handle_track_event(self, ev:NodeTrack) -> None:
        is_long_trail = ev.track_id in self.long_trails
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            if is_long_trail:
                self.long_trails.discard(ev.track_id)
                self._publish_event(ev)
            else:
                pendings = self.pending_dict.pop(ev.track_id, [])
                if pendings and self.logger and self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"drop short track events: track_id={ev.track_id}, length={len(pendings)}")
        elif is_long_trail:
            self._publish_event(ev)
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

    def __publish_pendings(self, pendings:list[NodeTrack]) -> None:
        for pev in pendings:
            self._publish_event(pev)
    
    def __repr__(self) -> str:
        return f"DropShortTrail(min_trail_length={self.min_trail_length})"