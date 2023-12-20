from __future__ import annotations

from typing import Optional
import logging

from dna import TrackId
from dna.event import SilentFrame, NodeTrack, EventProcessor
from .events import LineTrack


class ToLineTransform(EventProcessor):
    __slots__ = ( 'last_events', 'logger' )
    
    def __init__(self, *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)
        self.last_events:dict[TrackId,NodeTrack] = dict()
        self.logger = logger

    def close(self) -> None:
        self.last_events.clear()
        super().close()

    def handle_event(self, ev:object) -> None:
        if isinstance(ev, NodeTrack):
            self.handle_track_event(ev)
        elif isinstance(ev, SilentFrame):
            self.last_events.clear()
            self._publish_event(ev)
        else:
            self._publish_event(ev)

    def handle_track_event(self, ev:NodeTrack) -> None:
        if ev.is_deleted():
            self.last_events.pop(ev.track_id, None)
            self._publish_event(ev)
        else:           
            # track의 첫번재 이벤트인 경우는 last_event가 ev(자기 자신)이 됨.
            last_event = self.last_events.get(ev.track_id, ev)
            self._publish_event(LineTrack.from_events(last_event, ev))
            self.last_events[ev.track_id] = ev
            
    def __repr__(self) -> str:
        return f"ToLine"