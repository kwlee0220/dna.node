from __future__ import annotations

from typing import Optional
import logging

from dna import TrackId
from dna.event import SilentFrame, EventProcessor, NodeTrack
from .types import ZoneRelation


class ZoneEventRefiner(EventProcessor):
    __slots__ = ('locations', 'logger')

    def __init__(self, *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)

        self.locations:dict[TrackId,str] = dict()
        self.logger = logger
                
    def handle_event(self, track_ev:NodeTrack) -> None:
        if isinstance(track_ev, NodeTrack):
            if track_ev.is_deleted():
                # 삭제된 track의 location 정보를 삭제한다
                self.locations.pop(track_ev.track_id, None)
                # TrackEvent 이벤트를 re-publish한다
                self._publish_event(track_ev)
            else:
                self.handle_track_event(track_ev)
        if isinstance(track_ev, SilentFrame):
            self._publish_event(track_ev)
        else:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'unknown event: {track_ev}')
        
    def handle_track_event(self, track_ev:NodeTrack) -> None:
        current_zone = self.locations.get(track_ev.track_id)
        
        zone_expr = track_ev.zone_expr
        if zone_expr.relation == ZoneRelation.Unassigned:
            if current_zone:
                raise ValueError(f'invalid zone: expected={zone_expr.zone_id}, actual={current_zone}')
            self._publish_event(track_ev)
        elif zone_expr.relation == ZoneRelation.Left:
            # 추적 물체가 해당 zone에 포함되지 않은 상태면, 먼저 해당 물체를 zone 안에 넣는 event를 추가한다.
            if current_zone:
                if current_zone != zone_expr.zone_id:
                    raise ValueError(f'invalid zone: expected={zone_expr.zone_id}, actual={current_zone}')
                self.locations.pop(track_ev.track_id, None)
                self._publish_event(track_ev)
            else:
                if self.logger and self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'ignore a LEFT(track={track_ev.track_id}, zone={zone_expr.zone_id}, frame={track_ev.frame_index})')
        elif zone_expr.relation == ZoneRelation.Entered:
            if not current_zone:
                self.locations[track_ev.track_id] = track_ev.zone_expr.zone_id
                self._publish_event(track_ev)
            else:
                if self.logger and self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'ignore a ENTERED(track={track_ev.track_id}, zone={zone_expr.zone_id}, frame={track_ev.frame_index})')
        elif zone_expr.relation == ZoneRelation.Inside:
            if current_zone != zone_expr.zone_id:
                raise ValueError(f'incompatible zone: event({zone_expr.zone_id}) <-> managed({current_zone})')
            self._publish_event(track_ev)
        elif zone_expr.relation == ZoneRelation.Through:
            if current_zone:
                raise ValueError(f'invalid zone: expected=None, actual={current_zone}')
            self._publish_event(track_ev)
        else:
            raise ValueError(f'invalid NodeTrack event: {track_ev}')
        
    def __repr__(self) -> str:
        return f"RefineZoneEvents"
    