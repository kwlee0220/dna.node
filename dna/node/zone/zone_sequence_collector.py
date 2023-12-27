from __future__ import annotations

from typing import Union
import logging

from dna.event import EventNodeImpl
from ..types import SilentFrame
from ..node_track import NodeTrack
from .events import ZoneVisit, ZoneSequence

LOGGER = logging.getLogger('dna.node.zone.Turn')


class ZoneSequenceCollector(EventNodeImpl):
    def __init__(self) -> None:
        super().__init__()
        
        self.sequences:dict[str,ZoneSequence] = dict()
    
    def close(self) -> None:
        self.sequences.clear()
        super().close()

    def handle_event(self, ev:NodeTrack|SilentFrame) -> None:
        if isinstance(ev, NodeTrack):
            zseq = self.sequences.get(ev.track_id)
            if zseq:
                zseq.update(ev.frame_index, ev.ts, closed=ev.is_deleted())
                
            if ev.is_deleted():
                if zseq:
                    self.publish_event(zseq.duplicate())
            else:
                self._handle_zone_event(ev)
        else:
            pass
         
            
    def _handle_zone_event(self, ev:NodeTrack) -> None:
        if ev.zone_expr.is_inside() or ev.zone_expr.is_unassigned():
            return
        
        zseq = self.sequences.get(ev.track_id)
        if zseq is None:
            zseq = ZoneSequence(node_id=ev.node_id, track_id=ev.track_id, visits=[],
                                frame_index=ev.frame_index, ts=ev.ts)
            self.sequences[ev.track_id] = zseq
            
        if ev.zone_expr.is_entered():
            zseq.append(ZoneVisit.open(ev))
        elif ev.zone_expr.is_left():
            last:ZoneVisit = zseq[-1]
            assert last.is_open()
            last.close(frame_index=ev.frame_index, ts=ev.ts)
        elif ev.zone_expr.is_through():
            last = zseq[-1] if len(zseq) > 0 else None
            assert last is None or last.is_closed()

            last = ZoneVisit.open(ev)
            zseq.append(last)
            self.publish_event(zseq.duplicate())
            
            last.close_at_event(ev)
        self.publish_event(zseq.duplicate())
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: zseqs={self.sequences}"


class FinalZoneSequenceFilter(EventNodeImpl):
    def __init__(self) -> None:
        super().__init__()
        
        self.sequences:dict[str,ZoneSequence] = dict()
    
    def close(self) -> None:
        self.sequences.clear()
        super().close()
        
    def handle_event(self, ev:ZoneSequence|NodeTrack) -> None:
        if isinstance(ev, ZoneSequence):
            self.sequences[ev._track_id] = ev
        elif isinstance(ev, NodeTrack) and ev.is_deleted():
            zseq = self.sequences.pop(ev.track_id, None)
            if zseq:
                self.publish_event(zseq)
                
class ZoneSequenceWriter(EventNodeImpl):
    def __init__(self, log_file:str) -> None:
        super().__init__()

        from pathlib import Path
        parent = Path(log_file).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(log_file, 'w')
    
    def close(self) -> None:
        if self.fp:
            self.fp.close()
            self.fp = None
        super().close()

    def handle_event(self, zseq:ZoneSequence) -> None:
        if zseq.is_closed():
            line = ','.join((zseq.node_id, zseq.track_id, zseq.sequence_str(),
                            str(zseq.first_frame_index), str(zseq.first_ts),
                            str(zseq.frame_index), str(zseq.ts)))
            self.fp.write(line + '\n')