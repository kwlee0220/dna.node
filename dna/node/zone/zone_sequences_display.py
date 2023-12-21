from __future__ import annotations

from typing import Optional

import cv2

from dna import Box, color
from dna.camera import Frame, FrameUpdater, ImageProcessor, ImageCapture
from dna.event import EventListener
from .events import ZoneSequence


class ZoneSequenceDisplay(EventListener, FrameUpdater):
    def __init__(self) -> None:
        EventListener.__init__(self)
        FrameUpdater.__init__(self)
        
        self.sequence_count:dict[str,int] = dict()
        self.track_locations:dict[str,Box] = dict()
        self.motion_tracks:set[int] = set()

    def close(self) -> None:
        for key in self.sequence_count.keys():
            self.sequence_count[key] = 0
        
    def handle_event(self, zseq:ZoneSequence) -> None:
        if zseq.is_closed():
            seq_str = zseq.sequence_str()[1:-1]
            if seq_str not in self.sequence_count:
                self.sequence_count[seq_str] = 0
                self.sequence_count = dict(sorted(self.sequence_count.items()))
            self.sequence_count[seq_str] += 1
            self.motion_tracks.add(zseq.track_id)

    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None:
        for key in self.sequence_count.keys():
            self.sequence_count[key] = 0

    def close(self) -> None:
        pass

    def update(self, frame:Frame) -> Optional[Frame]:
        y_offset = 20
        convas = frame.image
        
        for track_id, loc in self.track_locations.items():
            if track_id in self.motion_tracks:
                convas = loc.draw(convas, color.RED, line_thickness=3)
        self.track_locations.clear()
        self.motion_tracks.clear()

        for seq, count in self.sequence_count.items():
            y_offset += 25
            convas = cv2.putText(convas, f'{seq:>3}: {count}',
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.RED, 2)
        return Frame(image=convas, index=frame.index, ts=frame.ts)

    def set_control(self, key:int) -> int:
        return key