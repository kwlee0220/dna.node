from __future__ import annotations
from typing import Optional

from pathlib import Path

from dna import color
from dna.camera import Frame, ImageProcessor, FrameProcessor
from . import ObjectDetector, Detection


class DetectingProcessor(FrameProcessor):
    __slots__ = 'detector', 'draw_detections', 'box_color', 'label_color', 'show_score', 'output', 'out_fp'
    
    def __init__(self, detector:ObjectDetector, *,
                output: Optional[str]=None,
                draw_detections: bool=False):
        self.detector = detector
        self.draw_detections = draw_detections
        self.box_color = color.RED
        self.label_color = color.WHITE
        self.show_score = True
        self.output = output
        self.out_fp = None

    @staticmethod
    def load(detector_uri:str, *, output: Optional[Path]=None, draw_detections: bool=False) -> DetectingProcessor:
        if not detector_uri:
            raise ValueError(f"detector id is None")

        parts = detector_uri.split(':', 1)
        id, query = tuple(parts) if len(parts) > 1 else (detector_uri, "")
        if id == 'file':
            from pathlib import Path
            from .object_detector import LogReadingDetector
            det_file = Path(query)
            detector = LogReadingDetector(det_file)
        else:
            import importlib
            loader_module = importlib.import_module(id)
            detector = loader_module.load(query)

        return DetectingProcessor(detector=detector, output=output, draw_detections=draw_detections)

    def on_started(self, proc: ImageProcessor) -> None:
        if self.output:
            Path(self.output).parent.mkdir(exist_ok=True)
            self.out_fp = open(self.output, "w")
        return self

    def on_stopped(self) -> None:
        if self.out_fp:
            self.out_fp.close()
            self.out_fp = None

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        img = frame.image
        frame_idx = frame.index

        for det in self.detector.detect(frame):
            if self.out_fp:
                self.out_fp.write(self._to_string(frame_idx, det) + '\n')
            if self.draw_detections:
                img = det.draw(img, color=self.box_color, label_color=self.label_color)

        return Frame(image=img, index=frame_idx, ts=frame.ts)

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.label_color = None if self.label_color else color.WHITE
        elif key == ord('c'):
            self.show_score = not self.show_score
        return key

    def _to_string(self, frame_idx: int, det: Detection) -> str:
        tlbr = det.bbox.tlbr
        return (f"{frame_idx},-1,{tlbr[0]:.3f},{tlbr[1]:.3f},{tlbr[2]:.3f},{tlbr[3]:.3f},"
                f"{det.score:.3f},-1,-1,-1,{det.label}")