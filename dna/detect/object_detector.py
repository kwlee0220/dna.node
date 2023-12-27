from __future__ import annotations

from typing import Optional
from pathlib import Path
from abc import ABCMeta, abstractmethod

import numpy as np

from dna import Box
from dna.camera import Frame
from .detection import Detection


class ObjectDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Detect objects from the image and returns their locations

        Args:
            image Image: an image from OpenCV
            frame_index (int, optional): frame index. Defaults to None.

        Returns:
            list[Detection]: a list of Detection objects
        """
        pass
    
    def detect_images(self, frames:list[Frame]) -> list[list[Detection]]:
        return [self.detect(frame) for frame in frames]


class ScoreFilteredObjectDetector(ObjectDetector):
    def __init__(self, detector: ObjectDetector, min_score: float) -> None:
        super().__init__()

        self.detector = detector
        if min_score < 0:
            raise ValueError(f'invalid score threshold: {min_score}')
        self.min_score = min_score

    def detect(self, frame: Frame) -> list[Detection]:
        return [det for det in self.detector.detect(frame)
                        if det.score < 0 or det.score >= self.min_score]


class LabelFilteredObjectDetector(ObjectDetector):
    def __init__(self, detector: ObjectDetector, accept_labels: list[str]) -> None:
        super().__init__()

        self.detector = detector
        self.labels = accept_labels

    def detect(self, frame: Frame) -> list[Detection]:
        return [det for det in self.detector.detect(frame)
                        if det.label in self.labels]


class BlindZoneObjectDetector(ObjectDetector):
    def __init__(self, detector: ObjectDetector, blind_zones: list[Box]) -> None:
        super().__init__()

        self.detector = detector
        self.blind_zones = blind_zones

    def detect(self, frame: Frame) -> list[Detection]:
        return [det for det in self.detector.detect(frame)
                        if not any(zone.contains(det.bbox) for zone in self.blind_zones)]


class LogReadingDetector(ObjectDetector):
    def __init__(self, det_file: Path) -> None:
        """Create an ObjectDetector object that issues detections from a detection file.

        Args:
            det_file (Path): Path to the detection file.
        """
        self.__file = open(det_file, 'r')
        self.look_ahead = self._look_ahead()

    @property
    def file(self) -> Path:
        return self.__file

    def detect(self, frame: Frame) -> list[Detection]:
        if not frame.index:
            return []

        if not self.look_ahead:
            return []

        idx = int(self.look_ahead[0])
        if idx > frame.index:
            return []

        # throw detection lines upto target_idx -
        while idx < frame.index:
            self.look_ahead = self._look_ahead()
            idx = int(self.look_ahead[0])

        detections = []
        while idx == frame.index and self.look_ahead:
            detections.append(self._parse_line(self.look_ahead))

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead:
                idx = int(self.look_ahead[0])
            else:
                idx += 1

        return detections

    def _look_ahead(self) -> Optional[list[str]]:
        line = self.__file.readline().rstrip()
        if line:
            return line.split(',')
        else:
            self.__file.close()
            return None

    def _parse_line(self, parts: list[str]) -> Detection:
        bbox = Box([float(v) for v in parts[2:6]])
        label: Optional[str] = parts[10] if len(parts) >= 11 else None
        score: float = float(parts[6])
        
        return Detection(bbox=bbox, label=label, score=score)

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"