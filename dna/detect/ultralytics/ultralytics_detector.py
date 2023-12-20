from __future__ import annotations

import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.utils import set_logging

import dna
from dna import Box, Image
from dna.camera import Frame
from dna.utils import parse_query
from dna.detect import ObjectDetector, Detection

from pathlib import Path
import logging
LOGGER = logging.getLogger(f"dna.detector.{Path(__file__).parent.stem}")

# detector: dna.detect.ultralytics:model=yolov8l&type=v8&score=0.1
def load(query: str):
    set_logging(verbose=False)

    args = parse_query(query)
    return UltralyticsDetector(**args)


class UltralyticsDetector(ObjectDetector):
    def __init__(self, **kwargs) -> None:
        model = kwargs.get('model', 'yolov8m')
        type = kwargs.get('type', 'v8')
        # self.model = YOLO(model=model, type=type)
        self.model = YOLO(model=model)
        self.names = self.model.names

        self.detect_args = dict()
        # self.detect_args = {'imgsz':640}
        if (v := kwargs.get('score')) is not None:
            self.detect_args['conf'] = float(v)
        if (v := kwargs.get('iou')) is not None:
            self.detect_args['iou'] = float(v)
        if (v := kwargs.get('max_det')) is not None:
            self.detect_args['max_det'] = int(v)
        if (v := kwargs.get('classes')) is not None:
            val_list = list(self.names.values())
            class_idxes = [val_list.index(cls) if isinstance(cls, str) else int(cls) for cls in v.split(',')]
            self.detect_args['classes'] = [idx for idx in class_idxes if idx < len(val_list)]
        if (v := kwargs.get('agnostic_nms')) is not None:
            self.detect_args['agnostic_nms'] = bool(v)
        if (v := kwargs.get('device')) is not None:
            self.detect_args['device'] = v

    def _resize(self, frames:list[Frame]) -> list[Frame]:
        return [self._resize_image(frame.image) for frame in frames]

    def _resize_image(self, image:Image) -> tuple[Image, tuple[float,float]]:
        h, w, d = image.shape
        ratio_h, ratio_w = h / 640, w / 640
        
        if h > 640 or w > 640:
            return cv2.resize(image, (640,640), interpolation=cv2.INTER_AREA), (ratio_w, ratio_h)
        elif h != 640 or w != 640:
            return cv2.resize(image, (640,640)), (ratio_w, ratio_h)
        else:
            return image, (1, 1)

    def detect(self, frame: Frame) -> list[Detection]:
        return self.detect_images([frame])[0]

    def detect_images(self, frames:list[Frame]) -> list[list[Detection]]:
        resizeds = self._resize(frames)
        batch = [resized[0] for resized in resizeds]
        wh_ratios = [resized[1] for resized in resizeds]

        results = self.model.predict(source=batch, **self.detect_args)

        return [self._to_detections(result, hw_ratio) for result, hw_ratio in zip(results, wh_ratios)]

    def _to_detections(self, result, wh_ratio:tuple[float,float]) -> list[Detection]:
        return [self._to_detection(pred.boxes[0], wh_ratio) for pred in result.boxes.cpu().numpy()]

    def _to_detection(self, pred:Boxes, wh_ratio:tuple[float,float]) -> list[Detection]:
        box = Box(pred[:4] * np.array([wh_ratio[0], wh_ratio[1], wh_ratio[0], wh_ratio[1]]))
        name = self.names[int(pred[5])]
        confi = pred[4]
        return Detection(box, name, confi)