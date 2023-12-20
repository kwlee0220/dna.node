from __future__ import annotations

from typing import Optional, Union
from pathlib import Path
from urllib.parse import parse_qs

import yaml
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from dna import Box, Image
from dna.camera import Frame
from dna.utils import parse_query
from dna.detect import ObjectDetector, Detection

from pathlib import Path
import logging
LOGGER = logging.getLogger(f"dna.detector.{Path(__file__).parent.stem}")


def load(query: str):
    args = parse_query(query)
    return Yolov5Detector(**args)

class Yolov5Detector(ObjectDetector):
    def __init__(self, **kwargs) -> None:
        model_id = 'yolov5' + kwargs.get('model', 's')
        LOGGER.info(f'Loading {Yolov5Detector.__name__}: model={model_id}')

        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', f'models/yolov5/{model_id}', verbose=False)
        self.model = torch.hub.load('ultralytics/yolov5', model_id, pretrained=True, verbose=False)
        self.names = self.model.names

        self.detect_args = dict()
        # self.detect_args = {'imgsz':640}
        if (v := kwargs.get('score')) is not None:
            self.model.conf = float(v)
        if (v := kwargs.get('iou')) is not None:
            self.model.iou = float(v)
        if (v := kwargs.get('max_det')) is not None:
            self.model.max_det = int(v)
        if (v := kwargs.get('classes')) is not None:
            from dna import utils
            class_names = self.names if utils.has_method(self.names, 'index') else list(self.names.values())
            class_idxes = [class_names.index(cls) if isinstance(cls, str) else int(cls) for cls in v.split(',')]
            self.model.classes = [idx for idx in class_idxes if idx < len(class_names)]
        if (v := kwargs.get('agnostic')) is not None:
            self.model.agnostic = bool(v)
        if (v := kwargs.get('device')) is not None:
            self.model.to(v)
            
    @property
    def score(self) -> float:
        return self.model.conf
    @score.setter
    def score(self, score:float) -> None:
        self.model.conf = score
            
    @property
    def classes(self) -> list[str]:
        return [self.names[cls_idx] for cls_idx in self.model.classes]
    @classes.setter
    def classes(self, classes:list[str]) -> None:
        class_idxes = [self.names.index(cls) if isinstance(cls, str) else int(cls) for cls in classes.split(',')]
        self.model.classes = [idx for idx in class_idxes if idx >= 0 and idx < len(self.names)]
            
    @property
    def iou(self) -> float:
        return self.model.iou
    @iou.setter
    def iou(self, iou:float) -> None:
        self.model.iou = iou
            
    @property
    def max_det(self) -> int:
        return self.model.max_det
    @max_det.setter
    def max_det(self, count:int) -> None:
        self.model.max_det = count
            
    def device(self, device:Union[int,str]) -> None:
        self.model.to(device)

    @torch.no_grad()
    def detect(self, frame: Frame) -> list[Detection]:
        batch = [self._preprocess(frame.image)]

        # inference
        result = self.model(batch, size=640)

        return self._to_detections(result.xyxy[0])

    # @torch.no_grad()
    # def detect_images(self, frames:list[Frame]) -> list[list[Detection]]:
    #     batch = [self._preprocess(frame.image) for frame in frames]

    #     # inference
    #     result = self.model(batch)

    #     return [self._to_detections(xyxy) for xyxy in result.xyxy]

    def _preprocess(self, image:Image) -> torch.Tensor:
        return image[:,:,::-1]
        # img = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # return np.ascontiguousarray(img)

    def _to_detections(self, xyxy) -> list[Detection]:
        return [self._to_detection(pred) for pred in xyxy.cpu().numpy()]

    def _to_detection(self, pred) -> list[Detection]:
        box = Box(pred[:4])
        name = self.names[int(pred[5])]
        confi = pred[4]
        return Detection(box, name, confi)