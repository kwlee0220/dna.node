from typing import Optional

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from dna import Box, Image
from dna.camera import Frame
from dna.detect import ObjectDetector, Detection
from dna.utils import parse_query

from pathlib import Path
import logging
LOGGER = logging.getLogger(f"dna.detector.{Path(__file__).stem}")

_DEFAULT_DETECT_THRESHOLD = 0.35
_DEFAULT_DEVICE = "cuda"

def load(query: str):
    LOGGER.info(f'Loading {Rcnn50FpnV2Detector.__name__}: query="{query}"')

    args = parse_query(query)
    return Rcnn50FpnV2Detector(**args)

class Rcnn50FpnV2Detector(ObjectDetector):
    def __init__(self, **kwargs) -> None:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.class_names = weights.meta["categories"]

        score_threshold = float(kwargs.get("score", _DEFAULT_DETECT_THRESHOLD))

        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_threshold)
        if device_name:=kwargs.get("device", _DEFAULT_DEVICE):
            if device_name.lower() == "cpu":
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
            else:
                self.device = torch.device(device_name)
                self.model = self.model.cuda(self.device)
        self.model.eval()

        self.transform = transforms.ToTensor()

    def detect(self, frame:Frame) -> list[Detection]:
        batch = [self._to_tensor(frame.image)]
        predictions = self.model(batch)
        return self._to_detections(predictions[0])

    def detect_images(self, frames:list[Frame]) -> list[list[Detection]]:
        batch = [self._to_tensor(frame.image) for frame in frames]
        return [self._to_detections(pred) for pred in self.model(batch)]

    def _to_tensor(self, image:Image) -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(img).to(self.device)
        return self.preprocess(img)

    def _to_detections(self, prediction) -> list[Detection]:
        labels = [self.class_names[i] for i in prediction["labels"]]
        boxes = prediction["boxes"].cpu().detach().numpy()
        scores = prediction["scores"].cpu().detach().numpy()

        return [self._to_detection(label, box, score)
                for label, box, score in zip(labels, boxes, scores)]

    def _to_detection(self, label, bbox, score):
        return Detection(bbox=Box(bbox), label=label, score=score)
        