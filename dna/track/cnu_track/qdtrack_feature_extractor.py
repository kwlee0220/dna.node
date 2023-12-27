# from typing import Optional

from pathlib import Path

import numpy as np

from dna import Box
from dna.camera import Image
from dna.detect import Detection
from ..types import MetricExtractor

from .models.config import add_configs
from .models.engine.predictor import CustomPredictor


def setup_config(device="cuda"):
    """
    :param config_file: Detectron2 config file for setup models
    :return:
    """
    cfg = add_configs()
    cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.125, 0.0625, 0.03125)
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TRAIN = (256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
    cfg.INPUT.MAX_SIZE_TRAIN = 900
    cfg.INPUT.MAX_SIZE_TEST = 736
    cfg.INPUT.MIN_SIZE_TEST = 512
    return cfg


class QDTrackMetricExtractor(MetricExtractor):
    def __init__(self, model_file:str) -> None:
        cfg = setup_config()
        # cfg.MODEL.WEIGHTS = 'model_.pth'
        cfg.MODEL.WEIGHTS = model_file
        cfg.freeze()
        self.predictor = CustomPredictor(cfg)
        
    def distance(self, metric1:np.ndarray, metric2:np.ndarray) -> float:
        # return 1. - np.dot(metric1, metric2.T)
        return 1 - np.inner(metric1, metric2)

    def extract_dets(self, image:Image, detections:list[Detection]):
        if detections:
            xywh_list = [det.bbox.xywh for det in detections]
            result = self.predictor(image, xywh_list)
            return result.cpu().numpy()
        else:
            return []
        
    def extract_boxes(self, image:Image, boxes:list[Box]) -> np.ndarray:
        bounding_boxes = [list(box.xywh) for box in boxes]
        result = self.predictor(image, bounding_boxes)
        return result.cpu().numpy()
    
    def extract_crops(self, crops:list[Image]) -> np.ndarray:
        raise ValueError("XXXXXXXXXX")