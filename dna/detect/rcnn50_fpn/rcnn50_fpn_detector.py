from typing import Optional

import torch
from detectron2 import model_zoo

from dna import get_logger, Box
from dna.camera import Frame
from dna.detect import ObjectDetector, Detection
from dna.utils import parse_query
from dna.detect import LOGGER
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


MODEL_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

def load(query: str):
    args = parse_query(query)

    score = args.get('score')
    if score is not None:
        score = float(score)
    else:
        score = 0.5

    LOGGER.info(f'Loading Rcnn50FpnDetector: query={query}')
    return Rcnn50FpnDetector(score)

class Rcnn50FpnDetector(ObjectDetector):
    def __init__(self, score:float) -> None:
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.merge_from_file(model_zoo.get_config_file(MODEL_FILE))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_FILE)
        self.predictor = DefaultPredictor(self.cfg)

        self.class_names = self.predictor.metadata.get("thing_classes")

    def detect(self, frame: Frame) -> list[Detection]:
        outputs = self.predictor(frame.image)
        insts = outputs["instances"].to("cpu")

        pred_classes = insts.pred_classes.numpy()
        pred_boxes = insts.pred_boxes.tensor.numpy()
        pred_scores = insts.scores.numpy()
        return [self._to_detection(cls, box, score)
                for cls, box, score in zip(pred_classes, pred_boxes, pred_scores)]

    def _to_detection(self, cls, bbox, score):
        return Detection(bbox=Box(bbox),
                        label=self.class_names[cls],
                        score=score)
        