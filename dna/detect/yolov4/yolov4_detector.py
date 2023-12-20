from __future__ import annotations

from dataclasses import dataclass
from re import L
from typing import Optional
from pathlib import Path
from urllib.parse import parse_qs

import yaml
import numpy as np
import cv2
import gdown

# import sys
# FILE = Path(__file__).absolute()
# _YOLOV4_DIR = str(Path(FILE.parents[3], 'dna-plugins/dna-yoloqv4-torch'))
# if not _YOLOV4_DIR in sys.path:
#     sys.path.append(_YOLOV4_DIR)

from dna import Box
from dna.camera import Frame
from dna.utils import gdown_file, parse_query, get_first_param
from dna.detect import Detection, ObjectDetector

from .tool.utils import load_class_names
from .tool.torch_utils import do_detect
from .tool.darknet2pytorch import Darknet

import logging
LOGGER = logging.getLogger('dna.detector.yolov4')


# def _download_file(url:str, file: str):
#     path = Path(file)
#     if not path.exists():
#         # create an empty 'weights' folder if not exists
#         path.parent.mkdir(parents=True, exist_ok=True)

#     gdown.download(url, file, quiet=False)

@dataclass(frozen=True, eq=True)
class YoloV4ModelDesc:
    class_names: list[str]
    cfg_file_path: str
    weights_file_path: str

def _download_model_descriptor(model_id: str, top_dir: Path) -> YoloV4ModelDesc:
    class_names_file = (top_dir / 'coco.names').as_posix()
    gdown_file('https://drive.google.com/uc?id=1HY_sDInWvjhBq1s3nM6IIZL9T2thh0ro', class_names_file)
    class_names = load_class_names(class_names_file)

    if model_id == 'yolov4': # YoloV4 normal
        # cfg file
        cfg_file_path = (top_dir / 'yolov4.cfg').as_posix()
        gdown_file('https://drive.google.com/uc?id=12MHD6crqkbrvCoS6jVGoMBwOEiE0B7mh', cfg_file_path)

        # weights file
        weights_file_path = (top_dir / 'yolov4.weights').as_posix()
        gdown_file('https://drive.google.com/uc?id=17SKxQtvhpVQbmlUP4n1yBP-0lyYVkgty', weights_file_path)
    elif model_id == 'yolov4-tiny':
        # cfg file
        cfg_file_path = (top_dir / 'yolov4-tiny.cfg').as_posix()
        gdown_file('https://drive.google.com/uc?id=1vEHC8ISK4vMuFSQ7uy5QbYZaOHBX8pzc', cfg_file_path)

        # weights file
        weights_file_path = (top_dir / 'yolov4-tiny.weights').as_posix()
        gdown_file('https://drive.google.com/uc?id=12AdGKIZqAUIZtTLxOvFMaS-wMkEp_THx', weights_file_path)
    else:
        raise ValueError(f'invalid YoloV4 model: {model_id}')

    return YoloV4ModelDesc(class_names=class_names, cfg_file_path=cfg_file_path,
                            weights_file_path=weights_file_path)

def load(query: str):
    args = parse_query(query)
    tag = args.get('model', '')
    model_id = 'yolov4' if tag == '' else f'yolov4-{tag}'

    import os
    top_dir = Path(os.getcwd()) / 'models' / 'yolov4'
    
    desc:Yolov4TorchDetector = _download_model_descriptor(model_id, top_dir)
    LOGGER.info((f'Loading Yolov4TorchDetector: cfg={desc.cfg_file_path}, weights={desc.weights_file_path}'))
    
    classes = args.get('classes')
    if classes:
        classes = set(classes.split(','))
    
    detector = Yolov4TorchDetector(desc, classes=classes)

    score = args.get('score')
    if score is not None:
        detector.conf_thres = float(score)

    return detector

class Yolov4TorchDetector(ObjectDetector):
    def __init__(self, desc: YoloV4ModelDesc,
                conf_thres=0.4,     # confidence threshold
                nms_thres=0.6,      # NMS IOU threshold
                classes:Optional[set[str]]=None,
                use_cuda = True
                ) -> None:
        self.model = Darknet(desc.cfg_file_path)
        self.model.load_weights(desc.weights_file_path)

        self.num_classes = self.model.num_classes
        self.class_names = desc.class_names
        self.target_classes = classes
        if classes is not None:
            self.target_classes =set(desc.class_names.index(cls_name) for cls_name in classes)
        else:
            self.target_classes = set(desc.class_names)
            
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
    # def __init__(self, desc: YoloV4ModelDesc,
    #             conf_thres=0.4,     # confidence threshold
    #             nms_thres=0.6,      # NMS IOU threshold
    #             use_cuda = True
    #             ) -> None:
    #     self.model = Darknet(desc.cfg_file_path)
    #     self.model.load_weights(desc.weights_file_path)

    #     self.num_classes = self.model.num_classes
    #     self.class_names = desc.class_names
    #     self.conf_thres = conf_thres
    #     self.nms_thres = nms_thres

    #     self.use_cuda = use_cuda
    #     if self.use_cuda:
    #         self.model.cuda()


    def detect(self, frame: Frame) -> list[Detection]:
        sized = cv2.resize(frame.image, (self.model.width, self.model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.image.shape
        batched_boxes = do_detect(self.model, sized, self.conf_thres, self.nms_thres, self.use_cuda)
        
        boxes = batched_boxes[0]
        if self.target_classes:
            boxes = [box for box in boxes if box[6] in self.target_classes]

        return [self.box_to_detection(box, w, h) for box in boxes]
    
    def is_valid_label(self, box) -> bool:
        if self.target_classes is None:
            return True
        else: 
            return self.class_names[box[6]] in self.target_classes
            

    def box_to_detection(self, box, w, h):
        coords = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        bbox = Box(coords)
        conf = box[5]
        label = self.class_names[box[6]]
        return Detection(bbox, label=label, score=conf)