from __future__ import annotations

from typing import Optional

from pathlib import Path
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf

from dna import Image, BGR, color
from dna.camera import Frame
from dna.support import plot_utils

FILE = Path(__file__).absolute()
DNA_TRACK_DIR = str(FILE.parents[0])
if not DNA_TRACK_DIR in sys.path:
    sys.path.append(DNA_TRACK_DIR)

import dna
from dna import Box, Size2d, Point
from dna.detect import ObjectDetector, Detection
from .types import ObjectTracker, MetricExtractor
from .dna_track_params import load_track_params, DNATrackParams
from .feature_extractor import DeepSORTMetricExtractor
from .cnu_track.qdtrack_feature_extractor import QDTrackMetricExtractor
from .dna_track import DNATrack
from .tracker import Tracker
from dna.event.node_track import NodeTrack

import logging
LOGGER = logging.getLogger('dna.tracker')

DSORT_REID_MODEL = "models/deepsort/model640.pt"
CNU_REID_MODEL = "models/cnu/cnu_reid_model.pth"
MODELS:dict[str,str] = {
    DSORT_REID_MODEL: 'https://drive.google.com/uc?id=160jJWtgIhyhHIBpgNOkAT52uvvtOYGly',
    CNU_REID_MODEL: 'https://drive.google.com/uc?id=1oXkiAhEoPtBrBil4H1hlhBghQWfZEpJc',
}


def load_feature_extractor(model_file:str=DSORT_REID_MODEL, normalize:bool=False):
        wt_path = Path(model_file).resolve()
        if not wt_path.exists():
            gdown_link = MODELS.get(model_file)
            if gdown_link is None:
                raise ValueError(f"unknown FeatureExtractor model id: {model_file}")
            dna.utils.gdown_file(gdown_link, wt_path)

        # loading this encoder is slow, should be done only once.
        if model_file == DSORT_REID_MODEL:
            return DeepSORTMetricExtractor(model_file, normalize)
        elif model_file == CNU_REID_MODEL:
            return QDTrackMetricExtractor(model_file)
        else:
            raise ValueError(f"unknown FeatureExtractor model id: {model_file}")


_DEEP_SORT_REID_MODEL = 'models/deepsort/model640.pt'
class DNATracker(ObjectTracker):
    def __init__(self, detector:ObjectDetector, tracker_conf:OmegaConf, /,
                 feature_extractor:Optional[MetricExtractor]=None) -> None:
        super().__init__()

        self.detector = detector

        #loading this encoder is slow, should be done only once.
        if not feature_extractor:
            # from .qdtrack.qdtrack_feature_extractor import QDTrackMetricExtractor
            # feature_extractor = QDTrackMetricExtractor("models/qdtrack/configs/CenterNet2_S4_DLA.yaml")
            model_file = tracker_conf.get('model_file', _DEEP_SORT_REID_MODEL)
            feature_extractor = load_feature_extractor(model_file, normalize=True)
        self.feature_extractor = feature_extractor
        
        self.params = load_track_params(tracker_conf)
        self.tracker = Tracker(self.params, LOGGER)
        self.last_track_events = []
        
        self.shrinked_rois = []
        self.roi_shifts = []
        for roi in self.params.magnifying_zones:
            shrinked = Box(roi.tlbr + np.array([5, 5, -5, -5]))
            if not shrinked.is_valid():
                raise ValueError(f'too small roi: {roi}')
            self.shrinked_rois.append(shrinked)
            self.roi_shifts.append(Size2d(roi.tl))

    @property
    def tracks(self) -> list[DNATrack]:
        return self.tracker.tracks

    def track(self, frame: Frame) -> tuple[list[DNATrack], list[NodeTrack]]:
        dna.DEBUG_FRAME_INDEX = frame.index
        self.last_event_tracks = []

        detections_list = self.detector.detect_images([frame] + self.crop_rois(frame))

        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        filterds = []
        for dets in detections_list:
            filterds.append([det for det in dets if det.label in self.params.detection_classes])
        detections_list = filterds

        # Detection box 크기에 따라 invalid한 detection들을 제거한다.
        filterds = []
        h, w, _ = frame.image.shape
        max_size = Size2d([w, h]) * 0.7
        for dets in detections_list:
            filterds.append([det for det in dets if _is_valid_detection_size(det, self.params, max_size)])
        detections_list = filterds

        detections = self.merge(detections_list)

        # track zone에 포함된 detection들만 선택한다.
        if self.params.track_zones:
            detections = [det for det in detections if self.params.find_track_zone(det.bbox) >= 0]

        # blind zone에 포함된 detection들은 무시한다.
        if self.params.blind_zones:
            detections = [det for det in detections if self.params.find_blind_zone(det.bbox) < 0]
        
        # 'drop_border_detection'이 enable된 경우, detection-box가 border와 근접한 경우 무시해야 하는데,
        # 여기서 해당 detection을 삭제해 버리면 matching 과정에서 해당 detection과 matching된 track이
        # 다른 detection과 무리하게 matching되는 결과가 발생된다.
        # 이 문제를 해결하기 위해 일단 border 근처의 detection들도 matching 과정에 참여하도록하고
        # matching 후처리 과정에서 해당 detection과 matching된 track을 delete하는 작업을 수행한다.
            
        # Detection끼리 너무 많이 겹치는 경우 low-score detection을 제거한다.
        def supress_overlaps(detections:list[Detection]) -> list[Detection]:
            remains = sorted(detections.copy(), key=lambda d: d.score, reverse=True)
            survivors = []
            while remains:
                head = remains.pop(0)
                survivors.append(head)
                remains = [det for det in remains if head.bbox.iou(det.bbox) < self.params.max_nms_score]
                pass
            return survivors
        detections = supress_overlaps(detections)

        # 모든 detection들에게 exit-zone 소속 여부에 따라 해당 exit-zone의 id를 부여한다.
        # Exit-zone에 포함되었다고 여기서 버리면, association시 다른 detection과 binding될 수 있기 때문에
        # 여기서는 해당 detection에 exit-zone id만 부여하고 나중에 처리한다.
        # 다만, exit-zone에 있는 모든 weak detection들은 무시한다
        def is_weak_exit_zone_detection(det:Detection) -> Detection:
            det.exit_zone = self.params.find_exit_zone(det.bbox)
            if det.exit_zone < 0 or self.params.is_strong_detection(det):
                return False
            else:
                return True
        detections = [det for det in detections if not is_weak_exit_zone_detection(det)]
        
        # Filtering을 마친 detection에 대해서는 영상 내의 해당 영역에서 feature를 추출하여 부여한다.
        metric_detections = [det for det in detections if self.params.is_metric_detection(det)]
        for det, feature in zip(metric_detections, self.feature_extractor.extract_dets(frame.image, metric_detections)):
            det.feature = feature

        if dna.DEBUG_SHOW_IMAGE:
            self.draw_detections(frame.image.copy(), 'detections', detections)
            pass

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"{frame.index}: ------------------------------------------------------")
        session, deleted_tracks, self.last_event_tracks = self.tracker.track(frame, detections)
        
        return self.tracker.tracks + deleted_tracks

    def crop_rois(self, frame:Frame) -> list[Frame]:
        return [Frame(roi.crop(frame.image), frame.index, frame.ts) for roi in self.params.magnifying_zones]

    def merge(self, detections_list:list[Detection]) -> list[Detection]:
        mergeds = []

        cropped_detections_list = detections_list[1:]
        for roi, shift, dets in zip(self.shrinked_rois, self.roi_shifts, cropped_detections_list):
            for det in dets:
                shifted_box = det.bbox.translate(shift)
                if roi.contains(shifted_box):
                    mergeds.append(Detection(shifted_box, det.label, det.score))

        for det in detections_list[0]:
            if all(not roi.contains(det.bbox) for roi in self.params.magnifying_zones):
                mergeds.append(det)

        return mergeds

    def draw_detections(self, convas:Image, title:str, detections:list[Detection], line_thickness=1):
        if self.params.draw:
            if 'track_zones' in self.params.draw:
                for zone in self.params.track_zones:
                    convas = zone.draw(convas, color.BLUE, 1)
            if 'blind_zones' in self.params.draw:
                for zone in self.params.blind_zones:
                    convas = zone.draw(convas, color.YELLOW, 1)
            if 'exit_zones' in self.params.draw:
                for zone in self.params.exit_zones:
                    convas = zone.draw(convas, color.RED, 1)
            if 'stable_zones' in self.params.draw:
                for zone in self.params.stable_zones:
                    convas = zone.draw(convas, color.BLUE, 1)
            if 'magnifying_zones' in self.params.draw:
                for roi in self.params.magnifying_zones:
                    roi.draw(convas, color.ORANGE, line_thickness=1)

        for idx, det in enumerate(detections):
            if not self.params.is_strong_detection(det):
                convas = det.draw(convas, color.RED, label=str(idx), label_color=color.WHITE,
                                    label_tl=Point(det.bbox.br.astype(int)), line_thickness=line_thickness)
        for idx, det in enumerate(detections):
            if self.params.is_strong_detection(det):
                convas = det.draw(convas, color.BLUE, label=str(idx), label_color=color.WHITE,
                                    label_tl=Point(det.bbox.br.astype(int)), line_thickness=line_thickness)
        cv2.imshow(title, convas)
        cv2.waitKey(1)

        return convas
    
    @staticmethod
    def load(tracker_conf: OmegaConf):
        from dna.detect.utils import load_object_detector

        detector_uri = tracker_conf.get("detector", dna.DEFAULT_DETECTOR_URI)
        detector = load_object_detector(detector_uri)

        return DNATracker(detector, tracker_conf)


def _is_valid_detection_size(det:Detection, params:DNATrackParams, max_size:Size2d) -> bool:
    size = det.bbox.size()
    if params.detection_min_size and not(size >= params.detection_min_size):
        return False
    if params.detection_max_size and not(size <= params.detection_max_size):
        return False
    return size <= max_size