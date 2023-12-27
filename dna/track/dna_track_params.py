from __future__ import annotations

from typing import Union, Optional
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import dna
from dna import Size2d, Box, config
from dna.detect import Detection
from dna.zone import Zone


@dataclass(frozen=True, eq=True)    # slots=True
class DistanceIoUThreshold:
    distance: float
    iou: float
    
    @staticmethod
    def from_config(conf:DictConfig, key:str, default=None) -> DistanceIoUThreshold:
        conf_value = OmegaConf.select(conf, key, default=None)
        if conf_value:
            npa = np.array(conf_value)
            return DistanceIoUThreshold(distance=npa[0], iou=npa[1])
        else:
            return default

DEFAULT_DETECTIION_CLASSES = ['car', 'bus', 'truck']
DEFAULT_DETECTION_CONFIDENCE = 0.37
DEFAULT_DETECTION_MIN_SIZE = Size2d([15, 15])
DEFAULT_DETECTION_MAX_SIZE = Size2d([768, 768])
DEFAULT_DETECTION_ROIS = []

DEFAULT_IOU_DIST_THRESHOLD = DistanceIoUThreshold(distance=70, iou=0.75)
DEFAULT_IOU_DIST_THRESHOLD_LOOSE = DistanceIoUThreshold(distance=90, iou=0.85)

# DEFAULT_METRIC_TIGHT_THRESHOLD = 0.3
DEFAULT_METRIC_THRESHOLD = 0.55
DEFAULT_METRIC_GATE_DISTANCE = 600
DEFAULT_METRIC_MIN_DETECTION_SIZE = Size2d([30, 30])
DEFAULT_METRIC_REGISTRY_MIN_DETECTION_SIZE = Size2d([35, 35])
DEFAULT_MAX_FEATURE_COUNT = 50

DEFAULT_N_INIT = 3
DEFAULT_MAX_AGE = 10
DEFAULT_NEW_TRACK_MIN_SIZE = [30, 20]
DEFAULT_MATCH_OVERLAP_SCORE = 0.75
DEFAULT_MAX_NMS_SCORE = 0.8

EMPTY_ZONES = []

def load_size2d(conf:OmegaConf, key:str, def_value:Optional[Size2d]) -> Optional[Size2d]:
    value = conf.get(key)
    return Size2d.from_expr(value) if value else def_value    


def parse_zones(conf:OmegaConf, key:str, default:Zone=EMPTY_ZONES) -> list[Zone]:
    zones_expr = OmegaConf.select(conf, key, default=default)
    return [Zone.from_coords(zone) for zone in zones_expr]
    
@dataclass(frozen=True, eq=True)    # slots=True
class DNATrackParams:
    detection_classes: set[str]
    detection_confidence: float
    detection_min_size: Size2d
    detection_max_size: Size2d
    drop_border_detections: bool

    iou_dist_threshold: DistanceIoUThreshold
    iou_dist_threshold_loose: DistanceIoUThreshold

    metric_threshold: float
    metric_gate_distance: float
    metric_min_detection_size: Size2d
    metric_registry_min_detection_size: Size2d
    max_feature_count: int

    n_init: int
    new_track_min_size: Size2d
    max_age: int
    match_overlap_score: float
    max_nms_score: float
    
    track_zones: list[Zone]
    magnifying_zones: list[Box]
    blind_zones: list[Zone]
    exit_zones: list[Zone]
    stable_zones: list[Zone]
    
    draw: list[str]

    def is_strong_detection(self, det:Detection) -> bool:
        return det.score >= self.detection_confidence
    
    def is_metric_detection(self, det:Detection) -> bool:
        return self.is_strong_detection(det) \
            and det.bbox.size() >= self.metric_min_detection_size \
            and det.exit_zone < 0
    
    def is_metric_detection_for_registry(self, det:Detection) -> bool:
        return self.is_strong_detection(det) \
            and det.bbox.size() >= self.metric_registry_min_detection_size \
            and det.exit_zone < 0

    def is_in_stable_zone(self, box:Box, zone_id:int) -> bool:
        return self.stable_zones[zone_id].contains_point(box.center())
            
    def find_track_zone(self, box:Box) -> int:
        return Zone.find_covering_zone(box.center(), self.track_zones)
    def find_stable_zone(self, box:Box) -> int:
        return Zone.find_covering_zone(box.center(), self.stable_zones)
    def find_exit_zone(self, box:Box) -> int:
        return Zone.find_covering_zone(box.center(), self.exit_zones)
    def find_blind_zone(self, box:Box) -> int:
        return Zone.find_covering_zone(box.center(), self.blind_zones)

def load_track_params(track_conf:DictConfig) -> DNATrackParams:
    detection_classes = set(track_conf.get('detection_classes', DEFAULT_DETECTIION_CLASSES))
    detection_confidence = track_conf.get('detection_confidence', DEFAULT_DETECTION_CONFIDENCE)
    detection_min_size = load_size2d(track_conf, 'detection_min_size', None)
    detection_max_size = load_size2d(track_conf, 'detection_max_size', None)
    drop_border_detections = track_conf.get('drop_border_detections', False)

    iou_dist_threshold = DistanceIoUThreshold.from_config(track_conf, 'iou_dist_threshold', DEFAULT_IOU_DIST_THRESHOLD)
    iou_dist_threshold_loose = DistanceIoUThreshold.from_config(track_conf, 'iou_dist_threshold_loose',
                                                                DEFAULT_IOU_DIST_THRESHOLD_LOOSE)

    metric_threshold = OmegaConf.select(track_conf, "metric_threshold", default=DEFAULT_METRIC_THRESHOLD)
    metric_gate_distance = OmegaConf.select(track_conf, "metric_gate_distance", default=DEFAULT_METRIC_GATE_DISTANCE)
    metric_min_detection_size = load_size2d(track_conf, 'metric_min_detection_size',
                                            DEFAULT_METRIC_MIN_DETECTION_SIZE)
    metric_registry_min_detection_size = load_size2d(track_conf, 'metric_registry_min_detection_size',
                                                     DEFAULT_METRIC_REGISTRY_MIN_DETECTION_SIZE)
    max_feature_count = OmegaConf.select(track_conf, "max_feature_count", default=DEFAULT_MAX_FEATURE_COUNT)

    n_init = int(track_conf.get('n_init', DEFAULT_N_INIT))
    max_age = int(track_conf.get('max_age', DEFAULT_MAX_AGE))
    max_nms_score = track_conf.get('max_nms_score', DEFAULT_MAX_NMS_SCORE)
    match_overlap_score = track_conf.get('match_overlap_score', DEFAULT_MATCH_OVERLAP_SCORE)
    new_track_min_size = Size2d.from_expr(track_conf.get('new_track_min_size', DEFAULT_NEW_TRACK_MIN_SIZE))
    
    track_zones = parse_zones(track_conf, 'track_zones')
    blind_zones = parse_zones(track_conf, 'blind_zones')
    exit_zones = parse_zones(track_conf, 'exit_zones')
    stable_zones = parse_zones(track_conf, 'stable_zones')
    zones_expr = OmegaConf.select(track_conf, 'magnifying_zones', default=[])
    magnifying_zones = [Box(zone) for zone in zones_expr]
    
    draw = track_conf.get("draw", [])

    return DNATrackParams(detection_classes=detection_classes,
                        detection_confidence=detection_confidence,
                        detection_min_size=detection_min_size,
                        detection_max_size=detection_max_size,
                        drop_border_detections=drop_border_detections,
                        
                        iou_dist_threshold=iou_dist_threshold,
                        iou_dist_threshold_loose=iou_dist_threshold_loose,
                        
                        metric_threshold=metric_threshold,
                        metric_gate_distance=metric_gate_distance,
                        metric_min_detection_size=metric_min_detection_size,
                        metric_registry_min_detection_size=metric_registry_min_detection_size,
                        max_feature_count=max_feature_count,
                        
                        n_init=n_init,
                        max_age=max_age,
                        max_nms_score=max_nms_score,
                        match_overlap_score=match_overlap_score,
                        new_track_min_size=new_track_min_size,
                        
                        track_zones=track_zones,
                        magnifying_zones=magnifying_zones,
                        blind_zones=blind_zones,
                        exit_zones=exit_zones,
                        stable_zones=stable_zones,
                        
                        draw=draw)