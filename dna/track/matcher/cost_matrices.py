from __future__ import absolute_import

import numpy as np
import numpy.typing as npt

import dna
from dna.detect import Detection
from ..types import ObjectTrack
from dna.track import utils
from ...track.kalman_filter import KalmanFilter
from .base import INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE, INVALID_METRIC_DISTANCE


def build_dist_cost(kf:KalmanFilter, tracks:list[ObjectTrack], detections:list[Detection]) -> np.ndarray:
    dist_matrix = np.ones((len(tracks), len(detections)))
    if tracks and detections:
        measurements = np.asarray([det.bbox.xyah for det in detections])
        for t_idx, track in enumerate(tracks):
            mahalanovis_dist = kf.gating_distance(track.mean, track.covariance, measurements)
            # detection과 여러 frame동안 association이 없는 track의 경우, detection들과의 거리 값이 다른 track들에
            # 비해 짧아지게되기 때문에 이를 보정한다.
            # 추후에는 mahalanovis distance를 사용하지 않는 버전을 수정해야 할 것 같다.
            dist_matrix[t_idx, :] = mahalanovis_dist * (1 + 0.75*(track.time_since_update-1))
    return dist_matrix


def build_iou_cost(tracks:list[ObjectTrack], detections:list[Detection]) -> np.ndarray:
    matrix = np.ones((len(tracks), len(detections)))
    if tracks and detections:
        for t_idx, track in enumerate(tracks):
            t_box = track.location
            for d_idx, det in enumerate(detections):
                matrix[t_idx,d_idx] = 1 - t_box.iou(det.bbox)
    return matrix


def gate_dist_iou_cost(dist_cost:np.ndarray, iou_cost:np.ndarray, \
                        tracks:list[ObjectTrack], detections:list[Detection]) -> tuple[np.ndarray, np.ndarray]:
    # track과 detection 사이 matching 과정에서 이 둘 사이의 크기에 많은 차이가 발생하는 경우
    # match되지 않도록 cost matrix의 해당 cell 값을 최대값으로 설정한다.
    # 'iou' zone 도입이후로 이 기능의 활용도가 떨어지는 것 같아서 나중에 없이질 수도 있음.
    validity_mask = build_task_det_ratio_mask(tracks, detections)
    gated_dist_cost = np.where(validity_mask, dist_cost, INVALID_DIST_DISTANCE)
    gated_iou_cost = np.where(validity_mask, iou_cost, INVALID_IOU_DISTANCE)
    return gated_dist_cost, gated_iou_cost


_AREA_RATIO_LIMITS = (0.3, 2.8)     # 크기가 일반적인 track의 location 대비 detection과의 크기 비율
_LARGE_AREA_RATIO_LIMITS = (0.5, 2) # 일정 크기 이상의 track의 location 대비 detection과의 크기 비율
def build_task_det_ratio_mask(tracks:list[ObjectTrack], detections:list[Detection],
                                area_ratio_limits:npt.ArrayLike=_AREA_RATIO_LIMITS):
    det_areas = np.array([det.bbox.area() for det in detections])
    
    area_ratio_limits = np.array(area_ratio_limits)
    large_area_ratio_limits = np.array(area_ratio_limits)

    mask = np.zeros((len(tracks), len(detections)), dtype=bool)
    for t_idx, track in enumerate(tracks):
        t_area = track.location.area()
        ratio_limits = area_ratio_limits if t_area < 100000 else large_area_ratio_limits
        limits = ratio_limits * t_area
        mask[t_idx,:] = (det_areas >= limits[0]) & (det_areas <= limits[1])
        
    return mask


def build_metric_cost(tracks:list[ObjectTrack], detections:list[Detection],
                        track_idxes:list[int], det_idxes:list[int]) -> np.ndarray:
    def build_matrix(tracks:list[ObjectTrack], detections:list[Detection]) -> np.ndarray:
        cost_matrix = np.ones((len(tracks), len(detections)))
        if tracks and detections:
            det_features = [det.feature for det in detections]
            for i, track in enumerate(tracks):
                if track.features:
                    distances = utils.cosine_distance(track.features, det_features)
                    cost_matrix[i, :] = distances.min(axis=0)
        return cost_matrix

    reduced_track_idxes = [i for i, track in utils.get_indexed_items(tracks, track_idxes) if track.features]
    reduced_det_idxes = [i for i, det in utils.get_indexed_items(detections, det_idxes) if det.feature is not None]
    reduced_matrix = build_matrix(utils.get_items(tracks, reduced_track_idxes), utils.get_items(detections, reduced_det_idxes))

    cost_matrix = np.ones((len(tracks), len(detections)))
    for row_idx, t_idx in enumerate(reduced_track_idxes):
        for col_idx, d_idx in enumerate(reduced_det_idxes):
            cost_matrix[t_idx, d_idx] = reduced_matrix[row_idx, col_idx]
    return cost_matrix

def gate_metric_cost(metric_costs:np.ndarray, dist_costs:np.ndarray,
                    gate_threshold:float) -> None:
    return np.where(dist_costs > gate_threshold, INVALID_METRIC_DISTANCE, metric_costs)


def print_cost_matrix(tracks:list[ObjectTrack], cost, trim_overflow=None):
    if trim_overflow:
        cost = cost.copy()
        cost[cost > trim_overflow] = trim_overflow

    for tidx, track in enumerate(tracks):
        dists = [int(round(v)) for v in cost[tidx]]
        track_str = f" {tidx:02d}: {track.id:03d}({track.state},{track.time_since_update:02d})"
        dist_str = ', '.join([f"{v:4d}" if v != trim_overflow else "    " for v in dists])
        print(f"{track_str}: {dist_str}")