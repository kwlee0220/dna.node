from __future__ import annotations

from typing import Optional
import logging

import numpy as np

from .base import Matcher, INVALID_METRIC_DISTANCE
from .hungarian_matcher import HungarianMatcher


class MetricCostMatcher(Matcher):
    def __init__(self, metric_cost:np.ndarray, threshold:float,
                 logger:Optional[logging.Logger]=None) -> None:
        self.metric_cost = metric_cost
        self.logger = logger
                
        self.metric_matcher = HungarianMatcher(metric_cost, threshold_name="metric",
                                                threshold=threshold,
                                                invalid_value=INVALID_METRIC_DISTANCE)
       
    def match(self, track_idxes:list[int], det_idxes:list[int]) -> list[tuple[int,int]]:
        return self.metric_matcher.match(track_idxes, det_idxes)
            