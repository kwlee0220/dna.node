
from __future__ import annotations

from typing import Optional
from collections.abc import Iterable

import logging
import numpy as np

from .base import Matcher
from ...track import utils


class MetricAssistedCostMatcher(Matcher):
    def __init__(self, tracks:list, cost_matrix:np.ndarray, threshold:float, metric_cost:np.ndarray,
                logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.cost_matrix = cost_matrix
        self.threshold = threshold
        self.metric_cost = metric_cost
        self.logger = logger
        
    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> list[tuple[int,int]]:
        unmatched_det_idxes = det_idxes.copy()
        
        matches = []
        for track_idx in track_idxes:
            min_cost_det_idx = self.select_min_cost_det_index(self.threshold, track_idx, unmatched_det_idxes)
            if min_cost_det_idx is not None:
                matches.append((track_idx, min_cost_det_idx))
                unmatched_det_idxes.remove(min_cost_det_idx)
                if not unmatched_det_idxes:
                    break

        return matches

    def select_min_cost_det_index(self, threshold:float, row_idx:int, det_idxes:Iterable[int]) -> int:
        row = self.cost_matrix[row_idx, :]
        qualified_det_idxes = np.where(row <= threshold)[0]
        qualified_det_idxes = utils.intersection(qualified_det_idxes, det_idxes)
        if len(qualified_det_idxes) == 0:
            return None
        elif len(qualified_det_idxes) == 1:
            return qualified_det_idxes[0]

        qualified_cols = self.metric_cost[row_idx, qualified_det_idxes]      
        min_col_idx = np.argmin(qualified_cols)
        weights = qualified_cols / qualified_cols[min_col_idx]
        weighted_row = self.cost_matrix[row_idx, qualified_det_idxes] * weights     
        min_col_idx = np.argmin(weighted_row)

        if self.logger.isEnabledFor(logging.DEBUG):
            track_id = self.tracks[row_idx].id
            cols_str = ", ".join([f'{i}({c1:.2f},{c2:.2f})' \
                                    for i, c1, c2 in zip(qualified_det_idxes, self.cost_matrix[row_idx, qualified_det_idxes],
                                                            weighted_row)])
            matching_str = f'   track[{track_id:>02}]: {cols_str} -> {qualified_det_idxes[min_col_idx]}'
            self.logger.debug(matching_str)
                                                  
        return qualified_det_idxes[min_col_idx]