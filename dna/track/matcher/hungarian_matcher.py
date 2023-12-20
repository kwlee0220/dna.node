from __future__ import annotations

from typing import Optional
from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import Matcher


class HungarianMatcher(Matcher):
    def __init__(self, cost_matrix:np.ndarray, threshold:float, invalid_value:float, threshold_name:Optional[str]=None) -> None:
        self.cost_matrix = cost_matrix
        self.threshold = threshold
        self.threshold_name = threshold_name
        self.invalid_value = invalid_value
        
    def match(self, row_idxes:Optional[Iterable[int]]=None, column_idxes:Optional[Iterable[int]]=None) \
         -> list[tuple[int,int]]:
        return self.matching_by_hungarian(self.threshold, row_idxes, column_idxes)
        
    def match_threshold(self, threshold:float, row_idxes:Optional[Iterable[int]]=None,
                        column_idxes:Optional[Iterable[int]]=None) \
        -> list[tuple[int,int]]:
        return self.matching_by_hungarian(threshold, row_idxes, column_idxes)

    def matching_by_hungarian(self, threshold:float, row_idxes:Optional[Iterable[int]], col_idxes:Optional[Iterable[int]]) \
        -> list[tuple[int,int]]:
        row_idxes = list(range(self.matrix.shape[0])) if row_idxes is None else row_idxes.copy()
        col_idxes = list(range(self.matrix.shape[1])) if col_idxes is None else col_idxes.copy()
        
        def _remove_by_index(list, idx):
            remains = list.copy()
            removed = remains.pop(idx)
            return removed, remains

        n_rows, n_cols = len(row_idxes), len(col_idxes)
        if n_rows == 0 or n_cols == 0:
            return []

        if n_rows == 1 and n_cols == 1:
            if self.cost_matrix[row_idxes[0], col_idxes[0]] <= threshold:
                return [(row_idxes[0], col_idxes[0])]
            else:
                return []
        elif n_cols == 1:   # row만 여러개
            reduced = self.cost_matrix[:,col_idxes[0]][row_idxes]
            tidx = np.argmin(reduced)
            if reduced[tidx] <= threshold:
                matched_track, unmatched_tracks = _remove_by_index(row_idxes, tidx)
                return [(matched_track, col_idxes[0])]
            else:
                return []
        elif n_rows == 1:       # column만 여러개
            reduced = self.cost_matrix[row_idxes[0],:][col_idxes]
            didx = np.argmin(reduced)
            if reduced[didx] <= threshold:
                matched_det, unmatched_dets = _remove_by_index(col_idxes, didx)
                return [(row_idxes[0], matched_det)]
            else:
                return []

        matrix = self.cost_matrix[np.ix_(row_idxes, col_idxes)]
        matrix[matrix > self.threshold] = self.invalid_value
        indices = linear_sum_assignment(matrix)
        indices = np.asarray(indices)
        indices = np.transpose(indices)

        matches = []
        for i, j in indices:
            tidx = row_idxes[i]
            didx = col_idxes[j]
            if self.cost_matrix[tidx, didx] <= threshold:
                matches.append((tidx, didx))
        
        return matches

    def __repr__(self) -> str:
        threshold_str = f'{self.threshold_name}={self.threshold}' if self.threshold_name else f'{self.threshold}'
        return f'hungarian({threshold_str})'