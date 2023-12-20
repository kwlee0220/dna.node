
from __future__ import annotations

from typing import Optional
from collections.abc import Iterable, Callable
import logging

import numpy as np
import numpy.typing as npt

import dna
from dna.detect import Detection
from .base import Matcher
from ...track import utils


SUPERIOR_FACTOR = 1.8


class ReciprocalCostMatcher(Matcher):
    def __init__(self, cost_matrix:np.ndarray, threshold:float, name:Optional[str]=None, \
                    logger:Optional[logging.Logger]=None) -> None:
        self.cost_matrix = cost_matrix
        self.threshold = threshold
        self.name = name if name else 'reciprocal'
        self.logger = logger
        
    def match(self, row_idxes:list[int], column_idxes:list[int]) -> list[tuple[int,int]]:
        unmatched_row_idxes = row_idxes.copy()
        unmatched_col_idxes = column_idxes.copy()
        
        matches = []
        match_count = 1
        while match_count > 0 and (unmatched_row_idxes and unmatched_col_idxes):
            match_count = 0
            row_idxes = unmatched_row_idxes.copy()
            for row_idx in row_idxes:
                top_col_idx = self.select_reciprocal_top(self.threshold, row_idx, unmatched_row_idxes, unmatched_col_idxes)
                if top_col_idx is not None:
                    matches.append((row_idx, top_col_idx))
                    unmatched_row_idxes.remove(row_idx)
                    unmatched_col_idxes.remove(top_col_idx)
                    match_count += 1
                    if not (unmatched_row_idxes and unmatched_col_idxes):
                        break
                        
        return matches

    def select_reciprocal_top(self, threshold:float, row_idx:int, row_idxes:list[int], col_idxes:list[int]) -> int:
        row = self.cost_matrix[row_idx, :]
        top1_col_idx, top2_col_idx = self.select_top2(row, threshold, col_idxes)

        if top1_col_idx is None:    # (None, None)
            return None
        if row_idx != self.argmin_column(top1_col_idx, row_idxes):
            return None
        if top2_col_idx is None:  # (top1, None)
            return top1_col_idx

        if row[top1_col_idx] * SUPERIOR_FACTOR < row[top2_col_idx]:
            # 독보적으로 작은 값이 존재하는 경우
            return top1_col_idx

        # 독보적으로 작은 값이 존재하지 않는 경우
        if row_idx == self.argmin_column(top2_col_idx, row_idxes):
            # top1, top2에 해당하는 detection들 모두 가장 가까운 track이 본인인 경우
            return top1_col_idx
        if row_idx == self.select_reciprocal_top_row(threshold, top1_col_idx, row_idxes, col_idxes):
            return top1_col_idx
        else:
            return None

    def select_reciprocal_top_row(self, threshod:float, col_idx:int, row_idxes:list[int], col_idxes:list[int]) -> int:
        column = self.cost_matrix[:, col_idx]
        top1_row_idx, top2_row_idx = self.select_top2(column, threshod, row_idxes)
        if col_idx != self.argmin_row(top1_row_idx, col_idxes):
            return None
        if top2_row_idx is None:  # (top1, None)
            return top1_row_idx
        if col_idx == self.argmin_row(top2_row_idx, col_idxes):
            # top1, top2에 해당하는 detection들 모두 가장 가까운 track이 본인인 경우
            return top1_row_idx
        return top1_row_idx if column[top1_row_idx] * SUPERIOR_FACTOR < column[top2_row_idx] else None

    def argmin_column(self, col_idx:int, row_idxes:list[int]):
        column = self.cost_matrix[:, col_idx]
        idx = column[row_idxes].argmin()
        return row_idxes[idx]

    def argmin_row(self, row_idx:int, col_idxes:list[int]):
        row = self.cost_matrix[row_idx,:]
        idx = row[col_idxes].argmin()
        return col_idxes[idx]

    def select_top2(self, values:np.ndarray, threshold:float, idxes:list[int]) -> tuple[int,int]:
        selected_idxes = np.where(values <= threshold)[0]
        selected_idxes = utils.intersection(selected_idxes, idxes)

        n_selecteds = len(selected_idxes)
        if n_selecteds == 0:
            return (None, None)
        if n_selecteds == 1:
            return (selected_idxes[0], None)
        elif n_selecteds == 2:
            first, second = values[selected_idxes[0]], values[selected_idxes[1]]
            return (selected_idxes[0], selected_idxes[1]) if first <= second else (selected_idxes[1], selected_idxes[0])
        else:
            selecteds = values[selected_idxes]
            idxes = np.argpartition(selecteds, 2)
            return (selected_idxes[idxes[0]], selected_idxes[idxes[1]])

    def __repr__(self) -> str:
        return f'{self.name}({self.threshold})'