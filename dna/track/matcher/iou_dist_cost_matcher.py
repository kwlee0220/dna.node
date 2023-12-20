from __future__ import annotations
from typing import Optional
from collections.abc import Iterable
import logging

import numpy as np
import numpy.typing as npt

import dna
from dna import Box
from dna.detect import Detection
from ..types import ObjectTrack
from dna.track import utils
from .matching_session import MatchingSession
from ..dna_track_params import DNATrackParams, DistanceIoUThreshold
from .base import matches_str, Matcher, chain
from .reciprocal_cost_matcher import ReciprocalCostMatcher


class IoUDistanceCostMatcher(Matcher):
    def __init__(self, tracks:list[ObjectTrack], detections:list[Detection],
                 params:DNATrackParams, dist_cost:np.ndarray, iou_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.logger = logger
        
        tight_thresholds = DistanceIoUThreshold(distance=20, iou=0.5)
        self.tight_matcher = self._create_chain(dist_cost, iou_cost, tight_thresholds)  
        self.matcher = self._create_chain(dist_cost, iou_cost, params.iou_dist_threshold)

    def match(self, track_idxes:list[int], det_idxes:list[int]) -> list[tuple[int,int]]:
        session = MatchingSession(self.tracks, self.detections, self.params, track_idxes, det_idxes)
        
        #   
        # Do matching between a set of tracks and a set of detections based on motion information (distance and IoU).
        # The matching is performed on hot tracks and tentative tracks only. The tracks that are unmatched more
        # than 2 times are left out in this matching, because we are not quite sure of their estimated locations.
        #
        matches0 = self.tight_matcher.match(session.unmatched_hot_track_idxes, session.unmatched_strong_det_idxes)
        if matches0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'hot, strong, {self.tight_matcher}: {matches_str(self.tracks, matches0)}')
            session.update(matches0)
                
        matches0 = self.matcher.match(session.unmatched_hot_track_idxes, session.unmatched_det_idxes)
        if matches0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'hot, all, {self.matcher}: {matches_str(self.tracks, matches0)}')
            session.update(matches0)

        if session.unmatched_tentative_track_idxes and session.unmatched_det_idxes:
            matches0 = self.matcher.match(session.unmatched_tentative_track_idxes, session.unmatched_det_idxes)
            if matches0:
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'tentative, all, {self.matcher}: {matches_str(self.tracks, matches0)}')
                session.update(matches0)
                
        return session.matches

    def __repr__(self) -> str:
        return repr(self.matcher)
        
    def _create_chain(self, dist_cost:np.array, iou_cost:np.array, thresholds) -> Matcher:
        dist_matcher = ReciprocalCostMatcher(dist_cost, thresholds.distance, name='dist')
        iou_matcher = ReciprocalCostMatcher(iou_cost, thresholds.iou, name='iou')
        return chain(dist_matcher, iou_matcher)