
from __future__ import annotations
from typing import Optional

from dna.track import utils

INVALID_DIST_DISTANCE = 9999
INVALID_IOU_DISTANCE = 1
INVALID_METRIC_DISTANCE = 1


def matches_str(tracks, matches):
    return ",".join([f'{match_str(tracks, m)}' for m in matches])

def match_str(tracks, match):
    return f'({tracks[match[0]].id}, {match[1]})'


from abc import ABCMeta, abstractmethod
class Matcher(metaclass=ABCMeta):
    @abstractmethod
    def match(self, track_idxes:list[int], det_idxes:list[int]) -> list[tuple[int,int]]:
        pass


class ChainedMatcher(Matcher):
    def __init__(self, *matchers):
        self.matchers = matchers
        
    def match(self, track_idxes:list[int], det_idxes:list[int]) -> list[tuple[int,int]]:
        unmatched_track_idxes = track_idxes.copy()
        unmatched_det_idxes = det_idxes.copy()

        matches = []
        for matcher in self.matchers:
            if not (unmatched_track_idxes and unmatched_det_idxes):
                break
            
            matches0 = matcher.match(unmatched_track_idxes, unmatched_det_idxes)
            if matches0:
                matches += matches0
                unmatched_track_idxes = utils.subtract(unmatched_track_idxes, utils.project(matches0, 0))
                unmatched_det_idxes = utils.subtract(unmatched_det_idxes, utils.project(matches0, 1))
        return matches

    def __repr__(self) -> str:
        return '+'.join([repr(m) for m in self.matchers])
    

def chain(*matchers) -> Matcher:
    return ChainedMatcher(*matchers)