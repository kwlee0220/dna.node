from __future__ import annotations
from typing import Optional

from dna.detect import Detection
from ..types import ObjectTrack
from dna.track import utils
from ..dna_track_params import DNATrackParams


class MatchingSession:
    def __init__(self, tracks:list[ObjectTrack], detections:list[Detection], params:DNATrackParams,
                track_idxes=None, det_idxes=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.matches = []
        self.unmatched_track_idxes = track_idxes.copy() if track_idxes else utils.all_indices(tracks)
        self.unmatched_det_idxes = det_idxes.copy() if det_idxes else utils.all_indices(detections)

    def update(self, matches0:list[tuple[int,int]]) -> None:
        self.matches += matches0
        self.unmatched_track_idxes = utils.subtract(self.unmatched_track_idxes, utils.project(matches0, 0))
        self.unmatched_det_idxes = utils.subtract(self.unmatched_det_idxes, utils.project(matches0, 1))

    def remove_det_idxes(self, idxes:list[int]) -> None:
        self.unmatched_det_idxes = utils.subtract(self.unmatched_det_idxes, idxes)

    def pull_out(self, match:tuple[int,int]) -> None:
        self.matches = [m for m in self.matches if m != match]
        self.unmatched_track_idxes.append(match[0])
        self.unmatched_det_idxes.append(match[1])

    def find_match_by_track(self, track_idx:int) -> Optional[tuple[int,int]]:
        founds = [m for m in self.matches if m[0] == track_idx]
        return founds[0] if founds else None

    def find_match_by_det(self, det_idx:int) -> Optional[tuple[int,int]]:
        founds = [m for m in self.matches if m[1] == det_idx]
        return founds[0] if founds else None
    
    def get_match(self, match:tuple[int,int]):
        return self.tracks[match[0]], self.detections[match[1]]

    @property
    def associations(self) -> list[tuple[ObjectTrack,Detection]]:
        return [(self.tracks[t_idx], self.detections[d_idx]) for t_idx, d_idx in self.matches]

    @property
    def bindings(self) -> str:
        return [(self.tracks[t_idx].id, d_idx) for t_idx, d_idx in self.matches]

    @property
    def unmatched_hot_track_idxes(self) -> list[int]:
        idx_tracks = ((i, self.tracks[i]) for i in self.unmatched_track_idxes)
        return [i for i, t in idx_tracks if t.is_confirmed() or (t.is_temporarily_lost() and t.time_since_update <= 3)]

    @property
    def unmatched_tlost_track_idxes(self) -> list[int]:
        idx_tracks = ((i, self.tracks[i]) for i in self.unmatched_track_idxes)
        return [i for i, t in idx_tracks if t.is_temporarily_lost()]

    @property
    def unmatched_tentative_track_idxes(self) -> list[int]:
        return [i for i in self.unmatched_track_idxes if self.tracks[i].is_tentative()]
    @property
    def unmatched_confirmed_track_idxes(self) -> list[int]:
        return [i for i in self.unmatched_track_idxes if self.tracks[i].is_confirmed()]
    @property
    def unmatched_non_tentative_track_idxes(self) -> list[int]:
        return [i for i in self.unmatched_track_idxes if not self.tracks[i].is_tentative()]
        
    @property
    def unmatched_strong_det_idxes(self) -> list[int]:
        return [i for i in self.unmatched_det_idxes if self.params.is_strong_detection(self.detections[i])]
        
    @property
    def unmatched_weak_det_idxes(self) -> list[int]:
        return [i for i in self.unmatched_det_idxes if not self.params.is_strong_detection(self.detections[i])]
        
    @property
    def unmatched_metric_det_idxes(self) -> list[int]:
        # metric 생성용 detection들 중에서 'exit_zone'에 포함된 것을 제외시킨다.
        return [i for i in self.unmatched_strong_det_idxes if self.params.is_metric_detection(self.detections[i])]

    def __repr__(self) -> str:
        um_track_idxes = [self.tracks[t_idx].id for t_idx in self.unmatched_track_idxes]
        return (f'matches={self.bindings}, unmatched: tracks={um_track_idxes}, det_idxes={self.unmatched_strong_det_idxes}')
    
         