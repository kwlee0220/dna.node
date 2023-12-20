from __future__ import annotations

from typing import Optional, TypeVar, Union
from collections.abc import Sequence, Iterable, Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt

from dna import Box, Point, Image, BGR
from dna.camera import Frame
from dna.detect import Detection
from dna.support import plot_utils
from dna.track import TrackState
from .types import ObjectTrack, ObjectTracker


T = TypeVar("T")

def subtract(list1:Iterable, list2:list) -> list:
    return [v for v in list1 if v not in list2]

def intersection(list1:Iterable, list2:list) -> list:
    return [v for v in list1 if v in list2]

def all_indices(values:Sequence):
    return list(range(len(values)))

def project(tuples: Iterable[tuple], elm_idx: int) -> list:
    return [t[elm_idx] for t in tuples]

def get_items(values:Iterable[T], idxes:Iterable[int]) -> list[T]:
    return [values[idx] for idx in idxes]

def get_indexed_items(values:Iterable[T], idxes:Iterable[int]) -> list[T]:
    return [(idx, values[idx]) for idx in idxes]
    
def cosine_distance(a:npt.ArrayLike, b:npt.ArrayLike, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        
    return 1. - np.dot(a, b.T)


class SimpleTrack(ObjectTrack):
    def __init__(self, id:int, state:TrackState, location:Box, frame_index:int, timestamp:float) -> None:
        super().__init__(id=id, state=state, location=location, frame_index=frame_index, timestamp=timestamp)
    
    @staticmethod
    def from_csv(csv) -> SimpleTrack:
        parts = csv.split(',')

        frame_idx = int(parts[0])
        track_id = int(parts[1])
        tlbr = np.array(parts[2:6]).astype('int32')
        bbox = Box(tlbr)
        state = TrackState(int(parts[6]))
        ts = int(parts[7]) / 1000
        return SimpleTrack(id=track_id, state=state, location=bbox, frame_index=frame_idx, timestamp=ts)


class LogFileBasedObjectTracker(ObjectTracker):
    def __init__(self, track_file: Path) -> None:
        self.__file = open(track_file, 'r')
        self.look_ahead = self._look_ahead()
        self._tracks:list[ObjectTrack] = []

    @property
    def file(self) -> Path:
        return self.__file

    def track(self, frame: Frame) -> list[ObjectTrack]:
        self._tracks = []

        if not self.look_ahead:
            return self._tracks

        if self.look_ahead.frame_index > frame.index:
            return self._tracks

        # throw track event lines upto target_idx -
        while self.look_ahead.frame_index < frame.index:
            self.look_ahead = self._look_ahead()

        while self.look_ahead.frame_index == frame.index:
            self._tracks.append(self.look_ahead)

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead is None:
                break

        return self._tracks

    @property
    def tracks(self) -> list[ObjectTrack]:
        return self._tracks
        
    def _look_ahead(self) -> ObjectTrack:
        csv = self.__file.readline().rstrip()
        if csv:
            return SimpleTrack.from_csv(csv)
        else:
            self.__file.close()
            return None

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"