from __future__ import annotations

from typing import Union
from collections.abc import Generator

from dna import TrackId
from dna.track import TrackState
from .node_track import NodeTrack


class Tracklet:
    def __init__(self, track_id:TrackId, tracks:list[NodeTrack], offset:int=0) -> None:
        self.track_id = track_id
        self.tracks:list[NodeTrack] = tracks

    def is_closed(self) -> bool:
        return len(self.tracks) > 0 and self.tracks[-1].state == TrackState.Deleted
    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __iter__(self):
        return (track for track in self.tracks)
    
    def __getitem__(self, index) -> Union[NodeTrack, Tracklet]:
        if isinstance(index, int):
            return self.tracks[index]
        else:
            return Tracklet(track_id=self.track_id, tracks=self.tracks[index])
    
    def __delitem__(self, idx) -> None:
        del self.tracks[idx]
        
    def sublist(self, begin_frame:int, end_frame:int) -> Tracklet:
        sub_tracks = [t for t in self.tracks if t.frame_index >= begin_frame and t.frame_index < end_frame]
        return Tracklet(self.track_id, sub_tracks)

    def append(self, track:NodeTrack) -> None:
        self.tracks.append(track)

    @staticmethod
    def from_tracks(tracks:list[NodeTrack]) -> Tracklet:
        if len(tracks) > 0:
            raise ValueError(f'empty track events')
        return Tracklet(tracks[0].track_id, tracks)

    @staticmethod
    def intersection(tracklet1:Tracklet, tracklet2:Tracklet) -> Tracklet:
        begin_frame_index = max(tracklet1[0].frame_index, tracklet2[0].frame_index)
        end_frame_index = min(tracklet1[-1].frame_index, tracklet2[-1].frame_index) + 1
        overlap1 = tracklet1.sublist(begin_frame_index, end_frame_index)
        overlap2 = tracklet2.sublist(begin_frame_index, end_frame_index)
        return overlap1, overlap2
    
    @staticmethod
    def sync(tracklet1:Tracklet, tracklet2:Tracklet) -> Generator[tuple[NodeTrack,NodeTrack], None, None]:
        try:
            iter1, iter2 = iter(tracklet1), iter(tracklet2)
            track1, track2 = next(iter1), next(iter2)
            while True:
                if track1.frame_index == track2.frame_index:
                    yield track1, track2
                    track1, track2 = next(iter1), next(iter2)
                elif track1.frame_index < track2.frame_index:
                    track1 = next(iter1)
                else:
                    track2 = next(iter2)
        except StopIteration:
            return

    def __repr__(self) -> str:
        seq_str = ''
        if len(self.tracks) == 1:
            seq_str = f'{self.tracks[0].frame_index}'
        elif len(self.tracks) > 1:
            seq_str = f'{self.tracks[0].frame_index}-{self.tracks[-1].frame_index}'
        state_str = f'[D]' if self.tracks and self.tracks[-1].is_deleted() else ''

        return f'{self.track_id}{state_str}:{len(self.tracks)}[{seq_str}]'
    

def read_tracklets(tracklet_gen:Generator[NodeTrack, None, None]) -> dict[TrackId, Tracklet]:
    tracklets:dict[TrackId, Tracklet] = dict()
    for track in tracklet_gen:
        tracklet = tracklets.get(track.track_id)
        if not tracklet:
            tracklet = Tracklet(track.track_id, [])
            tracklets[track.track_id] = tracklet
        tracklet.append(track)
    return tracklets