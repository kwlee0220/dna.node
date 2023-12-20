from __future__ import annotations

from typing import DefaultDict
from collections import defaultdict

from dna.support import iterables
from dna.event import JsonEvent
from . import stabilizer


_MAX_TS = 9999999999

class TrajectorySmoother:
    def __init__(self, look_ahead:int=10) -> None:
        self.look_ahead = look_ahead
        self.buffer:list[JsonEvent] = []
        self.running_track_starts:DefaultDict[str,int] = defaultdict(lambda:_MAX_TS)
        self.deleted_track_end:DefaultDict[str,int] = defaultdict()
        
    def __len__(self) -> int:
        return len(self.buffer)
            
    def smooth(self, track:JsonEvent) -> list[JsonEvent]:
        self.buffer.append(track)
        if track.is_deleted():
            start_ts = self.running_track_starts.pop(track.id, None)
            if start_ts:
                # Stabilize가 가능한 trajectoryfmf 찾아 stabilize를 수행한다.
                self.deleted_track_end[track.id] = track.ts
                self.smooth_traj(track.id, self.buffer)
                
                # running trajectory들 중에서 가장 작은 start ts를 다시 구한다.
                min_running_ts = min((ts for ts in self.running_track_starts.values()), default=_MAX_TS)
                
                # 'min_running_ts'를 기준으로 self.tracks를 둘로 나눈다
                break_idx, _ = iterables.argfind(self.buffer,
                                                 lambda t: t.ts >= min_running_ts,
                                                 default=(len(self.buffer), None))
                safe_tracks, self.buffer = self.buffer[:break_idx], self.buffer[break_idx:]
                
                self.deleted_track_end = { trk_id:end_ts for trk_id, end_ts in self.deleted_track_end.items()
                                                            if end_ts >= min_running_ts }
                
                return safe_tracks
        else:
            if track.ts < self.running_track_starts[track.id]:
                self.running_track_starts[track.id] = track.ts
        return []

    def smooth_traj(self, traj_id:str, tracks:list[JsonEvent]):
        idxed_samples = [(idx, trk) for idx, trk in enumerate(tracks) if trk.id == traj_id and not trk.is_deleted()]
        src_pts = [trk.location for _, trk in idxed_samples]
        smoothed_pts = stabilizer.stabilize(src_pts, look_ahead=self.look_ahead)
        for idx in range(len(src_pts)):
            sample_idx, sample = idxed_samples[idx]
            sample.location = smoothed_pts[idx]