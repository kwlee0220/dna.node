
from __future__ import annotations

from collections.abc import Generator
import sys
from dataclasses import dataclass
from collections import defaultdict
import itertools

import time
from dna import Point, initialize_logger
from dna.event import NodeTrack
from dna.support import iterables

import logging
LOGGER = logging.getLogger('dna.node.sync_frames')

@dataclass(frozen=True)    # slots=True
class Sample:
    location: Point
    frame_index: int
    
    @classmethod
    def from_event(cls, te: NodeTrack) -> Sample:
        return cls(te.location, te.frame_index)
    
    def distance_to(self, other: Sample) -> float:
        return self.location.distance_to(other.location)
    
@dataclass(frozen=True)    # slots=True
class Trajectory:
    luid: int
    samples: list[Sample]
    
    @property
    def length(self) -> int:
        return len(self.samples)
    
    @property
    def first_frame(self) -> int:
        return self.samples[0].frame_index
    
    @property
    def last_frame(self) -> int:
        return self.samples[-1].frame_index
    
    def frame_delta(self, other: Trajectory) -> int:
        return other.first_frame - self.first_frame
    
    def distance(self, other: Trajectory, too_far_dist=sys.float_info.max, outlier_ratio=0.1) -> float:
        dists = []
        for s1, s2 in zip(self.samples, other.samples):
            dist = s1.distance_to(s2)
            if dist >= too_far_dist:
                return sys.float_info.max
            dists.append(dist)
        dists = sorted(dists, reverse=True)
        n_outliers = round(len(dists) * outlier_ratio)
        dists = dists[n_outliers:]
        return sum(dists) / len(dists)
    
    def split_by_continuity(self) -> list[Trajectory]:
        samples:list[Sample] = self.samples
        split = [samples[0]]
        splits:list[list[Sample]] = [split]
        for idx in range(1, len(samples)-1):
            if samples[idx].frame_index - samples[idx-1].frame_index == 1:
                split.append(samples[idx])
            else:
                # 연속된 frame index를 갖지 않은 경우에는 새 split을 생성한다.
                split = [samples[idx]]
                splits.append(split)
        return [Trajectory(self.luid, split) for split in splits]
    
    def buffer(self, length:int) -> list[Trajectory]:
        if self.last_frame < length:
            return []
        
        return (Trajectory(self.luid, segment) for segment in iterables.buffer(self.samples, length, 1, length))
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}[luid={self.luid}, "
                f"frames=[{self.first_frame}-{self.last_frame}], "
                f"len={self.length}]")


def load_log_file(log_path:str, max_camera_dist:float) -> dict[int,list[NodeTrack]]:
    event_dict = defaultdict(list)
    with open(log_path, 'r') as fp:
        while True:
            line = fp.readline().rstrip()
            if len(line) <= 0:
                break
            
            # 'json' 문자열을 파싱하여 로딩한다
            te = NodeTrack.from_json(line)
            
            # 일부 TrackEvent의 "world_coord"와 "distance" 필드가 None 값을 갖기 때문에
            # 해당 event는 제외시킨다. 또한 대상 물체와 카메라와의 거리 (distance)가
            # 일정거리('max_camera_dist')보다 큰 경우도 추정된 실세계 좌표('woorld_dist') 값에
            # 정확도에 떨어져서 제외시킨다.
            if te.distance is not None and te.distance <= max_camera_dist and te.location is not None:
                # 'frame-index' 단위로 event들을 정리한다.
                event_dict[te.frame_index].append(te)
    return event_dict

def find_isolated_trajectories(events_by_frame:dict[int,list[NodeTrack]], sparse_dist:float) -> list[Trajectory]:
    def is_isolated(target:NodeTrack, events:list[NodeTrack]):
        # 같은 frame_index를 갖는 다른 track_event들과의 거리가 일정거리(sparse_dist) 이상인지를 판단함.
        for ev in events:
            if target.track_id != ev.track_id and target.location.distance_to(ev.location) < sparse_dist:
                return False
        return True
        
    # 각 frame별로 주변에 일정거리 이내에 위치를 갖는 track event가 없는 track event로 구성된 trajectory를 구성한다.
    trajectories = defaultdict(list)
    for events in events_by_frame.values():
        for te in events:
            if is_isolated(te, events):
                # 물체의 식별자('track_id')별로 track_event를 누적하여 trajectory를 구성함
                trajectories[te.track_id].append(Sample.from_event(te))
    return [Trajectory(luid, samples) for luid, samples in trajectories.items()]
    
def generate_segments(trajs:list[Trajectory], length:int):
    return iterables.flatten(traj.buffer(length) for traj in trajs)

def load_sparse_trajectories(log_file_path:str, camera_dist:float, sparse_distance:float,
                             min_traj_length:int) -> list[Trajectory]:
    events = load_log_file(log_file_path, camera_dist)
    trajs = (traj for traj in find_isolated_trajectories(events, sparse_distance) if traj.length >= min_traj_length)
    trajs = itertools.chain.from_iterable(traj.split_by_continuity() for traj in trajs)
    trajs = [traj for traj in trajs if traj.length >= min_traj_length]
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(f"load trajectories: path={log_file_path}, max_camera_dist={camera_dist:.1f}, count={len(trajs)}")
    
    return trajs

def match(trajs1:list[Trajectory], trajs2:list[Trajectory], max_frame_delta:int, segment_length:int,
          traj_distance:float):
    matches = defaultdict(list)
    for s1 in generate_segments(trajs1, segment_length):
        for s2 in generate_segments(trajs2, segment_length):
            frame_delta = s1.frame_delta(s2)
            if abs(frame_delta) < max_frame_delta:
                dist = s1.distance(s2, too_far_dist=10)
                # if dist != sys.float_info.max:
                #     print(f"seg1={s1}, seg2={s2}, delta={frame_delta}, dist={dist:.3f}")
                if dist < traj_distance:
                    matches[frame_delta].append(dist)
    return matches

def find_best_match(trajs1: list[Trajectory], trajs2: list[Trajectory], max_frame_delta: int, segment_length: int,
                    traj_distance:float):
    matches = matches(trajs1, trajs2, max_frame_delta, segment_length, traj_distance)
                
    matches = ((delta, sorted(dists)[:30]) for delta, dists in matches.items() if len(dists) >= 30)
    matches = ((delta, sum(dists) / len(dists)) for delta, dists in matches)
    return sorted(matches, key=iterables.get1)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Synchronize two videos")
    parser.add_argument("track_files", nargs='+', help="track json files")
    parser.add_argument("--max_camera_distance", type=float, metavar="meter", default=50,
                        help="max. distance from camera (default: 55)")
    parser.add_argument("--frame_delta", type=int, metavar="count", default=20, 
                        help="maximum frame difference between videos (default: 20)")
    parser.add_argument("--traj_distance", type=float, metavar="value", default=2, 
                        help="maximum valid distance between trajectories (default: 2)")
    parser.add_argument("--segment_length", type=int, metavar="count", default=10,
                        help="trajectory segment length (default: 10)")
    parser.add_argument("--sparse_distance", type=float, metavar="meter", default=10,
                        help="min. distance for sparse trajectories (default: 10)")
    parser.add_argument("--topk", type=int, metavar="count", default=5,
                        help="# of frame syncs (default: 5)")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

from collections import namedtuple
Matching = namedtuple('Matching', ['pair', 'frame_delta', 'cost'])

NULL_FRAME_OFFSET = -99999
def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    # load track files
    trajs_list = []
    for track_file in args.track_files:
        trajs_list.append(load_sparse_trajectories(track_file,
                                                    camera_dist=args.max_camera_distance,
                                                    sparse_distance=args.sparse_distance,
                                                    min_traj_length=args.segment_length))
    
    matchings_list = []
    pairs = list(itertools.combinations(range(len(args.track_files)), 2))
    for pair in pairs:
        trajs1, trajs2 = trajs_list[pair[0]], trajs_list[pair[1]]

        # 주어진 두 카메라 영상에서 검출된 trajectory들(trajs1, trajs2) 사이의 거리에 따라 matching 시킨다.
        # 유사도는 두 카메라 영상의 frame 번호 차이(frame_delta)를 기준으로 계산한다.
        matches = match(trajs1, trajs2, args.frame_delta, args.segment_length, args.traj_distance)

        # 각 frame_delta의 거리 값 중에서 가장 좋은 30개를 뽑아 평균을 구한다.
        frame_delta_costs = ((delta, iterables.mean(sorted(dists)[:30])) for delta, dists in matches.items() if len(dists) >= 30)

        # 평균 유사도 값을 기준으로 거리가 작을 값에서 큰 값으로 정렬시킨다.
        ranks = sorted(frame_delta_costs, key=iterables.get1)

        if LOGGER.isEnabledFor(logging.INFO):
            ranks_str = ', '.join(f"({frame_delta}:{cost:.3f})" for frame_delta, cost in ranks[:5])
            suffix_str = ', ...' if len(ranks) > 5 else ''
            LOGGER.info(f"match camera({pair[0]}, {pair[1]}): count={len(ranks)}, ranks={ranks_str}{suffix_str}")
        matchings_list.append([Matching(pair=pair, frame_delta=rank[0], cost=rank[1]) for rank in ranks[:10]])
        
    offset_costs = []
    for combi in itertools.product(*matchings_list):
        frame_syncs = [NULL_FRAME_OFFSET]*3
        for matching in combi:
            base_index, rel_index = matching.pair
            
            if frame_syncs[base_index] == NULL_FRAME_OFFSET:
                frame_syncs[base_index] = 0
                
            offset = frame_syncs[base_index] + matching.frame_delta
            if frame_syncs[rel_index] == NULL_FRAME_OFFSET:
                frame_syncs[rel_index] = offset
            elif frame_syncs[rel_index] == offset:
                final_cost = sum(m.cost for m in combi) / len(combi)
                offset_costs.append((frame_syncs, final_cost))
    offset_costs = sorted(offset_costs, key=lambda t: t[1])
    for frame_sync, cost in offset_costs[:args.topk]:
        shift = 0 - min(frame_sync)
        frame_sync = [offset + shift for offset in frame_sync]
        print(f"sync cameras={frame_sync}, cost={cost:.3f}")

if __name__ == '__main__':
    main()