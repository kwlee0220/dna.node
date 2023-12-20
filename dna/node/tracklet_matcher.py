from __future__ import annotations

from typing import Union, Optional
from dataclasses import dataclass, field, replace
import itertools

from dna.event import NodeTrack
from dna.track import TrackState
from dna.node import Tracklet

                
def match_tracklets(tracklets1:dict[str,Tracklet], tracklets2:dict[str,Tracklet],
                    ignore_matches:set[tuple[str,str]], \
                    min_overlap_length:int,
                    max_track_distance:float,
                    max_tracklet_distance:float) \
    -> tuple[dict[tuple[str,str],float], set[tuple[str,str]]]:
    matches = dict()
    dismatches = set()
    for tid1, tid2 in itertools.product(tracklets1.keys(), tracklets2.keys(), repeat=1):
        if (tid1, tid2) in ignore_matches:
            continue

        tracklet1, tracklet2 = Tracklet.intersection(tracklets1[tid1], tracklets2[tid2])
        if len(tracklet1) < min_overlap_length or len(tracklet2) < min_overlap_length:
            if tracklet1.is_closed() and tracklet2.is_closed():
                dismatches.add((tid1, tid2))
            elif tracklet1.is_closed() and (tracklet2[0].frame_index -tracklet1[-1].frame_index) < min_overlap_length:
                dismatches.add((tid1, tid2))
            elif tracklet2.is_closed() and (tracklet1[0].frame_index - tracklet2[-1].frame_index) < min_overlap_length:
                dismatches.add((tid1, tid2))
            # 그렇지 않은 경우는 이후 tracklet에 track이 추가됨에 따라 common tracklet이
            # 요구길이(min_overlap_length)보다 커질 수 있어 이번 match에서만 pskip함
            continue
        
        def distance(tracklet1:Tracklet, tracklet2:Tracklet):
            sum, count = 0, 0
            for left, right in Tracklet.sync(tracklet1, tracklet2):
                dist = left.world_coord.distance_to(right.world_coord)
                if dist > max_track_distance:
                    return -1
                sum, count = sum + dist, count + 1
            return sum / count

        avg_dist = distance(tracklet1, tracklet2)
        if avg_dist < 0 or avg_dist > max_tracklet_distance:
            dismatches.add((tid1, tid2))
        else:
            matches[(tid1, tid2)] = avg_dist
    return matches, dismatches


@dataclass(frozen=True)
class Node:
    id: str
    offset: int = field(default=0)
    tracklets: dict[str,Tracklet]=field(default_factory=lambda: dict())
    
    def track_ids(self) -> set[str]:
        return self.tracklets.keys()
    
    def append(self, ev:NodeTrack) -> None:
        tracklet = self.tracklets.get(ev.track_id)
        if tracklet is None:
            tracklet = Tracklet(ev.track_id, [])
            self.tracklets[ev.track_id] = tracklet           
        if self.offset != 0:
            ev = replace(ev, frame_index=ev.frame_index + self.offset)
        tracklet.append(ev)
            
    def purge_deleted_tracklet(self, frame_index:int, overlap_length:int) -> None:
        deleted_tracklet_ids = [tracklet.track_id for tracklet in self.tracklets.values() 
                                    if tracklet.is_closed() \
                                        and (frame_index - tracklet[-1].frame_index) > overlap_length]
        for tid in deleted_tracklet_ids:
            self.tracklets.pop(tid, None)
            # print(f'purge tracklet: node={self.id}, tracklet={deleted}, other_frame_index={frame_index}')


class TrackletMatcher:
    def __init__(self,
                 node_ids:tuple[str,str],
                 offset:int=0,
                 max_camera_distance:int=50,
                 sync_interval:int=30,
                 min_overlap_length:int=10,
                 max_track_distance:float=10,
                 max_tracklet_distance:float=7
                ) -> None:
        self.node_ids = node_ids 
        self.nodes:dict[str, Node] = {
            self.node_ids[0]: Node(self.node_ids[0]),
            self.node_ids[1]: Node(self.node_ids[1], offset=offset),
        }
        self.max_camera_distance = max_camera_distance
        self.min_track_overlap = min_overlap_length
        self.max_track_distance = max_track_distance
        self.max_tracklet_distance = max_tracklet_distance
        self.sync_interval = sync_interval
        self.batch_remains = sync_interval
        self.matches:dict[tuple[str,str],float] = dict()
        self.dismatches:set[tuple[str,str]] = set()
            
    def handle_event(self, ev:NodeTrack) -> None:
        def is_valid_track(ev:NodeTrack) -> bool:
            if ev.is_deleted():
                return True
            elif ev.state != TrackState.Confirmed and ev.state != TrackState.Tentative:
                return False
            return ev.distance and ev.distance <= self.max_camera_distance and ev.location
        if not is_valid_track(ev):
            return
        
        node = self.nodes.get(ev.node_id)
        if node is None:
            return
        
        node.append(ev)
        self.batch_remains -= 1
        if self.batch_remains > 0 or len(self.nodes) < 2:
            return
        
        tracklet1, tracklet2 = tuple(node.tracklets for node in self.nodes.values())
        ignore_matches = self.matches.keys() | self.dismatches
        matches, dismatches = match_tracklets(tracklet1, tracklet2,
                                              ignore_matches=ignore_matches,
                                              min_overlap_length=self.min_track_overlap,
                                              max_track_distance=self.max_tracklet_distance,
                                              max_tracklet_distance=self.max_tracklet_distance)
        for m, d in matches.items():
            self.replace_match(m, d)
        self.dismatches.update(dismatches)
        for node in self.nodes.values():
            if node.id != ev.node_id:
                offset = self.nodes[ev.node_id].offset
                frame_index = ev.frame_index + offset
                node.purge_deleted_tracklet(frame_index, self.min_track_overlap)
        self.batch_remains = self.sync_interval
        
    def replace_match(self, match:tuple[str,str], dist:float) -> None:
        prev_match, prev_dist = next(((m, d) for m, d in self.matches.items() if m[0] == match[0]),  (None,-1))
        if prev_match:
            if prev_dist > dist:
                self.matches.pop(prev_match, None)
                self.matches[match] = dist
                print(f'{match}: dist={dist:.3f}')
        else:
            self.matches[match] = dist
            print(f'{match}: dist={dist:.3f}')
