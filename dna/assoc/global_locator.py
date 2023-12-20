from __future__ import annotations

from typing import Optional, Union
from collections.abc import Iterable, Generator
from dataclasses import dataclass, field
from collections import namedtuple
from bisect import bisect_left, insort
import functools
import json
import logging

from dna import NodeId, TrackletId, Point
from dna.utils import get_or_else
from dna.support.func import Option
from dna.event import NodeTrack, KafkaEventPublisher, KafkaEvent, SimpleKafkaEvent, TimeElapsed
from dna.support import iterables


class GlobalLocation:
    __slots__ = ("contributors", "_coordinate_cache", "_cost_cache", "frame_index", "ts" )
    
    def __init__(self, *tracks:NodeTrack) -> None:
        self.contributors = { t.node_id:t for t in tracks }
        self._coordinate_cache = tracks[0].location if len(tracks) == 1 else None
        self._cost_cache = 0
        
        tup = max(((t.ts, t) for t in tracks), key=lambda t: t[0])
        self.frame_index = tup[1].frame_index
        self.ts = tup[1].ts
        
    def get(self, node_id:NodeId) -> Optional[NodeTrack]:
        return self.contributors.get(node_id)
        
    def update(self, new_track:NodeTrack) -> dict(str,object):
        self.contributors[new_track.node_id] = new_track
        self.frame_index = max(self.frame_index, new_track.frame_index)
        self.ts = max(self.ts, new_track.ts)
            
        # 기존 cache된 추정 위치 좌표를 무효화하여 다음번에 다시 계산되도록 한다.
        self._coordinate_cache = None
        self._cost_cache = None
        
    @property
    def mean(self) -> Point:
        if self._coordinate_cache is None:
            self._coordinate_cache = GlobalLocation._avg_coordinate(self.contributors.values())
        return self._coordinate_cache
        
    @property
    def error(self) -> float:
        """본 global location의 cost를 반환한다.
        Cost는 본 global location의 위치와 본 global location에 기여하는
        모든 point들의 위치들과의 평균 거리 값으로 정의된다.

        Returns:
            float: cost.
        """
        if not self._cost_cache:
            center = self.mean
            total = sum(center.distance_to(sample.location) for sample in self.contributors.values())
            self._cost_cache = total / len(self.contributors)
            
        return self._cost_cache
    
    def remove_track(self, track:NodeTrack) -> bool:
        if track in self:
            self.contributors.pop(track.node_id, None)
            
            # 기존 cache된 추정 위치 좌표를 무효화하여 다음번에 다시 계산되도록 한다.
            self._coordinate_cache = None
            self._cost_cache = None
            
            self.frame_index = max(self.frame_index, track.frame_index)
            self.ts = max(self.ts, track.ts)
            return True
        else:
            return False
        
    def distance(self, track:NodeTrack) -> float:
        return self.mean.distance_to(track.location)
        
    # def distance(self, track:TrackEvent) -> float:
    #     if track in self:
    #         # 주어진 track이 이미 contribution에 포함된 경우는 평균 위치로부터의 거리를 반환한다.
    #         return self.mean.distance_to(track.world_coord)
    #     else:
    #         # 주어진 track이 포함됐을 때의 평균 위치를 구하고, 이곳으로부터의 거리를 반환한다.
    #         samples = [contrib for contrib in self.contributors.values() if contrib.node_id != track.node_id]
    #         samples.append(track)
    #         avg_coordinate = GlobalLocation._avg_coordinate(samples)
    #         return avg_coordinate.distance_to(track.world_coord)
        
    def farthest(self) -> Optional[NodeTrack]:
        center = self.mean
        return max((t for t in self.contributors.values()), key=lambda te: center.distance_to(te.location))
        
    def closest(self) -> Optional[NodeTrack]:
        center = self.mean
        return min((t for t in self.contributors.value()), key=lambda te: center.distance_to(te.world_coord))
        
    def ordered_contributions(self, *, reversed:bool=False) -> list[NodeTrack]:
        center = self.mean
        return sorted((t for t in self.contributors.value()), key=lambda te: center.distance_to(te.location))
        
    @staticmethod
    def _avg_coordinate(samples:Iterable[NodeTrack]) -> Point:
        # total = sum(1/s.distance for s in samples)
        # weighted_coord = [s.world_coord * 1/(s.distance*total) for s in samples]
        # return functools.reduce(lambda pt1, pt2: pt1 + pt2, weighted_coord)
        
        w_coords = [s.location for s in samples]
        return functools.reduce(lambda pt1, pt2: pt1 + pt2, w_coords) / len(samples)
        
    def __len__(self) -> int:
        return len(self.contributors)
        
    def __bool__(self) -> bool:
        return bool(self.contributors)
    
    def __contains__(self, key:Union[NodeId,TrackletId,NodeTrack]) -> bool:
        if isinstance(key, NodeId.__supertype__):
            return key in self.contributors
        else:
            track = self.contributors.get(key.node_id)
            return track and track.track_id == key.track_id
        
    def __repr__(self) -> str:
        def contrib_str(contrib:NodeTrack):
            dist = contrib.location.distance_to(self.mean)
            return f"{contrib.tracklet_id}({dist:.1f})"
        
        if self.contributors:
            str = ','.join(contrib_str(contrib) for contrib in self.contributors.values())
            return f"[C={self.error:.1f}] {str}"
        else:
            return "{}"
        
    def _to_json_object(self) -> dict[str,object]:
        def _to_json_contribution_list(contribs:Iterable[NodeTrack]) -> list[dict[str,object]]:
            return [_to_json_contribution(c) for c in contribs]
        def _to_json_contribution(contrib:NodeTrack) -> dict[str,object]:
            return { 'node': contrib.node_id, 'track_id': contrib.track_id,
                    'location': list(contrib.location.xy)}
            
        return {
            'location': self.mean,
            'error': self.error,
            'contributors': _to_json_contribution_list(self.contributors),
            'frame_index': self.frame_index,
            'ts': self.ts,
        }


ValidCameraDistanceRange = namedtuple('ValidCameraDistanceRange', "enter_range, exit_range")
_DEFAULT_DISTANCE_RANGE = ValidCameraDistanceRange(47, 50)

class GlobalTrackLocator:
    __slots__ = ( "locations", "distinct_dist", "distance_ranges", "publisher", "logger" )
    
    def __init__(self, distinct_dist:float,
                 *,
                 distance_ranges: Optional[dict[str,ValidCameraDistanceRange]],
                 publisher: Optional[KafkaEventPublisher]=None,
                 logger: Optional[logging.Logger]=None) -> None:
        self.locations:list[GlobalLocation] = []
        self.distinct_dist = distinct_dist
        self.distance_ranges = distance_ranges
        self.publisher = publisher
        self.logger = logger
        
    def get(self, track:NodeTrack) -> Optional[GlobalLocation]:
        return iterables.first(gloc for gloc in self.locations if track in gloc)
        
    def update(self, track:NodeTrack) -> Optional[GlobalLocation]:
        # track이 포함된 global location을 검색한다.
        gloc = self.get(track)
        
        # track이 종료된 경우는 별도로 처리한다.
        if track.is_deleted():
            if gloc:
                self.__remove_track_with_msg(gloc, track)
            return gloc
        
        range = get_or_else(self.distance_ranges.get(track.node_id), _DEFAULT_DISTANCE_RANGE)
        
        # track의 카메라로부터의 거리 (distance)가 일정 거리 (exit_range)보다
        # 먼 경우 locator에서 무조건 삭제시킨다.
        if gloc and track.distance > range.exit_range:
            self.__remove_track_with_msg(gloc, track)
            return gloc
        
        # track이 등록되어 있지 않은 상태에서 카메라로부터의 거리 (distance)가
        # 일정 거리 (enter_range)보다 먼 경우는 무시한다.
        if not gloc and track.distance > range.enter_range:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"ignore unstable track:  {track.tracklet_id}#{track.frame_index}")
            return None
        
        return self.__assign(track, self.locations)
    
    def __assign(self, track:NodeTrack, locations:list[GlobalLocation]) -> GlobalLocation:
        def is_closer_than_competitor(gloc:GlobalLocation, track:NodeTrack, track_dist:float) -> bool:
            competitor = gloc.get(track.node_id)
            if competitor is None:
                return True
            elif competitor.tracklet_id == track.tracklet_id:
                return True
            elif len(gloc) == 1:
                # competitor가 해당 global location의 유일한 contribution인 경우는 제외시킨다.
                return False
            else:
                return gloc.distance(track) < gloc.distance(competitor)
            
        # 약간 loose한 distance threshold를 이용하여 거리가 먼 global location들을 사전에 제거한다.
        # 이 방법을 사용하는 이유는 'gloc.distance(track)'의 overhead가 크기 때문에
        # 거리가 확실히 먼 global location들을 걸러내기 위함이다.
        rough_dist = self.distinct_dist * 1.3
        locations = [gloc for gloc in self.locations if gloc.distance(track) <= rough_dist]
        
        # 주어진 track 과의 거리가 'distinct_dist' 이내의 global location들을 찾아 거리 순서대로 정렬시킨다.
        # 만일 검색된 global location에 동일 노드에서 검출된 다른 track이 존재하는 경우에는
        # 그 track보다 거리가 짧은 global location만 포함시킨다.
        close_gloc_tups = ((gloc, dist) for gloc in locations if (dist:=gloc.distance(track)) <= self.distinct_dist)
        close_gloc_tups = [(gloc,dist) for gloc, dist in close_gloc_tups if is_closer_than_competitor(gloc, track, dist)]
        sorted_gloc_list = [gloc for gloc, _ in sorted(close_gloc_tups, key=lambda t: t[1])]
        
        if sorted_gloc_list:
            return self.__select(track, sorted_gloc_list)
        
        # track과 근접한 global location이 없는 경우 새 global location을 생성한다.
        # 만일 track이 frame에 따라 위치가 급격히 변한 경우에는 동일 track을 포함하는 global location이
        # 존재할 수도 있기 때문에 삭제를 해본다.
        prev_gloc = self.get(track)
        if prev_gloc:
            if len(prev_gloc) == 1:
                prev_gloc.update(track)
                return prev_gloc
            else:
                self.__remove_track(prev_gloc, track)
                
        new_gloc = GlobalLocation(track)
        
        # 다른 global location에 포함된 다른 node들의 track 중에서 새로 만들어진
        # global location과 더 가까운 track이 있는가 조사하여 이들을 re-assign 시킨다.
        new_contrib_infos = ((gloc, contrib, new_dist)
                                for gloc, contrib in self.gen_contributions()
                                    if track.node_id != contrib.node_id
                                        and (new_dist:=new_gloc.distance(contrib)) < gloc.distance(contrib))
        new_contrib_infos = sorted(new_contrib_infos, key=lambda info: info[2])
        if new_contrib_infos:
            groups_by_node = iterables.groupby(new_contrib_infos, key_func=lambda t:t[1].node_id)
            for info_list in groups_by_node.values():
                prev_gloc, contrib, _ = info_list[0]
                self.__remove_track(prev_gloc, contrib)
                new_gloc.update(contrib)
        
        self.locations.append(new_gloc)
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"create a new global location: {new_gloc}#{track.frame_index}")
        if self.publisher:
            self.publisher.handle_event(GlobalLocationEvent.CREATED("etri", new_gloc))
            
        return new_gloc
    
    def __select(self, track:NodeTrack, sorted_valid_glocs:list[GlobalLocation]):
        prev_gloc = self.get(track)
        
        closest = sorted_valid_glocs[0]
        prev_track = closest.get(track.node_id)
        if prev_track is None:
            # 가장 가까운 global location에 대상 track이 없는 경우 추가시킨다.
            if prev_gloc and closest is not prev_gloc:
                self.__remove_track(prev_gloc, track)
            closest.update(track)
        elif track.tracklet_id == prev_track.tracklet_id:
            # 가장 가까운 gloc에 이미 동일 track이 존재하는 경우
            if len(closest) == 1 and len(sorted_valid_glocs) >= 2:
                # 만일 자신이 포함된 gloc이 자신만으로 구성된 location이고, 
                # 다른 가까운 gloc이 존재하는 경우 해당 gloc과 merge를 시도한다.
                self.__remove_track(prev_gloc, track)
                return self.__select(track, sorted_valid_glocs[1:])
            closest.update(track)
        else:
            assert closest.distance(track) < closest.distance(prev_track)
            
            # 가장 가까운 gloc에 동일 node의 다른 track이 존재하는 경우
            # 그 track을 다른 gloc에 할당시킨다.
            if prev_gloc and closest is not prev_gloc:
                self.__remove_track(prev_gloc, track)
            closest.update(track)
            self.__assign(prev_track, self.locations)
        
        for c in closest.contributors.values():
            if c.location.distance_to(closest.mean) > self.distinct_dist:
                # TODO: global location update 후, 몇몇 contribution과의 거리가 distinct_dist보다 멀어질 수 있기 때문에 이를 처리해야 한다.
                pass
        
        return closest
    
    def __remove_track_with_msg(self, gloc:GlobalLocation, track:NodeTrack) -> None:
        self.__remove_track(gloc, track)
        if len(gloc) == 0:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"removed the last track: track={track.tracklet_id}#{track.frame_index}, gloc={gloc}")
            if self.publisher:
                self.publisher.handle_event(GlobalLocationEvent.DELETED("etri", gloc))
        else:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"removed a track from a global location: track={track.tracklet_id}#{track.frame_index}, gloc={gloc}")
            if self.publisher:
                self.publisher.handle_event(GlobalLocationEvent.LOCATION_UPDATED("etri", gloc))
                
    def __remove_track(self, gloc:GlobalLocation, track:NodeTrack) -> None:
        gloc.remove_track(track)
        if len(gloc) == 0:
            # 제거 후 global location이 empty인 경우에는 그 global location도 함께 제거한다.
            self.locations.remove(gloc)
            
    def __len__(self):
        return len(self.locations)

    def __iter__(self):
        return iter(self.locations)
        
    def gen_contributions(self) -> Generator[tuple[GlobalLocation,NodeTrack],None,None]:
        for gloc in self.locations:
            for contrib in gloc.contributors.values():
                yield gloc, contrib
        
    def __repr__(self) -> str:
        return '{' + ", ".join(repr(gloc) for gloc in self.locations) + '}'


from enum import Enum
class GlobalLocationEventType(Enum):
    CREATED = 1
    LOCATION_UPDATED = 2
    DELETED = 3
  
def to_contribution_list_json(contrib_list:list[TrackletId]) -> dict[str,object]:
    return [[t.node_id, t.track_id] for t in contrib_list]

def from_contribution_list_json(contrib_list:list[list[object]]) -> list[TrackletId]:
    return [TrackletId(pair[0], pair[1]) for pair in contrib_list]  
    
@dataclass(frozen=True, eq=True, slots=True)     
class GlobalLocationEvent(KafkaEvent):
    overlap_area_id: str
    type: GlobalLocationEventType
    location: Point
    error: float
    contributions: list[TrackletId]
    frame_index: int
    ts: int

    def key(self) -> str:
        return self.overlap_area_id

    def serialize(self) -> object:
        return self.to_json().encode('utf-8')
    
    @staticmethod
    def create(id:str, type:GlobalLocationEventType, gloc:GlobalLocation) -> GlobalLocationEvent:
        return GlobalLocationEvent(
                    overlap_area_id = id,
                    type = type,
                    location = gloc.mean,
                    error = gloc.error,
                    contributions = [te.tracklet_id for te in gloc.contributors.values()],
                    frame_index = gloc.frame_index,
                    ts = gloc.ts,
                )
    
    @staticmethod
    def CREATED(id:str, gloc:GlobalLocation) -> GlobalLocationEvent:
        return GlobalLocationEvent(
                    overlap_area_id = id,
                    type = GlobalLocationEventType.CREATED,
                    location = gloc.mean,
                    error = gloc.error,
                    contributions = [te.tracklet_id for te in gloc.contributors.values()],
                    frame_index = gloc.frame_index,
                    ts = gloc.ts,
                )
    
    @staticmethod
    def LOCATION_UPDATED(id:str, gloc:GlobalLocation) -> GlobalLocationEvent:
        return GlobalLocationEvent(
                    overlap_area_id = id,
                    type = GlobalLocationEventType.LOCATION_UPDATED,
                    location = gloc.mean,
                    error = gloc.error,
                    contributions = [te.tracklet_id for te in gloc.contributors.values()],
                    frame_index = gloc.frame_index,
                    ts = gloc.ts,
                )
    
    @staticmethod
    def DELETED(id:str, gloc:GlobalLocation) -> GlobalLocationEvent:
        return GlobalLocationEvent(
                    overlap_area_id = id,
                    type = GlobalLocationEventType.DELETED,
                    location = gloc.mean,
                    error = gloc.error,
                    contributions = [te.tracklet_id for te in gloc.contributors.values()],
                    frame_index = gloc.frame_index,
                    ts = gloc.ts,
                )

    @staticmethod
    def from_json(json_str:str) -> GlobalLocationEvent:
        json_obj = json.loads(json_str)
        return GlobalLocationEvent(
                    overlap_area_id = json_obj['overlap_area_id'],
                    type = GlobalLocationEventType[json_obj['type']],
                    location = Point(json_obj['location']),
                    error = json_obj['error'],
                    contributions = from_contribution_list_json(json_obj['contributions']),
                    frame_index = json_obj['frame_index'],
                    ts = json_obj['ts']
                )

    def to_json(self) -> str:
        serialized = {'overlap_area_id': self.overlap_area_id,
                      'type': self.type.name,
                      'location': [round(v, 3) for v in tuple(self.location.xy)],
                      'error': self.error,
                      'contributions': to_contribution_list_json(self.contributions),
                      'frame_index': self.frame_index,
                      'ts': self.ts }
        return json.dumps(serialized, separators=(',', ':'))
    