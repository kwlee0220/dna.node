from __future__ import annotations

from typing import Union, Optional
from collections.abc import Iterable, Generator
from datetime import timedelta
import logging

import numpy as np

from dna import TrackletId, sub_logger
from dna.event import NodeTrack, TrackDeleted, EventProcessor
from dna.support import iterables
from dna.event.event_processors import TimeElapsedGenerator, EventRelay
from .windows import Window, TumblingWindowAssigner
from .association import Association, BinaryAssociation
from .schema import NodeAssociationSchema
        

class MotionBasedTrackletAssociator(EventProcessor):
    def __init__(self,
                 schema:NodeAssociationSchema,
                 window_interval_ms:int,
                 max_distance_to_camera:float,
                 max_track_distance:float,
                 idle_timeout:float=1,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.schema = schema
        self.max_distance_to_camera = max_distance_to_camera
        self.logger = logger
        
        self.ticker = TimeElapsedGenerator(interval=timedelta(seconds=idle_timeout))
        
        self.windows = TumblingWindowAssigner(nodes=schema.nodes,
                                              window_interval_ms=window_interval_ms,
                                              logger=sub_logger(logger, 'window'))
        self.ticker.add_listener(self.windows)
        
        self.associator = MotionBasedAssociator(schema=schema, max_track_distance=max_track_distance)
        self.windows.add_listener(self.associator)
        
        self.associator.add_listener(EventRelay(target=self))
        
    def close(self) -> None:
        self.ticker.stop()
        self.windows.close()
        super().close()
        
    def start(self) -> None:
        self.ticker.start()
        
    def handle_event(self, ev:NodeTrack) -> None:
        # 일정 거리 이상에서 추정된 물체 위치 정보는 사용하지 않는다.
        if (ev.is_confirmed() or ev.is_tentative()) and ev.distance > self.max_distance_to_camera:
            return
        self.windows.handle_event(ev)
            
    def __len__(self) -> int:
        return len(self.collector)
        
    def __iter__(self) -> Iterable[Association]:
        return iter(self.collector)
        
    def find(self, key:tuple[TrackletId,TrackletId]) -> Optional[Association]:
        closure = self.collector.find(key)
        return self.collector.find(key)

    def get_score(self, trk_pair:tuple[TrackletId, TrackletId], *, estimate:bool=False) -> Optional[float]:
        closure = self.collector.find(trk_pair[0])
        if closure is None:
            return None
        
        if trk_pair[1] in closure.tracklets():
            assoc = closure.find((trk_pair[0].node_id, trk_pair[1].node_id), estimate=estimate)
            return assoc.score if assoc else None
        else:
            return None


class MotionBasedAssociator(EventProcessor):
    def __init__(self, schema:NodeAssociationSchema, max_track_distance:float) -> None:
        super().__init__()
        
        self.schema = schema
        self.max_track_distance = max_track_distance
        
    def close(self) -> None:
        super().close()
        
    def handle_event(self, ev:Union[tuple[Window,Optional[TrackletId]],NodeTrack]) -> None:
        if isinstance(ev, NodeTrack):
            if ev.is_deleted():
                self._publish_event(TrackDeleted(node_id=ev.node_id, track_id=ev.track_id, frame_index=ev.frame_index, ts=ev.ts))
        else:
            for assoc in self.associate(ev[0].events, tracklet_id=ev[1]):
                self._publish_event(assoc)
        
    def associate(self, node_tracks:list[NodeTrack], *,
                  tracklet_id:Optional[TrackletId]=None) -> Generator[Association, None, None]:
        def get_tracklet_id(track:NodeTrack):
            return track.tracklet_id
        def calc_split_distance(events1:list[NodeTrack], events2:list[NodeTrack]) -> float:
            dist = np.mean([te1.location.distance_to(te2.location)
                            for te1, te2 in zip(events1, events2)])
            return dist
        def calc_distance(ev_list1:list[NodeTrack], ev_list2:list[NodeTrack]) -> Association:
            if len(ev_list1) >= len(ev_list2):
                splits = iterables.buffer_iterable(ev_list1, len(ev_list2), skip=1)
                base = ev_list2
            else:
                splits = iterables.buffer_iterable(ev_list2, len(ev_list1), skip=1)
                base = ev_list1
            return min(calc_split_distance(base, split) for split in splits)
        
        # 주어진 NodeTrack event들을 tracklet별로 그룹핑한다.
        tracklets = iterables.groupby(node_tracks, key_func=get_tracklet_id)
        
        # 검색 대상 tracklet 식별자가 주어진 경우, 이 식별자가 tracklet group에 포함되지 않은 경우에는
        # 바로 반환한다.
        if tracklet_id and tracklet_id not in tracklets:
            return
        
        t1 = tracklet_id if tracklet_id else iterables.first(tracklets.keys())
        while tracklets:
            # tracklet별로 모은 event들을 사용
            events1 = tracklets.pop(t1)
            for peer_node in self.schema.peers(t1.node_id):
                # 'tracklets' group들 중에서 t1과 overlap되는 node들의 식별자를 찾는다.
                peer_tracklet_ids = (t2 for t2 in tracklets.keys() if t2.node_id == peer_node)
                for t2 in peer_tracklet_ids:
                    # t1과 overlap되는 tracklet t2의 이벤트를 얻어, t1에 속한 event들 사이의
                    # 거리를 얻어 두 tracklet 사이의 거리를 계산한다.
                    # 이때, 두 tracklet의 거리는 t1과 t2에 속하는 event pair 중에서
                    # 그 거리 값이 가장 가까운 값으로 정의한다.
                    events2 = tracklets[t2]
                
                    dist = calc_distance(events1, events2)
                    if dist <= self.max_track_distance:
                        score = 1 - (dist / self.max_track_distance)
                        ts = max(events1[-1].ts, events2[-1].ts)
                        yield BinaryAssociation(t1, t2, score, ts)
                        
            if tracklet_id:
                return
            t1 = iterables.first(tracklets.keys())