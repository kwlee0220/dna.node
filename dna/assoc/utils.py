from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
import sys
import logging

from dna import TrackletId
from dna.support import iterables
from dna.event import EventProcessor, EventListener, TrackDeleted
from .association import Association, BinaryAssociation
from .collection import AssociationCollection, AssociationCollector
from .closure import AssociationClosure, AssociationClosureBuilder, PartialMatch
from .associator_motion import NodeAssociationSchema
                        

class AssociationCloser(EventListener):
    def __init__(self, collection:AssociationCollection) -> None:
        super().__init__()
        self.collection = collection
        
    def handle_event(self, track_deleted:TrackDeleted) -> None:
        if isinstance(track_deleted, TrackDeleted):
            trk = track_deleted.tracklet_id
            for assoc in self.collection.query(condition=PartialMatch(trk)):
                assoc.close(tracklet=trk)
                
                
class ClosedTrackletCollector(EventListener):
    def __init__(self) -> None:
        super().__init__()
        self.tracklets = set()
        
    def handle_event(self, ev:TrackDeleted) -> None:
        if isinstance(ev, TrackDeleted):
            self.tracklets.add(ev.tracklet_id)
            
    def update_association(self, assoc:Association) -> None:
        if not assoc.is_closed():
            if isinstance(assoc, AssociationClosure):
                for src_assoc in assoc.source_associations:
                    for trk in src_assoc:
                        if trk in self.tracklets:
                            src_assoc.close(tracklet=trk)
            else:
                for trk in assoc:
                    if trk in self.tracklets:
                        assoc.close(tracklet=trk)
        
                    
class ClosedAssociationPublisher(EventProcessor):
    def __init__(self, collection:AssociationCollection,
                 *,
                 publish_partial_close:bool=False) -> None:
        super().__init__()
        self.collection = collection
        self.publish_partial_close = publish_partial_close
        
    def handle_event(self, ev:TrackDeleted|object) -> None:
        def find_closed_associations(assoc_list:Iterable[Association], trk:TrackletId):
            return (assoc for assoc in assoc_list if trk in assoc and assoc.is_closed())
    
        if isinstance(ev, TrackDeleted):
            trk_id = ev.tracklet_id
            if self.publish_partial_close:
                for assoc in self.collection:
                    if isinstance(assoc, AssociationClosure):
                        for ca in find_closed_associations(assoc.source_associations, trk_id):
                            self._publish_event(ca)
            else:
                for ca in find_closed_associations(self.collection, trk_id):
                    self._publish_event(ca)
        

class FixedIntervalClosureBuilder(EventProcessor):
    __slots__ = ( 'interval_ms', 'first_ms', 'last_ms', 'closure_builder', 'logger' )
    
    def __init__(self, interval_ms:int,
                 *,
                 closer_collector:Optional[ClosedTrackletCollector]=None,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.interval_ms = interval_ms
        self.first_ms:int = sys.maxsize
        self.last_ms:int = 0
        self.closure_builder = AssociationClosureBuilder()
        self.closer_collector = closer_collector
        self.logger = logger

    def close(self) -> None:
        for selection in self.select_best_associations():
            self._publish_event(selection)
        super().close()
        
    def handle_event(self, ev:Association) -> None:
        if isinstance(ev, Association):
            # 추가된 association을 바로 closure 생성에 사용하지 않고, 'interval_ms' 동안 수집한다.
            self.closure_builder.handle_event(ev)
            self.first_ms = min(self.first_ms, ev.ts)
            self.last_ms = max(self.last_ms, ev.ts)
            
            if (self.last_ms - self.first_ms) > self.interval_ms:
                for selection in self.select_best_associations():
                    self._publish_event(selection)
                self.purge_associations(self.first_ms)
                self.first_ms = self.get_min_ts()
                self.last_ms = 0
        elif isinstance(ev, TrackDeleted):
            trk = ev.tracklet_id
            for c in self.closure_builder:
                for src_assoc in c.source_associations:
                    if trk in src_assoc:
                        src_assoc.close(tracklet=trk)
            pass
                
    def select_best_associations(self) -> tuple[list[Association], int]:
        selecteds = []
        sorted_closures = sorted(self.closure_builder.collection, key=lambda c:c.score, reverse=True)
        while sorted_closures:
            selected = sorted_closures[0]
            
            # 선택된 closure에 포함된 tracklet과 동일한 tracklet으로 구성된 closure들을 모두 제거한다.
            for trk in selected.tracklets:
                sorted_closures = [assoc for assoc in sorted_closures if trk not in assoc]
                
            if self.closer_collector:
                self.closer_collector.update_association(selected)
            selecteds.append(selected)
        return selecteds
    
    def purge_associations(self, upper_ts:int) -> None:
        def is_old(assoc:Association) -> bool:
            return assoc.ts <= upper_ts
        self.closure_builder.collection.remove_cond(is_old)
        
    def get_min_ts(self) -> int:
        return min((assoc.ts for assoc in self.closure_builder.collection), default=self.first_ms+1)


class FixedIntervalCollector(EventProcessor):
    __slots__ = ( 'interval_ms', 'first_ms', 'last_ms', 'collection', 'logger' )
    
    def __init__(self, interval_ms:int,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.interval_ms = interval_ms
        self.first_ms:int = sys.maxsize
        self.last_ms:int = 0
        self.collection = AssociationCollection()
        self.logger = logger

    def close(self) -> None:
        for assoc in self.collection:
            self._publish_event(assoc)
        self.collection.clear()
            
        super().close()
        
    def handle_event(self, assoc:Association|TrackDeleted) -> None:
        if isinstance(assoc, Association):
            # 추가된 association을 바로 closure 생성에 사용하지 않고, 'interval_ms' 동안 수집한다.
            self.collection.add(assoc)
            self.first_ms = min(self.first_ms, assoc.ts)    # 수집된 association들 중 가장 오래된 것의 timestamp
            self.last_ms = max(self.last_ms, assoc.ts)      # 수집된 association들 중 가장 최근 것의 timestamp
            
            if (self.last_ms - self.first_ms) > self.interval_ms and self.collection:
                self._publish_old_associations(self.first_ms)
                self.first_ms = self._get_min_ts()
                pass
        elif isinstance(assoc, TrackDeleted):
            trk = assoc.tracklet_id
            for assoc in self.collection:
                if isinstance(assoc, AssociationClosure):
                    for src_assoc in assoc.source_associations:
                        if trk in src_assoc:
                            src_assoc.close(tracklet=trk)
                elif trk in assoc:
                    assoc.close(tracklet=trk)
    
    def _publish_old_associations(self, ts:int) -> None:
        old_assoc_list = self.collection.remove_cond(lambda a: a.ts <= ts)
        for assoc in old_assoc_list:
            self._publish_event(assoc)
        pass
        
    def _get_min_ts(self) -> int:
        return min((assoc.ts for assoc in self.collection), default=self.first_ms+1)