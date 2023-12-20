from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
import logging 
from bisect import bisect_left, insort

import numpy as np

from dna import NodeId, TrackletId
from dna.support import iterables
from dna.event import EventProcessor
from .association import Association, BinaryAssociation
from .collection import AssociationCollection, PartialMatch
    

class AssociationClosure(Association):
    __slots__ = ('_tracklets', '_associations')
    
    def __init__(self) -> None:
        super().__init__()
        
        self._tracklets:set[TrackletId] = set()
        self._associations:list[BinaryAssociation] = []
        
    def validate(self) -> None:
        if len(self._tracklets) != len(self.nodes):
            raise ValueError(f'11111111')
        for src_assoc in self._associations:
            if not src_assoc.tracklets.issubset(self.tracklets):
                raise ValueError(f'22222')
        
    def is_closed(self) -> bool:
        return all(assoc.is_closed() for assoc in self._associations)
    
    @property
    def tracklets(self) -> set[TrackletId]:
        return self._tracklets
        
    @property
    def score(self) -> float:
        return np.mean([assoc.score for assoc in self._associations])
        
    @property
    def ts(self) -> int:
        if self._associations:
            return max(assoc.ts for assoc in self._associations)
        else:
            return 0
        
    @property
    def source_associations(self) -> list[Association]:
        return self._associations
    
    def min_ts(self) -> int:
        return min((sa.ts for sa in self._associations), default=0)
        
    def is_closed(self, *, node:Optional[NodeId]=None, tracklet:Optional[TrackletId]=None) -> bool:
        if tracklet:
            node = tracklet.node_id
        if node:
            for assoc in self._associations:
                if node in assoc:
                    return assoc.is_closed(node=node)
            raise ValueError(f'invalid node={node}')
        else:
            return all(assoc.is_closed() for assoc in self._associations)
    
    def close(self, *, node:Optional[NodeId]=None, tracklet:Optional[TrackletId]=None) -> bool:
        if tracklet:
            node = tracklet.node_id
        for src_assoc in self._associations:
            if node in src_assoc:
                src_assoc.close(node=node)
        
    def is_more_specific(self, assoc:Association) -> bool:
        if not isinstance(assoc, AssociationClosure):
            return super().is_more_specific(assoc)
        
        # 본 association을 구성하는 tracklet의 수가 'assoc'의 tracklet 수보다 작다면
        # 'more-specific'일 수 없기 때문에 'Fase'를 반환한다.
        if len(self) < len(assoc):
            return False
        
        if self.is_superset(assoc):
            if len(self) > len(assoc):
                return True
            
            if len(self._associations) > len(assoc._associations):
                return True
            elif len(self._associations) < len(assoc._associations):
                return False
            else:
                # self와 assoc은 서로 동일한 tracklet으로 구성된 closure인 경우
                return self.score > assoc.score
        else:
            return False
            
    def extend(self, assoc:Association) -> bool:
        def add_source_associations(assoc_list:Iterable[Association]) -> Iterable[TrackletId]:
            for assoc in assoc_list:
                self._associations.append(assoc)
                self._tracklets.update(assoc.tracklets)
        
        other_assocs = assoc._associations if isinstance(assoc, AssociationClosure) else [assoc]
        if len(self._associations) == 0:
            add_source_associations(other_assocs)
            return True
            
        replacements = []
        newbies = []
        for other in other_assocs:
            t_this = iterables.find(enumerate(self._associations), key=other, keyer=lambda t:t[1])
            if t_this:
                if t_this[1].score < other.score:
                    replacements.append((t_this[0], t_this[1], other))
            else:
                newbies.append(other)
        
        for idx, _, other in replacements:
            self._associations.pop(idx)
            self._associations.append(other)
        add_source_associations(newbies)
        
        return bool(replacements) or bool(newbies)
        
    def remove_node(self, node:NodeId) -> Optional[AssociationClosure]:
        key = self.tracklet(node)
        if key:
            new_assoc_list = [assoc for assoc in self._associations if key not in assoc]
            if new_assoc_list:
                removed = AssociationClosure()
                for assoc in new_assoc_list:
                    removed.extend(assoc)
                return removed
            else:
                return None
        else:
            raise ValueError(f'invalid node: {node}')
        
    def copy(self) -> AssociationClosure:
        dupl = AssociationClosure()
        dupl._tracklets = self._tracklets.copy()
        dupl._associations = self._associations.copy()
        return dupl
        
    def _find(self, key:list[TrackletId]) -> tuple[int,Optional[Association]]:
        for idx, assoc in enumerate(self.association):
            if assoc.tracklets == key:
                return idx, assoc
        return -1, None


from enum import Enum
class ExtendType(Enum):
    UNCHANGED = 1,
    EXTENDED = 2,
    CREATED = 3,

class AssociationClosureBuilder(EventProcessor):
    def __init__(self,
                 *,
                 collection:Optional[AssociationCollection]=None,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.collection = collection if collection is not None else AssociationCollection()
        self.logger = logger
        
    def close(self) -> None:
        super().close()
        
    def handle_event(self, ev:Association) -> None:
        if isinstance(ev, BinaryAssociation):
            self.build(ev)
        elif isinstance(ev, AssociationClosure):
            for src_assoc in ev.source_associations:
                self.build(src_assoc)
            pass
    
    def build(self, assoc:BinaryAssociation):
        def add_link(from_trk, to_trk, assoc) -> None:
            return [extended for c in self.collection.query(PartialMatch(from_trk)) if (extended := self.extend(c, to_trk, assoc))]
                      
        trks = list(assoc.tracklets)
        extendeds1 = add_link(trks[0], trks[1], assoc)
        extendeds2 = add_link(trks[1], trks[0], assoc)
        if not extendeds1 and not extendeds2:
            # Closure collection 내에 두 tracklet을 포함한 closure가 없는 경우
            # 'assoc'으로만 이루어진 새 closure를 추가한다.
            closure = AssociationClosure()
            closure.extend(assoc)
            closure.validate()
            if self.collection.add(closure):
                self._publish_event(closure)
                
    def extend(self, closure:AssociationClosure, to_trk:TrackletId, assoc:Association) -> AssociationClosure:
        prev_trk = closure.tracklet(to_trk.node_id)
        if prev_trk and prev_trk != to_trk:    # 이미 동일 node에 다른 tracklet과 연결된 경우.
            new_closure = closure.remove_node(to_trk.node_id)
            if new_closure is None:
                new_closure = AssociationClosure()
            new_closure.extend(assoc)
            new_closure.validate()
            
            if (ok := self.collection.add(new_closure)):
                self._publish_event(new_closure)
            return new_closure if ok else None
        else:
            if (ok := closure.extend(assoc)):
                # collection에 'keep_best_association_only' 속성이 설정된 경우
                # 구성 association들에 영향을 줄 수 있기 때문에 closure가 확장된 경우
                # collection에서 제거한 후 다시 add시킨다.
                self.collection.remove(closure.tracklets)
                closure.validate()
                if self.collection.add(closure):
                    self._publish_event(closure)
                    return True
            return False
        
    def __iter__(self) -> Iterable[Association]:
        return iter(self.collection)
        

class Extend(EventProcessor):
    def __init__(self, collection:AssociationCollection) -> None:
        super().__init__()
        self.left = collection
    
    def handle_event(self, closure:AssociationClosure) -> None:
        if isinstance(closure, AssociationClosure):
            if any(left_assoc.is_conflict(closure) for left_assoc in self.left):
                return
            
            for left_assoc in self.left:
                if left_assoc.is_mergeable(closure):
                    done = False
                    for src_assoc in closure.source_associations:
                        done = done or left_assoc.extend(src_assoc)
                    if done:
                        self._publish_event(left_assoc)