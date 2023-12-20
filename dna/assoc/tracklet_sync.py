from __future__ import annotations

from typing import Union, Optional

import numpy as np

from dna import NodeId, TrackId, TrackletId
from dna.event import NodeTrack, EventProcessor
from dna.assoc import Association
            
            
class AssociationClosure:
    def __init__(self) -> None:
        self.mappings:dict[NodeId,TrackId] = dict()
        
    def track_of(self, node_id:NodeId):
        return self.mappings.get(node_id)
    
    def add_mapping(self, tracklet_id:TrackletId) -> None:
        self.mappings[tracklet_id.node_id] = tracklet_id.track_id
    
    def __contains__(self, id:Union[NodeId,TrackletId]) -> bool:
        if isinstance(id, TrackletId):
            return id.track_id == self.mappings.get(id.node_id)
        elif isinstance(id, NodeId):
            return self.mappings.get(id) is not None
        else:
            raise ValueError(f"invalid id ({id}): neither NodeId or TrackletId")
        
    def __repr__(self) -> str:
        return '-'.join([f"{TrackletId(node_id,self.mappings[node_id])}" for node_id in sorted(self.mappings.keys())])


class AssociationAggregator(EventProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.global_associations:list[AssociationClosure] = []
        
    def close(self) -> None:
        super().close()
        
    def handle_event(self, ev:Union[Association,NodeTrack]) -> None:
        if isinstance(ev, Association):
            self.add_association(ev)
        elif isinstance(ev, NodeTrack):
            pass
            # if ev.is_deleted():
            #     self.handle_track_deleted(ev)
            # else:
            #     raise ValueError(f"unexpected TrackEvent, 'deleted' was expected, but {ev}")
        
    def add_association(self, assoc:Association) -> None:
        print(assoc)
        gassoc1 = self.find_global_assoc(assoc.tracklet1)
        gassoc2 = self.find_global_assoc(assoc.tracklet2)
        if not(gassoc1 or gassoc2):
            gassoc = AssociationClosure()
            gassoc.add_mapping(assoc.tracklet1)
            gassoc.add_mapping(assoc.tracklet2)
            self.global_associations.append(gassoc)
        elif gassoc1 and not gassoc2:
            gassoc1.add_mapping(assoc.tracklet2)
        elif not gassoc1 and gassoc2:
            gassoc2.add_mapping(assoc.tracklet1)
        elif gassoc1 != gassoc2:
            raise ValueError(f"error")
            
    def find_global_assoc(self, tracklet_id:Union[TrackletId,NodeId]) -> Optional[AssociationClosure]:
        for gassoc in self.global_associations:
            if tracklet_id in gassoc:
                return gassoc
        return None


class BestAssociationAggregator(EventProcessor):
    __slots__ = ( 'associations' )
    
    def __init__(self) -> None:
        super().__init__()
        
        self.associations:dict[tuple[TrackletId,TrackletId],float] = dict()
        
    def close(self) -> None:
        for (t1, t2), dist in self.associations:
            self._publish_event(Association(t1, t2, dist))
        self.associations.clear()
            
        super().close()
        
    def handle_event(self, ev:Union[Association,NodeTrack]) -> None:
        if isinstance(ev, Association):
            self.handle_association(ev)
        elif isinstance(ev, NodeTrack):
            if ev.is_deleted():
                self.handle_track_deleted(ev)
            else:
                raise ValueError(f"unexpected TrackEvent, 'deleted' was expected, but {ev}")
            
    def handle_association(self, assoc:Association) -> None:
        key, dist = BestAssociationAggregator.key_value(assoc)
        
        min_dist = self.associations.get(key)
        if min_dist is None or min_dist > dist:
            self.associations[key] = dist
            
    def handle_track_deleted(self, te:NodeTrack) -> None:
        fixeds = [(key, score) for key, score in self.associations.items() if te.tracklet_id in key]
        for key, _ in fixeds:
            del self.associations[key]
        for (t1, t2), score in fixeds:
            self._publish_event(Association(t1, t2, score))
           
    @staticmethod 
    def key_value(assoc:Association) -> tuple[tuple[TrackletId,TrackletId], float]:
        tracklet1, tracklet2, dist = assoc
        if tracklet1.node_id > tracklet2.node_id:
            tracklet1, tracklet2 = tracklet2, tracklet1
        return (tracklet1, tracklet2), dist