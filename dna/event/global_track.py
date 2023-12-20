from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
import json

from dna import ByteString, Point
from dna.event import KafkaEvent


@dataclass(frozen=True, eq=False, order=False, repr=False)   # slots=True
class LocalTrack:
    node: str           # node id
    track_id: str       # track id
    location: Point = field(hash=False)
    ts: int = field(hash=False)

    def key(self) -> str:
        return self.node
    
    def is_same_track(self, ltrack:LocalTrack) -> bool:
        return self.node == ltrack.node and self.track_id == ltrack.track_id

    @staticmethod
    def from_json_object(json_obj:dict[str,object]) -> LocalTrack:
        return LocalTrack(node=json_obj['node'], track_id=json_obj['track_id'],
                            location=Point(json_obj.get('location')),
                            ts=json_obj['ts'])

    def to_json_object(self) -> str:
        return {
            'node': self.node,
            'track_id': self.track_id,
            'location': list(self.location.xy),
            'ts': self.ts
        }

    def __repr__(self) -> str:
        return f"{self.node}[{self.track_id}]:{self.location}#{self.ts}"

    
from enum import Enum
class GlobalTrackType(Enum):
    ASSOCIATED = 'ASSOCIATED'
    MERGED = 'MERGED'
    ISOLATED = 'ISOLATED'
    DELETED = 'DELETED'

@dataclass(frozen=True, eq=False, order=False, repr=False)      # slots=True
class GlobalTrack(KafkaEvent):
    id: str                                                     # id
    state: GlobalTrackType                                      # state
    overlap_area: Optional[str] = field(hash=False)             # overlap area (nullable)
    location: Optional[Point] = field(hash=False)               # location (nullable)
    supports: Optional[list[LocalTrack]] = field(hash=False)    # nullable
    first_ts: int = field(hash=False)
    ts: int = field(hash=False)

    def key(self) -> str:
        return self.id
    
    def is_deleted(self) -> bool:
        return self.state == GlobalTrackType.DELETED
    
    def is_associated(self) -> bool:
        return self.state == GlobalTrackType.ASSOCIATED
    
    def is_in_overlap_area(self) -> bool:
        return self.overlap_area != None
    
    def is_same_track(self, ltrack:LocalTrack) -> bool:
        return self.node == ltrack.node and self.track_id == ltrack.track_id

    @staticmethod
    def from_json(json_str:str) -> GlobalTrack:
        json_obj = json.loads(json_str)
        
        state = GlobalTrackType(json_obj['state'])
        
        loc_obj = json_obj.get('location')
        location = Point(loc_obj) if loc_obj else None
            
        support_json_obj = json_obj.get('supports')
        supports = None if support_json_obj is None \
                        else [LocalTrack.from_json_object(sj) for sj in support_json_obj]
        
        return GlobalTrack(id=json_obj['id'], state=state, overlap_area=json_obj.get('overlap_area'),
                            location=location, supports=supports,
                            first_ts=json_obj['first_ts'], ts=json_obj['ts'])

    def to_json(self) -> str:
        serialized = {
            'id': self.id,
            'state': self.state.value,
        }
        if self.location:
            serialized['location'] = list(self.location)
        if self.overlap_area:
            serialized['overlap_area'] = self.overlap_area
        if self.supports:
            serialized['support'] = [s.to_json_object() for s in self.supports]
        serialized['first_ts'] = self.first_ts
        serialized['ts'] = self.ts
            
        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> bytes:
        return self.to_json().encode('utf-8')

    @staticmethod
    def deserialize(serialized:ByteString) -> GlobalTrack:
        return GlobalTrack.from_json(serialized.decode('utf-8'))
    
    def __lt__(self, other:GlobalTrack) -> bool:
        return self.ts < other.ts

    def __repr__(self) -> str:
        if self.is_deleted():
            return f"{self.id}: Deleted#{self.ts}"
        
        overlap_area_str = f'{self.overlap_area}:' if self.overlap_area else ''
        
        support_str = ""
        if self.supports:
            lt_str = '-'.join(f"{lt.node}[{lt.track_id}]" for lt in self.supports)
            support_str = f'{{{lt_str}}}'
            
        tail_str = f" - {overlap_area_str}{support_str}"
        return f"{self.id}:{self.location}#{self.ts}{tail_str}"