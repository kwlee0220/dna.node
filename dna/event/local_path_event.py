from dataclasses import dataclass, field
import json

from dna import TrackletId
from dna.event import KafkaEvent


@dataclass(frozen=True, eq=True, order=True, slots=True)
class LocalPathEvent(KafkaEvent):
    node_id: str
    track_id: str
    camera_path:str = field(compare=False, repr=False, hash=False)
    world_path:str = field(compare=False, repr=False, hash=False)
    first_frame: int
    last_frame: int
    continuation:bool = field(compare=False)

    def key(self) -> str:
        return str(TrackletId(self.node_id, self.track_id))
    
    def serialize(self) -> bytes:
        serialized = { 'node_id': self.node_id, 'track_id': self.track_id,
                        'camera_path': self.camera_path, 'world_path': self.world_path,
                        'first_frame': self.first_frame, 'last_frame': self.last_frame,
                        'continuation': self.continuation }
        return json.dumps(serialized, separators=(',', ':')).encode('utf-8')