from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from dna import ByteString, NodeId, TrackId, TrackletId
from .types import KafkaEvent
from .proto.reid_feature_pb2 import TrackFeatureProto


@dataclass(frozen=True, eq=True, order=False, repr=False)   # slots=True
class TrackFeature(KafkaEvent):
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int
    ts: int = field(hash=False)
    feature: Optional[np.ndarray] = field(default=None)

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)
    
    def is_deleted(self) -> bool:
        return self.feature is None

    def serialize(self) -> bytes:
        return self.to_bytes()

    def deserialize(binary_data:ByteString) -> TrackFeature:
        return TrackFeature.from_bytes(binary_data)

    def to_bytes(self) -> bytes:
        proto = TrackFeatureProto()
        proto.node_id = self.node_id
        proto.track_id = self.track_id
        if self.feature is not None:
            proto.feature.extend(self.feature.tolist())
        proto.frame_index = self.frame_index
        proto.ts = self.ts

        return proto.SerializeToString()

    @staticmethod
    def from_bytes(binary_data:bytes) -> TrackFeature:
        proto = TrackFeatureProto()
        proto.ParseFromString(binary_data)
        
        feature = np.array(proto.feature, dtype=np.float32) if len(proto.feature) > 0 else None
        
        return TrackFeature(node_id=proto.node_id, track_id=proto.track_id, feature=feature,
                            frame_index=proto.frame_index, ts=proto.ts)

    def __repr__(self) -> str:
        # dt = utc2datetime(self.ts)
        return f'{self.__class__.__name__}[id={self.node_id}[{self.track_id}], frame={self.frame_index}, ts={self.ts}]'