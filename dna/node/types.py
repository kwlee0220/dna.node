from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

from dna import NodeId, TrackId, TrackletId
from dna.utils import utc_now_millis


@dataclass(frozen=True, eq=True, unsafe_hash=True, slots=True)
class SilentFrame:
    frame_index: int
    ts: int = field(default_factory=utc_now_millis)


@dataclass(frozen=True, eq=True, slots=True)
class TrackDeleted:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int = field(hash=False)
    ts: int = field(hash=False)
    source:Any = field(default=None)

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: id={self.node_id}[{self.track_id}], "
                f"frame={self.frame_index}, ts={self.ts}")
