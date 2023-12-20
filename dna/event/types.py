from __future__ import annotations

from typing import TypeAlias, TypeVar, Any, Protocol, runtime_checkable
from collections.abc import Callable
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import json
import time

from dna import ByteString, NodeId, TrackId, TrackletId, Point
from dna.track import TrackState


@runtime_checkable
class Timestamped(Protocol):
    @property
    def ts(self) -> int: ...
    
TimestampedT = TypeVar("TimestampedT", bound=Timestamped)


@runtime_checkable
class TrackEvent(Protocol):
    @property
    def id(self) -> str: ...
    
    @property
    def state(self) -> str: ...
    
    @property
    def location(self) -> Point: ...
    
    @property
    def ts(self) -> int: ...
    
TrackEventT = TypeVar("TrackEventT", bound=Timestamped)


# conceptual inherits 'Timestamped'
@runtime_checkable
class KafkaEvent(Protocol):
    __slots__ = ()
    
    def key(self) -> str:
        """Returns key value for Kafka Producer record.

        Returns:
            str: key value for Kafka Producer record.
        """
    
    def serialize(self) -> bytes:
        """Returns encoded value for Kafka Producer record.

        Returns:
            bytes: encoded value for Kafka Producer record.
        """


@dataclass(frozen=True, eq=True)    # slots=True
class TrackDeleted:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int = field(hash=False)
    ts: int = field(hash=False)
    source:object = field(default=None)

    def key(self) -> str:
        return self.node_id

    def ts(self) -> int:
        return self.ts

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: id={self.node_id}[{self.track_id}], frame={self.frame_index}, ts={self.ts}")


@dataclass(frozen=True)
class TimeElapsed:
    ts: int = field(default_factory=lambda: int(round(time.time() * 1000)))
    

@dataclass(frozen=True, eq=True)
class SilentFrame:
    frame_index: int
    ts: int = field(default_factory=lambda: int(round(time.time() * 1000)))


class TrackEvent(Protocol):
    int:str
    location: Point
    ts:int
    

@runtime_checkable
class JsonEvent(Timestamped, Protocol):
    __slots__ = ()
    
    def to_json(self) -> str: ...
    
    @classmethod
    def from_json(cls, json_str:str) -> JsonEvent: ...
    
JsonEventT = TypeVar('JsonEventT', bound=JsonEvent)


KafkaEventT = TypeVar('KafkaEventT', bound=KafkaEvent)
KafkaEventDeserializer:TypeAlias = Callable[[Any], KafkaEvent]
KafkaEventSerializer:TypeAlias = Callable[[KafkaEvent], Any]
