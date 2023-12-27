from __future__ import annotations

from typing import TypeAlias, TypeVar, Any, Protocol, runtime_checkable
from abc import ABC
from collections.abc import Callable
from abc import ABC, abstractmethod

from kafka.consumer.fetcher import ConsumerRecord

from dna import Point, KeyValue


@runtime_checkable
class Timestamped(Protocol):
    __slots__ = ()
    
    @property
    def ts(self) -> int: ...
    
TimestampedT = TypeVar("TimestampedT", bound=Timestamped)


@runtime_checkable
class TrackEvent(Protocol):
    __slots__ = ()
    
    @property
    def id(self) -> str: ...
    
    @property
    def state(self) -> str: ...
    
    @property
    def location(self) -> Point: ...
    
    @property
    def ts(self) -> int: ...


class KafkaEvent(ABC):
    __slots__ = ()
    
    @abstractmethod
    def key(self) -> str:
        """Returns key value for Kafka Producer record.

        Returns:
            str: key value for Kafka Producer record.
        """
        ...
    
    @abstractmethod
    def to_kafka_record(self) -> KeyValue[bytes, bytes]:
        """Returns encoded value for Kafka Producer record.

        Returns:
            KeyValue[bytes, bytes]: encoded key and value for Kafka Producer record.
        """
        ...
        
    @staticmethod
    @abstractmethod
    def from_kafka_record(record:ConsumerRecord) -> KafkaEvent:
        ...


@runtime_checkable
class JsonEvent(Protocol):
    @property
    def ts(self) -> int: ...
    
    def to_json(self) -> str:
        ...
    
    @classmethod
    def from_json(cls, json_str:str) -> JsonEvent: ...
    
JsonEventT = TypeVar('JsonEventT', bound=JsonEvent)


KafkaEventT = TypeVar('KafkaEventT', bound=KafkaEvent)
KafkaEventDeserializer:TypeAlias = Callable[[Any], KafkaEvent]
KafkaEventSerializer:TypeAlias = Callable[[KafkaEvent], Any]
