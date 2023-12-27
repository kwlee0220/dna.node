from __future__ import annotations

from typing import Optional
from collections.abc import Callable
from enum import Enum

from dna import BytesSerializer, BytesDeserializer, BytesSerDeable, JsonSerializer, JsonDeserializer
from dna.event import KafkaEvent
from .global_track import GlobalTrack
from .node_track import NodeTrack
from .track_feature import TrackFeature
from dna.support import iterables


class NodeEventType(Enum):
    NODE_TRACK = ("node-tracks", NodeTrack)
    FEATURE = ("track-features", TrackFeature)
    GLOBAL_TRACK = ("global-tracks", GlobalTrack)

    def __init__(self, topic:str, event_type:type[BytesSerDeable]) -> None:
        self.topic = topic
        self.event_type = event_type

    def bytes_serializer(self) -> BytesSerializer:
        return self.event_type.serializer() # type: ignore

    def bytes_deserializer(self) -> BytesDeserializer:
        return self.event_type.deserializer()   # type: ignore

    def json_serializer(self) -> JsonSerializer:
        return lambda data: data.to_json()

    def json_deserializer(self) -> JsonDeserializer:
        return lambda bytes: self.event_type.from_json(bytes)   # type: ignore
    
    @classmethod
    def from_type_str(cls, type_str:str) -> NodeEventType:
        type_str = type_str.replace('_', '').replace('-','').lower()
        match type_str:
            case 'nodetrack':
                return NodeEventType.NODE_TRACK
            case 'globaltrack':
                return NodeEventType.FEATURE
            case 'trackfeature':
                return NodeEventType.GLOBAL_TRACK
            case _:
                raise ValueError('unknown event-type: {type_str}')

    @classmethod
    def from_topic(cls, topic:str) -> NodeEventType:
        for item in NodeEventType:
            if item.topic == topic:
                return item
        raise KeyError(f"unregistered topic: {topic}")

    @classmethod
    def from_event(cls, event:KafkaEvent) -> NodeEventType:
        for item in NodeEventType:
            if isinstance(event, item.event_type):
                return item
        raise KeyError(f"unregistered event_type: {event}")

    @classmethod
    def from_event_type(cls, event_type:type[KafkaEvent]) -> NodeEventType:
        for item in NodeEventType:
            if issubclass(event_type, item.event_type):
                return item
        raise KeyError(f"unregistered event_type: {event_type}")
    
    @staticmethod
    def find_topic(event:KafkaEvent) -> str:
        return NodeEventType.from_event_type(type(event)).topic