from __future__ import annotations
from collections.abc import Callable
from enum import Enum

from dna import ByteString
from dna.event import KafkaEvent, NodeTrack, TrackFeature, KafkaEventDeserializer, KafkaEventSerializer
from dna.assoc import GlobalTrack
from dna.support import iterables


class NodeEventType(Enum):
    NODE_TRACK = ("node-tracks", NodeTrack)
    FEATURE = ("track-features", TrackFeature)
    GLOBAL_TRACK = ("global-tracks", GlobalTrack)

    def __init__(self, topic:str, event_type:type[KafkaEvent]) -> None:
        self.topic = topic
        self.event_type = event_type

    @property
    def serializer(self) -> KafkaEventSerializer:
        return self.event_type.serialize

    @property
    def deserializer(self) -> KafkaEventDeserializer:
        return self.event_type.deserialize

    @classmethod
    def from_topic(cls, topic:str) -> NodeEventType:
        return iterables.first(node_ev for node_ev in cls if node_ev.topic == topic)

    @classmethod
    def from_event_type(cls, event_type:type[KafkaEvent]) -> NodeEventType:
        return iterables.first(node_ev for node_ev in cls if node_ev.event_type == event_type)
    
    @staticmethod
    def find_topic(event:KafkaEvent) -> str:
        return NodeEventType.from_event_type(type(event)).topic
    
    def deserialize(self, bytes:bytes) -> KafkaEvent:
        return self.event_type.deserialize(bytes)