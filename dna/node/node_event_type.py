from __future__ import annotations

from typing import Optional, NamedTuple, Any

from dna import Serializer, Deserializer, Serializable, Deserializable
from dna import JsonSerializable, JsonDeserializable, JsonSerializer, JsonDeserializer
from dna.event.json_event import JsonEventImpl
from .global_track import GlobalTrack
from .node_track import NodeTrack
from .track_feature import TrackFeature


class NodeEventType(NamedTuple):
    id: str
    topic: Optional[str]
    event_type: type
    
    
__NODE_EVENT_TYPES = {
    "nodetracks": NodeEventType(id="nodetracks", topic="node-tracks", event_type=NodeTrack),
    "trackfeatures": NodeEventType(id="trackfeatures", topic="track-features", event_type=TrackFeature),
    "globaltracks": NodeEventType(id="globaltracks", topic="global-tracks", event_type=GlobalTrack),
    "jsonevent": NodeEventType(id="jsonevent", topic=None, event_type=JsonEventImpl),
}


def find_node_event_type_by_event_type(event_type:type) -> NodeEventType:
    for id, net in __NODE_EVENT_TYPES.items():
        if net.event_type == event_type:
            return net
    raise KeyError(f"event_type: {event_type}")
def find_node_event_type_by_object(obj:Any) -> NodeEventType:
    for id, net in __NODE_EVENT_TYPES.items():
        if isinstance(obj, net.event_type):
            return net
    raise KeyError(f"object: {obj}")


def find_event_type_by_id(id:str) -> type:
    return __NODE_EVENT_TYPES[id].event_type

def find_event_type_by_type_str(type_str:str) -> type:
    id = type_str.replace('_', '').replace('-','').lower()
    try:
        return find_event_type_by_id(id)
    except KeyError:
        raise KeyError(f"unknown type_str: {type_str}")
        
def find_event_type_by_topic(topic:str) -> type:
    for id, net in __NODE_EVENT_TYPES.items():
        if net.topic == topic:
            return net.event_type
    raise KeyError(f"unknown topic: {topic}")


def find_serializer_by_topic(topic:str) -> Serializer:
    node_type = find_event_type_by_topic(topic)
    if issubclass(node_type, Serializable):
        return lambda o: o.serialize()
    else:
        raise ValueError(f"Target NodeType does not support Serializable: topic={topic}")
def find_deserializer_by_topic(topic:str) -> Deserializer:
    node_type = find_event_type_by_topic(topic)
    if issubclass(node_type, Deserializable):
        return node_type.deserialize
    else:
        raise ValueError(f"Target NodeType does not support Deserializable: topic={topic}")


def find_serializer_by_type_str(type_str:str) -> Serializer:
    node_type = find_event_type_by_type_str(type_str)
    if issubclass(node_type, Serializable):
        return lambda o: o.serialize()
    else:
        raise ValueError(f"Target NodeType does not support Serializable: type_str={node_type}")
def find_deserializer_by_type_str(type_str:str) -> Deserializer:
    node_type = find_event_type_by_type_str(type_str)
    if issubclass(node_type, Deserializable):
        return node_type.deserialize
    else:
        raise ValueError(f"Target NodeType does not support Deserializable: type_str={node_type}")


def find_json_serializer_by_type_str(type_str:str) -> JsonSerializer:
    node_type = find_event_type_by_type_str(type_str)
    if issubclass(node_type, JsonSerializable):
        return lambda o: o.to_json()
    else:
        raise ValueError(f"Target NodeType does not support JsonSerializable: {node_type}")
def find_json_serializer_by_topic(topic:str) -> JsonSerializer:
    node_type = find_event_type_by_topic(topic)
    if issubclass(node_type, JsonSerializable):
        return lambda o: o.to_json()
    else:
        raise ValueError(f"Target NodeType does not support JsonSerializable: topic={topic}")