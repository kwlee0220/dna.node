from __future__ import annotations

from typing import Any

import json

from dna import Point, SerDeable, JsonSerDeable


class JsonEventImpl(SerDeable['JsonEventImpl'],JsonSerDeable['JsonEventImpl']):
    __slots__ = ('json_obj')
    
    def __init__(self, json_obj:dict[str,Any]) -> None:
        self.json_obj = json_obj
        
    @property
    def id(self) -> str:
        return self.json_obj['id']
        
    def key(self) -> str:
        return self.json_obj['id']
    
    @property
    def ts(self) -> int:
        return self.json_obj['ts']
    
    @property
    def state(self) -> str:
        return self.json_obj['state']
    
    def is_deleted(self) -> bool:
        return self.state == 'DELETED'
    
    @property
    def location(self) -> Point:
        return Point(self.json_obj['location'])
    
    @location.setter
    def location(self, new_loc:Point) -> None:
        self.json_obj['location'] = list(new_loc)
        
    @staticmethod
    def from_json(json_str:str) -> JsonEventImpl:
        return JsonEventImpl(json.loads(json_str))
    
    def to_json(self) -> str:
        return json.dumps(self.json_obj)
    
    @staticmethod
    def deserialize(json_bytes:bytes) -> JsonEventImpl:
        return JsonEventImpl.from_json(json_bytes.decode('utf-8'))
    
    def serialize(self) -> bytes:
        return self.to_json().encode('utf-8')

    def __lt__(self, other) -> bool:
        if self.ts < other.ts:
            return True
        elif self.ts == other.ts:
            return self.id < other.id
        else:
            return False
    
    def __repr__(self) -> str:
        return f'{self.id}{{{self.state}, {self.ts}}}'

