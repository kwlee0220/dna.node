from __future__ import annotations

from typing import Any, Optional, Generator
from abc import ABCMeta, abstractmethod

import json


class Serde(metaclass=ABCMeta):
    @abstractmethod
    def serialize(self, data:Any) -> bytes: pass

    @abstractmethod
    def deserialize(self, body:bytes) -> Any: pass
    

class JsonSerde(Serde):
    def deserialize(self, body:bytes) -> Any:
        # json_str = body.decode('utf-8') if isinstance(body, bytes) else body
        return json.loads(body)

    def serialize(self, resp:Any) -> bytes:
        if isinstance(resp, str):
            resp = resp.encode('utf-8')
        elif not isinstance(resp, bytes):
            resp = json.dumps(resp, default=lambda o: o.__dict__).encode('utf-8')
        return resp