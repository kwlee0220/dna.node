from __future__ import annotations

from typing import Any, Generator
import uuid
import redis

from dna.event import EventQueue
from dna.execution import ExecutionState, InvocationError
from ..node_track import NodeTrack
from . import JsonSerde
    

class RedisNodeProcessorClient(EventQueue):
    def __init__(self, redis:redis.Redis, req_channel:str, node_id:str, camera_uri:str,
                 *,
                 sync:bool=False) -> None:
        self.redis = redis
        self.req_channel = req_channel
        self.serde = JsonSerde()
        
        self.app_id = str(uuid.uuid4())
        self.tracks_channel = f'{self.app_id}:node-tracks'
        self.progress_channel = f'{self.app_id}:progress'
        self.controls_channel = f'{self.app_id}:controls'
        self.request = {
            'id': self.app_id,
            'node': node_id,
            'action': 'track',
            'channels': {
                'node-tracks' : self.tracks_channel,
                'controls' : self.controls_channel,
                'progress_reports' : self.progress_channel
            },
            'camera': {
                'uri': camera_uri,
                'sync': sync
            }
        }
        self.request = self.serde.serialize(self.request)

    def node_tracks(self) -> Generator[NodeTrack, None,None]:
        with self.redis.pubsub() as pubsub:
            pubsub.subscribe([self.tracks_channel, self.progress_channel])  
            
            self.redis.publish(self.req_channel, self.request)
            while True:
                msg = pubsub.get_message()
                if msg and msg['type'] == 'message':
                    channel = msg['channel'].decode('utf-8')
                    json_str = msg['data'].decode('utf-8')
                    
                    if channel == self.tracks_channel:
                        yield NodeTrack.from_json(json_str)
                    elif channel == self.progress_channel:
                        progress = self.serde.deserialize(json_str)
                        state = progress['state']
                        if state == ExecutionState.COMPLETED.name or state == ExecutionState.STOPPED.name:
                            break
                        elif state == ExecutionState.FAILED.name:
                            raise InvocationError(progress['cause'])
            self.app_id = None
        
    def stop(self, details:str='client request') -> None:
        if self.app_id is None:
            return
        
        request = {
            "id": self.app_id,
            "action": "stop",
            "details": details
        }
        request = self.serde.serialize(request)
        self.redis.publish(self.controls_channel, request)