from __future__ import annotations

from typing import Optional, Any
import logging

import redis

from dna import InvalidStateError
from dna.event import KafkaEvent, SilentFrame, EventListener
    

class RedisEventPublisher(EventListener):
    def __init__(self, redis:redis.Redis, channel:str, *, logger:Optional[logging.Logger]=None) -> None:
        self.redis = redis
        self.channel = channel
        self.closed = False
        self.logger = logger
        
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"publishing node-track event to channel '{self.channel}'")

    def close(self) -> None:
        if not self.closed:
            self.closed = True

    def handle_event(self, ev:Any) -> None:
        if isinstance(ev, KafkaEvent):
            if self.closed:
                raise InvalidStateError("RedisEventPublisher has been closed already: {self}")
            self.redis.publish(self.channel, ev.to_json())
        elif isinstance(ev, SilentFrame):
            pass
        else:
            if self.logger and self.logger.isEnabledFor(logging.WARN):
                self.logger.warn(f"cannot publish non-Kafka event: {ev}")
        
    def __repr__(self) -> str:
        closed_str = ', closed' if self.closed else ''
        return f"RedisEventPublisher(channel={self.channel}{closed_str})"