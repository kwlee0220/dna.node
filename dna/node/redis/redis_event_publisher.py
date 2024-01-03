from __future__ import annotations

from typing import Optional, Any
import logging

import redis

from dna import InvalidStateError
from ..types import SilentFrame
from dna.event import KafkaEvent, EventListener
    

class RedisEventPublisher(EventListener):
    def __init__(self, redis:redis.Redis, channel:str, *, logger:Optional[logging.Logger]=None) -> None:
        self.redis = redis
        self.channel = channel
        self.completed = False
        self.logger = logger
        
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"publishing node-track event to channel '{self.channel}'")

    def on_completed(self) -> None:
        if not self.completed:
            self.completed = True

    def handle_event(self, ev:Any) -> None:
        if isinstance(ev, KafkaEvent):
            if self.completed:
                raise InvalidStateError("RedisEventPublisher has been closed already: {self}")
            self.redis.publish(self.channel, ev.to_json())
        elif isinstance(ev, SilentFrame):
            pass
        else:
            if self.logger and self.logger.isEnabledFor(logging.WARN):
                self.logger.warn(f"cannot publish non-Kafka event: {ev}")
        
    def __repr__(self) -> str:
        closed_str = ', closed' if self.completed else ''
        return f"RedisEventPublisher(channel={self.channel}{closed_str})"