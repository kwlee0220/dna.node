from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
from contextlib import suppress

from kafka import KafkaProducer
import logging

from dna import InvalidStateError
from ..event.types import KafkaEvent
from ..event.event_processor import EventListener
from .types import SilentFrame


class KafkaEventPublisher(EventListener):
    def __init__(self, kafka_brokers:Iterable[str], topic:str,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        if kafka_brokers is None or not isinstance(kafka_brokers, Iterable):
            raise ValueError(f'invalid kafka_brokers: {kafka_brokers}')
        
        self.logger = logger
        try:
            self.kafka_brokers = kafka_brokers
            self.topic = topic
            self.producer = KafkaProducer(bootstrap_servers=list(kafka_brokers))
            self.is_completed = False
            
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"connect kafka-servers: {kafka_brokers}, topic={self.topic}")
        except BaseException as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"fails to connect KafkaBrokers: {kafka_brokers}")
            raise e

    def on_completed(self) -> None:
        if not self.is_completed:
            with suppress(BaseException): super().on_completed()
            with suppress(BaseException): self.producer.close(1)
            self.is_completed = True

    def handle_event(self, ev:KafkaEvent) -> None:
        if self.is_completed:
            raise InvalidStateError(f"KafkaEventPublisher has been closed already: {self}")
        
        if isinstance(ev, KafkaEvent):
            key_bytes, value_bytes = ev.to_kafka_record()
            self.producer.send(self.topic, value=value_bytes, key=key_bytes)
            
            # tracklet의 마지막 이벤트인 경우 buffering 효과로 인해 바로 전달되지 않고
            # 오랫동안 대기하는 문제를 해결하기 위한 목적
            with suppress(Exception):
                if ev.is_deleted():    # type: ignore
                    self.producer.flush()
        elif isinstance(ev, SilentFrame):
            pass
        else:
            if self.logger and self.logger.isEnabledFor(logging.WARN):
                self.logger.warn(f"cannot publish non-Kafka event: {ev}")
            
    def flush(self) -> None:
        if self.is_completed:
            raise InvalidStateError(f"KafkaEventPublisher has been closed already: {self}")
        self.producer.flush()
        
    def __repr__(self) -> str:
        closed_str = ', closed' if self.is_completed else ''
        return f"KafkaEventPublisher(topic={self.topic}{closed_str})"