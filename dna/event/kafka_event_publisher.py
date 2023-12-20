from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
from contextlib import suppress

from kafka import KafkaProducer
import logging

from dna import InvalidStateError
from dna.utils import has_method
from .types import KafkaEvent, SilentFrame
from .event_processor import EventListener


class KafkaEventPublisher(EventListener):
    def __init__(self, kafka_brokers:Iterable[str], topic:str,
                 *,
                 print_elapsed:bool=False,
                 logger:Optional[logging.Logger]=None) -> None:
        if kafka_brokers is None or not isinstance(kafka_brokers, Iterable):
            raise ValueError(f'invalid kafka_brokers: {kafka_brokers}')
        
        try:
            self.kafka_brokers = kafka_brokers
            self.topic = topic
            self.logger = logger
            self.elapseds:list[int] = [] if print_elapsed else None
            
            self.producer = KafkaProducer(bootstrap_servers=list(kafka_brokers))
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"connect kafka-servers: {kafka_brokers}, topic={self.topic}")
                
            self.closed = False
        except BaseException as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"fails to connect KafkaBrokers: {kafka_brokers}")
            raise e

    def close(self) -> None:
        if not self.closed:
            with suppress(BaseException): super().close()
            with suppress(BaseException): self.producer.close(1)
            self.closed = True
            
        if self.elapseds is not None:
            elapsed_min = min(self.elapseds)
            elapsed_max = max(self.elapseds)
            elapsed_avg = sum(self.elapseds) / len(self.elapseds)
            print(f'elapsed event processing time: min={elapsed_min:.3f}, avg={elapsed_avg:.3f}, max={elapsed_max:.3f}')

    def handle_event(self, ev:object) -> None:
        if isinstance(ev, KafkaEvent):
            if self.closed:
                raise InvalidStateError(f"KafkaEventPublisher has been closed already: {self}")
            
            self.producer.send(self.topic, value=ev.serialize(), key=ev.key().encode('utf-8'))
            
            # tracklet의 마지막 이벤트인 경우 buffering 효과로 인해 바로 전달되지 않고
            # 오랫동안 대기하는 문제를 해결하기 위한 목적
            if has_method(ev, 'is_deleted') and ev.is_deleted():
                self.producer.flush()
        elif isinstance(ev, SilentFrame):
            pass
        else:
            if self.logger and self.logger.isEnabledFor(logging.WARN):
                self.logger.warn(f"cannot publish non-Kafka event: {ev}")
        
        if self.elapseds is not None:
            from dna.utils import utc_now_millis
            delta = (utc_now_millis() - ev.ts) / 1000
            self.elapseds.append(delta)
            

    def flush(self) -> None:
        if self.closed:
            raise InvalidStateError(f"KafkaEventPublisher has been closed already: {self}")
        self.producer.flush()
        
    def __repr__(self) -> str:
        closed_str = ', closed' if self.closed else ''
        return f"KafkaEventPublisher(topic={self.topic}{closed_str})"