from __future__ import annotations

from typing import Callable, Generator

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import NoBrokersAvailable


def open_kafka_consumer(brokers:list[str], offset_reset:str,
                        *,
                        key_deserializer:Callable[[bytes],str]=lambda k:k.decode('utf-8')) -> KafkaConsumer:
    try:
        return KafkaConsumer(bootstrap_servers=brokers, auto_offset_reset=offset_reset,
                             key_deserializer=key_deserializer)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')


def open_kafka_producer(brokers:list[str], *,
                        key_serializer:Callable[[str],bytes]=lambda k: k.encode('utf-8')) -> KafkaProducer:
    try:
        return KafkaProducer(bootstrap_servers=brokers, key_serializer=key_serializer)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')
    
    
def read_events_from_topics(consumer:KafkaConsumer, **poll_args) -> Generator[ConsumerRecord, None, None]:
    from contextlib import closing
    from .utils import find_event_deserializer
    
    records = read_topics(consumer,
                            initial_timeout_ms=poll_args['initial_timeout_ms'],
                            timeout_ms=poll_args['timeout_ms'],
                            stop_on_timeout=poll_args['stop_on_timeout'])
    deser = find_event_deserializer(poll_args['type'])
    for record in records:
        yield deser(record.value)


def read_topics(consumer:KafkaConsumer, **poll_args) -> Generator[ConsumerRecord, None, None]:
    # 'initial_timeout_ms'가 지정된 경우는 첫번째 poll() 메소드를 호출하는 경우의 timeout_ms은
    # 'timeout_ms'를 사용하지 않고, 이것을 사용하도록 일시적으로 변경시켜 사용한 후 다음부터는
    # 원래의 'timeout_ms'를 사용하도록 한다.
    
    initial_poll = False
    org_timeout_ms = poll_args.get('timeout_ms')
    
    initial_timeout_ms = poll_args.pop('initial_timeout_ms', None)
    if initial_timeout_ms:
        poll_args['timeout_ms'] = initial_timeout_ms
        initial_poll = True
        
    stop_on_timeout = poll_args.pop('stop_on_timeout', False)
    close_on_return = poll_args.pop('close_on_return', False)
        
    try:
        while True:
            partitions = consumer.poll(**poll_args)
            if partitions:
                for part_info, partition in partitions.items():
                    for record in partition:
                        yield record
                        
                if initial_poll:
                    if org_timeout_ms:
                        poll_args['timeout_ms'] = org_timeout_ms
                    else:
                        # 처음부터 'timeout_ms'가 설정되지 않은 경우에는
                        # poll_args에서 'timeout_ms'를 삭제한다.
                        poll_args.pop('timeout_ms', None)
                    initial_poll = False
                    
            elif stop_on_timeout:
                break
    finally:
        if close_on_return:
            consumer.close()