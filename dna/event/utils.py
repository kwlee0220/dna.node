from __future__ import annotations

from typing import Optional, Any
from collections.abc import Generator, Iterator, Sequence

from .types import TimestampedT, JsonEventT

        
def read_text_line_file(file:str) -> Generator[str, None, None]:
    """텍스트 파일에서 한 라인씩 읽은 line string을 하나씩 반환하는 generator를 생성한다.

    Args:
        file (str): 텍스트 파일 경로명.

    Yields:
        Generator[str, None, None]: Generator 객체.
    """
    with open(file) as f:
        for line in f.readlines():
            yield line


def read_json_event_file(file:str, event_type:type[JsonEventT]) -> Generator[JsonEventT, None, None]:
    """텍스트 파일에서 한 라인씩 읽은 line을 JSON 객체로 변환하는 generator를 생성한다.

    Args:
        file (str): JSON 텍스트 파일 경로명.
        event_type (type[JsonEventT]): JSON 객체 변환기

    Yields:
        Generator[JsonEventT, None, None]: Generator 객체.
    """
    import json
    with open(file) as f:
        for line in f.readlines():
            yield event_type.from_json(line)


def read_pickle_event_file(file:str) -> Generator[Any, None, None]:
    """Pickle 파일에 저장된 KafkaEvent 객체를 하나씩 반환하는 generator를 생성한다.

    Args:
        file (str): Pickle 파일 경로명.

    Yields:
        Generator[KafkaEvent, None, None]: Generator 객체.
    """
    import pickle
    with open(file, 'rb') as fp:
        try:
            while True:
                yield pickle.load(fp)
        except EOFError as e: pass

            
_SLEEP_OVERHEAD = 20
def synchronize_time(events:Iterator[TimestampedT],
                     *,
                     max_wait_ms:Optional[int]=None) -> Generator[TimestampedT, None, None]:
    import time
    from dna.support import iterables
    
    # 현재 시각과 events의 첫번째 event의 timestamp 값을 사용하여
    # 상대적인 time offset을 계산한다.
    if isinstance(events, Sequence):
        start_ts = events[0].ts
    elif isinstance(events, Iterator):
        events = iterables.to_peekable(events)
        start_ts = events.peek().ts
    else:
        raise ValueError(f"invalid events")
    now = round(time.time() * 1000)
    offset_ms = now - start_ts
        
    for ev in events:
        # 다음번 event issue할 때까지의 대기 시간을 계산한다.
        now = round(time.time() * 1000)
        wait_ms = (ev.ts + offset_ms) - now
        
        if max_wait_ms is not None:
            overflow = wait_ms - max_wait_ms
            if overflow > 0:
                offset_ms -= overflow
                wait_ms = max_wait_ms
        if wait_ms > _SLEEP_OVERHEAD:
            time.sleep(wait_ms / 1000)
        
        yield ev


def sort_events_with_fixed_buffer(source:Iterator[TimestampedT],
                                  *, heap_size:int=512) -> Generator[TimestampedT,None,None]:
    import heapq
        
    heap:list[tuple[int, TimestampedT]] = []
    for ev in source:
        heapq.heappush(heap, (ev.ts, ev))
        if len(heap) <= heap_size:
            continue
        yield heapq.heappop(heap)[1]
    
    # flush remaining heap
    while heap:
        yield heapq.heappop(heap)[1]