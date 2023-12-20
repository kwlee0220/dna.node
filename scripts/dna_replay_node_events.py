from __future__ import annotations

from contextlib import closing
from collections.abc import Sequence, Iterator
import dataclasses
from tqdm import tqdm
import argparse

from dna import config
from dna.support import iterables
from dna.event import KafkaEvent, open_kafka_producer, read_pickle_event_file, synchronize_time
from dna.event.node_event_type import NodeEventType


def define_args(parser):
    parser.add_argument("file", help="event file (pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--sync", action='store_true', default=False)
    parser.add_argument("--progress", action='store_true', default=False)
    parser.add_argument("--start_now", action='store_true', default=False)
    parser.add_argument('--begin_frames', nargs='+', default=["0", "0", "0"], type=str,
                        help='start frame number (starting from 0)')
    parser.add_argument("--max_wait_ms", metavar="millis-second", type=int, default=None,
                        help="maximum wait time in milli-seconds")


def update_event(ev, offset:int):
    if hasattr(ev, 'first_ts'):
        return dataclasses.replace(ev, ts=ev.ts+offset, first_ts=ev.first_ts+offset)
    else:
        return dataclasses.replace(ev, ts=ev.ts+offset)


def run(args):
    conf = config.to_conf(args)
    max_wait_ms = conf.get('max_wait_ms')
    
    offsets = [int(offset) for offset in args.begin_frames]
    def shift(ev:KafkaEvent) -> KafkaEvent:
        return dataclasses.replace(ev,
                                   track_id=str(int(ev.track_id) + offsets[0]),
                                   frame_index=(ev.frame_index) + offsets[1],
                                   ts=ev.ts + offsets[2])
    # print(f"loading events from the file '{args.file}'.")
    events = (shift(ev) for ev in read_pickle_event_file(args.file))
    
    if args.start_now:
        if isinstance(events, Sequence):
            start_ts = events[0].ts
        elif isinstance(events, Iterator):
            events = iterables.to_peekable(events)
            start_ts = events.peek().ts
        else:
            raise ValueError(f"invalid events")
        
        import time
        now = round(time.time() * 1000)
        offset = now - start_ts
        events = (update_event(ev, offset) for ev in events)
        
    print(f"publish events from the file '{args.file}'.")
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        progress = tqdm(desc='publishing node events') if args.progress else None
        
        if args.sync:
            events = synchronize_time(events, max_wait_ms=max_wait_ms)
            
        for ev in events:
            topic = NodeEventType.find_topic(ev)
            producer.send(topic, value=ev.serialize(), key=ev.key())
            if progress is not None:
                progress.update()
    
    
def main():
    parser = argparse.ArgumentParser(description="Feed Node events into relevant Kafka topics")
    define_args(parser)
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()