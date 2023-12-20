from __future__ import annotations

from typing import Any, Optional
from contextlib import closing
import pickle
import argparse

from dna.event import read_event_file , open_kafka_consumer
from dna.event.kafka_utils import read_events_from_topics
from scripts.utils import add_kafka_consumer_arguments


def define_args(parser) -> None:
    parser.add_argument("--files", nargs='+', help="track files to print")
    add_kafka_consumer_arguments(parser)
    parser.add_argument("--type", choices=['node-track', 'global-track', 'json-event', 'track-feature'],
                        default='json-event', help="event type ('node-track', 'global-track', 'json-event', 'track-feature')")
    parser.add_argument("--filter", metavar="expr", help="predicate expression", default=None)
  
     
def run(args) -> None:
    filter = compile(args.filter, "<string>", 'eval') if args.filter is not None else None
    
    if args.files is not None:
        for file in args.files:
            for ev in read_event_file(file):
                if not filter or eval(filter, {'ev':ev}):
                    print(ev)
    else:
        with closing(open_kafka_consumer(brokers=args.kafka_brokers, offset_reset=args.kafka_offset)) as consumer:
            consumer.subscribe(args.topic)
            for ev in read_events_from_topics(consumer, **vars(args)):
                if not filter or eval(filter, {'ev':ev}):
                    print(ev)
          
def main():
    parser = argparse.ArgumentParser(description="Print event file.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()