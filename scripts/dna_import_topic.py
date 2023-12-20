from __future__ import annotations

import sys
from contextlib import closing
from tqdm import tqdm
from pathlib import Path
import argparse

from dna import initialize_logger
from dna.event import open_kafka_producer, NodeTrack
from dna.assoc import GlobalTrack
from dna.event.utils import read_event_file
from scripts.utils import *


def define_args(parser):
    parser.add_argument("file", help="events file (json or pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--topic", metavar='name', required=True, help="target topic name")
    parser.add_argument("--type", choices=['node-track', 'global-track'], default=None,
                        help="event type ('node-track', 'global-track')")
    parser.add_argument("--progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")
    

def run(args):
    initialize_logger(args.logger)
    
    event_type = None
    match args.type:
        case 'node-track': event_type = NodeTrack
        case 'global-track': event_type = GlobalTrack
        case None: event_type = None
        case _: raise ValueError(f'unknown event type')
    events = read_event_file(args.file, event_type=event_type)
    
    print(f"Uploading events to the topic '{args.topic}' from the file '{args.file}'.")
    skip_count = 0
    import_count = 0
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        progress = tqdm(desc='publishing events') if args.progress else None
        for ev in events:
            if isinstance(ev, KafkaEvent):
                producer.send(args.topic, value=ev.serialize(), key=ev.key())
                import_count += 1
                if progress is not None:
                    progress.update()
            else:
                skip_count += 1
    if skip_count > 0:
        print("topic import done: importeds={import_count}, skippeds={skip_count}")
    

def main():
    parser = argparse.ArgumentParser(description="Import events into the Kafka topic.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)
            

if __name__ == '__main__':
    main()