from __future__ import annotations

import sys
from contextlib import closing
from tqdm import tqdm

import argparse

from dna import initialize_logger
from dna.event.kafka_utils import open_kafka_producer
from dna.event.utils import read_pickle_event_file


def define_args(parser):
    parser.add_argument("file", help="events file (pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--topic", metavar='name', required=True, help="target topic name")
    parser.add_argument("--progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")
    
    
def run(args):
    initialize_logger(args.logger)
    
    print(f"Uploading events to the topic '{args.topic}' from the file '{args.file}'.")
    
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        import_count = 0
        progress = tqdm(desc='publishing events') if args.progress else None
        for key, value in read_pickle_event_file(args.file):
            producer.send(args.topic, value=value, key=key)
            import_count += 1
            if progress is not None:
                progress.update()
    
def main():
    parser = argparse.ArgumentParser(description="Import events into the Kafka topic.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)
            

if __name__ == '__main__':
    main()