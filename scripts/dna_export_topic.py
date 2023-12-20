from __future__ import annotations

from contextlib import closing
import pickle
from tqdm import tqdm
from pathlib import Path
import argparse

from kafka.consumer.fetcher import ConsumerRecord

from dna import initialize_logger
from dna.event import open_kafka_consumer, read_topics
from scripts import *


class ConsumerRecordPickleWriter:
    def __init__(self, path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, 'wb')
        
    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None
            
    def write(self, record:ConsumerRecord) -> None:
        pickle.dump((record.key, record.value), self.fp)
        self.fp.flush()
        

class ConsumerRecordTextWriter:
    def __init__(self, path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, 'w')
        
    def close(self) -> None:
        if ( self.fp is not None ):
            self.fp.close()
            self.fp = None
            
    def write(self, record:ConsumerRecord) -> None:
        json = record.value.decode('utf-8')
        self.fp.write(json + '\n')
        

def define_args(parser):
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--topic", help="target topic name")
    parser.add_argument("--timeout_ms", metavar="milli-seconds", type=int, default=1000,
                        help="Kafka poll timeout in milli-seconds")
    parser.add_argument("--initial_timeout_ms", metavar="milli-seconds", type=int, default=5000,
                        help="initial Kafka poll timeout in milli-seconds")
    parser.add_argument("--stop_on_timeout", action='store_true', help="stop when a poll timeout expires")
    
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
        
    path = Path(args.output)
    match path.suffix:
        case '.json':
            writer = ConsumerRecordTextWriter(path)
        case '.pickle':
            writer = ConsumerRecordPickleWriter(path)
        case _:
            print(f'Unsupported output file format: {path.suffix}', file=sys.stderr)
    
    print(f"Reading Kafka ConsumerRecords from the topics '{args.topic}' and write to '{args.output}'.")
    with closing(open_kafka_consumer(brokers=args.kafka_brokers, offset_reset=args.kafka_offset)) as consumer, \
         closing(writer) as writer:
        consumer.subscribe(args.topic)
        
        records = read_topics(consumer,
                              initial_timeout_ms=args.initial_timeout_ms,
                              timeout_ms=args.timeout_ms,
                              stop_on_timeout=args.stop_on_timeout)
        for record in tqdm(records, desc='exporting records'):
            writer.write(record)
    

def main():
    parser = argparse.ArgumentParser(description="Export Kafka topic into a file.")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()