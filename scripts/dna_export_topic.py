from __future__ import annotations

from contextlib import closing
import pickle
import sys
from tqdm import tqdm
from pathlib import Path
import argparse

from kafka.consumer.fetcher import ConsumerRecord

from dna import initialize_logger
from dna.event.kafka_utils import open_kafka_consumer, read_topics
from dna.node import Jso
from dna.node.node_event_type import NodeEventType
from scripts.utils import add_kafka_consumer_arguments


class ConsumerRecordPickleWriter:
    def __init__(self, path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, 'wb')
        self.closed = False
        
    def close(self) -> None:
        if not self.closed:
            self.fp.close()
        self.closed = True
            
    def write(self, record:ConsumerRecord) -> None:
        pickle.dump((record.key, record.value), self.fp)
        self.fp.flush()
        

class ConsumerRecordJsonWriter:
    def __init__(self, path:Path, Json) -> None:
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
    add_kafka_consumer_arguments(parser)
    parser.add_argument("--type", choices=['node-track', 'global-track', 'json-event', 'track-feature'],
                        default='json-event',
                        help="event type ('node-track', 'global-track', 'json-event', 'track-feature')")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
        
    path = Path(args.output)
    match path.suffix:
        case '.json':
            event_type = NodeEventType.from_type_str(args.type)
            ser = event_type.json_serializer
            writer = ConsumerRecordJsonWriter(path, ser)
        case '.pickle':
            writer = ConsumerRecordPickleWriter(path)
        case _:
            print(f'Unsupported output file format: {path.suffix}', file=sys.stderr)
            sys.exit(-1)
    
    print(f"Reading Kafka ConsumerRecords from the topics '{args.topics}' and write to '{args.output}'.")
    params = vars(args)
    params['drop_poll_timeout'] = True
    with closing(open_kafka_consumer(**params)) as consumer, closing(writer):
        records = read_topics(consumer, **params)
        for record in tqdm(records, desc='exporting records'):
            assert isinstance(record, ConsumerRecord)
            writer.write(record) 
    

def main():
    parser = argparse.ArgumentParser(description="Export Kafka topic into a file.")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()