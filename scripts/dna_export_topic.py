from __future__ import annotations

from typing import Optional
from contextlib import closing, suppress
import pickle
import sys
from tqdm import tqdm
from pathlib import Path
import argparse

from kafka.consumer.fetcher import ConsumerRecord

from dna import initialize_logger
from dna import JsonSerializer, JsonSerializable, JsonSerDeable
from dna.event.kafka_utils import open_kafka_consumer, read_topics
from dna.node import node_event_type
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
    def __init__(self, path:Path, serialize:JsonSerializer) -> None:
        self.serialize = serialize
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, 'w')
        self.is_closed = False
        
    def close(self) -> None:
        if not self.is_closed:
            self.fp.close()
            self.is_closed = True
            
    def write(self, record:ConsumerRecord) -> None:
        json = self.serialize(record.value)
        self.fp.write(json + '\n')
        

def define_args(parser):
    add_kafka_consumer_arguments(parser)
    parser.add_argument("--type", choices=['node-track', 'global-track', 'json-event', 'track-feature'],
                        default=argparse.SUPPRESS,
                        help="event type ('node-track', 'global-track', 'json-event', 'track-feature')")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    params = vars(args)
    path = Path(args.output)
    if path.suffix == '.json':
        ser:Optional[JsonSerializer] = None
        if type := params.get('type'):
            ser = node_event_type.find_json_serializer_by_type_str(type)
            writer = ConsumerRecordJsonWriter(path, ser)
        elif topics := params.get('topics'):
            ser = node_event_type.find_json_serializer_by_topic(params['topics'][0])
            writer = ConsumerRecordJsonWriter(path, ser)
    elif path.suffix == '.pickle':
        writer = ConsumerRecordPickleWriter(path)
    else:
        print(f'Unsupported output file format: {path.suffix}', file=sys.stderr)
        sys.exit(-1)
    
    print(f"Reading Kafka ConsumerRecords from the topics '{args.topics}' and write to '{args.output}'.")
    params['drop_poll_timeout'] = True
    with closing(writer):   # type: ignore
        from scripts.utils import read_typed_topic
        for rec in tqdm(read_typed_topic(**params), desc='exporting records'):
            assert isinstance(rec, ConsumerRecord)
            writer.write(rec)    # type: ignore
    

def main():
    parser = argparse.ArgumentParser(description="Export Kafka topic into a file.")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()