from __future__ import annotations

from contextlib import closing
import pickle
from tqdm import tqdm
from pathlib import Path
import argparse

from kafka.consumer.fetcher import ConsumerRecord

from dna import initialize_logger
from dna.event.kafka_utils import open_kafka_consumer, read_topics
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
        

def define_args(parser):
    add_kafka_consumer_arguments(parser)
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    print(f"Reading Kafka ConsumerRecords from the topics '{args.topics}' and write to '{args.output}'.")
    params = vars(args)
    params['skip_poll_timeout'] = True
    with closing(open_kafka_consumer(**params)) as consumer, \
        closing(ConsumerRecordPickleWriter(args.output)) as writer:
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