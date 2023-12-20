from __future__ import annotations

from collections.abc import Iterable, Generator
import dataclasses
import itertools
import operator

import pickle
from contextlib import suppress
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import argparse
from tqdm import tqdm

from dna import initialize_logger
from dna.support import iterables
from dna.event import TrackFeature, NodeTrack
from scripts import update_namespace_with_environ

TOPIC_TRACK_EVENTS = "node-tracks"
TOPIC_MOTIONS = "track-motions"
TOPIC_FEATURES = 'track-features'


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("--kafka_brokers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def download(consumer:KafkaConsumer, topic:str, deserializer, timeout:int=2000):
    while True:
        partitions = consumer.poll(timeout=1000, max_records=100)
        if partitions:
            for partition in partitions.values():
                for serialized in partition:
                    yield deserializer(serialized.value)
        else:
            break

def backup(consumer:KafkaConsumer, topic:str, file:str, deserializer):
    print(f"loading records from the topic' {topic}'")
    consumer.subscribe(topic)
    
    count = 0
    with open(file, 'wb') as fp:
        for event in tqdm(download(consumer, topic, deserializer)):
            pickle.dump(event, fp)
            count += 1
    print(f"writing: {count} records into file '{file}'")


def main():
    args, _ = parse_args()
    args = update_namespace_with_environ(args)

    initialize_logger(args.logger)
    
    try:
        consumer = KafkaConsumer(bootstrap_servers=args.kafka_brokerss,
                                auto_offset_reset=args.kafka_offset,
                                key_deserializer=lambda k: k.decode('utf-8'))
        with suppress(EOFError): backup(consumer, TOPIC_TRACK_EVENTS, f"output/{TOPIC_TRACK_EVENTS}.pickle", NodeTrack.deserialize)
        with suppress(EOFError): backup(consumer, TOPIC_FEATURES, f"output/{TOPIC_FEATURES}.pickle", TrackFeature.deserialize)
        consumer.close()
        pass
    except NoBrokersAvailable as e:
        import sys
        print(f'fails to connect to Kafka: server={args.kafka_brokerss}', file=sys.stderr)
        raise e

if __name__ == '__main__':
    main()