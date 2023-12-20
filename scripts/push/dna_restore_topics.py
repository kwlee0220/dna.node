from __future__ import annotations

import time
import pickle
from tqdm import tqdm

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from dna import initialize_logger
from dna.event import NodeTrack, TrackFeature
from scripts import update_namespace_with_environ

TOPIC_TRACK_EVENTS = "node-tracks"
TOPIC_MOTIONS = "track-motions"
TOPIC_FEATURES = 'track-features'


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("files", nargs='+', help="event pickle files")
    parser.add_argument("--kafka_brokers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()
        
    
def topic_for(ev):
    if isinstance(ev, NodeTrack):
        return TOPIC_TRACK_EVENTS
    elif isinstance(ev, TrackFeature):
        return TOPIC_FEATURES
    else:
        raise AssertionError()


def main():
    args, _ = parse_args()
    args = update_namespace_with_environ(args)

    initialize_logger(args.logger)
    
    try:
        producer = KafkaProducer(bootstrap_servers=args.kafka_brokerss)
        for file in args.files:
            with open(file, 'rb') as fp:
                try:
                    while True:
                        event = pickle.load(fp)
                        topic = topic_for(event)
                        producer.send(topic, value=event.serialize(), key=event.key().encode('utf-8'))
                        # print(topic, event)
                except EOFError:
                    pass
        producer.close()
    except NoBrokersAvailable as e:
        import sys
        print(f'fails to connect to Kafka: server={args.kafka_brokerss}', file=sys.stderr)

if __name__ == '__main__':
    main()