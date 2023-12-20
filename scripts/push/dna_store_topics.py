from __future__ import annotations

from contextlib import closing

from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.event import NodeTrack, TrackFeature
from dna.support import sql_utils
from dna.event.tracklet_store import TrackletStore
from scripts import update_namespace_with_environ


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Store DNA Node kafka topics")
    
    parser.add_argument("--db_url", metavar="URL", help="PostgreSQL url", default='postgresql://dna:urc2004@localhost:5432/dna')
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument('-f', '--format', action='store_true', help='(re-)create tables necessary for Tracklet store.')
    
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    store = TrackletStore(sql_utils.SQLConnector.from_url(config.get(conf, 'db_url')))
    if conf.format:
        store.drop()
        store.format()
    
    consumer = KafkaConsumer(bootstrap_servers=config.get(conf, 'kafka_brokers'),
                             auto_offset_reset=config.get(conf, 'kafka_offset'),
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['node-tracks', 'track-motions', 'track-features'])
    
    while True:
        partitions = consumer.poll(timeout=500, max_records=100)
        if partitions:
            with closing(store.connect()) as conn:
                for topic_info, partition in partitions.items():
                    if topic_info.topic == 'node-tracks':
                        tracks = [NodeTrack.deserialize(serialized.value) for serialized in partition]
                        store.insert_track_events_conn(conn, tracks, batch_size=30)
                    elif topic_info.topic == 'track-features':
                        features = [TrackFeature.deserialize(serialized.value) for serialized in partition]
                        store.insert_track_features_conn(conn, features, batch_size=8)
                        

if __name__ == '__main__':
    main()