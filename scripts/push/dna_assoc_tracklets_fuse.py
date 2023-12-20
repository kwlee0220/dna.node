

from contextlib import closing
from datetime import timedelta
import logging

from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.event import NodeTrack, TrackFeature, EventListener, open_kafka_consumer
from dna.event.event_processors import PrintEvent
from dna.event.tracklet_store import TrackletStore
from dna.assoc import Association, AssociationCollection, AssociationCollector
from dna.assoc.associator_motion import NodeAssociationSchema, MotionBasedTrackletAssociator
from dna.assoc.associator_feature import FeatureBasedTrackletAssociator
from dna.assoc.closure import AssociationClosureBuilder, Extend
from dna.assoc.utils import FixedIntervalCollector
from dna.support.sql_utils import SQLConnector
from scripts import *

LOGGER = logging.getLogger('dna.assoc_tracklets')


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Associate tracklets by feature")
    
    parser.add_argument("node_pairs", nargs='+', help="target node ids")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='latest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--listen", nargs='+', help="listening nodes")
    parser.add_argument("--db_url", metavar="URL", help="PostgreSQL url", default='postgresql://dna:urc2004@localhost:5432/dna')
    parser.add_argument("--max_distance_to_camera", type=float, metavar="meter", default=55,
                        help="max. distance from camera (default: 55)")
    parser.add_argument("--max_track_distance", type=float, metavar="meter", default=5,
                        help="max. distance between two locations (default: 5)")
    parser.add_argument("--window_interval", type=float, metavar="seconds", default=1,
                        help="window interval seconds (default: 1)")
    parser.add_argument("--idle_timeout", type=float, metavar="seconds", default=10000,
                        help="idle timeout seconds (default: 1)")
    
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()

def consume_tracks_upto(consumer:KafkaConsumer, listener:EventListener, upto_ms:int):
    last_ts = 0
    while True:
        partitions = consumer.poll(timeout=1000, max_records=50)
        if partitions:
            for topic_info, partition in partitions.items():
                if topic_info.topic == 'node-tracks':
                    for serialized in partition:
                        ev = NodeTrack.deserialize(serialized.value)
                        listener.handle_event(ev)
                        last_ts = ev.ts
            if last_ts > upto_ms:
                return last_ts
        else:
            return None
    
def consume_features_upto(consumer:KafkaConsumer, listening_nodes, listener:EventListener, upto_ms:int):
    while True:
        partitions = consumer.poll(timeout=1000, max_records=10)
        if partitions:
            for topic_info, partition in partitions.items():
                if topic_info.topic == 'track-features':
                    for serialized in partition:
                        ev = TrackFeature.deserialize(serialized.value)
                        if ev.zone_relation == 'D':
                            listener.handle_event(ev)
                        elif serialized.key in listening_nodes:
                            listener.handle_event(ev)
                        last_ts = ev.ts
            if last_ts > upto_ms:
                return last_ts
        else:
            return None
        
def open_consumer(args) -> KafkaConsumer:
    return open_kafka_consumer(brokers=args.kafka_brokers,
                               offset_reset=args.kafka_offset,
                               key_deserializer=lambda k: k.decode('utf-8'))

def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    logger = logging.getLogger('dna.assoc.motion')
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    assoc_pairs = [tuple(pair_str.split('-')) for pair_str in args.node_pairs]
    schema = NodeAssociationSchema(assoc_pairs)
        
    window_interval_ms = round(args.window_interval * 1000)
    motion_associator = MotionBasedTrackletAssociator(schema,
                                               window_interval_ms=window_interval_ms,
                                               max_distance_to_camera=args.max_distance_to_camera,
                                               max_track_distance=args.max_track_distance,
                                               idle_timeout=args.idle_timeout)
    
    motion_closures = AssociationCollection(keep_best_association_only=True)
    motion_closure_builder = AssociationClosureBuilder(collection=motion_closures)
    motion_associator.add_listener(motion_closure_builder)
    
    #######################################################################################################
    
    store = TrackletStore.from_url(config.get(conf, 'db_url'))
    feature_associator = FeatureBasedTrackletAssociator(store=store,
                                              listen_nodes=args.listen,
                                              prefix_length=5,
                                              top_k=4,
                                              logger=LOGGER.getChild('associator'))
    
    interval_collector = FixedIntervalCollector(interval_ms=3*1000)
    feature_associator.add_listener(interval_collector)
    
    feature_closures = AssociationClosureBuilder()
    interval_collector.add_listener(feature_closures)
    
    #######################################################################################################
    
    extend = Extend(motion_closures)
    feature_closures.add_listener(extend)
    final_collector = AssociationCollector()
    extend.add_listener(final_collector)
    extend.add_listener(PrintEvent("**** "))
    
    #######################################################################################################
    
    motion_associator.start()
    
    with closing(open_consumer(args)) as track_consumer, closing(open_consumer(args)) as feature_consumer:
        track_consumer.subscribe(['node-tracks'])
        feature_consumer.subscribe(['track-features'])
        
        listening_nodes = set(args.listen)
        last_ts = 0
        while True:
            last_ts = consume_tracks_upto(track_consumer, motion_associator, last_ts)
            if last_ts is None: break
            last_ts = consume_features_upto(feature_consumer, listening_nodes, feature_associator, last_ts)
            if last_ts is None: break
    
    motion_associator.close()
    feature_associator.close()
    
    for assoc in motion_closures:
        print(assoc)
     
if __name__ == '__main__':
    main()