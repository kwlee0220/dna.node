from __future__ import annotations

from contextlib import closing
from datetime import timedelta
import logging

from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.event import NodeTrack, TrackFeature, open_kafka_consumer, read_topics
from dna.event.event_processors import PrintEvent
from dna.event.tracklet_store import TrackletStore
from dna.support.sql_utils import SQLConnector
from dna.assoc.associator_feature import FeatureBasedTrackletAssociator
from dna.assoc.utils import FixedIntervalCollector, AssociationCloser, FixedIntervalClosureBuilder, ClosedTrackletCollector
from dna.assoc import AssociationCollection, AssociationCollector, AssociationClosureBuilder
from scripts import *

LOGGER = logging.getLogger('dna.assoc.features')


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Associate tracklets by feature")
    
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='latest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--listen", nargs='+', help="listening nodes")
    parser.add_argument("--db_url", metavar="URL", help="PostgreSQL url", default='postgresql://dna:urc2004@localhost:5432/dna')
    parser.add_argument("--motion_score_threshold", type=float, metavar="0~1", default=0.35,
                        help="motion match score threshold (default: 0.35)")
    
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    store = TrackletStore.from_url(config.get(conf, 'db_url'))
    associator = FeatureBasedTrackletAssociator(store=store,
                                              listen_nodes=args.listen,
                                              prefix_length=5,
                                              top_k=4,
                                              logger=LOGGER.getChild('associator'))
    
    close_collector = ClosedTrackletCollector()
    associator.add_listener(close_collector)
    
    interval_collector = FixedIntervalClosureBuilder(interval_ms=5*1000,
                                                     closer_collector=close_collector,
                                                     logger=LOGGER.getChild('interval'))
    associator.add_listener(interval_collector)
    
    closures = AssociationCollection(keep_best_association_only=True)
    closuer_builder = AssociationClosureBuilder(collection=closures)
    closuer_builder.add_listener(PrintEvent("**** "))
    interval_collector.add_listener(closuer_builder)
    
    with closing(open_kafka_consumer(brokers=args.kafka_brokers,
                                     offset_reset=args.kafka_offset,
                                     key_deserializer=lambda k: k.decode('utf-8'))) as feature_consumer:
        feature_consumer.subscribe(['track-features'])
        
        listening_nodes = set(args.listen)
        for record in read_topics(feature_consumer, timeout=10*1000):
            feature = TrackFeature.deserialize(record.value)
            if record.key in listening_nodes or feature.zone_relation == 'D':
                associator.handle_event(feature)
          
    associator.close()
    for assoc in closures:
        print(assoc)                  
            
if __name__ == '__main__':
    main()