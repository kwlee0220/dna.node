from __future__ import annotations

from typing import Union, Optional
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np
import cv2
from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import Box, Image, BGR, color, Point, TrackletId, initialize_logger, config
from dna.camera import Frame, Camera
from dna.event import NodeTrack, read_topics, KafkaEventPublisher
from dna.node import stabilizer
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.support import plot_utils
from dna.track import TrackState
from dna.assoc.global_locator import GlobalLocation, GlobalTrackLocator
from scripts import update_namespace_with_environ

COLORS = {
    'etri:04': color.BLUE,
    'etri:05': color.GREEN,
    'etri:06': color.YELLOW,
    'etri:07': color.INDIGO,
    'global': color.RED
}
MAX_DISTS = {'etri:04': 45, 'etri:05': 45, 'etri:06': 45, 'etri:07': 45 }
LOCALIZERS:dict[str,WorldCoordinateLocalizer] = dict()
LOCALIZER = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                     camera_index=0, contact_point=ContactPointType.Simulation)

@dataclass(frozen=True)
class Location:
    luid: int
    point: Point
    distance: float

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


MAX_DISTS = {'etri:04': 45, 'etri:05': 45, 'etri:06': 45, 'etri:07': 45 }
def publish_global_locations(consumer:KafkaConsumer, publisher:KafkaEventPublisher,
                             *,
                             distinct_dist:float,
                             timeout:int):
    global_locations = GlobalTrackLocator(distinct_dist=6.0)
    for record in read_topics(consumer, timeout=500):
        track = NodeTrack.deserialize(record.value)
        
        if track.is_deleted():
            tracks.pop(track.tracklet_id, None)
            gloc = global_locations.update(track)
            if not gloc:
                tracks.pop(TrackletId("global", str(hash(gloc))), None)
        else:
            if track.distance <= MAX_DISTS[track.node_id]:
                global_locations.update(track)
            else:
                tracks.pop(track.tracklet_id, None)
                    
            tracks = {k:v for k, v in tracks.items() if k.node_id != 'global'}
            for idx, gloc in enumerate(global_locations):
                tracks[TrackletId("global", str(hash(gloc)))] = gloc.mean


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    world_image = cv2.imread("regions/etri_testbed/ETRI_221011.png", cv2.IMREAD_COLOR)
    localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                         camera_index=0, contact_point=ContactPointType.Simulation)
    drawer = MCLocationDrawer(title="Multiple Objects Tracking", localizer=localizer, world_image=world_image)
    
    consumer = KafkaConsumer(bootstrap_servers=args.kafka_brokers,
                             auto_offset_reset=args.kafka_offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['node-tracks'])
    
    done = False
    first_ts = -1
    first_rts = -1
    global_locations = GlobalTrackLocator(distinct_dist=6.0)
    tracks:dict[TrackletId,Point] = dict()
    while not done:
        for record in read_topics(consumer, timeout=500):
            track = NodeTrack.deserialize(record.value)
            
            if track.is_deleted():
                tracks.pop(track.tracklet_id, None)
                gloc = global_locations.update(track)
                if not gloc:
                    tracks.pop(TrackletId("global", str(hash(gloc))), None)
            else:
                if track.distance <= MAX_DISTS[track.node_id]:
                    tracks[track.tracklet_id] = track.location
                    global_locations.update(track)
                else:
                    tracks.pop(track.tracklet_id, None)
                        
                tracks = {k:v for k, v in tracks.items() if k.node_id != 'global'}
                for idx, gloc in enumerate(global_locations):
                    tracks[TrackletId("global", str(hash(gloc)))] = gloc.mean
            
            now_rts = int(time.time()*1000)
            if first_ts < 0:
                first_ts = track.ts
                first_rts = now_rts
                
            delay = (track.ts - first_ts) - (now_rts - first_rts)
            if delay > 30:
                # key = cv2.waitKey(1) & 0xFF
                key = cv2.waitKey(delay) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            last_ts = track.ts
            drawer.draw_tracks(tracks)
            
            if key == ord('q'):
                done = True
                break
        if not done:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                done = True
    drawer.close()

if __name__ == '__main__':
    main()