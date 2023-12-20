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

from dna import Box, Image, BGR, color, Frame, Point, TrackletId, initialize_logger, config
from dna.camera import Camera
from dna.event import NodeTrack, read_topics
from dna.node import stabilizer
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.support import plot_utils
from dna.track import TrackState
from scripts import update_namespace_with_environ

COLORS = {
    'etri:04': color.RED,
    'etri:05': color.BLUE,
    'etri:06': color.GREEN,
    'etri:07': color.YELLOW
}
MAX_DISTS = {'etri:04': 50, 'etri:05': 40, 'etri:06': 45, 'etri:07': 40 }
LOCALIZERS:dict[str,WorldCoordinateLocalizer] = dict()

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


class MCLocationDrawer:
    def __init__(self, title:str, localizers:dict[str,WorldCoordinateLocalizer], world_image:Image) -> None:
        self.title = title
        self.localizers = localizers
        self.world_image = world_image
        cv2.namedWindow(self.title)
        
    def close(self) -> None:
        cv2.destroyWindow(self.title)
    
    def draw_tracks(self, tracks:dict[TrackletId,NodeTrack]) -> Image:
        convas = self.world_image.copy()
        for track in tracks.values():
            color = COLORS[track.node_id]
            if track.distance < MAX_DISTS[track.node_id]:
                localizer = self.localizers[track.node_id]
                pt_m = localizer.from_world_coord(track.location)
                # localizer = self.localizers[track.node_id]
                # pt_m, dist = localizer.from_camera_box(track.location.tlbr)
                center = tuple(Point(localizer.to_image_coord(pt_m)).to_rint().xy)
                convas = cv2.circle(convas, center, radius=7, color=color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow(self.title, convas)
        return convas


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    world_image = cv2.imread("regions/etri_testbed/ETRI_221011.png", cv2.IMREAD_COLOR)
    LOCALIZERS['etri:04'] = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                                     camera_index=0, contact_point=ContactPointType.Simulation)
    LOCALIZERS['etri:05'] = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                                     camera_index=1, contact_point=ContactPointType.Simulation)
    LOCALIZERS['etri:06'] = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                                     camera_index=2, contact_point=ContactPointType.Simulation)
    LOCALIZERS['etri:07'] = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                                     camera_index=3, contact_point=ContactPointType.Simulation)
    drawer = MCLocationDrawer(title="Multiple Objects Tracking", localizers=LOCALIZERS, world_image=world_image)
    
    consumer = KafkaConsumer(bootstrap_servers=args.kafka_brokers,
                             auto_offset_reset=args.kafka_offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['node-tracks'])
    
    first_ts = -1
    first_rts = -1
    done = False
    tracks:dict[TrackletId,NodeTrack] = dict()
    while not done:
        for record in read_topics(consumer, timeout=500):
            track = NodeTrack.deserialize(record.value)
            
            if track.is_deleted():
                tracks.pop(track.tracklet_id)
            else:
                tracks[track.tracklet_id] = track
            
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
            drawer.draw_tracks(tracks)
            last_ts = track.ts
            
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