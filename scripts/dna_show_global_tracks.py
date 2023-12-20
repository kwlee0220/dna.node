from __future__ import annotations

from typing import Union, Optional, Iterable
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import heapq
from omegaconf import OmegaConf
from kafka import KafkaConsumer
import argparse

from dna import Box, Image, color, Point, TrackletId, initialize_logger, config, Size2d
from dna.utils import utc2datetime, datetime2str
from dna.camera import Camera, Frame
from dna.camera.opencv_video_writer import OpenCvVideoWriter
from dna.event import NodeTrack
from dna.event.utils import sort_events_with_fixed_buffer
from dna.event.json_event import JsonEventImpl
from dna.node import stabilizer
from dna.node.world_coord_localizer import ContactPointType, WorldCoordinateLocalizer
from dna.support import plot_utils
from dna.track import TrackState
from dna.assoc import GlobalTrack
import scripts
from scripts.utils import add_kafka_consumer_arguments


COLORS = {
    'etri:01': color.GOLD,
    'etri:02': color.WHITE,
    'etri:03': color.BLUE,
    'etri:04': color.ORANGE,
    'etri:05': color.GREEN,
    'etri:06': color.YELLOW,
    'etri:07': color.INDIGO,
    'global': color.RED
}

RADIUS_GLOBAL = 10
RADIUS_LOCAL = 6
FONT_SCALE = 0.6


def define_args(parser):
    parser.add_argument("--track_file", default=None, help="track event file (json or pickle format)")
    
    add_kafka_consumer_arguments(parser)
    # parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    # parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
    #                     help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    # parser.add_argument("--topic", help="target topic name")
    # parser.add_argument("--stop_on_timeout", action='store_true', help="stop when a poll timeout expires")
    # parser.add_argument("--timeout_ms", metavar="milli-seconds", type=int, default=1000,
    #                     help="Kafka poll timeout in milli-seconds")
    # parser.add_argument("--initial_timeout_ms", metavar="milli-seconds", type=int, default=5000,
    #                     help="initial Kafka poll timeout in milli-seconds")
    
    parser.add_argument("--show_supports", action='store_true', help="show the locations of supports")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--no_show", action='store_true', help="do not display convas on the screen")
    parser.add_argument("--output_video", metavar="path", help="output video file path")
    parser.add_argument("--progress", action='store_true', default=False)
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


class GlobalTrackDrawer:
    def __init__(self, title:str, localizer:WorldCoordinateLocalizer, world_image:Image,
                 *,
                 output_video:str=None,
                 show_supports:bool=False,
                 no_show:bool=False) -> None:
        self.title = title
        self.localizer = localizer
        self.world_image = world_image
        self.show_supports = show_supports
        
        if output_video:
            self.writer = OpenCvVideoWriter(Path(output_video).resolve(), 10, Size2d.from_image(world_image))
        else:
            self.writer = None
            
        self.no_show = no_show
        if not self.no_show:
            cv2.namedWindow(self.title)
        
    def close(self) -> None:
        if self.writer:
            self.writer.close()
        if not self.no_show:
            cv2.destroyWindow(self.title)
    
    def draw_tracks(self, gtracks:list[GlobalTrack]) -> Image:
        convas = self.world_image.copy()
        
        ts = max((gl.ts for gl in gtracks), default=None)
        dt_str = datetime2str(utc2datetime(ts))
        convas = cv2.putText(convas, f'{dt_str} ({ts})', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
    
        for gtrack in gtracks:
            gloc = self.to_image_coord(gtrack.location)
                
            label_pos = Point(gloc) + [-35, 35]
            convas = plot_utils.draw_label(convas, f'{gtrack.id}', label_pos.to_rint(), font_scale=FONT_SCALE,
                                       color=color.BLACK, fill_color=color.YELLOW, line_thickness=1)
            
            if gtrack.is_associated():
                convas = cv2.circle(convas, gloc, radius=RADIUS_GLOBAL, color=color.RED, thickness=-1, lineType=cv2.LINE_AA)
                if self.show_supports and gtrack.supports is not None:
                    for ltrack in gtrack.supports:
                        track_color = COLORS[ltrack.node]
                        sample = self.to_image_coord(ltrack.location)
                        convas = cv2.line(convas, gloc, sample, track_color, thickness=1, lineType=cv2.LINE_AA)
                        convas = cv2.circle(convas, sample, radius=RADIUS_LOCAL, color=track_color, thickness=-1, lineType=cv2.LINE_AA)
            else:
                node = TrackletId.from_string(gtrack.id).node_id
                track_color = COLORS[node]
                convas = cv2.circle(convas, gloc, radius=RADIUS_LOCAL, color=track_color, thickness=-1, lineType=cv2.LINE_AA)
            if self.writer:
                self.writer.write(convas)
            if not self.no_show:
                cv2.imshow(self.title, convas)
        
        return convas
        
    def to_image_coord(self, world_coord:Point) -> tuple[float,float]:
        pt_m = self.localizer.from_world_coord(world_coord)
        return tuple(Point(self.localizer.to_image_coord(pt_m)).to_rint())


def run(args):
    world_image = cv2.imread(scripts.WORLD_MAP_IMAGE_FILE, cv2.IMREAD_COLOR)
    localizer = WorldCoordinateLocalizer(scripts.LOCALIZER_CONFIG_FILE,
                                         camera_index=0, contact_point=ContactPointType.BottomCenter)
    drawer = GlobalTrackDrawer(title="Multiple Objects Tracking", localizer=localizer, world_image=world_image,
                                output_video=args.output_video, show_supports=args.show_supports,
                                no_show=args.no_show)
        
    last_ts = -1
    tracks:list[GlobalTrack] = []
    events = scripts.read_json_events(args, input_file=args.track_file, deserialize=GlobalTrack.from_json)
    for track in tqdm(sort_events_with_fixed_buffer(events, heap_size=64)):
        if track.is_deleted():
            continue
                
        if last_ts < 0:
            last_ts = track.ts
            
        if last_ts != track.ts:
            convas = drawer.draw_tracks(tracks)
            if not args.no_show:
                delay_ms = track.ts - last_ts if args.sync else 1
                if delay_ms <= 0:
                    delay_ms = 1
                elif delay_ms >= 5000:
                    delay_ms = 100
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    done = True
                    break
                
            tracks.clear()
            last_ts = track.ts
            
        tracks.append(track)
    drawer.close()
    

def main():
    parser = argparse.ArgumentParser(description="Display global-tracks.")
    define_args(parser)
    args = parser.parse_args()
    
    if args.topic is None:
        args.topic = ['global-tracks']

    initialize_logger(args.logger)
    run(args)
    return parser.parse_args()


if __name__ == '__main__':
    main()