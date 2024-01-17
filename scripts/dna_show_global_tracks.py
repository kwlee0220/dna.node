from __future__ import annotations

from typing import Union, Optional, Iterable, Generator
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import heapq
from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
import argparse

from dna import Box, color, Point, TrackletId, initialize_logger, config, Size2di
from dna.utils import utc2datetime, datetime2str
from dna.camera import Image
from dna.camera.opencv_video_writer import OpenCvVideoWriter
from dna.node import NodeTrack, GlobalTrack
from dna.event.kafka_utils import open_kafka_consumer, read_topics, PollTimeout, KafkaPollData
from dna.event.utils import sort_events_with_fixed_buffer
from dna.event.json_event import JsonEventImpl
from dna.node import stabilizer
from dna.node.world_coord_localizer import ContactPointType, WorldCoordinateLocalizer
from dna.support import plot_utils
from dna.track import TrackState
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
    
    parser.add_argument("--show_supports", action='store_true', help="show the locations of supports")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--no_show", action='store_true', help="do not display convas on the screen")
    parser.add_argument("--output_video", metavar="path", help="output video file path")
    parser.add_argument("--progress", action='store_true', default=False)
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


class GlobalTrackDrawer:
    def __init__(self, title:str, localizer:WorldCoordinateLocalizer, world_image:Image,
                 *,
                 output_video:Optional[str]=None,
                 show_supports:bool=False,
                 no_show:bool=False) -> None:
        self.title = title
        self.localizer = localizer
        self.world_image = world_image
        self.show_supports = show_supports
        
        if output_video:
            h, w, _ = world_image.shape
            self.writer = OpenCvVideoWriter(output_video, 10, Size2di(w, h))
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
        
        ts = max((gl.ts for gl in gtracks), default=0)
        dt_str = datetime2str(utc2datetime(ts))
        convas = cv2.putText(convas, f'{dt_str} ({ts})', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
    
        for gtrack in gtracks:
            assert gtrack.location
            gloc = self.to_image_coord(gtrack.location)
                
            label_pos = Point(gloc) + (-35, 35)
            convas = plot_utils.draw_label(convas, f'{gtrack.id}', label_pos, font_scale=FONT_SCALE,
                                           color=color.BLACK, fill_color=color.YELLOW, line_thickness=1)
            
            if gtrack.is_associated():
                convas = cv2.circle(convas, gloc, radius=RADIUS_GLOBAL, color=color.RED, thickness=-1,
                                    lineType=cv2.LINE_AA)
                if self.show_supports and gtrack.supports is not None:
                    for ltrack in gtrack.supports:
                        track_color = COLORS[ltrack.node]
                        sample = self.to_image_coord(ltrack.location)
                        convas = cv2.line(convas, gloc, sample, track_color, thickness=1, lineType=cv2.LINE_AA)
                        convas = cv2.circle(convas, sample, radius=RADIUS_LOCAL, color=track_color,
                                            thickness=-1, lineType=cv2.LINE_AA)
            else:
                node = TrackletId.from_string(gtrack.id).node_id
                track_color = COLORS[node]
                convas = cv2.circle(convas, gloc, radius=RADIUS_LOCAL, color=track_color, thickness=-1,
                                    lineType=cv2.LINE_AA)
            if self.writer:
                self.writer.write(convas)
        
        return convas
        
    def to_image_coord(self, world_coord:Point) -> tuple[int,int]:
        pt_m = self.localizer.from_world_coord(world_coord)
        return Point(self.localizer.to_image_coord(pt_m)).round()

def run(args:argparse.Namespace):
    world_image = cv2.imread(scripts.WORLD_MAP_IMAGE_FILE, cv2.IMREAD_COLOR)
    localizer = WorldCoordinateLocalizer(scripts.LOCALIZER_CONFIG_FILE,
                                         camera_index=0, contact_point=ContactPointType.BottomCenter)
    drawer = GlobalTrackDrawer(title="Multiple Objects Tracking", localizer=localizer, world_image=world_image,
                                output_video=args.output_video, show_supports=args.show_supports,
                                no_show=args.no_show)
        
    options = vars(args)
    upper_ts = -1
    tracks:list[GlobalTrack] = []
    convas = drawer.draw_tracks(tracks)
    
    consumer = open_kafka_consumer(value_deserializer=GlobalTrack.deserialize, **options)
    with closing(consumer), closing(drawer):
        for data in read_topics(consumer, **options):
            if isinstance(data, ConsumerRecord):
                track:GlobalTrack = data.value
                if upper_ts < 0:
                    upper_ts = track.ts + 100
                if track.is_deleted():
                    continue
                
                if track.ts < upper_ts:
                    tracks.append(track)
                else:
                    convas = drawer.draw_tracks(tracks)
                    if not args.no_show:
                        cv2.imshow(drawer.title, convas)
                        
                        delay_ms = 100 if args.sync else 1
                        key = cv2.waitKey(delay_ms) & 0xFF
                        if key == ord('q'):
                            break
                        
                    tracks.clear()
                    tracks.append(track)    
                    upper_ts += 100
            elif isinstance(data, PollTimeout) and not args.no_show:
                cv2.imshow(drawer.title, convas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

def main():
    parser = argparse.ArgumentParser(description="Display global-tracks.")
    define_args(parser)
    args = parser.parse_args()
    
    if args.topics is None:
        args.topics = ['global-tracks']

    initialize_logger(args.logger)
    run(args)
    return parser.parse_args()


if __name__ == '__main__':
    main()