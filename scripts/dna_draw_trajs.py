
from typing import Any, Protocol

import sys
import argparse
import cv2

from dna import Box, Image, color, Point, initialize_logger
from dna.support import iterables
from dna.event import NodeTrack, JsonEvent, read_kafka_event_file, parse_event_string, read_json_event_file
from dna.event.json_event import JsonEventImpl
from dna.node.world_coord_localizer import ContactPointType, WorldCoordinateLocalizer
from dna.node.running_stabilizer import RunningStabilizer
from dna.node.trajectory_drawer import TrajectoryDrawer, GlobalPointLocator, LocalPointLocator
from dna.assoc import GlobalTrack
import scripts


def define_args(parser):
    parser.add_argument("track_file")
    parser.add_argument("--type", metavar="type", type=str, default='point',
                        help="track type (e.g. node-track, global-track, json-event)")
    parser.add_argument("--bg_image", metavar="path", help="path to the background image file")
    
    # world coordinate localizer setting
    parser.add_argument("--camera_index", metavar="index", type=int, default=0, help="camera index")
    parser.add_argument("--contact_point", metavar="contact-point type", choices=_contact_point_choices,
                        type=str.lower, default='simulation', help="contact-point type")
    parser.add_argument("--world_map_image", metavar="path", default=scripts.WORLD_MAP_IMAGE_FILE,
                        help="path to the world map image file")
    
    # stabilizer setting
    parser.add_argument("--look_ahead", metavar='count', type=int, default=10, help="look-ahead/behind count")
    parser.add_argument("--smoothing", metavar='value', type=float, default=1, help="stabilization smoothing factor")
    
    # other setting
    parser.add_argument("--output", "-o", metavar="file path", help="output jpeg file path")
    parser.add_argument("--thickness", metavar="number", type=int, default=1, help="drawing line thickness")
    parser.add_argument("--color", metavar='color', default='RED', help="color for trajectory")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


_contact_point_choices = [t.name.lower() for t in ContactPointType]

def load_node_tracks(track_file:str, localizer:WorldCoordinateLocalizer) -> dict[str,list[Box]]:
    tracks = (track for track in read_kafka_event_file(track_file, event_type=NodeTrack) if not track.is_deleted())
    return iterables.groupby(tracks,
                             key_func=lambda t: str(t.tracklet_id),
                             value_func=lambda t: localizer.select_contact_point(t.bbox.tlbr))

def load_global_tracks(track_file:str) -> dict[str,list[Point]]:
    tracks = (track for track in read_kafka_event_file(track_file, event_type=GlobalTrack) if not track.is_deleted())
    return iterables.groupby(tracks, key_func=lambda t: t.id, value_func=lambda t: t.location)

def load_json_tracks(track_file:str) -> dict[str,list[Point]]:
    tracks = (track for track in read_json_event_file(track_file, event_type=JsonEventImpl) if not track.is_deleted())
    return iterables.groupby(tracks, key_func=lambda t: t.id, value_func=lambda t: t.location)


def run(args):
    initialize_logger(args.logger)
    
    bg_img = cv2.imread(args.bg_image, cv2.IMREAD_COLOR)
    
    world_map_image = cv2.imread(args.world_map_image, cv2.IMREAD_COLOR) if args.world_map_image else None
    contact_point = ContactPointType(_contact_point_choices.index(args.contact_point)) if args.contact_point else None
    localizer = WorldCoordinateLocalizer(scripts.LOCALIZER_CONFIG_FILE, args.camera_index, contact_point=contact_point)
        
    trajs = None
    locator = None
    event_type = parse_event_string(args.type)
    if event_type == NodeTrack:
        trajs = load_node_tracks(args.track_file, localizer)
        locator = LocalPointLocator(localizer=localizer, camera_image=bg_img, world_map_image=world_map_image)
    elif event_type == GlobalTrack:
        trajs = load_global_tracks(args.track_file)
        locator = GlobalPointLocator(localizer=localizer, bg_image=bg_img)
    elif event_type == JsonEvent:
        trajs = load_json_tracks(args.track_file)
        locator = LocalPointLocator(localizer=localizer, camera_image=bg_img, world_map_image=world_map_image)
    else:
        print(f'invalid track type: {args.type}', file=sys.stderr)
            
    stabilizer = None
    if args.look_ahead > 0 and args.smoothing > 0:
        stabilizer = RunningStabilizer(args.look_ahead, args.smoothing)
            
    traj_color = color.__dict__[args.color]
    drawer = TrajectoryDrawer(locator=locator, trajs=trajs, stabilizer=stabilizer, traj_color=traj_color)
    drawer.run()
    

def main():
    parser = argparse.ArgumentParser(description="Draw trajectories")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()