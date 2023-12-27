from __future__ import annotations

from typing import Optional, Generator, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import cv2
import argparse

from dna import Box, BGR, color, Point, initialize_logger
from dna.camera import Image
from dna.support import iterables
from dna.node import NodeTrack
from dna.event.utils import read_pickle_event_file
from dna.node.world_coord_localizer import ContactPointType, WorldCoordinateLocalizer
from dna.support import plot_utils
import scripts


def create_localizer(camera_index:int) -> WorldCoordinateLocalizer:
    return WorldCoordinateLocalizer(scripts.LOCALIZER_CONFIG_FILE,
                                     camera_index=camera_index,
                                     contact_point=ContactPointType.BottomCenter)

@dataclass(frozen=True,slots=True)
class Node:
    id:str
    localizer: WorldCoordinateLocalizer
    color:color.BGR
    max_dist:Optional[float] = field(default=None)
    frame_offset:int = field(default=0)
    
    
NODES:dict[str,Node] = {
    'etri:01': Node(id='etri:01', localizer=create_localizer(0), color=color.PURPLE),
    'etri:02': Node(id='etri:02', localizer=create_localizer(1), color=color.WHITE),
    'etri:03': Node(id='etri:03', localizer=create_localizer(2), color=color.GREEN),
    'etri:04': Node(id='etri:04', localizer=create_localizer(3), color=color.RED),
    'etri:05': Node(id='etri:05', localizer=create_localizer(4), color=color.BLUE),
    'etri:06': Node(id='etri:06', localizer=create_localizer(6), color=color.YELLOW),
    'etri:07': Node(id='etri:07', localizer=create_localizer(7), color=color.ORANGE),
    # 'etri:08': Node(id='etri:08', localizer=create_localizer(8), color=color.INDIGO),
}

@dataclass(frozen=True)
class Location:
    track_id: int
    point: Point
    distance: float


def define_args(parser):
    parser.add_argument("track_files", nargs='+', help="track files to display")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--start_frame", metavar="number", type=int, default=1, help="camera frame offsets")
    parser.add_argument("--interactive", "-i", action='store_true', help="show trajectories interactively")

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    
    

def load_scenes(track_files:list[str]) -> dict[int,list[Vehicle]]:
    def read_node_tracks(event_file:str) -> Generator[NodeTrack, None, None]:
        return (ev for ev in read_pickle_event_file(event_file )
                        if isinstance(ev, NodeTrack) and not ev.is_deleted())

    import itertools
    tracks:Iterable[NodeTrack] = itertools.chain.from_iterable((read_node_tracks(tfile) for tfile in track_files))
    return iterables.groupby(tracks,
                            key_func=lambda t: t.frame_index - NODES[t.node_id].frame_offset,
                            value_func=lambda t: Vehicle.from_track(t, NODES[t.node_id].localizer))


def run(args):
    initialize_logger(args.logger)

    if args.offsets is not None:
        offsets = [int(vstr) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.track_files)
    shift = 0 - min(offsets)
    offsets = [o + shift for o in offsets]

    for node, offset in zip(NODES.keys(), offsets):
        NODES[node] = replace(NODES[node], frame_offset=offset)
    scenes:dict[int,list[Vehicle]] = load_scenes(args.track_files)
        
    world_image = cv2.imread(scripts.WORLD_MAP_IMAGE_FILE, cv2.IMREAD_COLOR)
    drawer = MCLocationDrawer(scenes, world_image)
    drawer.draw(start_frame=args.start_frame)


class MCLocationDrawer:
    def __init__(self, scenes: dict[int,list[Vehicle]], world_image: Image) -> None:
        self.scenes = scenes
        self.world_image = world_image
        
        self.frames = list(scenes.keys())
        self.frames.sort()

    def draw_frame_index(self, convas: Image, base_frame_index:int, vehicles:list[Vehicle]) -> None:
        node_ids = list(iterables.groupby(vehicles, key_func=lambda v: v.node, init_dict=dict()).keys())
        node_ids.sort()
        
        cv2.putText(convas, f'{base_frame_index}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.RED, 2)
        for i in range(0, len(node_ids)):
            node_id = node_ids[i]
            frame_index = base_frame_index + NODES[node_id].frame_offset
            voff = 28 * (i+1)
            cv2.putText(convas, f'{node_id}: {frame_index}', (10, 25+voff), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.RED, 2)
    
    def draw_frame(self, cursor:int) -> Image:
        convas = self.world_image.copy()
        
        frame_index = self.frames[cursor]
        vehicles:list[Vehicle] = self.scenes[frame_index]
        
        self.draw_frame_index(convas, frame_index, vehicles)
        for vehicle in vehicles:
            node = NODES[vehicle.node]
            if not node.max_dist or vehicle.distance <= node.max_dist:
                self.draw_vehicle(convas, vehicle, fill_color=node.color)
        cv2.imshow("locations", convas)
        return convas
    
    def update_offset(self, index:int, delta:int):
        self.offsets[index] += delta
        shift = 0 - min(self.offsets)
        self.offsets = [o + shift for o in self.offsets]
        
    def find_index(self, frame_index:int):
        idx, _ = iterables.find_first(self.frames, lambda f: f > frame_index)
        return max(idx-1, 0)
        
    def draw(self, title='locations', start_frame:int=1, interactive:bool=True) -> None:
        cursor = self.find_index(start_frame)
        self.draw_frame(cursor)
        
        while True:
            delay = 1 if interactive else 100
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif  key == ord(' '):
                interactive = not interactive
            elif interactive:
                if key == ord('n'):
                    if cursor < len(self.frames)-1:
                        cursor += 1
                        self.draw_frame(cursor)
                elif key == ord('p'):
                    if cursor > 0:
                        cursor -= 1
                        self.draw_frame(cursor)
                elif key == ord('s'):
                    image = self.draw_frame(cursor)
                    cv2.imwrite("output/output.png", image)
                else:
                    delta = key - ord('1')
                    if delta >= 0 and delta < 5:
                        self.update_offset(delta, 1)
                        self.draw_frame(cursor)
                    elif delta >= 5 and delta < 10:
                        self.update_offset(delta-5, -1)
                        self.draw_frame(cursor)
            else:
                if key == 0xFF:
                    if cursor < len(self.frames)-1:
                        cursor += 1
                        self.draw_frame(cursor)
        cv2.destroyWindow(title)

    def draw_vehicle(self, convas:Image, vehicle:Vehicle, fill_color:BGR) -> Image:
        convas = cv2.circle(convas, center=tuple(vehicle.point.round()), radius=8,
                            color=fill_color, thickness=-1, lineType=cv2.LINE_AA)
        loc = vehicle.point - (-20, 0)
        convas = plot_utils.draw_label(convas, f'{vehicle.id}', loc, font_scale=0.7,
                                       color=color.BLACK, fill_color=fill_color, line_thickness=1)
        return convas


@dataclass(frozen=True)
class Vehicle:
    node:str
    id: str
    point: Point
    distance: float
    
    @classmethod
    def from_track(cls, track:NodeTrack, localizer:WorldCoordinateLocalizer) -> Vehicle:
        assert track.location and track.distance
        
        id = f"{track.node_id[5:]}[{track.track_id}]"
        pt_m = localizer.from_world_coord(track.location)
        point = Point(localizer.to_image_coord(pt_m))
        return Vehicle(node=track.node_id, id=id, point=point, distance=track.distance)
    
@dataclass(frozen=True)
class Scene:
    vehicles: list[Vehicle]
    

def main():
    parser = argparse.ArgumentParser(description="Show locations from multiple cameras")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()