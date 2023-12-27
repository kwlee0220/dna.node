from __future__ import annotations

from typing import DefaultDict
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import cv2
import argparse

from dna import Point, Size2d, color, Trajectory, config
from dna.camera import Image, Frame, ImageCapture, create_camera, OpenCvVideoWriter
from dna.event import NodeTrack, read_json_event_file
from dna.node.world_coord_localizer import WorldCoordinateLocalizer
from dna.support import iterables
from dna.track import TrackState
from scripts import WORLD_MAP_IMAGE_FILE, LOCALIZER_CONFIG_FILE


def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("file", help="events file (json or pickle format)")
    parser.add_argument("--input_video", metavar="path", help="input video file path")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--output_video", metavar="path", help="output video file path")
    
    return parser.parse_known_args()


class ImageReader:
    def __init__(self, video_file:Path, sync:bool=False) -> None:
        self.video = create_camera(uri=video_file, sync=sync)
        self.capture:ImageCapture = None
        self.current_frame:Frame = None
        
    def close(self) -> None:
        if self.capture is None:
            return
        
        self.capture.close()
        self.capture = None
        self.current_frame = None
        
    def __enter__(self):
        self.capture = self.video.open()
        self.current_frame:Frame = self.capture()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        with suppress(Exception): self.close()
        
    def read(self, frame_index:int) -> Frame:
        if self.capture is None:
            raise ValueError(f'not open')
        if self.current_frame is None:
            raise ValueError(f'end-of-capture already')
        if frame_index < self.current_frame.index:
            raise ValueError(f'too old frame index: current={self.current_frame_index}, requested={frame_index}')
        
        while self.current_frame.index < frame_index:
            self.current_frame = self.capture()
            if self.current_frame is None:
                return None
        return self.current_frame


class ImageCoordinateTrajectories:
    __slots__ = ('trajectories', 'localizer')

    def __init__(self, localizer:WorldCoordinateLocalizer) -> None:
        self.localizer = localizer
        self.trajectories:dict[str,Trajectory] = dict()
    
    def get_or_create_trajectory(self, track_id:str) -> Trajectory:
        traj = self.trajectories.get(track_id)
        if traj is None:
            traj = Trajectory()
            self.trajectories[track_id] = traj
        return traj

    def update(self, tracks:list[NodeTrack]) -> None:      
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                self.__add_track(track)
            elif track.state == TrackState.Deleted:
                self.trajectories.pop(track.track_id, None)
    
    def __add_track(self, track:NodeTrack) -> None:
        pt_m = self.localizer.from_world_coord(track.location.xy)
        pt_img = self.localizer.to_image_coord(pt_m)
        sample = Trajectory.Sample(Point(pt_img), track.ts)
        traj = self.get_or_create_trajectory(track.track_id)
        traj.append(sample)


def main():
    args, _ = parse_args()
    
    localizer = WorldCoordinateLocalizer(LOCALIZER_CONFIG_FILE, 0, 'EPSG:5186')
    world_image = cv2.imread(WORLD_MAP_IMAGE_FILE, cv2.IMREAD_COLOR)
    trajs = ImageCoordinateTrajectories(localizer)
    
    writer:OpenCvVideoWriter = None
    if args.output_video:
        writer = OpenCvVideoWriter(Path(args.output_video).resolve(), 10, Size2d.from_image(world_image))
        writer.open()
    
    tracks = (track for track in read_json_event_file(args.file, event_type=NodeTrack))
    groups:DefaultDict[int,list[NodeTrack]] = iterables.groupby(tracks, key_func=lambda ev:ev.frame_index,
                                                                init_dict=defaultdict(list))

    with ImageReader(args.input_video, sync=args.sync) as reader:
        idx = 1
        cv2.namedWindow('trajectories')
        
        while True:
            frame = reader.read(idx)
            if frame is None:
                break
            
            # convas = trajs.draw(groups[idx])
            trajs.update(groups[idx])
            convas = world_image.copy()
            convas = cv2.putText(convas, f'frame={frame.index}',
                                (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
            for trj in trajs.trajectories.values():
                convas = trj.draw(convas)
            
            if writer:
                writer.write(convas)
            
            cv2.imshow('trajectories', convas)
            key = cv2.waitKey(int(1)) & 0xFF
            if key == ord('q'):
                break
            idx += 1
            
        cv2.destroyWindow("trajectories")
        if writer:
            writer.close()

if __name__ == '__main__':
    main()