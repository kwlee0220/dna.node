from __future__ import annotations

from typing import Optional
from collections.abc import Generator
from pathlib import Path
from collections import defaultdict
import itertools

import cv2

import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna import Point, Box, Size2d, Frame, TrackId
from dna.config import load_node_conf2
from dna.camera import ImageProcessor, FrameProcessor, create_camera_from_conf
from dna.track.track_state import TrackState
from dna.event import NodeTrack, EventProcessor
from dna.node import TrackEventPipeline
from dna.node.zone import ZoneEvent, ZonePipeline
from dna.event.utils import read_tracks_json
from dna.zone import Zone


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate ReID training data set")
    parser.add_argument("track_video")
    parser.add_argument("track_file")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--margin", metavar="pixels", type=int, default=0, help="margin pixels for cropped image")
    parser.add_argument("--matches", metavar="path", default=None, help="path to the tracklet match file")
    parser.add_argument("--start_gidx", metavar="number", type=int, default=0, help="starting global tracjectory number")
    parser.add_argument("--min_size", metavar="2d-size", type=str, default='70x70', help="minimum training crop image size")
    parser.add_argument("--output", "-o", metavar="directory path", help="output training data directory path")
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


def load_tracklets_by_frame(tracklet_gen:Generator[NodeTrack, None, None]) -> dict[int,list[NodeTrack]]:
    tracklets:dict[int,list[NodeTrack]] = dict()
    for track in tracklet_gen:
        tracks = tracklets.get(track.frame_index)
        if tracks is None:
            tracklets[track.frame_index] = [track]
        else:
            tracklets[track.frame_index].append(track)

    return tracklets


class TrackletCropWriter(FrameProcessor):
    def __init__(self, tracks_per_frame_index:dict[int, list[NodeTrack]],
                 global_tracklet_mappings: dict[TrackId,tuple[str,str,str]],
                 output_dir:str,
                 margin:int=5,
                 min_size:Size2d=Size2d([80, 80])) -> None:
        self.tracks_per_frame_index = tracks_per_frame_index
        self.global_tracklet_mappings = global_tracklet_mappings
        self.margin = margin
        self.min_size = min_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        self.non_global_tracklets = set()
        self.global_tracklet_seqnos = defaultdict(itertools.count)
        
    def on_started(self, proc:ImageProcessor) -> None: pass
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        tracks = self.tracks_per_frame_index.pop(frame.index, None)
        if tracks is None:
            return frame
        
        for track in tracks:
            if track.is_deleted():
                self.sessions.pop(track.track_id, None)
            elif track.is_confirmed() or track.is_tentative():
                crop_file = self.crop_file_path(track)
                if crop_file:
                    crop_file.parent.mkdir(parents=True, exist_ok=True)

                    h, w, _ = frame.image.shape
                    border = Box.from_size((w,h))
                    
                    assert track.detection_box
                    # crop_box = track.location.expand(self.margin).to_rint()
                    crop_box = track.detection_box.expand(self.margin).to_rint()
                    crop_box = crop_box.intersection(border)
                    if crop_box.size() > self.min_size:
                        track_crop = crop_box.crop(frame.image)
                        cv2.imwrite(str(crop_file), track_crop)

        return frame

    def crop_file_path(self, track:NodeTrack) -> str:
        gid = self.global_tracklet_mappings.get((track.node_id, track.track_id))
        if not gid:
            if (track.node_id, track.track_id) not in self.non_global_tracklets:
                print(f'cannot find global track: {track.node_id}, {track.track_id}')
                self.non_global_tracklets.add((track.node_id, track.track_id))
            return None
        
        node_id = track.node_id.replace(':', '_')
        seqno = next(self.global_tracklet_seqnos[gid])
        return self.output_dir / f'{gid}' / f'{node_id}_{seqno:05}.png'


def load_tracklet_matches(file:str, start_index:int=0) -> dict[(str, TrackId),str]:
    import csv

    global_tracklet_mappings:dict[(str, TrackId),str] = dict()
    with open(file, 'r') as fp:
        csv_reader = csv.DictReader(fp)
        for idx, fields in enumerate(csv_reader, start=start_index):
            for node, track_id in fields.items():
                if track_id != 'X':
                    global_tracklet_mappings[(node, track_id)] = f'g{idx:05}'
    return global_tracklet_mappings


def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf2(args, ['show_progress'])
        
    tracks = (track for track in read_tracks_json(args.track_file) if not track.is_deleted())
    tracks_per_frame_index = load_tracklets_by_frame(tracks)
        
    tracklet_match_file = args.matches
    global_tracklet_mappings = load_tracklet_matches(tracklet_match_file, args.start_gidx) if tracklet_match_file else None
    
    # 카메라 설정 정보 추가
    conf.camera = {'uri': args.track_video, 'sync': False, }
    camera = create_camera_from_conf(conf.camera)

    min_size = Size2d.parse_string(args.min_size)
    
    img_proc = ImageProcessor(camera.open(), conf)
    training_data_writer = TrackletCropWriter(tracks_per_frame_index, global_tracklet_mappings,
                                              args.output, margin=args.margin, min_size=min_size)
    img_proc.add_frame_processor(training_data_writer)
    img_proc.run()

if __name__ == '__main__':
	main()