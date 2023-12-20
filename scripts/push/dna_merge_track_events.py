from __future__ import annotations

from typing import Union, Optional
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass, replace
import itertools
import pickle

import numpy as np
import cv2
from omegaconf import OmegaConf
import json
from pathlib import Path

from dna.event import NodeTrack, read_json_event_file
from dna.support import iterables

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("track_files", nargs='+', help="track files to display")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    return parser.parse_known_args()

def read_adjusted_track_events(track_file:str, offset:int):
    def adjust_frame_index(track:NodeTrack):
        return replace(track, frame_index=track.frame_index + offset)
    return (adjust_frame_index(track) for track in read_json_event_file(track_file, NodeTrack))

def shift_events(events:list[NodeTrack], offset:int):
    frame_delta = events[offset].frame_index - events[0].frame_index
    ts_delta = events[offset].ts - events[0].ts
    return [replace(ev, frame_index = ev.frame_index-frame_delta, ts = ev.ts-ts_delta) for ev in events[offset:]]
    
def main():
    args, _ = parse_args()

    if args.offsets is not None:
        offsets = [int(vstr) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.track_files)
    shift = 0 - min(offsets)
    offsets = [o + shift for o in offsets]
    
    full_events:list[NodeTrack] = []
    for i, track_file in enumerate(args.track_files):
        events:list[NodeTrack] = list(read_json_event_file(track_file, NodeTrack))
        full_events.extend(shift_events(events, offsets[i]))
    
    print(f"sorting {len(full_events)} events by timestamp.")
    full_events.sort(key=lambda ev:ev.ts)
    
    print(f"writing: {len(full_events)} records into file '{args.output}'")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as fp:
        for ev in full_events:
            pickle.dump(ev, fp)
    
if __name__ == '__main__':
    main()