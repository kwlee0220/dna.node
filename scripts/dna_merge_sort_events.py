from __future__ import annotations

from typing import DefaultDict, Any, Optional
from collections import defaultdict
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, replace
import pickle

from pathlib import Path
from tqdm import tqdm
import argparse

from dna import NodeId
from dna.event import KafkaEvent
from dna.support import iterables


def define_args(parser):
    parser.add_argument("track_files", nargs='+', help="track files to merge")
    parser.add_argument("--node_offsets", metavar="csv", default=None, help="node ids and camera frame offsets")
    parser.add_argument("--output", metavar="path", default=None, help="output file.")


def parse_node_offset_args(node_offsets:str) -> dict[NodeId,int]:
    if node_offsets is not None:
        def parse_node_offset(node_offset:str) -> tuple[str,int]:
            idx = node_offset.rindex(':')
            return node_offset[:idx].strip(), int(node_offset[idx+1:].strip())
        
        node_offsets:dict[NodeId,int] = { node_id:offset for node_id, offset
                                                in ( parse_node_offset(part) for part in node_offsets.split(',') ) }
    else:
        node_offsets:dict[NodeId,int] = dict()
        
    shift = 0 - min(node_offsets.values(), default=0)
    return  { node:offset+shift for node, offset in node_offsets.items() }


def calc_ts_delta(node_events:list[KafkaEvent], offset:int, base_idx:int) -> Optional[int]:
    base_event = node_events[base_idx]
    _, found = iterables.find_first(node_events, lambda ev: (ev.frame_index - base_event.frame_index) == offset)
    if found is not None:
        return found.ts - base_event.ts
    else:
        return None

def shift_backward(events:Iterable[KafkaEvent], node_offsets:dict[NodeId,int]):
    def shift_event(ev:KafkaEvent, frame_index_delta:int, ts_delta:int) -> KafkaEvent:
        if hasattr(ev, 'first_ts'):
            return replace(ev, frame_index=ev.frame_index-frame_index_delta, first_ts=ev.first_ts-ts_delta,
                           ts=ev.ts-ts_delta)
        else:
            return replace(ev, frame_index=ev.frame_index-frame_index_delta, ts=ev.ts-ts_delta)
    
    shifted_events:list[KafkaEvent] = []
    node_event_groups:dict[NodeId,list[KafkaEvent]] = iterables.groupby(events, lambda ev: ev.node_id)
    for node_id, offset in node_offsets.items():
    # for node_id, node_events in node_event_groups.items():
        node_events = node_event_groups.get(node_id)
        if node_events is None:
            continue
        
        # 계산된 frame_index offset과 timestamp offset을 이용하여 지정된 node에서 생성된 모든 event들을 calibration한다.
        # 'NodeTrack'의 경우에는 first_ts가 존재하기 때문에 이것도 함께 보정해야 한다.
        if offset > 0:
            # 첫번째 event의 frame_index를 기준으로 삼아, 이것보다 offset만큼 큰 frame_index를 갖는 event를 찾아서
            # 두 event 사이의 timestamp 차이를 구한다.
            for i in range(50):
                ts_delta = calc_ts_delta(node_events, offset, i)
                if ts_delta is not None:
                    break
                
            print(f"\tshift backward: node={node_id}, frame={offset}, ts={ts_delta}")
            shifteds = [shift_event(ev, frame_index_delta=offset, ts_delta=ts_delta)
                                for ev in node_events if ev.frame_index >= offset]
            shifted_events.extend(shifteds)
        else:
            print(f"\tshift backward: node={node_id}, frame=0, ts=0")
            shifted_events.extend(node_events)
            
    return shifted_events
       
    
from dna.event import read_pickle_event_file  
def run(args):    
    # 노드간 event들의 offset 정보를 얻는다.
    node_offsets = parse_node_offset_args(args.node_offsets)
    
    # read track files
    full_events:list[KafkaEvent] = []
    for track_file in args.track_files:
        full_events.extend(read_pickle_event_file(track_file))
    
    # 전체 event를 node별로 grouping하고, node별로 설정된 offset 정보를
    # 이용하여 모든 event들의 frame, ts 정보를 수정한다.
    shifted_events = shift_backward(full_events, node_offsets)
    print(f"shifted {len(full_events)} events.")
    
    shifted_events.sort(key=lambda ev:ev.ts)
    print(f"sorted {len(shifted_events)} events by timestamp.")
    
    print(f"writing: {len(shifted_events)} records into file '{args.output}'")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as fp:
        for ev in tqdm(shifted_events):
            pickle.dump(ev, fp)

    
def main():
    parser = argparse.ArgumentParser(description="Read/merge per-node node events and sort them into output file.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()