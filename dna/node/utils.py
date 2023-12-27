from __future__ import annotations

from typing import Optional, Any, Protocol, runtime_checkable
from collections import defaultdict
from collections.abc import Callable, Generator
from pathlib import Path
import pickle
import logging

from dna.event.event_processor import EventNodeImpl, EventListener
from dna.event.json_event import JsonEventImpl
from .types import SilentFrame
from .node_track import NodeTrack
from .track_feature import TrackFeature
from .global_track import GlobalTrack
            

def read_tracks_csv(track_file:str) -> Generator[NodeTrack, None, None]:
    import csv
    with open(track_file) as f:
        reader = csv.reader(f)
        for row in reader:
            yield NodeTrack.from_csv(row)
            
class NodeEventWriter(EventListener):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(file_path, 'wb')

    def on_completed(self) -> None:
        self.fp.close()

    def handle_event(self, ev:NodeTrack|TrackFeature) -> None:
        match ev:
            case NodeTrack() | TrackFeature():
                pickle.dump(ev, self.fp)
            case SilentFrame():
                pass
            case _:
                raise ValueError(f"unexpected event: {ev}")


@runtime_checkable
class MinFrameIndexGauge(Protocol):
    def min_frame_index(self) -> int: ...


class GroupByFrameIndex(EventNodeImpl):
    def __init__(self, min_frame_index_gauge:MinFrameIndexGauge,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()

        self.groups:dict[int,list[NodeTrack]] = defaultdict(list)  # frame index별로 TrackEvent들의 groupp
        self.min_frame_index_gauge = min_frame_index_gauge
        self.max_published_index = 0
        self.logger = logger

    def on_completed(self) -> None:
        while self.groups:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self.publish_event(group)
        super().on_completed()

    # TODO: 만일 track이 delete 된 후, 한동안 물체가 검출되지 않으면
    # 이 delete event는 계속적으로 publish되지 않는 문제를 해결해야 함.
    # 궁극적으로는 이후 event가 발생되지 않아서 'handle_event' 메소드가 호출되지 않아서 발생하는 문제임.
    def handle_event(self, ev:NodeTrack|SilentFrame) -> None:
        if isinstance(ev, NodeTrack):
            # 만일 새 TrackEvent가 이미 publish된 track event group의 frame index보다 작은 경우
            # late-arrived event 문제가 발생하여 예외를 발생시킨다.
            if ev.frame_index <= self.max_published_index:
                raise ValueError(f'A late TrackEvent: {ev}, already published upto {self.max_published_index}')

            # 이벤트의 frame 번호에 해당하는 group을 찾아 추가한다.
            group = self.groups[ev.frame_index]
            group.append(ev)

            # pending된 TrackEvent group 중에서 가장 작은 frame index를 갖는 group을 검색.
            frame_index = min(self.groups.keys())
            group = self.groups[frame_index]
            # frame_index, group = min(self.groups.items(), key=lambda t: t[0])

            # 본 GroupByFrameIndex 이전 EventProcessor들에서 pending된 TrackEvent 들 중에서
            # 가장 작은 frame index를 알아내어, 이 frame index보다 작은 값을 갖는 group의 경우에는
            # 이후 해당 group에 속하는 TrackEvent가 더 이상 도착하지 않을 것이기 때문에 그 group들을 publish한다.
            min_frame_index = self.min_frame_index_gauge.min_frame_index()
            if not min_frame_index:
                min_frame_index = ev.frame_index

            for idx in range(frame_index, min_frame_index):
                group = self.groups.pop(idx, None)
                if group:
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'publish TrackEvent group: frame_index={idx}, count={len(group)}')
                    self.publish_event(group)
                    self.max_published_index = max(self.max_published_index, idx)
        elif isinstance(ev, SilentFrame):
            for frame_idx in sorted(self.groups.keys()):
                self.publish_event(self.groups[frame_idx])
            self.groups.clear()
            self.max_published_index = ev.frame_index
        else:
            print(f'unexpected event for grouping: {ev}')

    def __repr__(self) -> str:
        keys = list(self.groups.keys())
        range_str = f'[{keys[0]}-{keys[-1]}]' if keys else '[]'
        return f"{self.__class__.__name__}[max_published={self.max_published_index}, range={range_str}]"

                    
def find_event_deserializer(event_type_str:str) -> Callable[[bytes],Any]:
    event_type_str = event_type_str.replace('_', '').replace('-','').lower()
    match event_type_str:
        case 'nodetrack':
            return NodeTrack.deserialize
        case 'globaltrack':
            return GlobalTrack.deserialize
        case 'trackfeature':
            return TrackFeature.deserialize
        case 'jsonevent':
            return JsonEventImpl.deserialize
        case _:
            raise ValueError('unknown event-type: {event_type_str}')


def find_event_serializer(event_type_str:str) -> Callable[[Any],bytes]:
    event_type_str = event_type_str.replace('_', '').replace('-','').lower()
    match event_type_str:
        case 'nodetrack':
            return NodeTrack.serialize
        case 'globaltrack':
            return GlobalTrack.serialize
        case 'trackfeature':
            return TrackFeature.serialize
        case 'jsonevent':
            return JsonEventImpl.serialize
        case _:
            raise ValueError('unknown event-type: {event_type_str}')