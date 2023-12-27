from __future__ import annotations

from omegaconf import OmegaConf
from shapely.geometry import LineString
import cv2

from dna import Point
from dna.color import BGR
from dna.track.track_state import TrackState
from ..event.local_path_event import LocalPathEvent
from dna.event import NodeTrack, EventNodeImpl

class Session:
    def __init__(self, node_id:str, luid:str) -> None:
        self.node_id = node_id
        self.luid = luid

        self.points:list[Point] = []
        self.location:list[Point] = []
        self.first_frame = -1
        self.last_frame = -1

    @property
    def length(self) -> int:
        return len(self.points)

    def append(self, ev: NodeTrack) -> None:
        self.points.append(ev.bbox.center())
        self.location.append(ev.location)
        if self.first_frame < 0:
            self.first_frame = ev.frame_index
        self.last_frame = ev.frame_index

    def build_local_path(self, length: int, cont: bool) -> LocalPathEvent:
        camera_path = LineString([tuple(pt.xy) for pt in self.points[:length]]).wkb_hex
        world_path = LineString([tuple(pt.xy) for pt in self.location[:length]]).wkb_hex
        self.points = self.points[length:]
        self.location = self.location[length:]

        return LocalPathEvent(node_id=self.node_id, track_id=self.luid,
                              camera_path=camera_path, world_path=world_path,
                              first_frame=self.first_frame, last_frame=self.last_frame,
                              continuation=cont)

class LocalPathGenerator(EventNodeImpl):
    MAX_PATH_LENGTH = 100

    def __init__(self, conf:OmegaConf) -> None:
        EventNodeImpl.__init__(self)

        self.max_path_length = conf.get('max_path_length', LocalPathGenerator.MAX_PATH_LENGTH)
        self.sessions = dict()

    def close(self) -> None:
        super().close()
        
        # build local paths from the unfinished sessions and upload them
        for session in self.sessions.values():
            pev = session.build_local_path(cont=False)
            self.publish_event(pev)
        self.sessions.clear()


    def handle_event(self, ev: NodeTrack) -> None:
        session = self.sessions.get(ev.track_id, None)
        if session is None:
            session = Session(ev.node_id, ev.track_id)
            self.sessions[ev.track_id] = session

        if ev.state == TrackState.Deleted:
            self.sessions.pop(ev.track_id, None)
            if session.length > 0:
                pev = session.build_local_path(length=session.length, cont=False)
                self.publish_event(pev)
        else:
            if session.length >= self.max_path_length + 10:
                pev = session.build_local_path(length=self.max_path_length, cont=True)
                self.publish_event(pev)

                # refresh the current session
                # session = Session(ev.node_id, ev.luid)
                # self.sessions[ev.luid] = session
            session.append(ev)