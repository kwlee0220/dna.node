from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import json

import numpy as np

from dna import Box, Point, ByteString, NodeId, TrackId, TrackletId
from dna.track import TrackState
from dna.support import sql_utils
from .types import KafkaEvent, Timestamped
# from dna.node.zone.types import ZoneExpression
from .proto.reid_feature_pb2 import NodeTrackProto, LocalizationProto


_WGS84_PRECISION = 7
_DIST_PRECISION = 3

        
def to_loc_bytes(loc:Point, dist:float) -> LocalizationProto:
    proto = LocalizationProto()
    proto.location_x = loc.x
    proto.location_y = loc.y
    proto.distance = dist
    return proto.SerializeToString()


def from_loc_bytes(binary_data:bytes) -> tuple[Point,float]:
    proto = LocalizationProto()
    proto.ParseFromString(binary_data)
    return Point((proto.location_x, proto.location_y)), proto.distance


@dataclass(frozen=True, eq=True, order=False, repr=False, slots=True)   # slots=True
class NodeTrack:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    state: TrackState   # tracking state
    bbox: Box = field(hash=False)
    first_ts: int = field(hash=False)
    frame_index: int
    ts: int = field(hash=False)
    location: Optional[Point] = field(default=None, repr=False, hash=False)
    distance: Optional[float] = field(default=None, repr=False, hash=False)
    zone_expr: Optional[ZoneExpression]  = field(default=None)
    detection_box: Optional[Box] = field(default=None)  # local-only

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def __lt__(self, other) -> bool:
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index == other.frame_index:
            return self.track_id < other.luid
        else:
            return False

    @staticmethod
    def from_row(row:tuple[str,str,TrackState,Box,Point,float,str,int,int]) -> NodeTrack:
        from dna.node.zone import ZoneExpression
        return NodeTrack(node_id=row[1],
                            track_id=row[2],
                            state=TrackState.from_abbr(row[3]),
                            bbox=sql_utils.from_sql_box(row[4]),
                            location=sql_utils.from_sql_point(row[5]),
                            distance=row[6],
                            zone_expr=ZoneExpression.parse_str(row[7]),
                            frame_index=row[8],
                            ts=row[9])

    def to_row(self) -> tuple[str,str,str,str,str,float,str,int,int]:
        return (self.node_id, self.track_id, self.state.abbr,
                sql_utils.to_sql_box(self.bbox.to_rint()),
                sql_utils.to_sql_point(self.location),
                self.distance, str(self.zone_expr),
                self.frame_index, self.ts)

    @staticmethod
    def from_json(json_str:str) -> NodeTrack:
        from dna.node.zone import ZoneExpression
        def json_to_box(tlbr_list:Optional[Iterable[float]]) -> Box:
            return Box(tlbr_list) if tlbr_list else None

        json_obj = json.loads(json_str)

        location = json_obj.get('location', None)
        if location is not None:
            location = Point(location)
        distance = json_obj.get('distance', None)
        zone_expr = ZoneExpression.parse_str(json_obj.get('zone_expr', None))
        # detection_box = json_to_box(json_obj.get('detection_box', None))

        return NodeTrack(node_id=json_obj['node'],
                            track_id=json_obj['track_id'],
                            state=TrackState[json_obj['state']],
                            bbox=json_to_box(json_obj['bbox']),
                            location=location,
                            distance=distance,
                            zone_expr=zone_expr,
                            first_ts = json_obj['first_ts'],
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'])

    def to_json(self) -> str:
        def box_to_json(box:Box) -> list[float]:
            return [round(v, 2) for v in box.tlbr.tolist()] if box else None

        serialized = {
            'id': f'{self.node_id}[{self.track_id}]',
            'node': self.node_id,
            'track_id': self.track_id,
            'state': self.state.name,
            'bbox': box_to_json(self.bbox)
        }
        if self.location is not None:
            serialized['location'] = [round(v, _WGS84_PRECISION) for v in tuple(self.location.xy)]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)
        if self.zone_expr:
            serialized['zone_expr'] = str(self.zone_expr)
        serialized['frame_index'] = self.frame_index
        serialized['ts'] = self.ts
        serialized['first_ts'] = self.first_ts

        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> str:
        return self.to_json().encode('utf-8')

    @staticmethod
    def deserialize(serialized:ByteString) -> NodeTrack:
        return NodeTrack.from_json(serialized.decode('utf-8'))

    def updated(self, **kwargs:object) -> NodeTrack:
        fields = asdict(self)
        for key, value in kwargs.items():
            fields[key] = value
        return NodeTrack(**fields)

    def to_csv(self) -> str:
        vlist = [self.node_id, self.track_id, self.state.name] \
                + self.bbox.tlbr.tolist() \
                + [self.frame_index, self.ts]
        if self.location is not None:
            vlist += np.round(self.location.xy, _WGS84_PRECISION).tolist() + [round(self.distance, _DIST_PRECISION)]
        else:
            vlist += ['', '']

        return ','.join([str(v) for v in vlist])

    @staticmethod
    def from_csv(csv: str) -> NodeTrack:
        parts = csv.split(',')

        node_id = parts[0]
        track_id = parts[1]
        state = TrackState[parts[2]]
        loc = Box([float(s) for s in parts[3:7]])
        frame_idx = int(parts[7])
        first_ts = int(parts[8])
        ts = int(parts[9])
        xy_str = parts[10:12]
        if len(xy_str[0]) > 0:
            location = Point(np.array([float(s) for s in xy_str]))
            dist = float(parts[12])
        else:
            location = None
            dist = None

        return NodeTrack(node_id=node_id, track_id=track_id, state=state, bbox=loc, first_ts=first_ts,
                            frame_index=frame_idx, ts=ts, location=location, distance=dist)
        

    def to_bytes(self) -> bytes:
        proto = NodeTrackProto()
        proto.node_id = self.node_id
        proto.track_id = self.track_id
        proto.state = self.state.abbr
        if self.tlbr is not None:
            proto.bbox_tlbr.extend(self.bbox.tlbr.tolist())
        proto.first_ts = self.first_ts
        proto.frame_index = self.frame_index
        proto.ts = self.ts
        if self.location is not None:
            proto.localization = to_loc_bytes(self.location, self.distance)
        if self.zone_expr is not None:
            proto.zone_expr = self.zone_expr

        return proto.SerializeToString()

    @staticmethod
    def from_bytes(binary_data:bytes) -> NodeTrack:
        proto = NodeTrackProto()
        proto.ParseFromString(binary_data)
        
        bbox = Box(np.array(proto.bbox_tlbr, dtype=np.float32)) if len(proto.bbox_tlbr) > 0 else None
        location, dist = from_loc_bytes(proto.localization) if len(proto.localization) else (None, None)
        
        return NodeTrack(node_id=proto.node_id, track_id=proto.track_id, state=proto.state,
                         bbox=bbox, first_ts=proto.first_ts, frame_index=proto.frame_index, ts=proto.ts,
                         location=location, distance=dist, zone_expr=proto.zone_expr)

    def __repr__(self) -> str:
        age = (self.ts - self.first_ts) / 1000
        return (f"NodeTrack[id={self.node_id}[{self.track_id}]({self.state.abbr}), loc={self.location}, "
                f"bbox={self.bbox}, zone={self.zone_expr}, frame={self.frame_index}, ts={self.ts}({age:.1f})]")