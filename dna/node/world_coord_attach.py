
from __future__ import annotations
from typing import Union, Optional
from collections import namedtuple

import logging

from omegaconf import OmegaConf

from dna import Point
from dna.event import NodeTrack, TimeElapsed, EventProcessor
from .world_coord_localizer import WorldCoordinateLocalizer


# CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes,cylinder_table,cuboid_table')
CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes')
class WorldCoordinateAttacher(EventProcessor):
    def __init__(self, conf:OmegaConf, *, logger:logging.Logger) -> None:
        super().__init__()
        

        from .world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
        camera_index = conf.get('camera_index', 0)
        epsg_code = conf.get('epsg_code', 'EPSG:5186')
        self.contact_point = conf.get('contact_point', ContactPointType.BottomCenter.name)
        self.contact_point = ContactPointType[self.contact_point]
        self.localizer = WorldCoordinateLocalizer(conf.camera_geometry, camera_index, epsg_code,
                                                  contact_point=self.contact_point,
                                                  logger=logger)
        
        self.logger = logger
        self.logger.info((f'created: WorldCoordinateAttacher: '
                          f'config_file={conf.camera_geometry}, '
                          f'camera_index={camera_index}, '
                          f'epsg_code={epsg_code}, '
                          f'contact_point={self.contact_point}'))

    def handle_event(self, ev:Union[NodeTrack, TimeElapsed]) -> None:
        if isinstance(ev, NodeTrack):
            pt_m, dist = self.localizer.from_camera_box(ev.bbox.tlbr)
            world_coord = self.localizer.to_world_coord(pt_m) if pt_m is not None else None
            if world_coord is not None:
                world_coord = Point(world_coord)
            updated = ev.updated(location=world_coord, distance=dist)
            self._publish_event(updated)
        else:
            self._publish_event(ev)
    
    def __repr__(self) -> str:
        return f"AttachWorldCoordinate(contact={self.contact_point})"