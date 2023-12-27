# from __future__ import annotations
from typing import Optional, Iterable, Any
from abc import ABCMeta, abstractmethod
# from enum import Enum

import cv2
import numpy as np
import re

from dna import Point, color, Box, Size2d
from dna.camera import Image
from dna.support import plot_utils
from dna.event import NodeTrack
from dna.assoc import GlobalTrack
from .running_stabilizer import RunningStabilizer
from .world_coord_localizer import ContactPointType, WorldCoordinateLocalizer



_WINDOW_TITLE = 'trajectory'
_DEFAULT_CAPTURE_FILE = 'output.png'


class Locator(metaclass=ABCMeta):
    @abstractmethod
    def location(self, sample:Any) -> Point: pass
    
    @property
    @abstractmethod
    def mode(self) -> int: pass
    
    @mode.setter
    @abstractmethod
    def mode(self, idx:int) -> None: pass
    
    @abstractmethod
    def mode_count(self) -> int: pass
    
    @abstractmethod
    def background_image(self) -> Image: pass


class BoxLocator(Locator):
    __slot__ = ('localizer', 'bg_images', 'mode_idx')
    
    def __init__(self, localizer:Optional[WorldCoordinateLocalizer],
                 camera_image:Image, world_map_image:Optional[Image]):
        self.localizer = localizer
        self.mode_idx = 0
        self.bg_images = [camera_image, world_map_image] if world_map_image is not None else [camera_image]
        
    def location(self, sample:Box) -> Point:
        if self.mode == 0:
            return sample.center()
        elif self.mode == 1:
            pt_m, _ = self.localizer.from_camera_box(sample)
            pt_img = self.localizer.to_image_coord(pt_m)
            return pt_img
        else:
            raise ValueError(f'invalid mode: {self.mode}')
        
    @property
    def mode(self) -> int:
        return self.mode_idx
       
    @mode.setter 
    def mode(self, idx:int) -> None:
        if self.localizer:
            if idx >= 0 and idx <= 1:
                self.mode_idx = idx
            else:
                raise ValueError(f'invalid mode: {self.mode}')
        elif idx != 0:
            raise ValueError(f'invalid mode: {idx}')
                
    def mode_count(self) -> int:
        return 2
    
    def background_image(self) -> Image:
        return self.bg_images[self.mode]


class GlobalPointLocator(Locator):
    __slot__ = ('bg_image', 'mode_idx')
    
    def __init__(self, localizer:Optional[WorldCoordinateLocalizer], bg_image:Image):
        self.localizer = localizer
        self.bg_image = bg_image
        
    def location(self, sample:Point) -> Point:
        pt_m = self.localizer.from_world_coord(sample)
        return self.localizer.to_image_coord(pt_m)
        
    @property
    def mode(self) -> int:
        return 0
        
    @mode.setter 
    def mode(self, idx:int) -> None:
        if idx != 0:
            raise ValueError(f'invalid mode: {idx}')
    
    def mode_count(self) -> int:
        return 1
    
    def background_image(self) -> Image:
        return self.bg_image


class LocalPointLocator(Locator):
    __slot__ = ('localizer', 'bg_images', 'mode_idx')
    
    def __init__(self, localizer:Optional[WorldCoordinateLocalizer],
                 camera_image:Image, world_map_image:Optional[Image]):
        self.localizer = localizer
        self.mode_idx = 0
        self.bg_images = [camera_image, world_map_image] if world_map_image is not None else [camera_image]
        
    def location(self, sample:Point) -> Point:
        if self.mode_idx == 0:
            return sample
        elif self.mode_idx == 1:
            pt_m, _ = self.localizer.from_camera_coord(sample)
            return self.localizer.to_image_coord(pt_m)
        else:
            raise ValueError(f'invalid mode: {self.mode}')
        
    @property
    def mode(self) -> int:
        return self.mode_idx
       
    @mode.setter 
    def mode(self, idx:int) -> None:
        if self.localizer:
            if idx >= 0 and idx <= 1:
                self.mode_idx = idx
            else:
                raise ValueError(f'invalid mode: {self.mode}')
        elif idx != 0:
            raise ValueError(f'invalid mode: {idx}')
    
    def mode_count(self) -> int:
        return 2
    
    def background_image(self) -> Image:
        return self.bg_images[self.mode]


def sort_traj_ids(traj_ids:Iterable[str]) -> list[str]:
    p = re.compile('(.+)\[(.+)\]')
    def to_tuple(traj_id:str) -> tuple[str,int]:
        m = p.search(str(traj_id))
        return m.group(1), int(m.group(2))
    return [f'{node_id}[{track_id}]' for node_id, track_id in sorted(to_tuple(trj_id) for trj_id in traj_ids)]
    

class TrajectoryDrawer:
    def __init__(self,
                 locator:BoxLocator,
                 trajs: dict[str,list[Any]],
                 *,
                 stabilizer:Optional[RunningStabilizer]=None,
                 traj_color:color=color.RED,
                 line_thickness:int=2,
                 capture_file:str=_DEFAULT_CAPTURE_FILE) -> None:
        self.locator = locator
        self.trajs = trajs
        self.stabilizer = stabilizer
        self.traj_color = traj_color
        self.line_thickness = line_thickness
        self.show_stabilized = False
        self.capture_file = capture_file
        self.mode = 0
        self.trj_ids = sort_traj_ids(trajs.keys())

    def run(self):
        try:
            idx = 0
            while True:
                trk_id = self.trj_ids[idx]
                convas = self.locator.background_image().copy()
                convas = self._put_text(convas, trk_id)
                convas = self.draw_trajectory(convas, self.trajs[trk_id])
                cv2.imshow(_WINDOW_TITLE, convas)
                
                cont = True
                while cont:
                    key = cv2.waitKey(1) & 0xFF
                    cont, idx = self.handle_control(key, idx, convas)
        except StopIteration: pass
        finally:
            cv2.destroyWindow(_WINDOW_TITLE)
    
    def handle_control(self, ctrl:int, idx:int, convas:Image) -> tuple[bool,int]:
        if ctrl == ord('q'):
            raise StopIteration('end')
        elif ctrl == ord('n'):
            idx = (idx + 1) % len(self.trj_ids)
            return False, idx
        elif ctrl == ord('p'):
            idx = (idx + len(self.trj_ids) - 1) % len(self.trj_ids)
            return False, idx
        elif ctrl == ord('t'):
            self.show_stabilized = not self.show_stabilized
            return False, idx
        elif ctrl == ord('s'):
            cv2.imwrite(self.capture_file, convas)
        elif ctrl == ord('w'):
            self.locator.mode = (self.locator.mode + 1) % self.locator.mode_count()
            return False, idx
            
        return True, idx

    def draw_trajectory(self, convas:Image, samples:list[Any]) -> Image:
        pts = [self.locator.location(pt) for pt in samples]
        if self.show_stabilized:
            pts = self.stabilize(pts)

        pts = np.rint(np.array(pts)).astype('int32')
        return cv2.polylines(convas, [pts], False, self.traj_color, self.line_thickness)

    def stabilize(self, traj:list[Point]) -> list[Point]:
        pts_s = []
        for pt in traj:
            pt_s = self.stabilizer.transform(pt)
            if pt_s is not None:
                pts_s.append(pt_s)
        pts_s.extend(self.stabilizer.get_tail())
        self.stabilizer.reset()
        return pts_s
        
    def draw_to_file(self, outfile:str) -> None:
        convas = self.draw(pause=False)
        cv2.imwrite(outfile, convas)

    def _put_text(self, convas:Image, trj_id:str):
        stabilized_flag = f', stabilized({self.stabilizer.smoothing_factor})' if self.show_stabilized else ''
        
        return plot_utils.draw_label2(convas, f'id={trj_id}{stabilized_flag}',
                                      (10, 20),
                                      font_face=cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale=0.65,
                                      color=color.RED,
                                      fill_color=color.WHITE,
                                      bg_margin=Size2d((0, 4)))