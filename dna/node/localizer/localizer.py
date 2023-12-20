from __future__ import annotations

from typing import Optional
from collections import namedtuple

import numpy as np
import numpy.typing as npt
import cv2

from dna import Point
from .config_common import load_config, conv_meter2pixel, get_marker_palette
from .object_localize import get_uncertainty


_BASE_EPSG = 'EPSG:5186'
CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes,cylinder_table,cuboid_table')

    
class Localizer:
    def __init__(self, config_file:str,
                 *,
                 camera_index:int) -> None:
        self.satellite, cameras, _ = load_config(config_file)
        camera_params = cameras[camera_index]
        self.geometry = CameraGeometry(camera_params['K'], camera_params['distort'],
                                        camera_params['ori'], camera_params['pos'],
                                        camera_params['polygons'], self.satellite['planes'],
                                        camera_params['cylinder_table'], camera_params['cuboid_table'])

        # utm_origin 값 설정
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', _BASE_EPSG)
        self.origin_utm = transformer.transform(*(self.satellite['origin_lonlat'][::-1]))[::-1]
    
    def from_camera_coord(self, pt:npt.ArrayLike) -> tuple[np.ndarray,np.double]:
        pt_m, dist = self.localize_point(np.array(pt))
        return (pt_m[:2], dist) if pt_m is not None else (None, dist) 
            
    def from_camera_box(self, tlbr:npt.ArrayLike) -> tuple[np.ndarray,np.double]:
        # pt = self.select_contact_point(tlbr)
        pt = get_bbox_bottom_mid(tlbr)
        return self.from_camera_coord(pt)
    
    def from_image_coord(self, pt_m:npt.ArrayLike) -> np.ndarray:
        return conv_pixel2meter(np.array(pt_m), self.satellite['origin_pixel'], self.satellite['meter_per_pixel'])
    
    def from_world_coord(self, pt_5186:npt.ArrayLike) -> np.ndarray:
        pt_5186 = np.array(pt_5186)
        return pt_5186[0:2] - self.origin_utm
        
    def to_world_coord(self, pt_m:npt.ArrayLike) -> np.ndarray:
        pt_m = np.array(pt_m)
        pt_5186 = pt_m[0:2] + self.origin_utm
        pt_world = pt_5186 if self.transformer is None else self.transformer.transform(*pt_5186[::-1])[::-1]
        return pt_world

    def to_image_coord(self, pt_m:npt.ArrayLike) -> np.ndarray:
        return conv_meter2pixel(pt_m, self.satellite['origin_pixel'], self.satellite['meter_per_pixel'])
    
    def localize_point(self, pt:np.ndarray) -> tuple[Optional[np.ndarray], np.double]:
        '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
        # Make a ray aligned to the world coordinate
        pt = pt.xy if isinstance(pt, Point) else pt
        pt_n = cv2.undistortPoints(np.array(pt, dtype=self.geometry.K.dtype), self.geometry.K, self.geometry.distort).flatten()
        r = self.geometry.ori @ np.append(pt_n, 1) # A ray with respect to the world coordinate
        scale = np.linalg.norm(r)
        r = r / scale

        # Get a plane if 'pt' exists inside of any 'polygons'
        n, d = np.array([0, 0, 1]), 0
        plane_idx = check_polygons(pt, self.geometry.polygons)
        if (plane_idx >= 0) and (plane_idx < len(self.geometry.planes)):
            n, d = self.geometry.planes[plane_idx][0:3], self.geometry.planes[plane_idx][-1]

        # Calculate distance and position on the plane
        denom = n.T @ r
        if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
            return None, None
        distance = -(n.T @ self.geometry.pos + d) / denom
        r_c = self.geometry.ori.T @ (np.sign(distance) * r)
        if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
            return None, None
        position = self.geometry.pos + distance * r
        return position, np.fabs(distance)


def get_bbox_bottom_mid(bbox):
    '''Get the bottom middle point of the given bounding box'''
    tl_x, tl_y, br_x, br_y = bbox
    return np.array([(tl_x + br_x) / 2, br_y])

def check_polygons(pt, polygons):
    '''Check whether the given point belongs to polygons (index) or not (-1)'''
    if len(polygons) > 0:
        for idx, polygon in polygons.items():
            if cv2.pointPolygonTest(polygon, np.array(pt, dtype=np.float32), False) >= 0:
                return idx
    return -1

def conv_pixel2meter(pt, origin_pixel, meter_per_pixel):
    '''Convert image position to metric position on the satellite image'''
    x = (pt[0] - origin_pixel[0]) * meter_per_pixel
    y = (origin_pixel[1] - pt[1]) * meter_per_pixel
    z = 0
    if len(pt) > 2:
        z = pt[2]
    if type(pt) is np.ndarray:
        return np.array([x, y, z])
    return [x, y, z]

def conv_meter2pixel(pt, origin_pixel, meter_per_pixel):
    '''Convert metric position to image position on the satellite image'''
    u = pt[0] / meter_per_pixel + origin_pixel[0]
    v = origin_pixel[1] - pt[1] / meter_per_pixel
    if type(pt) is np.ndarray:
        return np.array([u, v])
    return [u, v]
