from __future__ import annotations

from typing import Optional
from enum import Enum
from collections import namedtuple
import json, pickle

from omegaconf import OmegaConf
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from pathlib import Path
import cv2

from dna import Point
from dna.node.localizer.config_common import load_config

import logging

class ContactPointType(Enum):
    Centroid = 0
    BottomCenter = 1
    Simulation = 2
    NearBottom = 3


_BASE_EPSG = 'EPSG:5186'
CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos,polygons,planes')
class WorldCoordinateLocalizer:
    def __init__(self, config_file:str, camera_index:int, epsg_code:str=_BASE_EPSG,
                 *,
                 contact_point:ContactPointType=ContactPointType.Simulation,
                 logger:Optional[logging.Logger]=None) -> None:
        try:
            self.satellite, cameras, _ = load_config(config_file)
            camera_params = cameras[camera_index]
            self.geometry = CameraGeometry(camera_params['K'], camera_params['distort'],
                                            camera_params['ori'], camera_params['pos'],
                                            camera_params['polygons'], self.satellite['planes'])
        except Exception as e:
            if logger :
                logger.error(f'fails to load localization config: config_file={config_file}')
            raise

        # utm_origin 값 설정
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', _BASE_EPSG)
        self.origin_utm = transformer.transform(*(self.satellite['origin_lonlat'][::-1]))[::-1]

        # 위경도 좌표계 생성을 위한 좌표계 변환 객체 생성
        self.transformer = None
        if epsg_code != _BASE_EPSG:
            self.transformer = Transformer.from_crs(_BASE_EPSG, epsg_code)
        self.contact_point_type = contact_point
        
    def from_camera_coord(self, pt:npt.ArrayLike) -> tuple[np.ndarray,np.double]:
        pt_m, dist = self.localize_point(np.array(pt))
        return (pt_m[:2], dist) if pt_m is not None else (None, dist)

    def from_camera_box(self, tlbr:npt.ArrayLike) -> tuple[np.ndarray,np.double]:
        pt = self.select_contact_point(np.array(tlbr))
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
        
    def localize_point(self, pt) -> tuple[Optional[np.ndarray], np.double]:
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

    def select_contact_point(self, tlbr:np.ndarray) -> np.ndarray:
        '''Get the bottom middle point of the given bounding box'''
        tl_x, tl_y, br_x, br_y = tlbr
        
        mean_x = (tl_x + br_x) / 2
        if self.contact_point_type == ContactPointType.Centroid:
            return np.array([mean_x, (tl_y + br_y) / 2])
        elif self.contact_point_type == ContactPointType.NearBottom:
            delta = (br_y - tl_y) * 0.1
            return np.array([mean_x, br_y - delta])
        
        pt_bc = np.array([mean_x, br_y])
        if self.contact_point_type == ContactPointType.BottomCenter:
            return pt_bc
        elif self.contact_point_type == ContactPointType.Simulation:
            delta = predict_center_from_table(pt_bc, self.geometry.cuboid_table)
            return pt_bc + delta
        else:
            raise ValueError(f"unknown contact-point type={self.contact_point_type}")

def load_config_file(json_file:str):
    '''Load the satellite and multi-camera configuration together from a JSON file'''
    conf_dir = Path(json_file).parent
    with open(json_file, 'r') as f:
        config = json.load(f)
        if ('satellite' in config) and ('cameras' in config):
            conv_satellite_config(config['satellite'])
            conv_camera_config(conf_dir, config['cameras'])
            return config['satellite'], config['cameras']

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

def conv_camera_config(conf_dir:Path, cameras):
    '''Pre-process the multi-camera configuration'''
    for cam in cameras:
        for key in ['K', 'distort', 'rvec', 'tvec', 'ori', 'pos']:
            if key in cam:
                cam[key] = np.array(cam[key])
        if ('focal' in cam) and ('center' in cam):
            cam['K'] = np.array([[cam['focal'][0], 0, cam['center'][0]], [0, cam['focal'][1], cam['center'][1]], [0, 0, 1]])
        if ('rvec' in cam) and ('tvec' in cam):
            cam['ori'] = Rotation.from_rotvec(cam['rvec']).as_matrix().T
            cam['pos'] = -cam['ori'] @ cam['tvec']
        if 'polygons' in cam:
            cam['polygons'] = {int(key): np.array(value).reshape(-1, 2) for key, value in cam['polygons'].items()}
        else:
            cam['polygons'] = {}
        if 'cylinder_file' in cam:
            with open(conf_dir / cam['cylinder_file'], 'rb') as f:
                cam['cylinder_table'] = pickle.load(f)
        if 'cuboid_file' in cam:
            with open(conf_dir / cam['cuboid_file'], 'rb') as f:
                cam['cuboid_table'] = pickle.load(f)

def conv_satellite_config(satellite):
    '''Pre-process the satellite configuration'''
    for key in ['pts', 'planes']:
        if key in satellite:
            satellite[key] = np.array(satellite[key])
    if 'planes' not in satellite:
        satellite['planes'] = []
    if 'roads' in satellite:
        satellite['roads'] = [np.array(road).reshape(-1, 2) for road in satellite['roads']]
        roads_data = []
        for road in satellite['roads']:
            road_m = np.array([conv_pixel2meter(pt, satellite['origin_pixel'], satellite['meter_per_pixel']) for pt in road])
            road_v = road_m[1:] - road_m[:-1]
            road_n = np.linalg.norm(road_v, axis=1)
            roads_data.append(np.hstack((road_m[:-1], road_v, road_n.reshape(-1, 1))))
        satellite['roads_data'] = np.vstack(roads_data)
    else:
        satellite['roads'] = []
        satellite['roads_data'] = []

def predict_center_from_table(bottom_mid, table, dist_threshold=100):
    '''Predict a foot point using the given lookup table and nearest search'''
    x, y = bottom_mid
    dist = np.fabs(table[:,0] - x) + np.fabs(table[:,1] - y)
    min_idx = np.argmin(dist)
    if dist[min_idx] < dist_threshold:
        return table[min_idx,2:4]
    return np.zeros(2)

def check_polygons(pt, polygons):
    '''Check whether the given point belongs to polygons (index) or not (-1)'''
    if len(polygons) > 0:
        for idx, polygon in polygons.items():
            if cv2.pointPolygonTest(polygon, np.array(pt, dtype=np.float32), False) >= 0:
                return idx
    return -1

if __name__ == '__main__':
    from dna import color, Box
    import scripts

    config_file = scripts.LOCALIZER_CONFIG_FILE
    camera_index = 5
    localizer = WorldCoordinateLocalizer(Path(config_file), camera_index, 'EPSG:5186',
                                          contact_point=ContactPointType.BottomCenter)
    # box = Box([323,679,715,995])
    # pt = box.center().xy
    pt = [1551.1,269.11]

    pt_m, dist = localizer.from_camera_coord(pt)
    coord = localizer.to_world_coord(pt_m)
    print(coord, dist)

    img = cv2.imread(scripts.WORLD_MAP_IMAGE_FILE, cv2.IMREAD_COLOR)
    img_coord = Point(localizer.to_image_coord(pt_m)).round()
    img = cv2.circle(img, img_coord, 5, color.RED, -1)
    while ( True ):
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow('image')