import numpy as np
import json, pickle, copy
from scipy.spatial.transform import Rotation
from pyproj import Transformer
import matplotlib.colors as mcolors

def save_camera_config(json_file, cameras, keys_to_save=['name', 'focal', 'center', 'distort', 'rvec', 'tvec']):
    '''Save the multi-camera configuration as a JSON file'''
    with open(json_file, 'w') as f:
        cameras_to_save = []
        for cam in cameras:
            cam_to_save = {}
            for key in keys_to_save:
                if key in cam:
                    if type(cam[key]) == np.ndarray:
                        cam_to_save[key] = cam[key].tolist()
                    else:
                        cam_to_save[key] = cam[key]
            cameras_to_save.append(cam_to_save)
        json.dump(cameras_to_save, f, indent=4)

def postprocess_camera_config(cameras):
    '''Post-process the multi-camera configuration'''
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
            with open(cam['cylinder_file'], 'rb') as f:
                cam['cylinder_table'] = pickle.load(f)
        if 'cuboid_file' in cam:
            with open(cam['cuboid_file'], 'rb') as f:
                cam['cuboid_table'] = pickle.load(f)

def load_camera_config(json_file, cameras=None):
    '''Load the multi-camera configuration from a JSON file'''
    with open(json_file, 'r') as f:
        cameras_from_file = json.load(f)
        if cameras is None:
            cameras = cameras_from_file
        else:
            for (src, dst) in zip(cameras, cameras_from_file):
                src.update(dst)
        postprocess_camera_config(cameras)
        return cameras

def postprocess_satellite_config(satellite):
    '''Post-process the satellite configuration'''
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

def load_satellite_config(json_file, satellite=None):
    '''Load the satellite configuration from a JSON file'''
    with open(json_file, 'r') as f:
        satellite_from_file = json.load(f)
        if satellite is None:
            satellite = satellite_from_file
        else:
            satellite.update(satellite_from_file)

        postprocess_satellite_config(satellite)
        return satellite

def get_default_config():
    config = {
        'detector_name'     : 'YOLOv5',
        'detector_option'   : {},
        'tracker_name'      : 'DeepSORT',
        'tracker_option'    : {},
        'tracker_margin'    : 1.2,
        'filter_classes'    : [0, 2],
        'filter_min_conf'   : 0.5,
        'filter_rois'       : [],
        'filter_max_dist'   : 50.0,
        'multicam_name'     : 'Simple',
        'multicam_option'   : {},
        'zoom_level'        : 1.0,
        'frame_offset'      : (10, 10),
        'frame_color'       : (0, 255, 0),
        'frame_font_scale'  : 0.7,
        'label_offset'      : (-8, -24),
        'label_font_scale'  : 0.5,
        'circle_radius'     : 8,
        'bbox_thickness'    : 3,
        'bbox_skip_color'   : (127, 127, 127)
    }
    return config

def postprocess_config(config):
    if 'filter_rois' in config:
        if (len(config['filter_rois']) > 0) and (type(config['filter_rois']) is not dict):
            config['filter_rois'] = {idx: np.array(polygon).astype(np.float32).reshape(-1, 2) for idx, polygon in enumerate(config['filter_rois'])}

def load_config(json_file):
    '''Load the satellite and multi-camera configuration together from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
        if ('satellite' in config) and ('cameras' in config) and ('config' in config):
            postprocess_satellite_config(config['satellite'])
            postprocess_camera_config(config['cameras'])
            default_cfg = get_default_config()
            default_cfg.update(config['config'])
            config['config'] = default_cfg
            postprocess_config(config['config'])
            for cam in config['cameras']:
                # Copy empty options from the global options
                cam_cfg = copy.deepcopy(config['config'])
                cam_cfg.update(cam['config'])
                cam['config'] = cam_cfg
                postprocess_config(cam['config'])
            return config['satellite'], config['cameras'], config['config']
    return {}, [], {}

def load_3d_points(csv_file, trans_code='', origin_id=-1):
    '''Load 3D points (e.g. road markers) from a CSV file'''
    # Read the CSV file
    id_pts = np.loadtxt(csv_file, delimiter=',')
    pts = {int(id): np.array(pt) for id, *pt in id_pts}

    # Transform the given data to the specific coordinate
    if trans_code:
        transformer = Transformer.from_crs('EPSG:4326', trans_code)
        for id, (lon, lat, alt) in pts.items():
            y, x = transformer.transform(lat, lon)
            pts[id] = np.array([x, y, float(alt)])

    # Assign the origin using the given index
    if origin_id >= 0:
        origin = pts[origin_id]
        for id, pt in pts.items():
            pts[id] = pt - origin
    return pts

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

def load_3d_points_from_satellite(json_file, origin_id=-1):
    '''Load 3D points (e.g. road markers) from 2D points defined on the satellite image'''
    satellite = load_satellite_config(json_file)
    if ('id_pts' in satellite) and ('meter_per_pixel' in satellite):
        # Copy points from the given 'satellite'
        pts = {}
        for id, u, v in satellite['id_pts']:
            pts[int(id)] = satellite['meter_per_pixel'] * np.array([u, -v, 0])

        # Assign the origin using the given index
        if origin_id >= 0:
            origin = pts[origin_id]
            for id, pt in pts.items():
                pts[id] = pt - origin
        return pts

def get_marker_palette(int_type=False, bgr=False):
    '''Load the pre-defined palette for consistent coloring'''
    # Use 'TABLEAU_COLORS' palette by default
    palette = [mcolors.ColorConverter.to_rgb(rgb) for rgb in mcolors.TABLEAU_COLORS.values()]
    palette[7] = (0., 0., 0.) # Make gray to black for better visibility
    if int_type:
        palette = [(int(255* r), int(255* g), int(255* b)) for r, g, b in palette]
    if bgr:
        palette = [(b, g, r) for r, g, b in palette]
    return palette



if __name__ == '__main__':
    # Test 'load_3d_points()'
    markers3d = load_3d_points('data/ETRITestbed/markers45_QGIS+MMS.csv', trans_code='EPSG:5186', origin_id=23)

    # Test 'load_3d_points_satellit()'
    markers3d_sate = load_3d_points_from_satellite('data/ETRITestbed/markers45_satellite.json', origin_id=23)
