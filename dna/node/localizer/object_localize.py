import numpy as np
import cv2 as cv
from object_track import check_polygons

def localize_point(pt, K, distort=None, ori=np.eye(3), pos=np.zeros((3, 1)), polygons={}, planes=[]):
    '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
    # Make a ray aligned to the world coordinate
    pt_n = cv.undistortPoints(np.array(pt, dtype=K.dtype), K, distort).flatten()
    r = ori @ np.append(pt_n, 1) # A ray with respect to the world coordinate
    scale = np.linalg.norm(r)
    r = r / scale

    # Get a plane if 'pt' exists inside of any 'polygons'
    n, d = np.array([0, 0, 1]), 0
    plane_idx = check_polygons(pt, polygons)
    if (plane_idx >= 0) and (plane_idx < len(planes)):
        n, d = planes[plane_idx][0:3], planes[plane_idx][-1]

    # Calculate distance and position on the plane
    denom = n.T @ r
    if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
        return None, None
    distance = -(n.T @ pos + d) / denom
    r_c = ori.T @ (np.sign(distance) * r)
    if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
        return None, None
    position = pos + distance * r
    return position, np.fabs(distance)

def get_uncertainty(pt, sigma, K, distort=None, ori=np.eye(3), pos=np.zeros((3, 1)), polygons={}, planes=[], n=32):
    '''Get an uncertainty ellipse on the given point (unit: [pixel]) with the standard deviation (unit: [pixel])'''
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    circle_p = pt + sigma * np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
    circle_m = [localize_point(p, K, distort, ori, pos, polygons, planes)[0] for p in circle_p]
    return circle_p, np.array(circle_m)

def get_bbox_bottom_mid(bbox):
    '''Get the bottom middle point of the given bounding box'''
    tl_x, tl_y, br_x, br_y = bbox
    return np.array([(tl_x + br_x) / 2, br_y])

def test_localize(image_file, config_file, camera_index=0, show_uncertainty=True):
    '''Test point localization'''

    import numpy as np
    import cv2 as cv
    import opencx as cx
    from config_common import load_config, conv_meter2pixel, get_marker_palette
    from object_localize import get_uncertainty

    # A callback function to save the clicked point
    def click_camera_image(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0] = x
            param[1] = y

    # Configure parameters
    sigma_px = 8
    sigma_color = (255, 0, 0)
    cursor_width = 10
    curosr_color = (0, 0, 255)
    label_offset = (-5, 15)
    label_color = (0, 0, 255)

    # Load configuration and images
    satellite, cameras, _ = load_config(config_file)
    cam = cameras[camera_index]
    topview = cv.imread(satellite['file'])
    camview = cv.imread(image_file)
    if ('polygons' in cam) and (len(cam['polygons']) > 0):
        palette = get_marker_palette(int_type=True, bgr=True)
        for idx, pts in cam['polygons'].items():
            cv.polylines(camview, [pts.astype(np.int32)], True, palette[idx % len(palette)], 2)

    # Get a point and show it on two images
    click_curr, click_prev = np.array([0, 0]), np.array([0, 0])
    cv.imshow('test_localize: Satellite View', topview)
    cv.imshow('test_localize: Camera View', camview)
    cv.setMouseCallback('test_localize: Camera View', click_camera_image, click_curr)
    while True:
        if not np.array_equal(click_curr, click_prev):
            click_prev = click_curr.copy()

            # Show the point and uncertainty on two images
            pt_m, dist_m = localize_point(click_curr, cam['K'], cam['distort'], cam['ori'], cam['pos'], cam['polygons'], satellite['planes'])
            if pt_m is not None:
                topview_viz = topview.copy()
                camview_viz = camview.copy()

                # Draw uncertainty
                if show_uncertainty:
                    uncertain_p, uncertain_m = get_uncertainty(click_curr, sigma_px, cam['K'], cam['distort'], cam['ori'], cam['pos'], cam['polygons'], satellite['planes'])
                    if uncertain_m.ndim == 2: # If all elements are not None ('dtype' is not 'object')
                        uncertain_m2p = [conv_meter2pixel(p, satellite['origin_pixel'], satellite['meter_per_pixel']) for p in uncertain_m]
                        uncertain_m2p = np.array(uncertain_m2p)
                        cv.polylines(topview_viz, [uncertain_m2p.astype(np.int32)], True, sigma_color, 2)
                        cv.polylines(camview_viz, [uncertain_p.astype(np.int32)], True, sigma_color, 2)

                # Draw 'click_curr' and 'pt_m' as a cross mark
                pt_m2p = conv_meter2pixel(pt_m, satellite['origin_pixel'], satellite['meter_per_pixel']).astype('int32')
                cv.line(topview_viz, pt_m2p-[cursor_width, 0], pt_m2p+[cursor_width, 0], curosr_color, 2)
                cv.line(topview_viz, pt_m2p-[0, cursor_width], pt_m2p+[0, cursor_width], curosr_color, 2)
                cv.line(camview_viz, click_curr-[cursor_width, 0], click_curr+[cursor_width, 0], curosr_color, 2)
                cv.line(camview_viz, click_curr-[0, cursor_width], click_curr+[0, cursor_width], curosr_color, 2)

                # Draw 'pt_m', 'pt_u', and 'pt_l' as text
                label = f'XYZ: ({pt_m[0]:.3f}, {pt_m[1]:.3f}, {pt_m[2]:.3f}), Dist: {dist_m:.3f}'
                cx.putText(topview_viz, label, pt_m2p+label_offset, color=label_color)
                cx.putText(camview_viz, label, click_curr+label_offset, color=label_color)

                cv.imshow('test_localize: Satellite View', topview_viz)
                cv.imshow('test_localize: Camera View', camview_viz)
            else:
                print('* Warning) The clicked point is out of the reference plane.')

        key = cv.waitKey(1)
        if key == 27: # ESC
            break

    cv.destroyAllWindows()



if __name__ == '__main__':
    # test_localize('camera_1.png',  'config_etri_testbed_2023.json', camera_index=0)
    # test_localize('camera_2.png',  'config_etri_testbed_2023.json', camera_index=1)
    # test_localize('camera_3.png',  'config_etri_testbed_2023.json', camera_index=2)
    # test_localize('camera_4.png',  'config_etri_testbed_2023.json', camera_index=3)
    # test_localize('camera_5.png',  'config_etri_testbed_2023.json', camera_index=4)
    test_localize('camera_6.png',  'config_etri_testbed_2023.json', camera_index=5)
    # test_localize('camera_7.png',  'config_etri_testbed_2023.json', camera_index=6)
    # test_localize('camera_8.png',  'config_etri_testbed_2023.json', camera_index=7)
    # test_localize('camera_9.png',  'config_etri_testbed_2023.json', camera_index=8)
    # test_localize('camera_10.png', 'config_etri_testbed_2023.json', camera_index=9)