from contextlib import closing

import cv2
import numpy as np
from omegaconf import OmegaConf

from dna import Box, Size2d, Point, BGR, color, camera
from dna.support import plot_utils
from dna.support.rectangle_drawer import RectangleDrawer
from dna.support.polygon_drawer import PolygonDrawer
from dna.zone import Zone

img = None

SHIFT = 50

def create_blank_image(size:Size2d, *, clr:BGR=color.WHITE) -> np.ndarray:
    from dna.color import WHITE, BLACK
    blank_img = np.zeros([size.height, size.width, 3], dtype=np.uint8)
    if color == WHITE:
        blank_img.fill(255)
    elif color == BLACK:
        pass
    else:
        blank_img[:,:,0].fill(clr[0])
        blank_img[:,:,1].fill(clr[1])
        blank_img[:,:,2].fill(clr[2])
    
    return blank_img

# camera_conf.uri = "data/2022/etri_041.mp4"
uri = "D:/Dropbox/Temp/20231109T1300-20231109T1330/videos/etri_05.mp4"
# camera_conf.uri = "output/output.mp4"
# camera_conf.uri = "data/ai_city/ai_city_t3_c01.avi"
# camera_conf.uri = "data/crossroads/crossroad_04.mp4"
# camera_conf.uri = "data/shibuya_7_8.mp4"
begin_frame = 25
cam = camera.load_camera(uri, begin_frame=begin_frame)

localizer = None
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
# localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', 0, contact_point=ContactPointType.Simulation)

track_zones = [
    [[85, 593], [674, 367], [735, 333], [725, 175], [1245, 258], [1677, 384], [1677, 557], [1420, 534], [1187, 933], [235, 876]]
]
blind_zones = [
    # [170,130,810,770]
    # [[1674, 615], [1021, 323], [1164, 274], [1743, 514]],
]
exit_zones = [
    # [[943, 363], [959, 353], [1170, 374], [1166, 388]],
    # [[1692, 208], [1482, 145], [1505, 120], [1703, 179]],
    # [[1692, 208], [1482, 145], [1505, 120], [1703, 179]],
]

zones = [
    # [[890, 372], [1179, 407], [1155, 523], [687, 446], [890, 372]],
    # [[543, 492], [1180, 603], [1103, 976], [109, 660], [543, 492]],
    # [[155, 116], [196, 119], [262, 234], [217, 265], [155, 116]],
]

with cam.open() as cap:
    src_img = cap().image
    # src_img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)
    
    box = Box.from_image(src_img)
    
    img = create_blank_image(box.expand(50).size(), clr=color.BLACK)
    roi = box.translate([SHIFT, SHIFT])
    roi.update_roi(img, src_img)
# img = cv2.imread("output/2023/etri_06_trajs.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("output/ETRI_221011.png", cv2.IMREAD_COLOR)

def shift(coords, amount=SHIFT):
    if not coords:
        return []
    elif isinstance(coords[0], list):
        return [[c[0]+amount, c[1]+amount] for c in coords]
    else:
        return [c+amount for c in coords]

for coords in track_zones:
    img = plot_utils.draw_polygon(img, shift(coords), color.GREEN, 1)
for coords in exit_zones:
    img = Zone.from_coords(shift(coords)).draw(img, color.BLUE, line_thickness=1)
for coords in blind_zones:
    img = Zone.from_coords(shift(coords)).draw(img, color.RED, line_thickness=1)
for coords in zones:
    img = Zone.from_coords(shift(coords), as_line_string=True).draw(img, color.RED, line_thickness=1)

def image_to_world(localizer:WorldCoordinateLocalizer, pt_p):
    pt_m = localizer.from_image_coord(pt_p)
    return localizer.to_world_coord(pt_m).astype(int)

polygon = []
polygon = [[85, 593], [674, 367], [735, 333], [725, 175], [1245, 258], [1677, 384], [1677, 557], [1420, 534], [1187, 933], [235, 876]]
coords = PolygonDrawer(img, shift(polygon)).run()
coords = shift(coords, -SHIFT)
if localizer:
    coords = [list(image_to_world(localizer, coord)) for coord in coords]

print(coords)

cv2.destroyAllWindows()