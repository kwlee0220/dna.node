
from typing import Tuple
from contextlib import closing

import cv2
import numpy as np

from dna import Box
from dna.camera import create_camera
from dna.utils import RectangleDrawer

img = None
camera = create_camera("data/etri/etri_053.mp4")
with closing(camera.open()) as cap:
    img = cap().image

image, box = RectangleDrawer(img).run()
print(box.top_left(), box.bottom_right())

cv2.destroyAllWindows()    