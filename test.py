
from typing import Optional
import cv2
from omegaconf import OmegaConf

from dna import Size2d, camera

conf = OmegaConf.load("conf/etri_testbed/etri_01.yaml")
print(type(conf))
x = OmegaConf.create()
pass

# cam = camera.load_camera("output/test.mp4", sync=True, end_frame=51)
# result = camera.process_images(cam, show=True, title='frame+fps', output_video='output/output.mp4')
# print(result)