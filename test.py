
import cv2

from dna import Size2d, camera

cam = camera.load_camera("output/test.mp4", sync=True, end_frame=51)
result = camera.process_images(cam, show=True, title='frame+fps', output_video='output/output.mp4')
print(result)