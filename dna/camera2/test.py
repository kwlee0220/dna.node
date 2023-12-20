
#import cv2

from dna.camera2 import cre
# from dna import Size2d
# from dna.camera2.image_processor import ImageProcessor
# from dna.camera2.show_frame import ShowFrame

cam = camera2.create_camera("output/dets.mp4", fps=20, sync=True, end_frame=101)
# proc = ImageProcessor(cam, show=Size2d((640,480)))

# proc.add_frame_reader(ShowFrame('test_window'))
# result = proc.run()
# print(result)

