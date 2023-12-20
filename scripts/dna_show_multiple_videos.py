
from pathlib import Path

import argparse
import cv2
import numpy as np

from dna import initialize_logger, Size2d, color, Frame, Image, Box, Point
from dna.support import plot_utils
from dna.camera import Camera, ImageCapture, create_camera
from dna.camera.opencv_video_writer import OpenCvVideoWriter
from dna.camera.utils import multi_camera_context


def define_args(parser):
    parser.add_argument("video_uris", nargs='+', help="video uris to display")
    parser.add_argument("--begin_frames", metavar="csv", help="first frame indexes")
    parser.add_argument("--start", default=0, type=int, help="start frame index")
    parser.add_argument("--output_video", metavar="path", help="output video file path")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


class MultipleCameraConvas:
    def __init__(self, captures:list[ImageCapture], *, output_video:str=None) -> None:
        self.captures = captures

        size:Size2d = captures[0].size
        self.convas:Image = np.zeros((size.height*2, size.width*2, 3), np.uint8)
        self.blank_image:Image = np.zeros((size.height, size.width, 3), np.uint8)
        self.last_frames:list[Frame] = [None] * len(self.captures)
        self.offset:int = 0

        roi:Box = Box.from_size(size)
        self.rois = [roi,
                     roi.translate(Size2d([size.width, 0])),
                     roi.translate(Size2d([0, size.height])),
                     roi.translate(size)]
        
        if output_video:
            self.writer = OpenCvVideoWriter(Path(output_video).resolve(), 10, Size2d.from_image(self.convas))
            self.writer.open()
        else:
            self.writer = None
        
    def close(self) -> None:
        if self.writer:
            self.writer.close()

    @property
    def size(self) -> int:
        return len(self.captures)
    
    def show(self, title:str) -> None:
        cv2.imshow(title, self.convas)
        if self.writer:
            self.writer.write(self.convas)

    def reset_offset(self) -> None:
        self.offset = min(frame.index for frame in self.last_frames)
        for idx in range(self.size):
            self.draw(idx)

    def update(self, idx:int) -> bool:
        self.last_frames[idx] = self.captures[idx]()
        return self.draw(idx)

    def update_all(self) -> list[bool]:
        return [self.update(idx) for idx in range(len(self.captures))]
        
    def draw(self, idx:int) -> bool:
        frame = self.last_frames[idx]
        roi = self.rois[idx]
        if frame is not None:
            image = self.draw_frame_index(idx, frame.image.copy())
            roi.update_roi(self.convas, image)
            return True
        else:
            roi.update_roi(self.convas, self.blank_image)
            return False

    def draw_frame_index(self, idx:int, convas: Image) -> Image:
        index = self.last_frames[idx].index - self.offset
        msg = f'{idx}: frames={index}'
        # return cv2.putText(convas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.RED, 2)
        return plot_utils.draw_label2(convas, label=msg, pos=Point((10,20)), font_face=cv2.FONT_HERSHEY_SIMPLEX,
                                      color=color.RED, fill_color=color.WHITE, line_thickness=1, font_scale=0.7,
                                      bg_margin=Size2d((0,3)))


def loop_in_still_images(display:MultipleCameraConvas, title:str) -> int:
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord(' '):
            return 1
        elif key == ord('q'):
            return key
        elif key - ord('0') < display.size:
            camera_idx = key - ord('0')
            display.update(camera_idx)
            display.show(title)
        elif key == ord('n'):
            if not any(display.update_all()):
                return ord('q')
            display.show(title)
        elif key == ord('r'):
            display.reset_offset()
            display.show(title)


def run(args):
    initialize_logger(args.logger)

    if args.begin_frames is not None:
        begin_frames = [int(vstr) for vstr in args.begin_frames.split(',')]
    else:
        begin_frames = [0] * len(args.video_uris)
    offset = args.start - min(begin_frames)
    begin_frames = [idx+offset for idx in begin_frames]

    size:Size2d = None
    camera_list:list[Camera] = []
    for idx, uri in enumerate(args.video_uris):
        camera = create_camera(uri, begin_frame=begin_frames[idx])
        if idx == 0:
            size = (camera.size * 0.7).to_rint()
        camera_list.append(camera.resize(size))

    with multi_camera_context(camera_list) as caps:
        display:MultipleCameraConvas = MultipleCameraConvas(caps, output_video=args.output_video)
        while any(display.update_all()):
            display.show("multiple cameras")
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                key = loop_in_still_images(display, "multiple cameras")
                
            if key == ord('q'):
                break
            elif key == ord('r'):
                display.reset_offset()
        display.close()
        cv2.destroyWindow("multiple cameras")
    

def main():
    parser = argparse.ArgumentParser(description="Show multiple videos")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()