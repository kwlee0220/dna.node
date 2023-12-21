from __future__ import annotations
from typing import Optional

import cv2

from dna import Size2d, Image
from .camera import Frame, Camera, ImageCapture


class ResizingCamera(Camera):
    def __init__(self, camera: Camera, target_size: Size2d):
        self.base_camera = camera
        self.__size = target_size

    @property
    def uri(self) -> str:
        return self.base_camera.uri

    def open(self) -> ImageCapture:
        src_capture = self.base_camera.open()
        return ResizingImageCapture(self, src_capture) if src_capture.size != self.__size else src_capture

    @property
    def size(self) -> Size2d:
        return self.__size

    def __repr__(self) -> str:
        end_frame = self.end_frame if self.end_frame else ""
        return f"Range({self.base_camera}, {self.begin_frame}:{end_frame})"


class ResizingImageCapture(ImageCapture):
    def __init__(self, camera:ResizingCamera, capture:ImageCapture) -> None:
        self.__cap = capture
        self.__size = camera.size

        src_size = capture.size
        if self.__size == src_size:
            raise ValueError(f"target size({self.__size}) is equal to the source size({src_size})")
        elif self.__size.area() < src_size.area():
            self.interpolation = cv2.INTER_AREA
        elif self.__size.area() > src_size.area():
            self.interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError(f"invalid target size: {self.__size}")

    def close(self) -> None:
        self.__cap.close()

    def is_open(self) -> bool:
        return self.__cap.is_open()

    def __call__(self) -> Optional[Image]:
        frame:Frame = self.__cap()
        if frame:
            mat = cv2.resize(frame.image, dsize=tuple(self.__size.wh), interpolation=self.interpolation)
            frame = Frame(image=mat, index=frame.index, ts=frame.ts)
        return frame

    @property
    def camera(self) -> Camera:
        return self.__cap.camera

    @property
    def size(self) -> Size2d:
        return self.__size

    @property
    def fps(self) -> int:
        return self.__cap.fps

    @property
    def frame_index(self) -> int:
        return self.__cap.frame_index

    @property
    def sync(self) -> bool:
        return self.__cap.sync

    @sync.setter
    def sync(self, flag) -> None:
        return self.__cap.sync(flag)

    @property
    def repr_str(self) -> str:
        return f'{self.__cap.repr_str}, target_size={self.__size}'

    def __repr__(self) -> str:
        return f'{self.__camera}({self.repr_str})'