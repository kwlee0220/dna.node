from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from .types import Image, Frame, ImageCapture
from .ts_generator import TimestampGenerator


class SyncableImageCapture(ImageCapture):
    __slots__ =  ( '__frame_index', '__fps', '__ts_gen', )

    def __init__(self, fps:int, sync:bool, init_ts_expr:str, init_frame_index:int) -> None:
        self.__ts_gen = TimestampGenerator.parse(init_ts_expr, fps=fps, sync=sync)

        self.__frame_index = init_frame_index-1
        self.__fps = fps

    def __next__(self) -> Frame:
        image = self.grab_image()
        if image is None:
            raise StopIteration()

        ts = self.__ts_gen.generate(self.__frame_index)
        self.__frame_index += 1

        return Frame(image=image, index=self.__frame_index, ts=ts)

    @abstractmethod
    def grab_image(self) -> Optional[Image]:
        """Grab an image frame from a camera.
        If it fails to capture an image, this method returns None.

        Returns:
            Image: captured image (OpenCv format).
        """
        pass

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def sync(self) -> bool:
        return self.__ts_gen.sync

    @property
    def initial_ts(self) -> int:
        return self.__ts_gen.initial_ts