from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from contextlib import suppress

from .types import Image, Frame, ImageCapture
from .ts_generator import TimestampGenerator


class SyncableImageCapture(ImageCapture):
    __slots__ =  ( '__frame_index', '__ts_gen', 'init_ts_expr', '__closed')

    def __init__(self, init_ts_expr:str, init_frame_index:int) -> None:
        self.__frame_index = init_frame_index-1
        self.init_ts_expr = init_ts_expr
        self.__ts_gen:Optional[TimestampGenerator] = None
            
    @abstractmethod
    def close_in_guard(self) -> None: pass

    def close(self) -> None:
        if not self.__closed:
            with suppress(Exception): self.close_in_guard()
            self.__closed = False
        
    def is_closed(self) -> bool:
        return self.__closed

    def __next__(self) -> Frame:
        image = self.grab_image()
        if image is None:
            raise StopIteration()
        
        if self.__ts_gen is None:
            self.__ts_gen = TimestampGenerator.parse(self.init_ts_expr, fps=self.fps, sync=self.sync)

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
    @abstractmethod
    def sync(self) -> bool:
        pass

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    @property
    def initial_ts(self) -> int:
        if self.__ts_gen:
            return self.__ts_gen.initial_ts
        else:
            raise ValueError(f"not initialized")