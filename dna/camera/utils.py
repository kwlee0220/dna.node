from __future__ import annotations
from abc import abstractmethod
from typing import Optional

from contextlib import ExitStack, closing, contextmanager

from dna import Image
from .types import Frame, ImageCapture
        

@contextmanager
def multi_camera_context(camera_list):
    with ExitStack() as stack:
        yield [stack.enter_context(closing(camera.open())) for camera in camera_list]


class SyncImageCapture(ImageCapture):
    __slots__ =  ( '_frame_index', '__fps', '__ts_sync', )

    def __init__(self, fps:int, ts_sync_expr:str, sync:bool=False) -> None:
        from .time_synchronizer import TimeSynchronizer
        self.__ts_sync = TimeSynchronizer.parse(ts_sync_expr, sync=sync)
        self.__ts_sync.start(fps)

        self._frame_index = 0
        self.__fps = fps

    def __call__(self) -> Optional[Frame]:
        if not self.is_open():
            raise IOError(f"{self.__class__.__name__}: not opened")

        image = self.capture_a_image()
        if image is None: return None

        ts = self.__ts_sync.wait(self._frame_index)
        self._frame_index += 1

        return Frame(image=Image(image), index=self._frame_index, ts=ts)

    @abstractmethod
    def capture_a_image(self) -> Optional[Image]:
        """Grab an image frame from a camera.
        If it fails to capture an image, this method returns None.

        Returns:
            Image: captured image (OpenCv format).
        """
        pass

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def sync(self) -> bool:
        return self.__ts_sync.sync