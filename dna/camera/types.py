from __future__ import annotations

from typing import TypeAlias, Any
from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass, field

import numpy as np

from ..size2di import Size2di

from cv2.typing import MatLike
Image:TypeAlias = MatLike


@dataclass(frozen=True, eq=True, slots=True)
class Frame:
    """Frame captured by ImageCapture.
    a Frame object consists of image (OpenCv format), frame index, and timestamp.
    """
    image: Image = field(repr=False, compare=False, hash=False)
    index: int
    ts: int
    
    def __repr__(self) -> str:
        h, w, _ = self.image.shape
        return f'{self.__class__.__name__}[image={w}x{h}, index={self.index}, ts={self.ts}]'
    

class CameraOptions(UserDict):
    KEYS = {'camera_uri', 'fps', 'sync', 'init_ts', 'begin_frame', 'end_frame'}
    
    def __init__(self, **options):
        super().__init__()
        
        for key, value in options.items():
            self.__setitem__(key, value)
            
    def __getitem__(self, key:str) -> Any:
        return self.data[key]
                
    def __setitem__(self, key:str, item:Any) -> None:
        match key:
            case 'camera':
                assert isinstance(item, str)
                self.data['camera_uri'] = item
            case 'fps':
                assert isinstance(item, int)
                self.data['fps'] = item
            case 'sync':
                assert isinstance(item, bool)
                self.data['sync'] = item
            case 'init_ts':
                assert isinstance(item, str|int)
                self.data['init_ts'] = item
            case 'begin_frame':
                assert isinstance(item, int)
                self.data['begin_frame'] = item
            case 'end_frame':
                assert isinstance(item, int)
                self.data['end_frame'] = item
            case _:
                self.data[key] = item
    

class Camera(ABC):
    @property
    @abstractmethod
    def uri(self) -> str:
        """Returns the identification of this camera.

        Returns:
            str: camera URI.
        """
        pass

    @abstractmethod
    def open(self) -> ImageCapture:
        """Open this camera.

        Returns:
            ImageCapture: image capture object.
        """
        pass
    
    @property
    @abstractmethod
    def image_size(self) -> Size2di:
        """Returns the size of the image captured from this camera.

        Returns:
            Size2di: image size.
        """
        pass
    
    @property
    @abstractmethod
    def fps(self) -> int:
        """Returns the fps of this camera.

        Returns:
            int: number of frames per second.
        """
        pass


class ImageCapture(ABC):
    @abstractmethod
    def close(self) -> None:
        """Closes this ImageCapture.
        """
        pass
    
    @abstractmethod
    def camera(self) -> Camera:
        pass
    
    def __iter__(self) -> ImageCapture:
        return self

    @abstractmethod
    def __next__(self) -> Frame:
        """Captures an OpenCV image frame.
        If there is no more frames to capture, this method raises StopIteration exception.

        Returns:
            Frame: captured frame.
        """
        pass

    @property
    @abstractmethod
    def image_size(self) -> Size2di:
        pass
    
    @property
    @abstractmethod
    def fps(self) -> int:
        pass
    
    @property
    @abstractmethod
    def initial_ts(self) -> int:
        pass
        
    def __enter__(self) -> ImageCapture:
        return self
        
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        from contextlib import suppress
        with suppress(Exception): self.close()
        return False
    

class VideoWriter(ABC):
    @abstractmethod
    def close(self) -> None:
        """Closes this VideoWriter.
        """
        pass
        
    @abstractmethod
    def is_open(self) -> bool:
        pass
    
    @abstractmethod
    def write(self, image:Image) -> None:
        """Write image.
        """
        pass

    @property
    @abstractmethod
    def image_size(self) -> Size2di:
        """Returns the size of the images.

        Returns:
            Size2di: (width, height)
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Returns the fps of this VideoWriter.

        Returns:
            int: frames per second.
        """
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        from contextlib import suppress
        with suppress(Exception): self.close()


from enum import Enum
class CRF(Enum):
    OPENCV = 0
    FFMPEG = 23
    LOSSLESS = 17
        
    @classmethod
    def from_name(cls, name:str) -> CRF:
        name = name.upper()
        for item in CRF:
            if item.name == name:
                return item
        raise KeyError(f"invalid CRF: {name}")
    
    @staticmethod
    def names() -> list[str]:
        return [member.name for member in CRF]
