from __future__ import annotations

from typing import NewType, Protocol, Optional, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from dna import Size2d


Image = NewType('Image', np.ndarray)

@dataclass(frozen=True, eq=True, slots=True)
class Frame:
    """Frame captured by ImageCapture.
    a Frame object consists of image (OpenCv format), frame index, and timestamp.
    """
    image: Image = field(repr=False, compare=False, hash=False)
    index: int
    ts: float
    
    def __repr__(self) -> str:
        h, w, d = self.image.shape
        return f'{self.__class__.__name__}[image={w}x{h}, index={self.index}, ts={self.ts}]'

  
@runtime_checkable
class ImageGrabber(Protocol):
    def grab_image(self) -> Optional[Image]: ...
    

class Camera(ABC):
    @property
    @abstractmethod
    def uri(self) -> str:
        raise NotImplementedError("Camera.uri")
    
    @property
    @abstractmethod
    def image_size(self) -> Size2d:
        raise NotImplementedError("Camera.image_size")
    
    @property
    @abstractmethod
    def fps(self) -> int:
        raise NotImplementedError("Camera.fps")

    @abstractmethod
    def open(self) -> ImageCapture:
        raise NotImplementedError("Camera.open")


# extends (Iterator, Closeable)
class ImageCapture(ABC):
    @abstractmethod
    def close(self) -> None:
        """Closes this ImageCapture.
        """
        raise NotImplementedError("ImageCapture.close")
    
    @abstractmethod
    def camera(self) -> Camera:
        raise NotImplementedError("ImageCapture.camera")
    
    def __iter__(self) -> ImageCapture:
        return self

    @abstractmethod
    def __next__(self) -> Frame:
        """Captures an OpenCV image frame.
        If it fails to capture an image, this method returns None.

        Returns:
            Frame: captured image frame.
        """
        raise NotImplementedError("ImageCapture.__next__")
    
    @property
    @abstractmethod
    def initial_ts(self) -> int:
        raise NotImplementedError("ImageCapture.initial_ts")
        
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
    def image_size(self) -> Size2d:
        """Returns the size of the images.

        Returns:
            Size2d: (width, height)
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