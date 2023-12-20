from __future__ import annotations

from typing import Optional
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from dna import Size2d, Image


@dataclass(frozen=True, eq=True, slots=True)
class Frame:
    """Frame captured by ImageCapture.
    a Frame object consists of image (OpenCv format), frame index, and timestamp.
    """
    image: Image = field(repr=False, compare=False, hash=False)
    index: int
    ts: float
    

class Camera(metaclass=ABCMeta):
    @abstractmethod
    def open(self) -> ImageCapture:
        """Open this camera.

        Returns:
            ImageCapture: an ImageCapture object that captures images from this camera.
        """
        pass

    @property
    @abstractmethod
    def uri(self) -> str:
        """Returns the URI of this Camera.

        Returns:
            str: URI of this camera.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> Size2d:
        """The image size captured from this Camera.

        Returns:
            Size2d: the image size captured from this Camera
        """
        pass

    def resize(self, size:Size2d) -> Camera:
        """Returns a Camera that captures the resized images.

        Args:
            size (Size2d): target image size.

        Returns:
            Camera: Camera
        """
        from .resized_camera import ResizingCamera
        return ResizingCamera(self, size)


class ImageCapture(metaclass=ABCMeta):
    @abstractmethod
    def close(self) -> None:
        """Closes this ImageCapture.
        """
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Returns whether this is opened or not.

        Returns:
            bool: True if this is opened, False otherwise.
        """
        pass

    @abstractmethod
    def __call__(self) -> Optional[Frame]:
        """Captures an OpenCV image frame.
        If it fails to capture an image, this method returns None.

        Returns:
            Frame: captured image frame.
        """
        pass

    @property
    @abstractmethod
    def camera(self) -> Camera:
        """Returns source camera object that this capture is from.

        Returns:
            Camera: a camera object.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> Size2d:
        """Returns the size of the images that this ImageCapture captures.

        Returns:
            Size2d: (width, height)
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Returns the fps of this ImageCapture.

        Returns:
            int: frames per second.
        """
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        """Returns the total count of images this ImageCapture has captured so far.

        Returns:
            int: The number of frames
        """
        pass

    @property
    @abstractmethod
    def sync(self) -> bool:
        """Returns whether frames are captured on its fps or immediately as they are captured from the camera.

        Returns:
            bool: synchronized capture or not
        """
        pass

    @property
    @abstractmethod
    def repr_str(self) -> str:
        pass
    

class VideoWriter(metaclass=ABCMeta):
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