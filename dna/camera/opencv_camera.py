from __future__ import annotations

from typing import NewType, Iterator, Optional

import cv2

from dna import Size2d
from dna.utils import try_supply
from .types import Camera, Frame, ImageCapture, Image
from .utils import SyncableImageCapture


def _get_image_size(cap:cv2.VideoCapture) -> Size2d:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return Size2d([width, height])

def _set_image_size(cap:cv2.VideoCapture, size:Size2d) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size.height)

def _get_fps(cap:cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FPS))

def _set_fps(cap:cv2.VideoCapture, fps:int) -> None:
    cap.set(cv2.CAP_PROP_FPS, fps)

def _get_frame_count(cap:cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def _set_frame_index(cap:cv2.VideoCapture, index:int) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    

class OpenCvCamera(Camera):
    __slot__ = ( '__uri' '__image_size', '__fps', '__sync', '__init_ts_expr')
    
    def __init__(self, uri:str, **options:object):
        super().__init__()

        self.__uri = uri
        self.__image_size = options.get('image_size')
        self.__fps = options.get('fps')
        self.__sync = options.get('sync')
        self.__init_ts_expr = options.get('init_ts', 'open')
        
    def open(self) -> OpenCvImageCapture:
        return OpenCvImageCapture(self, camera=self, capture=cv2.VideoCapture(self.uri))
        
    @property
    def uri(self) -> str:
        return self.__uri
        
    @property
    def image_size(self) -> str:
        if self.__image_size is None:
            cap = cv2.VideoCapture(self.uri)
            self.__image_size = _get_image_size(cap)
            self.__fps = _get_fps(cap)
            cap.release()
        return self.__image_size
        
    @property
    def fps(self) -> int:
        if self.__fps is None:
            cap = cv2.VideoCapture(self.uri)
            self.__image_size = _get_image_size(cap)
            self.__fps = _get_fps(cap)
            cap.release()
        return self.__fps
        
    @property
    def sync(self) -> bool:
        return self.__sync
        
    @property
    def init_ts_expr(self) -> str:
        return self.__init_ts_expr


class OpenCvImageCapture(SyncableImageCapture):
    __slots__ = ( '__camera', '__capture', '__image_size' )

    def __init__(self, camera:OpenCvCamera, capture:cv2.VideoCapture, init_frame_index:int=1) -> None:
        super().__init__(fps=camera.fps, sync=camera.sync, init_ts_expr=camera.init_ts_expr, init_frame_index=init_frame_index)
        
        if capture is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        
        self.__camera = camera
        self.__capture = capture            # None if closed
        self.__image_size = _get_image_size(capture)

    def close(self) -> None:
        if self.__capture:
            self.__capture.release()
            self.__capture = None
            
    def __iter__(self) -> OpenCvImageCapture:
        return self

    def is_open(self) -> bool:
        return self.__capture is not None

    @property
    def camera(self) -> OpenCvCamera:
        return self.__camera
    
    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self.__capture

    @property
    def image_size(self) -> Size2d:
        return self.__image_size

    @property
    def repr_str(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return f'{state}, size={self.image_size}, fps={self.fps:.0f}/s, sync={self.sync}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.repr_str})'
    
    def grab_image(self) -> Optional[Image]:
        if self.__capture is None:
            return None
        return self.__capture.read()[1]


class VideoFile(OpenCvCamera):
    __slot__ = ( '__begin_frame', '__end_frame' )
    
    def __init__(self, uri:str, **options:object):
        super().__init__(uri, **options)
        
        self.__begin_frame = options.get('begin_frame', 1)
        self.__end_frame = options.get('end_frame')
        
    def open(self) -> VideoFileCapture:
        capture = cv2.VideoCapture(self.uri)
        if not capture.isOpened():
            raise ValueError(f"fails to open VideFile: {self.uri}")
        return VideoFileCapture(self, capture)
        
    @property
    def begin_frame(self) -> int:
        return self.__begin_frame
        
    @property
    def end_frame(self) -> int:
        return self.__end_frame
    

class VideoFileCapture(OpenCvImageCapture):
    def __init__(self, camera:VideoFile, capture:cv2.VideoCapture) -> None:
        super().__init__(camera, capture, init_frame_index=camera.begin_frame)
        
        if camera.begin_frame <= 0 and camera.begin_frame > self.total_frame_count:
            raise ValueError(f'index({camera.begin_frame}) should be between 1 and {self.total_frame_count}')
        
        if camera.begin_frame > 0:
            _set_frame_index(capture, camera.begin_frame-1)
            
    def __iter__(self) -> VideoFileCapture:
        return self
            
    def __next__(self) -> Frame:
        # 지정된 마지막 프레임 번호보다 큰 경우는 image capture를 종료시킨다.
        if self.camera.end_frame is not None and (self.frame_index+1) >= self.camera.end_frame:
            raise StopIteration()
        return super().__next__()

    @property
    def total_frame_count(self) -> int:
        return _get_frame_count(self.video_capture)

    @property
    def repr_str(self) -> str:
        return f'{super().repr_str}, sync={self.sync}'
