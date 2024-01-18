from __future__ import annotations

from typing import Optional, Any, cast

import cv2

from ..size2di import Size2di
from .types import Camera, Image, Frame, CameraOptions
from .utils import SyncableImageCapture


def _get_image_size(cap:cv2.VideoCapture) -> Size2di:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return Size2di(width=width, height=height)

def _set_image_size(cap:cv2.VideoCapture, size:Size2di) -> None:
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
    
    def __init__(self, camera_uri:str, options:CameraOptions):
        super().__init__()

        self.__uri = camera_uri
        self.__sync:bool = options.get('sync', True)
        self.__init_ts_expr:str = options.get('init_ts', 'open')
        
        self.__image_size = options.get('image_size')
        self.__fps = options.get('fps')
        
    def open(self) -> OpenCvImageCapture:
        return OpenCvImageCapture(self, capture=self._create_video_capture())
        
    @property
    def uri(self) -> str:
        return self.__uri
        
    def image_size(self) -> Optional[Size2di]:
        return self.__image_size
        
    def fps(self) -> Optional[int]:
        return self.__fps
        
    @property
    def sync(self) -> bool:
        return self.__sync
        
    @property
    def init_ts_expr(self) -> str:
        return self.__init_ts_expr
    
    def _create_video_capture(self) -> cv2.VideoCapture:
        if self.uri.isnumeric():
            return cv2.VideoCapture(int(self.uri))
        else:
            return cv2.VideoCapture(self.uri)


class OpenCvImageCapture(SyncableImageCapture):
    __slots__ = ( '__camera', '__capture', '__image_size', '__fps', '__opened' )

    def __init__(self, camera:OpenCvCamera, capture:cv2.VideoCapture, init_frame_index:int=1) -> None:
        super().__init__(init_ts_expr=camera.init_ts_expr, init_frame_index=init_frame_index)
        
        if capture is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        
        self.__camera = camera
        self.__capture = capture            # None if closed
        self.__image_size = camera.image_size()
        if self.__image_size is None:
            self.__image_size = _get_image_size(capture)
        else:
            _set_image_size(capture, self.__image_size)
        self.__fps = camera.fps()
        if self.__fps is None:
            self.__fps = _get_fps(capture)
        self.__opened = True

    def close(self) -> None:
        if self.__opened:
            self.__capture.release()
            self.__opened = False
            
    def __iter__(self) -> OpenCvImageCapture:
        return self

    def is_open(self) -> bool:
        return self.__opened

    @property
    def camera(self) -> OpenCvCamera:
        return self.__camera
    
    @property
    def video_capture(self) -> cv2.VideoCapture:
        assert self.__opened
        return self.__capture

    @property
    def image_size(self) -> Size2di:
        return self.__image_size

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def sync(self) -> bool:
        return False

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
    
    def __init__(self, uri:str, options:CameraOptions):
        super().__init__(uri, options)
        
        self.__begin_frame:int = options.get('begin_frame', 1)
        self.__end_frame:Optional[int] = options.get('end_frame')
        
    def open(self) -> VideoFileCapture:
        capture = self._create_video_capture()
        if not capture.isOpened():
            raise ValueError(f"fails to open VideFile: {self.uri}")
        return VideoFileCapture(self, capture)
        
    @property
    def begin_frame(self) -> int:
        return self.__begin_frame
        
    @property
    def end_frame(self) -> Optional[int]:
        return self.__end_frame
    

class VideoFileCapture(OpenCvImageCapture):
    def __init__(self, video_file:VideoFile, capture:cv2.VideoCapture) -> None:
        super().__init__(video_file, capture, init_frame_index=video_file.begin_frame)
        
        if video_file.begin_frame <= 0 and video_file.begin_frame > self.total_frame_count:
            raise ValueError(f'index({video_file.begin_frame}) should be between 1 and {self.total_frame_count}')
        if video_file.begin_frame > 0:
            _set_frame_index(capture, video_file.begin_frame-1)
            
        self.__end_frame = video_file.end_frame
            
    def __iter__(self) -> VideoFileCapture:
        return self
            
    def __next__(self) -> Frame:
        # 지정된 마지막 프레임 번호보다 큰 경우는 image capture를 종료시킨다.
        if self.__end_frame is not None and (self.frame_index+1) >= self.__end_frame:
            raise StopIteration()
        return super().__next__()

    @property
    def total_frame_count(self) -> int:
        return _get_frame_count(self.video_capture)

    @property
    def repr_str(self) -> str:
        return f'{super().repr_str}, sync={self.sync}'
