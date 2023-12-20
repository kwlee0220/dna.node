from __future__ import annotations

from typing import Union, Optional, Any
import dataclasses
import time

from tqdm import tqdm
import numpy as np
import cv2
from omegaconf import OmegaConf

import dna
from dna import Size2d, Image
from .types import Frame, Camera
from .utils import SyncImageCapture
from .camera import is_local_camera, is_video_file


class OpenCvCamera(Camera):
    def __init__(self, uri:str, **options:object):
        super().__init__()

        self._uri = uri
        self._size = Size2d.from_expr(options.get('size'))
        self._target_size = self._size
        self.__init_ts_expr = options.get('init_ts', 'open')
        
    def open(self) -> OpenCvImageCapture:
        """Open this camera and set ready for captures.

        Returns:
            ImageCapture: a image capturing session from this camera.
        """
        uri = int(self.uri) if is_local_camera(self.uri) else self.uri
        vid = self._open_video_capture(uri)

        from_video_file = is_video_file(self.uri)
        if from_video_file:
            return VideoFileCapture(self, vid, init_ts_expr=self.__init_ts_expr)
        else:
            return OpenCvImageCapture(self, vid, init_ts_expr=self.__init_ts_expr)

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def size(self) -> Size2d:
        if self._target_size is not None:
            return self._target_size
        
        if self._size is None:
            # if the image size has not been set yet, open this camera and find out the image size.
            vid = self._open_video_capture(self._uri)
            try:
                width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._size = Size2d([width, height])
            finally:
                vid.release()
            
        return self._size
    
    def _open_video_capture(self, uri:Union[str,int]) -> cv2.VideoCapture:
        vid = cv2.VideoCapture(uri)
        if self._target_size is not None:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_size.width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_size.height)
            
        return vid

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str})"


class OpenCvVideFile(OpenCvCamera):
    def __init__(self, uri:str, **options):
        super().__init__(uri, **options)

        self.sync = options.get('sync', True)
        self.begin_frame = options.get('begin_frame', 1)
        self.end_frame = options.get('end_frame')

    def open(self) -> VideoFileCapture:
        import os
        if not os.path.exists(self.uri):
            raise IOError(f"invalid video file path: {self.uri}")
        
        return super().open()

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync})"


class OpenCvImageCapture(SyncImageCapture):
    __slots__ = '__camera', '_vid', '__size'

    def __init__(self, camera:OpenCvCamera, vid:cv2.VideoCapture,
                 *,
                 sync:bool=False,
                 init_ts_expr:str='open',
                 init_frame_index:int=0) -> None:
        """Create a OpenCvImageCapture object.

        Args:
            uri (str): Resource identifier.
        """
        self.__camera = camera
        if vid is None:
            raise ValueError(f'cv2.VideoCapture is invalid')
        self._vid:cv2.VideoCapture = vid            # None if closed
        
        super().__init__(fps = self._vid.get(cv2.CAP_PROP_FPS), sync=sync, ts_sync_expr=init_ts_expr)

        width = int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__size = Size2d([width, height])

    def close(self) -> None:
        if self._vid:
            self._vid.release()
            self._vid = None
            
    def capture_a_image(self) -> Optional[Image]:
        return self._vid.read()[1]

    def is_open(self) -> bool:
        return self._vid is not None

    @property
    def camera(self) -> OpenCvCamera:
        return self.__camera

    @property
    def cv2_video_capture(self) -> cv2.VideoCapture:
        return self._vid

    @property
    def size(self) -> Size2d:
        return self.__size

    @property
    def repr_str(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return f'{state}, size={self.size}, fps={self.fps:.0f}/s, sync={self.sync}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.repr_str})'


class VideoFileCapture(OpenCvImageCapture):
    __slots__ = ('__end_frame_index', )

    def __init__(self, camera:OpenCvVideFile, vid:cv2.VideoCapture,
                 *,
                 init_ts_expr:Optional[str]='open') -> None:
        super().__init__(camera, vid, sync=camera.sync, init_ts_expr=init_ts_expr)
        
        if camera.begin_frame <= 0 and camera.begin_frame > self.total_frame_count:
            raise ValueError(f'index({camera.begin_frame}) should be between 1 and {self.total_frame_count}')
        
        self._frame_index = camera.begin_frame - 1
        if self._frame_index > 0:
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)
            
        if camera.end_frame:
            self.__end_frame_index = camera.end_frame
        else:
            self.__end_frame_index = int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 100

    def __call__(self) -> Optional[Frame]:
        # 지정된 마지막 프레임 번호보다 큰 경우는 image capture를 종료시킨다.
        if (self._frame_index+1) >= self.__end_frame_index:
            return None
        return super().__call__()

    @property
    def total_frame_count(self) -> int:
        return int(self.cv2_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def repr_str(self) -> str:
        return f'{super().repr_str}, sync={self.sync}'
    
    
class RTSPOpenCvCamera(OpenCvCamera):
    def __init__(self, uri:str, **options):
        super().__init__(uri, **options)
        
        # self.use_rtsp_relay = False
        # if uri.find("&end=") >= 0 or uri.find("start=") >= 0:
        #     if not ffmpeg_utils.is_initialize():
        #         ffmpeg_utils.init(ffmpeg_path=options.get('ffmpeg_path'))
        #     self.use_rtsp_relay = True
        
    def open(self) -> OpenCvImageCapture:
        # if self.use_rtsp_relay:
        #     proc, local_rtsp_url = ffmpeg_utils.relay_to_local_stream(self.uri, id=uuid.uuid1())
            
        #     from contextlib import suppress
        #     with suppress(TimeoutExpired):
        #         ret = proc.wait(5)
        #         raise ValueError(f"fails to start RTSP relay server: ret-code={ret}")

        #     while True:
        #         vcap = self._open_video_capture(local_rtsp_url)
        #         ret, _ = vcap.read()
        #         if ret:
        #             return RTSPOpenCvImageCapture(self, vcap, proc)
        #         else:
        #             with suppress(TimeoutExpired):
        #                 ret = proc.wait(1)
        #                 raise ValueError(f"fails to start RTSP relay server: ret-code={ret}")
        # else:
        #     return super().open()
        return super().open()

    def __repr__(self) -> str:
        size_str = f', size={self._size}' if self._size is not None else ''
        ffmpeg_str = f', ffmpeg={self.ffmpeg_path}' if self.ffmpeg_path else ""
        return f"{__class__.__name__}(uri={self.uri}{size_str}, sync={self.sync}{ffmpeg_str})"
    

from subprocess import Popen
class RTSPOpenCvImageCapture(OpenCvImageCapture):
    __slots__ = ( '_proc', )

    def __init__(self, camera:RTSPOpenCvCamera, vid:cv2.VideoCapture, proc:Popen) -> None:
        super().__init__(camera, vid)
        
        self._proc = proc
        
    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
        super().close()

    def __repr__(self) -> str:
        return f'RTSPCapture({self.repr_str})'