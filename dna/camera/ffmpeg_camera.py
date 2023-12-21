from __future__ import annotations

from typing import Optional

import ffmpeg
import numpy as np
import cv2

from dna import Size2d, Image
from .types import Camera
from .utils import SyncableImageCapture
from dna.support import iterables


def eval_ratio(ratio_str:str) -> float:
    num, denom = ratio_str.split('/')
    return eval(num) / eval(denom)
    

class FFMPEGCamera(Camera):
    __slots__ = '_uri', 'cap_info', '_size', '_fps', '_pipeline', '__init_ts_expr'
    
    def __init__(self, uri:str, **options:object):
        super().__init__()

        self._uri = uri
        probe = ffmpeg.probe(uri)
        _, self.cap_info = iterables.find_first(probe['streams'], lambda s: s['codec_name'] == 'h264')
        self._size = Size2d((self.cap_info['width'], self.cap_info['height']))
        self._fps = int(eval_ratio(self.cap_info['r_frame_rate']))
        self._pipeline = (
            ffmpeg.input(self._uri, rtsp_transport = 'tcp')
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                    .overwrite_output()
        )
        
        self.__init_ts_expr = options.get('init_ts', 'open')

    def open(self) -> FFMPEGCameraCapture:
        return FFMPEGCameraCapture(self, self.cap_info, self.pipeline, init_ts_expr=self._init_ts_expr)

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def size(self) -> Size2d:
        return self._size

    @property
    def fps(self) -> int:
        return self._fps
        
    @property
    def init_ts_expr(self) -> str:
        return self.__init_ts_expr

    @property
    def pipeline(self):
        return self._pipeline
        

class FFMPEGCameraCapture(SyncableImageCapture):
    __slots__ = ( '__camera', '__process', '__image_bytes' )

    def __init__(self, camera:FFMPEGCamera, cap_info, pipeline) -> None:
        super().__init__(fps=camera.fps, sync=camera.sync, init_ts_expr=camera.init_ts_expr)
        
        self.__camera = camera
        self.__process = pipeline.run_async(pipe_stdout=True)
        self._image_size = camera.image_size
        self.__image_bytes = camera.image_size.area() * 3

    def close(self) -> None:
        if self.__process:
            self.__process.stdout.close()
            self.__process.wait()
            self.__process = None

    def is_open(self) -> bool:
        return self.__process is not None

    @property
    def camera(self) -> FFMPEGCamera:
        return self.__camera

    @property
    def image_size(self) -> Size2d:
        return self.__camera.image_size
            
    def __iter__(self) -> FFMPEGCameraCapture:
        return self

    @property
    def repr_str(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return f'{state}, size={self.size}, fps={self.fps:.0f}/s'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.repr_str})'
    
    def grab_image(self) -> Optional[Image]:
        image_bytes = self.__process.stdout.read(self.__image_bytes)
        if not image_bytes:
            return None
        
        in_frame = (np.frombuffer(image_bytes, np.uint8)
                        .reshape([self.size.height, self.size.width, 3]))
        return cv2.resize(in_frame, tuple(self.camera.size))