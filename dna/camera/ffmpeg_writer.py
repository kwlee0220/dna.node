from __future__ import annotations
from typing import Optional

from pathlib import Path
from contextlib import suppress
import logging
import ffmpeg

from dna import Image, Size2d
from dna.camera import Frame, VideoWriter, ImageCapture
from .image_processor import FrameReader, ImageProcessor


CRF_VISUALLY_LOSSLESS = 17
CRF_FFMPEG_DEFAULT = 23

class FFMPEGWriter(VideoWriter):
    __slots__ = ( '__path', '__fps', '__image_size', 'process' )
    
    def __init__(self, video_file:str, fps:int, size:Size2d,
                 *,
                 crf:int=CRF_FFMPEG_DEFAULT) -> None:
        super().__init__()
        
        self.__path = Path(video_file).resolve()
        self.__path.parent.mkdir(exist_ok=True)
        self.__fps = fps
        self.__image_size = size
        self.process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{size.width}x{size.height}', r=fps)
                    # .output(video_file, f='mp4', vcodec='mpeg4')
                    .output(str(video_file), f='mp4', vcodec='libx264', pix_fmt='yuv420p', crf=crf)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
        )
        
    def close(self) -> None:
        if self.is_open():
            from subprocess import TimeoutExpired
            
            self.process.stdin.close()
            try:
                self.process.wait(3)
            except TimeoutExpired:
                self.process.terminate()
        
    def is_open(self) -> bool:
        return self.process is not None
        
    @property
    def path(self) -> Path:
        return self.__path
        
    @property
    def fps(self) -> int:
        return self.__fps
        
    @property
    def image_size(self) -> Size2d:
        return self.__image_size

    def write(self, image:Image) -> None:
        assert self.is_open(), "not opened."
        self.process.stdin.write(image.tobytes())


class FFMPEGWriteProcessor(FrameReader):
    __slots__ = ( 'path', '__options', 'logger', '__writer' )
    
    def __init__(self, path:Path, **options) -> None:
        self.path = path.resolve()
        self.logger = options.get('logger')
        self.__options = options
        self.__options.pop('logger', None)
        self.__writer = None
        
    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'opening video file: {self.path}')
        self.__writer = FFMPEGWriter(self.path.resolve(), img_proc.capture.fps, img_proc.capture.size, **self.__options)

    def close(self) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'closing video file: {self.path}')
        with suppress(Exception):
            self.__writer.close()
            self.__writer = None

    def read(self, frame:Frame) -> None:
        if self.__writer is None:
            raise ValueError(f'OpenCvWriteProcessor has not been started')
        self.__writer.write(frame.image)