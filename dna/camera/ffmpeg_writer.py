from __future__ import annotations
from typing import Optional

from enum import Enum
from pathlib import Path
from contextlib import suppress
import logging
import ffmpeg

from dna import Size2di
from dna.camera import Image, Frame, VideoWriter, CRF
from .image_processor import FrameReader, ImageProcessor

                        
class FFMPEGWriter(VideoWriter):
    __slots__ = ( '__path', '__fps', '__image_size', 'process' )
    
    def __init__(self, video_file:str, fps:int, size:Size2di,
                 *,
                 crf:CRF=CRF.FFMPEG) -> None:
        super().__init__()
        
        path = Path(video_file).resolve()
        path.parent.mkdir(exist_ok=True)
        self.__path = str(path)
        self.__fps = fps
        self.__image_size = size
        self.process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{size.width}x{size.height}', r=fps)
                    # .output(video_file, f='mp4', vcodec='mpeg4')
                    .output(self.__path, f='mp4', vcodec='libx264', pix_fmt='yuv420p', crf=crf)
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
    def path(self) -> str:
        return self.__path
        
    @property
    def fps(self) -> int:
        return self.__fps
        
    @property
    def image_size(self) -> Size2di:
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
        
    def open(self, img_proc:ImageProcessor) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'opening video file: {self.path}')
        cap = img_proc.capture
        self.__writer = FFMPEGWriter(cap.camera.path, cap.fps, cap.image_size, **self.__options)

    def close(self) -> None:
        if self.__writer is not None:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f'closing video file: {self.path}')
            with suppress(Exception):
                self.__writer.close()
                self.__writer = None

    def read(self, frame:Frame) -> None:
        if self.__writer is None:
            raise ValueError(f'OpenCvWriteProcessor has not been started')
        self.__writer.write(frame.image)