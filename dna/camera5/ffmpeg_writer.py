from __future__ import annotations
from typing import Optional

from pathlib import Path
from contextlib import suppress
import ffmpeg
import logging

from dna import Image, Size2d
from .types import Frame, VideoWriter
from .image_processor import FrameProcessor, ImageProcessor


CRF_VISUALLY_LOSSLESS = 17
CRF_FFMPEG_DEFAULT = 23

class FFMPEGWriter(VideoWriter):
    __slots__ = ( '_path', '_fps', '_image_size', 'process' )
    
    def __init__(self, video_file:str, fps:int, size:Size2d,
                 *,
                 crf:int=CRF_FFMPEG_DEFAULT) -> None:
        super().__init__()
        
        self._path = Path(video_file).resolve()
        self._path.parent.mkdir(exist_ok=True)
        self._fps = fps
        self._image_size = size
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
        return self._path
        
    @property
    def fps(self) -> int:
        return self._fps
        
    @property
    def image_size(self) -> Size2d:
        return self._image_size

    def write(self, image:Image) -> None:
        assert self.is_open(), "not opened."
        self.process.stdin.write(image.tobytes())


class FFMPEGWriteProcessor(FrameProcessor):
    __slots__ = ( 'path', '_options', 'logger', '_writer' )
    
    def __init__(self, path:Path, **options) -> None:
        self.path = path.resolve()
        self.logger = options.get('logger')
        self._options = options
        self._options.pop('logger', None)
        self._writer = None
        
    def on_started(self, img_proc:ImageProcessor) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'opening video file: {self.path}')
        self._writer = FFMPEGWriter(self.path.resolve(), img_proc.capture.fps, img_proc.capture.size, **self._options)

    def on_stopped(self) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'closing video file: {self.path}')
        with suppress(Exception):
            self._writer.close()
            self._writer = None

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if self._writer is None:
            raise ValueError(f'OpenCvWriteProcessor has not been started')
        self._writer.write(frame.image)
        return frame