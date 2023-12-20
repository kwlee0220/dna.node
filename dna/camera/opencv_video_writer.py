from __future__ import annotations

from typing import Optional
from contextlib import suppress

import logging
from pathlib import Path

import cv2

from dna import Image, Size2d
from .types import Frame, VideoWriter
from .image_processor import FrameProcessor, ImageProcessor


class OpenCvVideoWriter(VideoWriter):
    __slots__ = ('fourcc', '_path', '_fps', '_image_size', '_video_writer')
    FOURCC_MP4V = 'mp4v'
    FOURCC_XVID = 'XVID'
    FOURCC_DIVX = 'DIVX'
    FOURCC_WMV1 = 'WMV1'
    
    def __init__(self, video_file:str, fps:int, image_size:Size2d) -> None:
        path = Path(video_file)

        self.fourcc = None
        ext = path.suffix.lower()
        if ext == '.mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*OpenCvVideoWriter.FOURCC_MP4V)
        elif ext == '.avi':
            self.fourcc = cv2.VideoWriter_fourcc(*OpenCvVideoWriter.FOURCC_DIVX)
        elif ext == '.wmv':
            self.fourcc = cv2.VideoWriter_fourcc(*OpenCvVideoWriter.FOURCC_WMV1)
        else:
            raise IOError("unknown output video file extension: 'f{ext}'")
        self._path = path.resolve()
        
        self._fps = fps
        self._image_size = image_size
        self._path.parent.mkdir(exist_ok=True)
        self._video_writer = cv2.VideoWriter(str(self._path), self.fourcc, self._fps,
                                             tuple(self._image_size.to_rint()))
        
    def close(self) -> None:
        if self.is_open():
            self._video_writer.release()
            self._video_writer = None
        
    def is_open(self) -> bool:
        return self._video_writer is not None
        
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
        self._video_writer.write(image)
        


class OpenCvWriteProcessor(FrameProcessor):
    __slots__ = ( 'path', 'logger', '_writer' )
    
    def __init__(self, path: Path,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        self.path = path.resolve()
        self.logger = logger
        self._writer = None
        
    def on_started(self, img_proc:ImageProcessor) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'opening video file: {self.path}')
        self._writer = OpenCvVideoWriter(self.path.resolve(), img_proc.capture.fps, img_proc.capture.size)

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