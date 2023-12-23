from __future__ import annotations

from typing import Optional
from contextlib import suppress

import logging
from pathlib import Path

import cv2

from dna import Image, Size2di
from dna.camera import Frame, ImageCapture, VideoWriter
from .image_processor import FrameReader, ImageProcessor


class OpenCvVideoWriter(VideoWriter):
    __slots__ = ('fourcc', '__path', '__fps', '__image_size', '__video_writer')
    FOURCC_MP4V = 'mp4v'
    FOURCC_XVID = 'XVID'
    FOURCC_DIVX = 'DIVX'
    FOURCC_WMV1 = 'WMV1'
    
    def __init__(self, video_file:str, fps:int, image_size:Size2di) -> None:
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
        self.__path = str(path.resolve())
        
        self.__fps = fps
        self.__image_size = image_size
        path.parent.mkdir(exist_ok=True)
        self.__video_writer = cv2.VideoWriter(self.__path, self.fourcc, self.__fps, self.__image_size.tuple())
        
    def close(self) -> None:
        if self.is_open():
            self.__video_writer.release()
            self.__video_writer = None
        
    def is_open(self) -> bool:
        return self.__video_writer is not None
        
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
        self.__video_writer.write(image)
        


class OpenCvWriteProcessor(FrameReader):
    __slots__ = ( 'path', 'logger', '__writer' )
    
    def __init__(self, path: str,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        self.path = path
        self.logger = logger
        self.__writer = None
        
    def open(self, img_proc:ImageProcessor) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'opening video file: {self.path}')
        img_sz = img_proc.show_size if img_proc.show_size else img_proc.image_size
        self.__writer = OpenCvVideoWriter(self.path, img_proc.capture.fps, img_sz)

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