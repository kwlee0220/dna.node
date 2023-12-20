from __future__ import annotations

from typing import Optional
from contextlib import suppress
import logging

from datetime import timedelta, datetime
import cv2

from dna import Size2d, color
from dna.camera2 import Frame, ImageCapture
from .image_processor import FrameReader, FrameUpdater, ImageProcessor


from collections import namedtuple
TitleSpec = namedtuple('TitleSpec', 'date, time, ts, frame, fps')
    

class DrawFrameTitle(FrameUpdater):
    def __init__(self, title_spec:set[str], bg_color:Optional[color.BGR]=None) -> None:
        super().__init__()
        self.title_spec = TitleSpec('date' in title_spec, 'time' in title_spec, 'ts' in title_spec,
                                    'frame' in title_spec, 'fps' in title_spec)
        self.bg_color = bg_color
        
    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None:
        self.proc = proc
        
    def close(self) -> None: pass

    def update(self, frame:Frame) -> Frame:
        ts_sec = frame.ts / 1000.0
        date_str = datetime.fromtimestamp(ts_sec).strftime('%Y-%m-%d')
        time_str = datetime.fromtimestamp(ts_sec).strftime('%H:%M:%S.%f')[:-4]
        ts_str = f'ts:{frame.ts}'
        frame_str = f'#{frame.index}'
        fps_str = f'fps:{self.proc.fps_measured:.2f}'
        str_list = [date_str, time_str, ts_str, frame_str, fps_str]
        message = ' '.join([msg for is_on, msg in zip(self.title_spec, str_list) if is_on])
        
        convas = frame.image
        if self.bg_color:
            (msg_w, msg_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            convas = cv2.rectangle(convas, (7, 3), (msg_w+11, msg_h+11), self.bg_color, -1)
        convas = cv2.putText(convas, message, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
        return Frame(image=convas, index=frame.index, ts=frame.ts)


class ShowFrame(FrameReader):
    _PAUSE_MILLIS = int(timedelta(hours=1).total_seconds() * 1000)

    def __init__(self, window_name:str,
                 *,
                 window_size:Optional[Size2d]=None,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        self.window_name = window_name
        self.window_size = window_size
        self.logger = logger

    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None:
        win_size = self.window_size if self.window_size else capture.image_size
        
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'create window: {self.window_name}, size=({win_size})')
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, win_size.width, win_size.height)

    def close(self) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'destroy window: {self.window_name}')
        with suppress(Exception): cv2.destroyWindow(self.window_name)

    def read(self, frame:Frame) -> bool:
        img = cv2.resize(frame.image, tuple(self.window_size.wh), cv2.INTER_AREA) if self.window_size else frame.image
        cv2.imshow(self.window_name, img)

        key = cv2.waitKey(int(1)) & 0xFF
        while True:
            if key == ord('q'):
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'interrupted by a key-stroke')
                raise StopIteration(f"Requested to quit")
            elif key == ord(' '):
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'paused by a key-stroke')
                while True:
                    key = cv2.waitKey(ShowFrame._PAUSE_MILLIS) & 0xFF
                    if key == ord(' '):
                        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(f'resumed by a key-stroke')
                        key = 1
                        break
                    elif key == ord('q'):
                        raise StopIteration(f"Requested to quit")
            elif key != 0xFF:
                for proc in self.processors:
                    key = proc.set_control(key)
                return
            else: 
                return