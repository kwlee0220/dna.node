from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import field
from contextlib import suppress
import time
from datetime import timedelta

import cv2

from dna import Frame, Camera, ImageCapture, Size2d, color, sub_logger
from dna.utils import datetime2utc, utc_now_millis

  
@runtime_checkable
class FrameReader(Protocol):
    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None: ...
    def read(self, frame:Frame) -> bool: ...
    def close(self) -> None: ...


@runtime_checkable
class FrameUpdater(Protocol):
    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None: ...
    def update(self, frame:Frame) -> Optional[Frame]: ...
    def close(self) -> None: ...


class FrameProcessor(ABC):
    @abstractmethod
    def on_started(self, proc:ImageProcessor) -> None:
        pass

    @abstractmethod
    def on_stopped(self) -> None:
        pass

    @abstractmethod
    def process_frame(self, frame:Frame) -> Optional[Frame]:
        pass

    def set_control(self, key:int) -> int:
        return key


from dataclasses import dataclass
@dataclass(slots=True)
class Result:
    elapsed: timedelta
    frame_count: int
    fps_measured: float
    failure_cause: Exception = field(default=None)


__ALPHA = 0.2
class ImageProcessor:
    def __init__(self, camera:Camera, **options) -> None:
        self.camera = camera
        self.clean_frame_readers:list[FrameReader] = []
        self.frame_updaters:list[FrameUpdater] = []
        self.suffix_frame_readers:list[FrameReader] = []
        self.fps_measured = 0.0
        
        def parse_show_option(show) -> Optional[Size2d]:
            if isinstance(show, bool):
                return camera.size if show else None
            else:
                return Size2d.from_expr(show).to_rint() if show else None
        self.show_size = parse_show_option(options['show'])

        output_video = options.get('output_video')
        title = options.get('title')
        self.is_drawing:bool = self.show_size is not None or output_video is not None
        if self.is_drawing and title:
            specs:set[str] = set(title.split('+'))
            self.suffix_frame_readers.append(DrawFrameTitle(specs, bg_color=color.WHITE))
            
        if output_video:
            write_processor = None
            crf_opt = options.get('crf', 'opencv')
            if crf_opt == 'opencv':
                from dna.camera.opencv_video_writer import OpenCvWriteProcessor
                write_processor = OpenCvWriteProcessor(output_video, logger=sub_logger(self.logger, 'image_writer'))
            else:
                from dna.camera.ffmpeg_writer import FFMPEGWriteProcessor, CRF_FFMPEG_DEFAULT, CRF_VISUALLY_LOSSLESS
                crf = CRF_FFMPEG_DEFAULT if crf_opt == 'ffmpeg' else CRF_VISUALLY_LOSSLESS
                write_processor = FFMPEGWriteProcessor(output_video, crf=crf,
                                                       logger=sub_logger(self.logger, 'image_writer'))
            self.suffix_processors.append(write_processor)

        self.show_processor:ShowFrame = None
        if self.show_size:
            # 여기서 'show_processor'를 생성만 하고, 실제 등록은
            # 'run_work()' 메소드 수행 시점에서 추가시킨다.
            self.show_processor = ShowFrame(window_name=f'camera={camera.uri}',
                                            window_size=self.show_size,
                                            logger=sub_logger(self.logger, 'show_frame'))

        if options.get('progress', False):
            # 카메라 객체에 'begin_frame' 속성과 'end_frame' 속성이 존재하는 경우에만 ShowProgress processor를 추가한다.
            self.suffix_processors.append(ShowProgress())

    def add_clean_frame_reader(self, frame_reader:FrameReader) -> None:
        self.clean_frame_readers.append(frame_reader)

    def add_frame_updater(self, frame_updater:FrameUpdater) -> None:
        self.frame_updaters.append(frame_updater)

    def add_suffix_frame_reader(self, frame_reader:FrameReader) -> None:
        self.suffix_frame_readers.append(frame_reader)
    
    def run(self) -> Result:
        with self.camera.open() as capture:
            if self.show_processor:
                self.suffix_frame_readers.append(self.show_processor)
            
            # 등록된 모든 frame 처리기를 초기화시킨다.
            for proc in [*self.clean_frame_readers, *self.frame_updaters, *self.suffix_frame_readers]:
                proc.open(self, capture)
            
            started = time.time()
            capture_count = 0
            self.fps_measured = 0.
            
            try:
                started_10th = 0
                for frame in capture:
                    capture_count += 1
                    self.__process_frame(frame)
                    
                    now = time.time()
                    if capture_count == 10:
                        started_10th = now
                    elif capture_count > 10:
                        elapsed = now - started_10th
                        self.fps_measured = 1 / (elapsed / (capture_count-10))
                    else:
                        elapsed = now - started
                        self.fps_measured = 1 / (elapsed / capture_count)
                    
                return Result(elapsed=timedelta(seconds=time.time() - started),
                              frame_count=capture_count,
                              fps_measured=self.fps_measured,
                              failure_cause=None)
            except StopIteration: pass
            finally:
                # 등록된 모든 frame 처리기를 종료화시킨다.
                for proc in [*self.clean_frame_readers, *self.frame_updaters, *self.suffix_frame_readers]:
                    proc.close()
        
    def __process_frame(self, frame:Frame) -> None:
        for reader in self.clean_frame_readers:
            reader.read(frame)
            
        for updater in self.frame_updaters:
            frame = updater.update(frame)
            
        for reader in self.suffix_frame_readers:
            reader.read(frame)


class ShowProgress(FrameReader):
    __slots__ = ( 'last_frame_index', )
    
    def __init__(self) -> None:
        super().__init__()
        self.last_frame_index = 0

    def open(self, img_proc:ImageProcessor, capture:ImageCapture) -> None:
        from tqdm import tqdm
        
        begin, total = (-1, -1)
        with suppress(Exception): begin = capture.begin_frame
        with suppress(Exception): total = capture.total_frame_count
        self.tqdm = tqdm(total=total) if total >= 0 else tqdm()
        if begin > 0:
            self.tqdm.update(begin - 1)

    def close(self) -> None:
        with suppress(Exception):
            self.tqdm.close()

    def read(self, frame:Frame) -> None:
        self.tqdm.update(frame.index - self.last_frame_index)
        self.tqdm.refresh()
        self.last_frame_index = frame.index
        

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