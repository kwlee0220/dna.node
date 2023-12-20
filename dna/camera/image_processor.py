from __future__ import annotations

from typing import Optional, Any
from abc import ABCMeta, abstractmethod
from contextlib import suppress

import logging
from pathlib import Path
import time
from datetime import timedelta, datetime

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from dna import color, utils, Size2d, sub_logger
from .types import Frame, ImageCapture
from dna.execution import AbstractExecution, ExecutionContext, CancellationError


class FrameProcessor(metaclass=ABCMeta):
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


class ImageProcessor(AbstractExecution):
    __ALPHA = 0.2
    __slots__ = ('capture', 'clean_frame_readers', 'frame_processors', 'suffix_processors',
                 'show_processor', 'is_drawing', 'fps_measured', 'logger')

    from dataclasses import dataclass
    @dataclass(frozen=True, slots=True)
    class Result:
        elapsed: float
        frame_count: int
        fps_measured: float
        failure_cause: Exception

        def __repr__(self):
            return (f"elapsed={timedelta(seconds=self.elapsed)}, "
                    f"frame_count={self.frame_count}, fps={self.fps_measured:.1f}")

    def __init__(self, cap:ImageCapture, **options):
        super().__init__(context=options.get('context'))
        
        self.capture = cap
        self.clean_frame_readers: list[FrameProcessor] = [] # read-only frame processor
        self.frame_processors: list[FrameProcessor] = []
        self.suffix_processors: list[FrameProcessor] = []
        self.logger = logging.getLogger('dna.image_processor')

        self.show_processor:ShowFrame = None
        
        def parse_show_option(show) -> Optional[Size2d]:
            if isinstance(show, bool):
                return cap.size if show else None
            else:
                return Size2d.from_expr(show).to_rint() if show else None
        show_size = parse_show_option(options['show'])

        output_video = options.get('output_video')
        title = options.get('title')
        self.is_drawing:bool = show_size is not None or output_video is not None
        if self.is_drawing and title:
            specs:set[str] = set(title.split('+'))
            self.suffix_processors.append(DrawFrameTitle(specs, bg_color=color.WHITE))
            
        if output_video:
            write_processor = None
            crf_opt = options.get('crf', 'opencv')
            if crf_opt == 'opencv':
                from dna.camera.opencv_video_writer import OpenCvWriteProcessor
                write_processor = OpenCvWriteProcessor(Path(output_video), logger=sub_logger(self.logger, 'image_writer'))
            else:
                from dna.camera.ffmpeg_writer import FFMPEGWriteProcessor, CRF_FFMPEG_DEFAULT, CRF_VISUALLY_LOSSLESS
                crf = CRF_FFMPEG_DEFAULT if crf_opt == 'ffmpeg' else CRF_VISUALLY_LOSSLESS
                write_processor = FFMPEGWriteProcessor(Path(output_video), crf=crf,
                                                       logger=sub_logger(self.logger, 'image_writer'))
            self.suffix_processors.append(write_processor)

        if show_size:
            # 여기서 'show_processor'를 생성만 하고, 실제 등록은
            # 'run_work()' 메소드 수행 시점에서 추가시킨다.
            window_name = f'camera={cap.camera.uri}'
            self.show_processor = ShowFrame(window_name,
                                            window_size=tuple(show_size.wh) if show_size else None,
                                            logger=sub_logger(self.logger, 'show_frame'))

        if options.get('progress', False):
            # 카메라 객체에 'begin_frame' 속성과 'end_frame' 속성이 존재하는 경우에만 ShowProgress processor를 추가한다.
            camera = self.capture.camera
            if ( hasattr(camera, 'begin_frame')
                and hasattr(camera, 'end_frame')
                and hasattr(self.capture, 'total_frame_count') ):
                self.suffix_processors.append(ShowProgress())

        self.fps_measured = 0.
        
    def close(self) -> None:
        self.stop("close requested", nowait=True)
        
    def add_clean_frame_reader(self, reader:FrameProcessor) -> None:
        self.clean_frame_readers.append(reader)

    def add_frame_processor(self, frame_proc: FrameProcessor) -> None:
        self.frame_processors.append(frame_proc)
        
    def run_work(self) -> Result:
        started = time.time()
        
        # frame_processor들을 정리한다.
        # 'show_processor'가 생성된 경우에 등록시킨다.
        full_frame_processors = self.clean_frame_readers + self.frame_processors + self.suffix_processors
        if self.show_processor is not None:
            full_frame_processors.append(self.show_processor)
        
        # 등록된 모든 frame processor들의 'on_started()' 메소드를 호출하여 ImageProcessor가 시작됨을 알린다.
        for frame_proc in full_frame_processors:
            frame_proc.on_started(self)

        capture_count = 0
        self.fps_measured = 0.
        failure_cause = None
        try:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f'start: ImageProcess[cap={self.capture}]')
            while self.capture.is_open():
                # 사용자에 의해 동작 멈춤이 요청된 경우 CallationError 예외를 발생시킴
                self.check_stopped()
                
                # ImageCapture에서 처리할 이미지를 읽어 옴.
                frame: Frame = self.capture()
                if frame is None: break
                capture_count += 1

                # 등록된 모든 frame-processor를 capture된 image를 이용해 'process_frame' 메소드를 차례대로 호출한다.
                # process_frame() 호출시 첫번째 processor는 capture된 image를 입력받지만,
                # 이후 processor들은 자신 바로 전에 호출된 process_frame()의 반환값을 입력으로 받는다.
                # 만일 어느 한 frame-processor의 process_frame() 호출 결과가 None인 경우는 이후 frame-processor 호출은 중단되고
                # 전체 image-processor의 수행이 중단된다.
                for frame_proc in full_frame_processors:
                    frame = frame_proc.process_frame(frame)
                    if frame is None:
                        self.stop(nowait=True)
                        break
                if frame is None:
                    break

                elapsed = time.time() - started
                fps = 1 / (elapsed / capture_count)
                weight = ImageProcessor.__ALPHA if capture_count > 300 else 0.5
                self.fps_measured = weight*fps + (1-weight)*self.fps_measured
        except CancellationError as e:
            failure_cause = e
        except Exception as e:
            failure_cause = e
            self.logger.error(e, exc_info=True)
        finally:
            # 등록된 순서의 역순으로 'on_stopped()' 메소드를 호출함
            for frame_proc in reversed(full_frame_processors):
                try:
                    frame_proc.on_stopped()
                except Exception as e:
                    self.logger.error(e, exc_info=True)

        return ImageProcessor.Result(time.time()-started, capture_count, self.fps_measured, failure_cause)
                
    def stop_work(self) -> None: pass

    def finalize(self) -> None:
        self.capture.close()


from collections import namedtuple
TitleSpec = namedtuple('TitleSpec', 'date, time, ts, frame, fps')
    

class DrawFrameTitle(FrameProcessor):
    def __init__(self, title_spec:set[str], bg_color:Optional[color.BGR]=None) -> None:
        super().__init__()
        self.title_spec = TitleSpec('date' in title_spec, 'time' in title_spec, 'ts' in title_spec,
                                    'frame' in title_spec, 'fps' in title_spec)
        self.bg_color = bg_color
        
    def on_started(self, proc:ImageProcessor) -> None:
        self.proc = proc
    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
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


class ShowFrame(FrameProcessor):
    _PAUSE_MILLIS = int(timedelta(hours=1).total_seconds() * 1000)

    def __init__(self, window_name:str, window_size:Optional[tuple[int,int]],
                 *, logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        self.window_name = window_name
        self.window_size:Optional[tuple[int,int]] = window_size if window_size != (0,0) else None
        self.processors: list[FrameProcessor] = []
        self.logger = logger
        
    def add_processor(self, proc:FrameProcessor) -> None:
        self.processors.append(proc)

    def on_started(self, proc:ImageProcessor) -> None:
        win_size = self.window_size if self.window_size else tuple(proc.capture.size.wh)
        
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'create window: {self.window_name}, size=({win_size[0]}x{win_size[1]})')
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, win_size[0], win_size[1])

    def on_stopped(self) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'destroy window: {self.window_name}')
        with suppress(Exception): cv2.destroyWindow(self.window_name)

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        img = cv2.resize(frame.image, self.window_size, cv2.INTER_AREA) if self.window_size else frame.image
        cv2.imshow(self.window_name, img)

        key = cv2.waitKey(int(1)) & 0xFF
        while True:
            if key == ord('q'):
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'interrupted by a key-stroke')
                return None
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
                        return None
            elif key != 0xFF:
                for proc in self.processors:
                    key = proc.set_control(key)
                return frame
            else: 
                return frame


class ShowProgress(FrameProcessor):
    __slots__ = ( 'begin_frame_index', 'last_frame_index', 'tqdm' )
    
    def __init__(self) -> None:
        super().__init__()
        
        self.begin_frame_index = -1
        self.last_frame_index = -1
        self.tqdm = None

    def on_started(self, proc:ImageProcessor) -> None:
        camera = proc.capture.camera
        self.begin_frame_index = camera.begin_frame
        end_frame_index = camera.end_frame if camera.end_frame is not None else proc.capture.total_frame_count+1
        self.tqdm = tqdm(total=end_frame_index-1)

    def on_stopped(self) -> None:
        with suppress(Exception):
            self.tqdm.close()
            self.tqdm = None

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if self.last_frame_index < 0:
            self.last_frame_index = 0
        
        self.tqdm.update(frame.index - self.last_frame_index)
        self.tqdm.refresh()
        self.last_frame_index = frame.index
        return frame