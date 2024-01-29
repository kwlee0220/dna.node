from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, overload
from dataclasses import field, replace
from contextlib import suppress
import logging
import time
from datetime import timedelta

import cv2

from .. import color
from ..utils import utc_now_millis, sub_logger
from ..size2di import Size2di
from ..execution import AbstractExecution, ExecutionContext, CancellationError
from .types import Frame, Camera, ImageCapture, CRF, CameraOptions

_DEFAULT_WINDOW_SIZE = Size2di(0, 0)


class ImageProcessorOptions(CameraOptions):
    PROCESSOR_ONLY_KEYS = { 'show', 'output_video', 'title', 'progress', 'crf' }
    KEYS = CameraOptions.KEYS.union(PROCESSOR_ONLY_KEYS)
    
    def __init__(self, **options):
        super().__init__(**options)
                
    def __setitem__(self, key: str, item: Any) -> None:
        match key:
            # case 'camera_uri':
            #     assert isinstance(item, str)
            #     self.data[key] = item
            case 'show':
                if isinstance(key, str):
                    self.data['show'] = Size2di.from_expr(item)
                elif isinstance(key, bool):
                    self.data['show'] = Size2di(0, 0)
                elif isinstance(key, Size2di):
                    self.data['show'] = item
                else:
                    raise ValueError(f"invalid option value: show={item}")
            case 'output_video':
                if item is not None:
                    from pathlib import Path
                    assert isinstance(item, str|Path)
                    self.data[key] = str(item)
            case 'title':
                assert isinstance(item, str)
                self.data[key] = item
            case 'progress':
                assert isinstance(item, bool)
                self.data[key] = item
            case 'crf':
                if isinstance(key, str):
                    self.data[key] = CRF.from_name(item)
                elif isinstance(key, CRF):
                    self.data['show'] = item
                else:
                    raise ValueError(f"invalid option value: {key}={item}")
            case _:
                super().__setitem__(key, item)
    
    def to_camera_options(self) -> CameraOptions:
        opts = { k:v for k, v in self.items() if k not in ImageProcessorOptions.PROCESSOR_ONLY_KEYS }
        return CameraOptions(**opts)

  
@runtime_checkable
class FrameReader(Protocol):
    def open(self, img_proc:ImageProcessor) -> None: ...
    def read(self, frame:Frame) -> bool: ...
    def close(self) -> None: ...


@runtime_checkable
class FrameUpdater(Protocol):
    def open(self, img_proc:ImageProcessor) -> None: ...
    def update(self, frame:Frame) -> Frame: ...
    def close(self) -> None: ...


@runtime_checkable
class FrameProcessor(Protocol):
    def open(self, img_proc:ImageProcessor) -> None: ...
    def process(self, frame:Frame) -> Frame: ...
    def close(self) -> None: ...
    

@overload
def create_image_processor(camera_obj:Camera, /,
                            context:Optional[ExecutionContext]=None,
                            frame_processor:Optional[FrameProcessor]=None,
                            **options) -> ImageProcessor:
    pass

@overload
def create_image_processor(camera_uri:str, /,
                            context:Optional[ExecutionContext]=None,
                            frame_processor:Optional[FrameProcessor]=None,
                            **options) -> ImageProcessor:
    pass

def create_image_processor(_camera:Camera|str, /,
                            context:Optional[ExecutionContext]=None,
                            frame_processor:Optional[FrameProcessor]=None,
                            **options) -> ImageProcessor:
    img_proc_opts = ImageProcessorOptions(**options)
    if isinstance(_camera, str):
        from .camera import load_camera
        camera_opts = dict(img_proc_opts.to_camera_options())
        _camera = load_camera(_camera, **camera_opts)
    capture = _camera.open()
        
    return ImageProcessor(capture, img_proc_opts,
                          context=context,
                          frame_processor=frame_processor)

@overload
def process_images(camera_obj:Camera, /,
                   context:Optional[ExecutionContext]=None,
                   frame_processor:Optional[FrameProcessor]=None,
                   **options) -> Result:
    pass
@overload
def process_images(camera_uri:str, /,
                   context:Optional[ExecutionContext]=None,
                   frame_processor:Optional[FrameProcessor]=None,
                   **options) -> Result:
    pass

def process_images(_camera:Camera|str, /,
                   context:Optional[ExecutionContext]=None,
                   frame_processor:Optional[FrameProcessor]=None,
                   **options) -> Result:
    proc = create_image_processor(_camera, context=context, frame_processor=frame_processor, **options)
    return proc.run()


from dataclasses import dataclass
@dataclass(slots=True)
class Result:
    elapsed_ms: int
    frame_count: int
    fps_measured: float
    failure_cause: Optional[Exception] = field(default=None)

    def __repr__(self) -> str:
        elapsed = timedelta(milliseconds=self.elapsed_ms)
        return f"fps={self.fps_measured:.1f}, count={self.frame_count}, elapsed={str(elapsed)[:-4]}"


__ALPHA = 0.1
class ImageProcessor(AbstractExecution):
    def __init__(self, capture:ImageCapture, options:ImageProcessorOptions,
                 *,
                 context:Optional[ExecutionContext]=None,
                 frame_processor:Optional[FrameProcessor]=None) -> None:
        super().__init__(context=context)
        
        self.capture = capture
        
        self.frame_processor:Optional[FrameProcessor] = frame_processor
        self.clean_frame_readers:list[FrameReader] = []
        self.frame_updaters:list[FrameUpdater] = []
        self.final_frame_readers:list[FrameReader] = []
        self.fps_measured = 0.0
        self.is_drawing = False
        self.logger = logging.getLogger('dna.image_processor')
        
        self.show_size = self.__get_show_size(options)
        self.__set_show_title(options)
        self.__set_output_video(options)
        self.__set_show_frame(options)
        self.__set_progress(options)
        
    @property
    def image_size(self) -> Size2di:
        return self.capture.image_size
            
    def set_frame_processor(self, proc:FrameProcessor) -> None:
        self.frame_processor = proc

    def add_clean_frame_reader(self, frame_reader:FrameReader) -> None:
        self.clean_frame_readers.append(frame_reader)

    def add_frame_updater(self, frame_updater:FrameUpdater) -> None:
        self.frame_updaters.append(frame_updater)

    def add_final_frame_reader(self, frame_reader:FrameReader) -> None:
        self.final_frame_readers.append(frame_reader)
    
    def run_work(self) -> Result:
        started_ms = utc_now_millis()
        capture_count:int = 0
        self.fps_measured:float = 0.
        failure_cause:Optional[Exception] = None
            
        with self.capture:
            if self.show_processor:
                self.final_frame_readers.append(self.show_processor)
            
            # 등록된 모든 frame 처리기를 초기화시킨다.
            if self.frame_processor:
                self.frame_processor.open(self)
            for proc in [*self.clean_frame_readers, *self.frame_updaters, *self.final_frame_readers]:
                proc.open(self)
            
            try:
                started_ms_10th = 0
                for frame in self.capture:
                    capture_count += 1
                    self.__process_frame(frame)
                    
                    now = utc_now_millis()
                    if capture_count == 10:
                        started_ms_10th = now
                    elif capture_count > 10:
                        elapsed = now - started_ms_10th
                        self.fps_measured = 1000 / (elapsed / (capture_count-10))
                    else:
                        elapsed = now - started_ms
                        self.fps_measured = 1000 / (elapsed / capture_count)
            except StopIteration: pass
            except CancellationError as e:
                failure_cause = e
            except Exception as e:
                failure_cause = e
                self.logger.error(e, exc_info=True)
            finally:
                # 등록된 모든 frame 처리기를 종료화시킨다.
                for proc in [*self.clean_frame_readers, *self.frame_updaters, *self.final_frame_readers]:
                    with suppress(Exception): proc.close()
                if self.frame_processor:
                    self.frame_processor.close()
                    
        return Result(elapsed_ms=utc_now_millis() - started_ms,
                      frame_count=capture_count,
                      fps_measured=self.fps_measured,
                      failure_cause=failure_cause)
                
    def stop_work(self) -> None: pass

    def finalize(self) -> None: pass
        
    def __process_frame(self, frame:Frame) -> None:
        for reader in self.clean_frame_readers:
            reader.read(frame)
            
        if self.frame_processor is not None:
            frame = self.frame_processor.process(frame)
            
        for updater in self.frame_updaters:
            frame = updater.update(frame)
            
        for reader in self.final_frame_readers:
            reader.read(frame)
            
    def __get_show_size(self, options:ImageProcessorOptions) -> Optional[Size2di]:
        def parse_show_option(show:bool|str) -> Optional[Size2di]:
            if isinstance(show, bool):
                return self.capture.image_size if show else None
            else:
                sz = Size2di.from_expr(show)
                if sz == _DEFAULT_WINDOW_SIZE:
                    sz = self.capture.image_size
                return sz
        return parse_show_option(options.get('show', False))
        
    def __set_show_title(self, options:ImageProcessorOptions) -> None:
        output_video = options.get('output_video')
        title = options.get('title')
        self.is_drawing = self.show_size is not None or output_video is not None
        if self.is_drawing and title:
            specs:set[str] = set(title.split('+'))
            self.frame_updaters.append(DrawFrameTitle(specs, bg_color=color.WHITE))

    def __set_output_video(self, options:ImageProcessorOptions) -> None:
        output_video = options.get('output_video')
        if output_video:
            write_processor = None
            crf = options.get('crf', CRF.OPENCV)
            if crf == CRF.OPENCV:
                from .opencv_video_writer import OpenCvWriteProcessor
                write_processor = OpenCvWriteProcessor(output_video, logger=sub_logger(self.logger, 'image_writer'))
            else:
                from .ffmpeg_writer import FFMPEGWriteProcessor
                write_processor = FFMPEGWriteProcessor(output_video, crf=crf,
                                                       logger=sub_logger(self.logger, 'image_writer'))
            self.final_frame_readers.append(write_processor)
            
    def __set_show_frame(self, options:ImageProcessorOptions) -> None:
        self.show_processor:Optional[ShowFrame] = None
        if self.show_size:
            # 여기서 'show_processor'를 생성만 하고, 실제 등록은
            # 'run_work()' 메소드 수행 시점에서 추가시킨다.
            self.show_processor = ShowFrame(window_name=f'camera={self.capture.camera.uri}',
                                            logger=sub_logger(self.logger, 'show_frame'))
    
    def __set_progress(self, options:ImageProcessorOptions) -> None:
        if options.get('progress', False):
            self.final_frame_readers.append(ShowProgress())

class ShowProgress(FrameReader):
    __slots__ = ( 'last_frame_index', )
    
    def __init__(self) -> None:
        super().__init__()
        self.last_frame_index = 0

    def open(self, img_proc:ImageProcessor) -> None:
        from tqdm import tqdm
        
        begin, total = (-1, -1)
        with suppress(Exception): begin = capture.begin_frame       # type: ignore
        with suppress(Exception): total = capture.total_frame_count # type: ignore
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
        
    def open(self, img_proc:ImageProcessor) -> None:
        self.image_proc = img_proc
        
    def close(self) -> None: pass

    def update(self, frame:Frame) -> Frame:
        from datetime import datetime
        
        ts_sec = frame.ts / 1000.0
        date_str = datetime.fromtimestamp(ts_sec).strftime('%Y-%m-%d')
        time_str = datetime.fromtimestamp(ts_sec).strftime('%H:%M:%S.%f')[:-4]
        ts_str = f'ts:{frame.ts}'
        frame_str = f'#{frame.index}'
        fps_str = f'fps:{self.image_proc.fps_measured:.2f}'
        str_list = [date_str, time_str, ts_str, frame_str, fps_str]
        message = ' '.join([msg for is_on, msg in zip(self.title_spec, str_list) if is_on])
        
        convas = frame.image
        if self.bg_color:
            (msg_w, msg_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)
            convas = cv2.rectangle(convas, (7, 3), (msg_w+11, msg_h+11), self.bg_color, -1)
        convas = cv2.putText(convas, message, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
        return replace(frame, image=convas)


class ShowFrame(FrameReader):
    _PAUSE_MILLIS = int(timedelta(hours=1).total_seconds() * 1000)

    def __init__(self, window_name:str,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        self.window_name = window_name
        self.logger = logger

    def open(self, img_proc:ImageProcessor) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        assert img_proc.show_size is not None
        self.window_size = img_proc.image_size if img_proc.show_size == _DEFAULT_WINDOW_SIZE else img_proc.show_size
        cv2.resizeWindow(self.window_name, self.window_size.width, self.window_size.height)
        
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'create window: {self.window_name}, size=({self.window_size})')
            
    def close(self) -> None:
        if self.logger and self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f'destroy window: {self.window_name}')
        with suppress(Exception): cv2.destroyWindow(self.window_name)

    def read(self, frame:Frame) -> None:
        if not self.window_size:
            img = frame.image
        else:
            img = cv2.resize(frame.image, dsize=self.window_size, interpolation=cv2.INTER_AREA)
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
            else: 
                return