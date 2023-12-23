from .types import Box, Size2d, Size2di, Point, Image, ByteString, InvalidStateError, NodeId, TrackId, TrackletId, Trajectory
from .color import BGR
from .utils import initialize_logger, sub_logger

__version__ = '3.0.0'

DEBUG_FRAME_INDEX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = True


DEFAULT_DETECTOR_URI_YOLOv6 = "dna.detect.yolov5:model=l6&score=0.1&agnostic=True&max_det=50&classes=car,bus,truck"
DEFAULT_DETECTOR_URI_YOLOv8 = "dna.detect.ultralytics:model=yolov8l&type=v8&score=0.1&classes=car,bus,truck&agnostic_nms=True"
DEFAULT_DETECTOR_URI = DEFAULT_DETECTOR_URI_YOLOv6