from contextlib import closing
from collections import ChainMap
import sys

import argparse
from omegaconf import OmegaConf

import dna
from dna import config, camera, initialize_logger
from dna.camera import CameraOptions, ImageProcessorOptions
from dna.detect.detecting_processor import DetectingProcessor
from scripts.utils import add_image_processor_arguments

__DEFAULT_DETECTOR_URI = 'dna.detect.yolov5:model=l&score=0.4'
# __DEFAULT_DETECTOR_URI = 'dna.detect.yolov4'


def define_args(parser):
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    add_image_processor_arguments(parser)

    parser.add_argument("--detector", help="Object detection algorithm.", default=argparse.SUPPRESS)
    parser.add_argument("--output", "-o", metavar="csv file", default=argparse.SUPPRESS,
                        help="output detection file.")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # camera uri를 선택한다.
    options = dict(ChainMap(vars(args), dict(conf.camera)))
    camera_uri = options.get('camera', options.get('uri'))
    if camera_uri is None:
        raise ValueError(f"camera uri is not specified")
    
    # image processor 를 생성한다.
    img_proc = camera.create_image_processor(camera_uri, **options)
    
    # detector 설정 정보
    detector_uri = args.detector
    if detector_uri is None:
        detector_uri = config.get(conf, "tracker.dna_deepsort.detector")
    if detector_uri is None:
        detector_uri = __DEFAULT_DETECTOR_URI
    detector = DetectingProcessor.load(detector_uri=detector_uri,
                                        output=options.get('output'),
                                        draw_detections=img_proc.is_drawing)
    img_proc.set_frame_processor(detector)
    
    result = img_proc.run()
    print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
	main()