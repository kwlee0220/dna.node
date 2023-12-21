from contextlib import closing
import sys

import argparse
from omegaconf import OmegaConf

import dna
from dna import config, camera, initialize_logger
from dna.detect.detecting_processor import DetectingProcessor
from scripts.utils import to_camera_options, add_image_processor_arguments, to_image_processor_options

__DEFAULT_DETECTOR_URI = 'dna.detect.yolov5:model=l&score=0.4'
# __DEFAULT_DETECTOR_URI = 'dna.detect.yolov4'


def define_args(parser):
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    add_image_processor_arguments(parser)

    parser.add_argument("--detector", help="Object detection algorithm.", default=None)
    parser.add_argument("--output", "-o", metavar="csv file", default=None, help="output detection file.")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # 카메라 설정 정보 추가
    config.update(conf, 'camera', to_camera_options(args))
    cam = camera.load_camera(**dict(conf.camera))
        
    # args에 포함된 ImageProcess 설정 정보를 추가한다.
    config.update_values(conf, config.to_conf(args))
    options = to_image_processor_options(conf)
    img_proc = camera.create_image_processor(camera=cam, **options)
    
    # detector 설정 정보
    detector_uri = args.detector
    if detector_uri is None:
        detector_uri = config.get(conf, "tracker.dna_deepsort.detector")
    if detector_uri is None:
        detector_uri = __DEFAULT_DETECTOR_URI
    detector = DetectingProcessor.load(detector_uri=detector_uri,
                                        output=args.output,
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