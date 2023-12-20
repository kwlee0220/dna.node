from __future__ import annotations

import sys
from contextlib import closing
import argparse
from omegaconf import OmegaConf

from dna import config, initialize_logger
from dna.camera import ImageProcessor, create_camera_from_conf
from scripts.utils import add_image_processor_arguments, to_image_processor_options


def define_args(parser):
    parser.add_argument("uri", metavar="uri", help="target camera uri")
    add_image_processor_arguments(parser)
    
    parser.add_argument("--hide", action='store_true')
    parser.add_argument("--nosync", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    conf = config.to_conf(args)
    config.update(conf, 'sync', not conf.nosync)
    config.update(conf, 'show', not conf.hide)
    camera = create_camera_from_conf(conf)

    # Camera 설정에 필용한 옵션들을 수집하고, camera를 개방시킨다.
    with closing(camera.open()) as capture:
        # ImageProcessor 객체를 생성하고 구동시킨다.
        options = to_image_processor_options(conf)
        img_proc = ImageProcessor(capture, **options)
        result: ImageProcessor.Result = img_proc.run()
        print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Display images from camera source")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()