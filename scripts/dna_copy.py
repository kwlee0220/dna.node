from __future__ import annotations

import sys
from contextlib import closing
import argparse

from dna import config, initialize_logger
from dna.camera import ImageProcessor, create_camera_from_conf


def define_args(parser):
    parser.add_argument("uri", metavar="uri", help="target camera uri")
    parser.add_argument("output_video", metavar="file", help="output video file.")
    parser.add_argument("--crf", metavar='crf', choices=['opencv', 'ffmpeg', 'lossless'], default='opencv', help="constant rate factor (crf).")
    parser.add_argument("--show", nargs='?', const='0x0', default=None)
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--progress", help="display progress bar.", action='store_true')
    parser.add_argument("--title", metavar="titles", help="title message (date+time+ts+fps+frame)", default=None)
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    conf = config.to_conf(args)
    camera = create_camera_from_conf(conf)

    # Camera 설정에 필용한 옵션들을 수집하고, camera를 개방시킨다.
    with closing(camera.open()) as capture:
        # ImageProcessor 객체를 생성하고 구동시킨다.
        options = config.to_dict(config.filter(conf, 'show', 'output_video', 'title', 'progress', 'crf'))
        img_proc = ImageProcessor(capture, **options)
        result: ImageProcessor.Result = img_proc.run()
        print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Copy camera image to output video file.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()