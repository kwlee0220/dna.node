from __future__ import annotations

from typing import Any
import argparse

from dna import initialize_logger, camera
from scripts.utils import add_image_processor_arguments


def define_args(parser):
    parser.add_argument("camera_uri", metavar="uri", help="target camera uri")
    add_image_processor_arguments(parser)
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args:argparse.Namespace):
    initialize_logger(args.logger)
    
    result = camera.process_images(args.camera_uri, **vars(args))
    print(result)
    
def main():
    parser = argparse.ArgumentParser(description="Display images from camera source")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()