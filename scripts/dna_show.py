from __future__ import annotations

from typing import Any
import sys
from contextlib import closing
import argparse
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from dna import config, initialize_logger, camera
from dna.camera import ImageProcessorOptions
from scripts.utils import add_image_processor_arguments


def define_args(parser):
    parser.add_argument("uri", metavar="uri", help="target camera uri")
    add_image_processor_arguments(parser)
    
    parser.add_argument("--hide", action='store_true')
    parser.add_argument("--nosync", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    options = ImageProcessorOptions(vars(args))
    result = camera.process_images(options)
    print(result)
    
def main():
    parser = argparse.ArgumentParser(description="Display images from camera source")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()