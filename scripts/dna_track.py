import warnings
warnings.filterwarnings("ignore")

from contextlib import closing
from datetime import timedelta
from collections import ChainMap

import argparse
from omegaconf import OmegaConf

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna import config, camera
from dna.camera import ImageProcessorOptions
from dna.track import TrackingPipeline
from scripts.utils import add_image_processor_arguments


def define_args(parser):
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    add_image_processor_arguments(parser)

    parser.add_argument("--output", "-o", metavar="csv file", default=argparse.SUPPRESS, help="output detection file.")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    dna.initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # tracking pipeline을 생성한다.
    tracker_conf = config.get_or_insert_empty(conf, 'tracker')
    config.update_values(tracker_conf, args, 'output')
    tracking_pipeline = TrackingPipeline.load(tracker_conf)
    
    # camera uri를 선택한다.
    options = dict(ChainMap(vars(args), dict(conf.camera)))
    camera_uri = options.get('camera', options.get('uri'))
    if camera_uri is None:
        raise ValueError(f"camera uri is not specified")
    
    # image processor 를 생성한다.
    img_proc = camera.create_image_processor(camera_uri, frame_processor=tracking_pipeline,
                                             **options)
    if tracking_pipeline.info_drawer:
        img_proc.add_frame_updater(tracking_pipeline.info_drawer)
    result = img_proc.run()
    print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()