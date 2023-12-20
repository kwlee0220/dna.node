import warnings
warnings.filterwarnings("ignore")

from contextlib import closing
from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna import config
from dna.camera import ImageProcessor, create_camera_from_conf
from dna.track import TrackingPipeline
from scripts.utils import filter_camera_conf


def define_args(parser):
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--begin_frame", type=int, metavar="number", default=1, help="the first frame number")
    parser.add_argument("--end_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the last frame number")

    parser.add_argument("--output", "-o", metavar="csv file", default=argparse.SUPPRESS, help="output detection file.")
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default=None)
    parser.add_argument("--progress", help="display progress bar.", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    dna.initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # 카메라 설정 정보 추가
    config.update(conf, 'camera', filter_camera_conf(args))
    camera = create_camera_from_conf(conf.camera)

    # args에 포함된 ImageProcess 설정 정보를 추가한다.
    config.update_values(conf, args, 'show', 'output_video', 'progress')
    
    options = config.to_dict(config.filter(conf, 'show', 'output_video', 'progress'))
    img_proc = ImageProcessor(camera.open(), **options)
    
    tracker_conf = config.get_or_insert_empty(conf, 'tracker')
    config.update_values(tracker_conf, args, 'output')
    
    tracking_pipeline = TrackingPipeline.load(tracker_conf)
    img_proc.add_frame_processor(tracking_pipeline)
    if tracking_pipeline.info_drawer:
        img_proc.add_frame_processor(tracking_pipeline.info_drawer)

    result: ImageProcessor.Result = img_proc.run()
    print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()