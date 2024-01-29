from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
import argparse
from collections import ChainMap

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

from dna import config, initialize_logger, camera
from scripts.utils import add_image_processor_arguments
from dna.node.node_processor import build_node_processor


def define_args(parser):
    parser.add_argument("conf", metavar="path", help="node configuration file path")
    add_image_processor_arguments(parser)

    parser.add_argument("--output", metavar="pickle file", help="track/feature event pickle file.", default=None)
    parser.add_argument("--zseq_log", metavar="csv file", help="create Zone Sequence log file.", default=None)
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", help="Kafka broker hosts list", default=None)
    parser.add_argument("--silent_kafka", action='store_true')
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
    
    if args.silent_kafka:
        config.remove(conf, 'publishing.publish_kafka')
    else:
        if args.kafka_brokers:
            # 'kafka-brokers'가 설정된 경우 publishing 작업에서 이 broker로 접속하도록 설정한다.
            config.update(conf, 'publishing.plugins.kafka_brokers', args.kafka_brokers)
    if args.output:
        # 'output'이 설정되어 있으면, track 결과를 frame 단위로 출력할 수 있도록 설정을 수정함.
        config.update(conf, "publishing.output", args.output)
    # if args.track_output:
    #     # 'output'이 설정되어 있으면, track 결과를 frame 단위로 출력할 수 있도록 설정을 수정함.
    #     config.update(conf, "tracker.output", args.track_output)
    if args.zseq_log and config.exists(conf, 'publishing.zone_pipeline'):
        config.update(conf, 'publishing.zone_pipeline.zone_seq_log', args.zseq_log)
    build_node_processor(img_proc, conf)
    result = img_proc.run()
    print(result)
    

def main():
    parser = argparse.ArgumentParser(description="Track objects and publish their locations to Kafka topics")
    define_args(parser)
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
	main()