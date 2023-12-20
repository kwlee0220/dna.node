from __future__ import annotations

from typing import Generator, Any, Optional, Callable
from contextlib import closing

import argparse
from omegaconf import OmegaConf

from dna import config
from dna.event import KafkaEvent
from dna.event.json_event import JsonEventImpl


def read_json_events(args:argparse.Namespace, *,
                     input_file:Optional[str]=None,
                     deserialize:Callable[[str],KafkaEvent]=JsonEventImpl.from_json) -> Generator[KafkaEvent,None,None]:
    if input_file is None:
        from dna.event import open_kafka_consumer, read_topics
        
        consumer = open_kafka_consumer(args.kafka_brokers, args.kafka_offset)
        if isinstance(args.topic, list):
            consumer.subscribe(args.topic)
        else:
            consumer.subscribe([args.topic])
            
        records = read_topics(consumer,
                              initial_timeout_ms=args.initial_timeout_ms,
                              timeout_ms=args.timeout_ms,
                              stop_on_timeout=args.stop_on_timeout,
                              close_on_return=True)
        return (deserialize(rec.value.decode('utf-8')) for rec in records)
    else:
        from dna.event import read_text_line_file
        return map(deserialize, read_text_line_file(input_file))


def filter_camera_conf(args:dict[str,object]|argparse.Namespace) -> OmegaConf:
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    conf = OmegaConf.create()
    if (v := args.get("camera", None)):
        conf.uri = v
    if (v := args.get("begin_frame", None)):
        conf.begin_frame = v
    if (v := args.get("end_frame", None)):
        conf.end_frame = v
    if (v := args.get("init_ts", None)):
        conf.init_ts = v
    if (v := args.get("sync", None)) is not None:
        conf.sync = v
    if (v := args.get("nosync", None)) is not None:
        conf.sync = not v
    if (v := args.get("title", None)):
        conf.title = v
    if (v := args.get("show", None)) is not None:
        conf.show = v
    if (v := args.get("hide", None)) is not None:
        conf.show = not v
    return conf


def parse_true_false_string(truth:str):
    truth = truth.lower()
    if truth in ['yes', 'true', 'y', 't', '1']:
        return True
    elif truth in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        return None


def update_namespace_with_environ(args:argparse.Namespace) -> argparse.Namespace:
    import os
    import logging
    from typing import Optional
    from collections.abc import Callable
    
    def set_from_environ(args:argparse.Namespace, env_name:str, key:str, *, handler:Optional[Callable[[str],object]]=None) -> None:
        if value := os.environ.get(env_name):
            if handler:
                value = handler(value)
            args[key] = value
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"use environment: name='{env_name}', value='{value}'")
    
    logger = logging.getLogger('dna.envs')

    args = vars(args)
    set_from_environ(args, 'DNA_NODE_CONF', 'conf')
    set_from_environ(args, 'DNA_NODE_CAMERA', 'camera')
    set_from_environ(args, 'DNA_NODE_SYNC', 'sync', handler=parse_true_false_string)
    set_from_environ(args, 'DNA_NODE_BEGIN_FRAME', 'begin_frame', handler=lambda s:int(s))
    set_from_environ(args, 'DNA_NODE_END_FRAME', 'end_frame', handler=lambda s:int(s))
    set_from_environ(args, 'DNA_NODE_OUTPUT', 'output')
    set_from_environ(args, 'DNA_NODE_OUTPUT_VIDEO', 'output_video')
    set_from_environ(args, 'DNA_NODE_SHOW_PROGRESS', 'show_progress', handler=parse_true_false_string)
    
    def parse_size(size_str:str) -> Optional[str]:
        truth = parse_true_false_string(size_str)
        if truth is None:
            return truth
        elif truth is True:
            return '0x0'
        else:
            return None
    set_from_environ(args, 'DNA_NODE_SHOW', 'show', handler=parse_size)
            
    set_from_environ(args, 'DNA_NODE_KAFKA_BROKERS', 'kafka_brokers', handler=lambda s:s.split(','))
    set_from_environ(args, 'DNA_NODE_LOGGER', 'logger')
    set_from_environ(args, 'DNA_NODE_CONF_ROOT', 'conf_root')
    set_from_environ(args, 'DNA_NODE_FFMPEG_PATH', 'ffmpeg_path')
    set_from_environ(args, 'DNA_NODE_RABBITMQ_URL', 'rabbitmq_url')
        
    return argparse.Namespace(**args)

def count_lines(file:str) -> int:
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b
    with open(file, 'r') as f:
        return sum(bl.count('\n') for bl in blocks(f))
    
    
def add_image_processor_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--init_ts", metavar="timestamp", default=argparse.SUPPRESS, help="initial timestamp (eg. 0, now)")
    parser.add_argument("--sync", action='store_true')
    parser.add_argument("--show", nargs='?', const='0x0', default=None)
    parser.add_argument("--begin_frame", metavar="number", type=int, default=argparse.SUPPRESS,
                        help="the first frame number to show. (inclusive)")
    parser.add_argument("--end_frame", metavar="number", type=int, default=argparse.SUPPRESS,
                        help="the last frame number to show. (exclusive)")
    parser.add_argument("--title", metavar="titles", default=None, help="title message (date+time+ts+fps+frame)")
    
    parser.add_argument("--output_video", metavar="mp4 file", default=None, help="output video file.")
    parser.add_argument("--crf", metavar='crf', choices=['opencv', 'ffmpeg', 'lossless'],
                        default='opencv', help="constant rate factor (crf).")
    
    parser.add_argument("--progress", help="display progress bar.", action='store_true')
    

def add_kafka_consumer_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='latest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--topic", nargs='+', help="topic names")
    parser.add_argument("--stop_on_timeout", action='store_true', help="stop when a poll timeout expires")
    parser.add_argument("--timeout_ms", metavar="milli-seconds", type=int, default=1000,
                        help="Kafka poll timeout in milli-seconds")
    parser.add_argument("--initial_timeout_ms", metavar="milli-seconds", type=int, default=5000,
                        help="initial Kafka poll timeout in milli-seconds")
    

def to_image_processor_options(conf:OmegaConf) -> dict[str,Any]:
    return config.to_dict(config.filter(conf, 'show', 'output_video', 'title', 'progress', 'crf'))