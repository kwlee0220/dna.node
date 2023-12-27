from __future__ import annotations

from typing import Any

import argparse

from dna.camera import CRF


def parse_true_false_string(truth:str):
    truth = truth.lower()
    if truth in ['yes', 'true', 'y', 't', '1']:
        return True
    elif truth in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        return None
    
    
def add_image_processor_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", metavar="uri", default=argparse.SUPPRESS, help="target camera uri")
    parser.add_argument("--init_ts", metavar="timestamp", default=argparse.SUPPRESS,
                        help="initial timestamp (eg. 0, now)")
    parser.add_argument("--sync", action='store_true')
    parser.add_argument("--show", nargs='?', const='0x0', default=argparse.SUPPRESS)
    parser.add_argument("--begin_frame", metavar="number", type=int, default=argparse.SUPPRESS,
                        help="the first frame number to show. (inclusive)")
    parser.add_argument("--end_frame", metavar="number", type=int, default=argparse.SUPPRESS,
                        help="the last frame number to show. (exclusive)")
    parser.add_argument("--title", metavar="titles", default=argparse.SUPPRESS,
                        help="title message (date+time+ts+fps+frame)")
    
    parser.add_argument("--output_video", metavar="mp4 file", default=argparse.SUPPRESS,
                        help="output video file.")
    parser.add_argument("--crf", metavar='crf', choices=[name.lower() for name in CRF.names()],
                        default='opencv', help="constant rate factor (crf).")
    
    parser.add_argument("--progress", help="display progress bar.", action='store_true')
    

def add_kafka_consumer_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='latest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--topics", nargs='+', help="topic names")
    parser.add_argument("--timeout", metavar="milli-seconds", type=int, default=argparse.SUPPRESS,
                        help="Kafka poll timeout in milli-seconds")
    parser.add_argument("--poll_timeout", metavar="milli-seconds", type=int, default=1000,
                        help="Kafka poll timeout in milli-seconds")
    parser.add_argument("--initial_poll_timeout", metavar="milli-seconds", type=int, default=5000,
                        help="initial Kafka poll timeout in milli-seconds")
    parser.add_argument("--drop_poll_timeout", action='store_true')