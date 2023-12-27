
import sys
import logging
import argparse

from dna import initialize_logger
from dna.node.redis import RedisExecutionServer

LOGGER = logging.getLogger('dna.node.redis')


def define_args(parser):
    parser.add_argument("--conf_root", metavar="dir", help="Root directory for configurations", default="conf")
    parser.add_argument("--req_channel", metavar="name", help="Track request channel name", default='track-requests')
    parser.add_argument("--redis", metavar="URL", help="Redis server URL", default="redis://localhost:6379?db=0")
    parser.add_argument("--show", nargs='?', const='0x0', default=None)
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")


def run(args):
    initialize_logger(args.logger)
    
    server = RedisExecutionServer(redis_url=args.redis, request_channel=args.req_channel,
                                  args=args, logger=LOGGER)
    server.run()
    

def main():
    parser = argparse.ArgumentParser(description="Run a node processor.")
    define_args(parser)
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()