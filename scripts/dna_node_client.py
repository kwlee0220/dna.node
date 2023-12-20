import json
from contextlib import closing

from dna import config
from dna.support import redis as dna_redis
from dna.node.redis import JsonSerde, RedisNodeProcessorClient


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--redis", metavar="URL", help="Redis server URL", default="redis://localhost:6379?db=0")
    parser.add_argument("--req_channel", metavar="name", help="Track request channel name", default='track-requests')
    parser.add_argument("--node", metavar="id", required=True, help="target DNA node id")
    parser.add_argument("--camera", metavar="uri", required=True, help="target camera uri")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    return parser.parse_args()


def main():
    args = parse_args()
        
    serde = JsonSerde()
    with closing(dna_redis.connect(args.redis)) as redis:
        client = RedisNodeProcessorClient(redis=redis, req_channel=args.req_channel,
                                          node_id=args.node, camera_uri=args.camera,
                                          sync=args.sync)
        for track in client.node_tracks():
            print(track)


if __name__ == '__main__':
	main()