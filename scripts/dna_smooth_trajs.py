from __future__ import annotations

import argparse
from tqdm import tqdm

from dna import initialize_logger
from dna.event import sort_events_with_fixed_buffer
from dna.event.json_event import JsonEventImpl
from dna.node.trajectory_smoother import TrajectorySmoother
import scripts


class TextLineWriter:
    def __init__(self, path:str) -> None:
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(path, 'w')
        
    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None
            
    def write(self, line:str) -> None:
        self.fp.write(line + '\n')
        
        
def define_args(parser):
    parser.add_argument("--track_file", default=None, help="track event file (json or pickle format)")
    
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--topic", help="target topic name")
    parser.add_argument("--stop_on_timeout", action='store_true', help="stop when a poll timeout expires")
    parser.add_argument("--timeout_ms", metavar="milli-seconds", type=int, default=1000,
                        help="Kafka poll timeout in milli-seconds")
    parser.add_argument("--initial_timeout_ms", metavar="milli-seconds", type=int, default=5000,
                        help="initial Kafka poll timeout in milli-seconds")
    
    # stabilizer setting
    parser.add_argument("--look_ahead", metavar='count', type=int, default=10, help="look-ahead/behind count")
    parser.add_argument("--smoothing", metavar='value', type=float, default=1, help="stabilization smoothing factor")
    
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")


HEAP_SIZE = 500
def run(args):
    writer = TextLineWriter(args.output)
    progress = tqdm(desc='smoothing track events')

    def write_events(tracks:list[JsonEventImpl]) -> None:
        for track in tracks:
            writer.write(track.to_json())
            progress.update()
            
    smoother = TrajectorySmoother(look_ahead=args.look_ahead)
    events = scripts.read_json_events(args, input_file=args.track_file)
    for ev in sort_events_with_fixed_buffer(events, heap_size=HEAP_SIZE):
        write_events(smoother.smooth(ev))
    progress.close()
    

def main():
    parser = argparse.ArgumentParser(description="Smooth global trajectories")
    define_args(parser)
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()