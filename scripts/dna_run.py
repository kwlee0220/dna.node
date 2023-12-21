from __future__ import annotations

import argparse

from scripts import dna_node, dna_replay_node_events, dna_node_server
from scripts import dna_show, dna_detect, dna_track, dna_show_global_tracks, dna_show_mc_locations
from scripts import dna_draw_trajs, dna_show_multiple_videos, dna_smooth_trajs
from scripts.push import dna_download_node_events


def parse_show_args(parser_show):
    subparsers_show = parser_show.add_subparsers(dest='show_target', help='show target')
    
    parser_show_camera = subparsers_show.add_parser("camera", help="Display images from camera source")
    dna_show.define_args(parser_show_camera)
    
    parser_show_globaltracks = subparsers_show.add_parser("global-tracks", help="Display global-tracks.")
    dna_show_global_tracks.define_args(parser_show_globaltracks)
    
    parser_draw_trajs = subparsers_show.add_parser("trajs", help="Draw trajectories")
    dna_draw_trajs.define_args(parser_draw_trajs)
    
    parser_multi_videos = subparsers_show.add_parser("videos", help="Show multiple videos")
    dna_show_multiple_videos.define_args(parser_multi_videos)
    
    parser_mc_locations = subparsers_show.add_parser("mc_locations", help="Show locations from multiple cameras")
    dna_show_mc_locations.define_args(parser_mc_locations)
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-level DNA command")
    
    subparsers = parser.add_subparsers(dest='command', help='DNA commands')
    
    parser_show = subparsers.add_parser("show", help="Show sub-commands")
    parse_show_args(parser_show)
    
    parser_node = subparsers.add_parser("node", help="Track objects and publish their locations to Kafka topics.")
    dna_node.define_args(parser_node)
    
    parser_copy = subparsers.add_parser("copy", help="Copy camera image to output video file.")
    dna_copy.define_args(parser_copy)
    
    parser_relay = subparsers.add_parser("replay", help="Feed Node events into relevant Kafka topics")
    dna_replay_node_events.define_args(parser_relay)
    
    parser_download = subparsers.add_parser("download", help="Download node events from relevant Kafka topics.")
    dna_download_node_events.define_args(parser_download)
    
    parser_server = subparsers.add_parser("server", help="Run a node server.")
    dna_node_server.define_args(parser_server)
    
    parser_export = subparsers.add_parser("export", help="Export Kafka topic into a file.")
    dna_export_topic.define_args(parser_export)
    
    parser_import = subparsers.add_parser("import", help="Import events into the Kafka topic.")
    dna_import_topic.define_args(parser_import)
    
    parser_detect = subparsers.add_parser("detect", help="Detect objects in an video")
    dna_detect.define_args(parser_detect)
    
    parser_track = subparsers.add_parser("track", help="Track objects from a camera")
    dna_track.define_args(parser_track)
    
    parser_smooth = subparsers.add_parser("smooth", help="Smooth global trajectories")
    dna_smooth_trajs.define_args(parser_smooth)
    
    import sys
    return parser.parse_args()


def run_show_command(args):
    match args.show_target:
        case 'camera':
            dna_show.run(args)
        case 'global-tracks':
            dna_show_global_tracks.run(args)
        case 'trajs':
            dna_draw_trajs.run(args)
        case 'videos':
            dna_show_multiple_videos.run(args)
        case 'mc_locations':
            dna_show_mc_locations.run(args)


def main():
    args = parse_args()
    match args.command:
        case 'show':
            run_show_command(args)
        case 'copy':
            dna_copy.run(args)
        case 'node':
            dna_node.run(args)
        case 'replay':
            dna_replay_node_events.run(args)
        case 'download':
            dna_download_node_events.run(args)
        case 'node_processor':
            dna_node_server.run(args)
        case 'detect':
            dna_detect.run(args)
        case 'export':
            dna_export_topic.run(args)
        case 'import':
            dna_import_topic.run(args)
        case 'track':
            dna_track.run(args)
        case 'smooth':
            dna_smooth_trajs.run(args)
        case _:
            import sys
            print(f"unsupported command: '{args.command}'", file=sys.stderr)

if __name__ == '__main__':
    main()