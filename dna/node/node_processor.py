from __future__ import annotations

from typing import Optional
import logging

from omegaconf.dictconfig import DictConfig

from dna import config
from dna.camera import ImageProcessor
from dna.track.track_pipeline import TrackingPipeline
from .node_event_pipeline import NodeEventPipeline
 

def build_node_processor(image_processor:ImageProcessor, conf: DictConfig,
                         *,
                         tracking_pipeline:Optional[TrackingPipeline]=None) \
    -> tuple[TrackingPipeline, NodeEventPipeline]:
    # TrackingPipeline 생성하고 ImageProcessor에 등록함
    if not tracking_pipeline:
        tracker_conf = config.get_or_insert_empty(conf, 'tracker')
        tracking_pipeline = TrackingPipeline.load(tracker_conf)
    image_processor.set_frame_processor(tracking_pipeline)
    
    if tracking_pipeline.info_drawer:
        image_processor.add_frame_updater(tracking_pipeline.info_drawer)

    # TrackEventPipeline 생성하고 TrackingPipeline에 등록함
    publishing_conf:DictConfig = config.get_or_insert_empty(conf, 'publishing')
    logger = logging.getLogger("dna.node.event")
    track_event_pipeline = NodeEventPipeline(conf.id, publishing_conf=publishing_conf,
                                                  image_processor=image_processor,
                                                  logger=logger)
    tracking_pipeline.add_track_processor(track_event_pipeline)
    
    return tracking_pipeline, track_event_pipeline
    