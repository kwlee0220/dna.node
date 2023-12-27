from __future__ import annotations

from typing import Optional
from dataclasses import replace

import logging
from omegaconf.dictconfig import DictConfig

from dna import config, sub_logger, NodeId, Size2d
from dna.camera import Frame, ImageProcessor
from dna.event import EvenNodePipeline
from dna.track.types import TrackProcessor, ObjectTrack
from dna.track.dna_tracker import DNATracker
from .types import SilentFrame
from .refine_track_event import RefineTrackEvent
from .drop_short_trail import DropShortTrail
from .world_coord_attach import WorldCoordinateAttacher
from .stabilizer import TrackletSmoothProcessor
from .reid_features import PublishReIDFeatures
from .zone.zone_pipeline2 import ZonePipeline2
from .utils import NodeEventWriter, GroupByFrameIndex, MinFrameIndexGauge

_DEFAULT_BUFFER_SIZE = 30
_DEFAULT_BUFFER_TIMEOUT = 5.0
_DEEP_SORT_REID_MODEL = 'models/deepsort/model640.pt'


class CompositeMinFrameIndexGauges:
    def __init__(self) -> None:
        self.gauges:list[MinFrameIndexGauge] = []
        self.holder_idx = -1
        self.min = -1
        
    def add(self, gauge:MinFrameIndexGauge) -> None:
        self.gauges.append(gauge)
        self.min = min(self.min, gauge.min_frame_index())
        
    def min_frame_index(self) -> int:
        import sys
        
        if self.holder_idx >= 0:
            new_min = self.gauges[self.holder_idx].min_frame_index()
            if self.min == new_min:
                return new_min
            
        self.holder_idx, self.min = min(enumerate(gauge.min_frame_index() for gauge in self.gauges), key=lambda t: t[1])
        return self.min
    

class NodeTrackEventPipeline2(EvenNodePipeline, TrackProcessor):
    def __init__(self, node_id:NodeId, publishing_conf:DictConfig,
                 image_processor:ImageProcessor,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        EvenNodePipeline.__init__(self)
        TrackProcessor.__init__(self)
        
        self.node_id = node_id
        self.conf = publishing_conf
        self.__group_event_queue:Optional[GroupByFrameIndex] = None
        self.gauges = CompositeMinFrameIndexGauges()
        self.logger = logger
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        refine_tracks = self.load_refine_tracks()
        if refine_tracks:
            self.append(refine_tracks)
            self.gauges.add(refine_tracks)

        # drop too-short tracks of an object
        drop_short_tail = self.load_drop_short_trail()
        if drop_short_tail:
            self.append(drop_short_tail)
            self.gauges.add(drop_short_tail)
        
        # attach world-coordinates to each track
        attach_world_coordinates = self.load_attach_world_coordinate()
        if attach_world_coordinates:
            self.append(attach_world_coordinates)
        
        # stabilize world-coordinates
        stabilization = self.load_stabilizer()
        if stabilization:
            self.append(stabilization)
            self.gauges.add(stabilization)
        
        zone_pipeline = self.load_zone_pipeline(image_processor)
        if zone_pipeline:
            self.append(zone_pipeline)
        
        reid_features = self.load_reid_features(image_processor)
        if reid_features:
            # self.add(reid_features, name='reid_features')
            self.group_event_queue.add_listener(reid_features)
            
        self.load_kafka_publishers(reid_features)
        self.load_output_writer(reid_features)
        
    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None: pass
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:list[ObjectTrack]) -> None:
        if len(tracker.last_event_tracks) > 0:
            for ev in tracker.last_event_tracks:
                ev = replace(ev, node_id=self.node_id)
                self.handle_event(ev)
        else:
            self.handle_event(SilentFrame(frame_index=frame.index, ts=frame.ts))
    
    @property
    def group_event_queue(self) -> GroupByFrameIndex:
        if not self.__group_event_queue:
            from dna.node.utils import GroupByFrameIndex
            self.__group_event_queue = GroupByFrameIndex(self.gauges)
            self.add_listener(self.__group_event_queue)
        return self.__group_event_queue
    
    def min_pending_frame_index(self) -> int:
        return min(named.value.min_frame_index() for named in self.event_nodes
                                                    if isinstance(named.value, MinFrameIndexGauge))

    def load_refine_tracks(self) -> Optional[RefineTrackEvent]:
        refine_track_conf = config.get(self.conf, 'refine_tracks')
        if not refine_track_conf:
            return None
        buffer_size = config.get(refine_track_conf, 'buffer_size', default=_DEFAULT_BUFFER_SIZE)
        buffer_timeout = config.get(refine_track_conf, 'buffer_timeout', default=_DEFAULT_BUFFER_TIMEOUT)
        return RefineTrackEvent(buffer_size=buffer_size,
                                buffer_timeout=buffer_timeout,
                                logger=sub_logger(self.logger, "refine"))
    
    def load_drop_short_trail(self) -> Optional[DropShortTrail]:
        min_path_length = config.get(self.conf, 'min_path_length', default=-1)
        if min_path_length < 0:
            return None
        return DropShortTrail(min_path_length, logger=sub_logger(self.logger, 'drop_short_tail'))
    
    def load_attach_world_coordinate(self) -> Optional[WorldCoordinateAttacher]:
        attach_conf = config.get(self.conf, 'attach_world_coordinates')
        if not attach_conf:
            return None
        return WorldCoordinateAttacher(attach_conf, logger=sub_logger(self.logger, 'localizer'))
            
    def load_stabilizer(self) -> Optional[TrackletSmoothProcessor]:
        if not config.exists(self.conf, 'stabilization'):
            return None
        return TrackletSmoothProcessor(self.conf.stabilization)
            
    def load_zone_pipeline(self, image_processor:ImageProcessor) -> Optional[ZonePipeline2]:
        zone_pipeline_conf = config.get(self.conf, 'zone_pipeline')
        if not zone_pipeline_conf:
            return None
        zone_logger = logging.getLogger('dna.node.zone')
        return ZonePipeline2(zone_pipeline_conf, image_processor=image_processor, logger=zone_logger)
                    
    def load_reid_features(self, image_processor:ImageProcessor) -> Optional[PublishReIDFeatures]:
        reid_features_conf = config.get(self.conf, 'reid_features')
        if not reid_features_conf:
            return None
            
        from dna.track.dna_tracker import load_feature_extractor
        distinct_distance = reid_features_conf.get('distinct_distance', 0.0)
        min_crop_size = Size2d.from_expr(reid_features_conf.get('min_crop_size', '80x80'))
        max_iou = reid_features_conf.get('max_iou', 1)
        model_file = reid_features_conf.get('model_file', _DEEP_SORT_REID_MODEL)
        gen_features = PublishReIDFeatures(extractor=load_feature_extractor(model_file, normalize=True),
                                           distinct_distance=distinct_distance,
                                           min_crop_size=min_crop_size,
                                           max_iou=max_iou,
                                           logger=sub_logger(self.logger, 'features'))
        image_processor.add_clean_frame_reader(gen_features)
        return gen_features

    def load_kafka_publishers(self, reid_features:Optional[PublishReIDFeatures]) -> None:
        publish_tracks_conf = config.get(self.conf, 'publish_kafka.publish_node_tracks')
        if publish_tracks_conf:
            self.load_plugin_publish_tracks(publish_tracks_conf, logger=sub_logger(self.logger, 'tracks'))
        
        publish_features_conf = config.get(self.conf, "publish_kafka.publish_track_features")
        if publish_features_conf:
            # 'PublishReIDFeatures' plugin은 ImageProcessor가 지정된 경우에만 등록시킴
            if reid_features:
                assert isinstance(reid_features, PublishReIDFeatures)
                self.load_plugin_publish_features(publish_features_conf, reid_features,
                                                  logger=sub_logger(self.logger, 'features'))
            else:
                raise ValueError(f'ReIDFeatures are not generated')
            
    def load_plugin_publish_tracks(self, publish_tracks_conf:DictConfig, logger:Optional[logging.Logger]=None) -> None:
        from dna.node.kafka_event_publisher import KafkaEventPublisher
        
        kafka_brokers = config.get(publish_tracks_conf, 'kafka_brokers')
        topic = config.get(publish_tracks_conf, 'topic', default='node-tracks')
        publisher = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=sub_logger(logger, 'tracks'))
        self.add_listener(publisher)
        
    def load_plugin_publish_features(self, publish_features_conf:DictConfig,
                                     reid_features:PublishReIDFeatures,
                                     *,
                                     logger:Optional[logging.Logger]=None) -> None:
        from dna.node.kafka_event_publisher import KafkaEventPublisher
        
        kafka_brokers = config.get(publish_features_conf, 'kafka_brokers')
        topic = config.get(publish_features_conf, 'topic', default='track-features')
        publisher = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger)
        reid_features.add_listener(publisher)

    def load_output_writer(self, reid_features:Optional[PublishReIDFeatures]) -> Optional[NodeEventWriter]:
        output_file = config.get(self.conf, 'output')
        if output_file is None:
            return None
            
        writer = NodeEventWriter(output_file)
        self.add_listener(writer)
        if reid_features:
            reid_features.add_listener(writer)
        return writer