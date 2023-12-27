from __future__ import annotations

from typing import Optional, Any
from dataclasses import replace
import logging
from datetime import timedelta

import numpy as np
from omegaconf.dictconfig import DictConfig

from dna import Size2d, config, NodeId, sub_logger, TrackletId
from dna.camera import Frame, ImageProcessor
from dna.event import TimeElapsed, SilentFrame, MultiStagePipeline, EventQueue, EventNodeImpl
from dna.event.event_processors import DropEventByType, TimeElapsedGenerator
from dna.track.types import TrackProcessor, ObjectTrack
from dna.track.dna_tracker import DNATracker
from dna.node.utils import GroupByFrameIndex
from .zone.zone_pipeline import ZonePipeline

_DEFAULT_BUFFER_SIZE = 30
_DEFAULT_BUFFER_TIMEOUT = 5.0


class MinFrameIndexComposer:
    def __init__(self) -> None:
        self.processors:list[EventNodeImpl] = []
        self.min_indexes:list[int] = []
        self.min_holder = -1
        
    def append(self, proc:EventNodeImpl) -> None:
        self.processors.append(proc)
        self.min_indexes.append(None)
        
    def min_frame_index(self) -> int:
        import sys
        
        if self.min_holder >= 0:
            min = self.processors[self.min_holder].min_frame_index()
            if min == self.min_indexes[self.min_holder]:
                return min
        for idx, proc in enumerate(self.processors):
            min = proc.min_frame_index()
            self.min_indexes[idx] = min if min else sys.maxsize
        
        self.min_holder = np.argmin(self.min_indexes)
        min = self.min_indexes[self.min_holder]
        if min != sys.maxsize:
            return min
        else:
            self.min_holder = -1
            return None


class NodeTrackEventPipeline(MultiStagePipeline, TrackProcessor):
    def __init__(self, node_id:NodeId, publishing_conf:DictConfig,
                 image_processor:ImageProcessor,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        MultiStagePipeline.__init__(self)
        TrackProcessor.__init__(self)

        self.node_id = node_id
        self.services:dict[str,tuple[Any,bool]] = dict()
        self._tick_gen = None
        
        self.__group_event_queue:Optional[GroupByFrameIndex] = None
        self.logger = logger
        
        self.min_frame_indexers:MinFrameIndexComposer = MinFrameIndexComposer()
        
        # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
        refine_track_conf = config.get(publishing_conf, 'refine_tracks')
        if refine_track_conf:
            from .refine_track_event import RefineTrackEvent
            buffer_size = config.get(refine_track_conf, 'buffer_size', default=_DEFAULT_BUFFER_SIZE)
            buffer_timeout = config.get(refine_track_conf, 'buffer_timeout', default=_DEFAULT_BUFFER_TIMEOUT)
            refine_tracks = RefineTrackEvent(buffer_size=buffer_size, buffer_timeout=buffer_timeout,
                                             logger=sub_logger(logger, "refine"))
            self.add_stage("refine_tracks", refine_tracks)
            self.min_frame_indexers.append(refine_tracks)

        # drop too-short tracks of an object
        min_path_length = config.get(publishing_conf, 'min_path_length', default=-1)
        if min_path_length > 0:
            from .drop_short_trail import DropShortTrail
            drop_short_trail = DropShortTrail(min_path_length, logger=sub_logger(logger, 'drop_short_tail'))
            self.add_stage("drop_short_trail", drop_short_trail)
            self.min_frame_indexers.append(drop_short_trail)

        # attach world-coordinates to each track
        if config.exists(publishing_conf, 'attach_world_coordinates'):
            from .world_coord_attach import WorldCoordinateAttacher
            attacher = WorldCoordinateAttacher(publishing_conf.attach_world_coordinates,
                                               logger=sub_logger(logger, 'localizer'))
            self.add_stage("attach_world_coordinates", attacher)

        if config.exists(publishing_conf, 'stabilization'):
            from .stabilizer import TrackletSmoothProcessor
            stabilizer = TrackletSmoothProcessor(publishing_conf.stabilization)
            self.add_stage("stabilization", stabilizer)
            self.min_frame_indexers.append(stabilizer)
            
        self.add_stage("drop TimeElapsed", DropEventByType([TimeElapsed]))
        
        # generate zone-based events
        zone_pipeline_conf = config.get(publishing_conf, 'zone_pipeline')
        if zone_pipeline_conf:
            zone_logger = logging.getLogger('dna.node.zone')
            image_size = None
            # image_size = image_processor.capture.size
            zone_pipeline = ZonePipeline(zone_pipeline_conf, image_size=image_size, logger=zone_logger)
            self.add_stage('zone_pipeline', zone_pipeline)
            
            # zone sequence 수집 여부를 결정한다.
            # 수집 여부는 zone sequence의 출력이 필요하거나 zone sequence의 로깅 여부에 따른다.
            # 둘 중 하나라도 필요한 경우넨 zone sequence collector를 추가시킨다.
            
            # ZoneSequence 요약 정보 출력 여부 확인
            draw_zone_seqs = config.get(zone_pipeline_conf, 'draw', default=False)
            # ZoneSequence 로깅 여부 확인
            zone_log_path = config.get(zone_pipeline_conf, 'zone_seq_log')
            if (image_processor.is_drawing and draw_zone_seqs) or zone_log_path is not None:
                # ZoneSequence collector를 생성시킨다.
                from .zone.zone_sequence_collector import ZoneSequenceCollector
                collector = ZoneSequenceCollector()
                zone_pipeline.add_listener(collector)
                
                if zone_log_path is not None:
                    from .zone.zone_sequence_collector import ZoneSequenceWriter
                    collector.add_listener(ZoneSequenceWriter(zone_log_path))
            
                if image_processor.is_drawing and draw_zone_seqs:
                    from .zone.zone_sequences_display import ZoneSequenceDisplay
                    display = ZoneSequenceDisplay()
                    collector.add_listener(display)
                    image_processor.add_frame_updater(display)
                    
        reid_features_conf = config.get(publishing_conf, 'reid_features')
        if reid_features_conf:
            load_reid_feature_generator(reid_features_conf, self, image_processor, logger=sub_logger(logger, 'features'))
    
        # 알려진 TrackEventPipeline의 plugin 을 생성하여 등록시킨다.
        kafka_conf = config.get(publishing_conf, "publish_kafka")
        if kafka_conf:
            load_kafka_publisher(kafka_conf, self, logger=self.logger)

        tick_interval = config.get(publishing_conf, 'tick_interval', default=-1)
        if tick_interval > 0:
            self._tick_gen = TimeElapsedGenerator(timedelta(seconds=tick_interval))
            self._tick_gen.add_listener(self)
            self._tick_gen.start()
    
        output_file = config.get(publishing_conf, 'output')
        if output_file is not None:
            load_output_writer(output_file, self)

    def close(self) -> None:
        if self._tick_gen:
            self._tick_gen.stop()
        
        for svc, call_close in reversed(self.services.values()):
            if call_close:
                svc.close()
        
        MultiStagePipeline.close(self)
    
    @property
    def group_event_queue(self) -> EventQueue:
        if not self.__group_event_queue:
            from dna.node.utils import GroupByFrameIndex
            self.__group_event_queue = GroupByFrameIndex(self.min_frame_indexers.min_frame_index)
            self.add_listener(self.__group_event_queue)
        return self.__group_event_queue
        
    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()
        
    def process_tracks(self, tracker:DNATracker, frame:Frame, tracks:list[ObjectTrack]) -> None:
        if len(tracker.last_event_tracks) > 0:
            for ev in tracker.last_event_tracks:
                ev = replace(ev, node_id=self.node_id)
                self.handle_event(ev)
        else:
            self.handle_event(SilentFrame(frame_index=frame.index, ts=frame.ts))

    def _append_processor(self, proc:EventNodeImpl) -> None:
        self._tail.add_listener(proc)
        self._tail = proc
        
  
_DEEP_SORT_REID_MODEL = 'models/deepsort/model640.pt'      
def load_reid_feature_generator(conf:DictConfig,
                                pipeline:NodeTrackEventPipeline,
                                image_processor:ImageProcessor,
                                *, logger:Optional[logging.Logger]=None):
    from dna.track.dna_tracker import load_feature_extractor
    from .reid_features import PublishReIDFeatures
    
    distinct_distance = conf.get('distinct_distance', 0.0)
    min_crop_size = Size2d.from_expr(conf.get('min_crop_size', '80x80'))
    max_iou = conf.get('max_iou', 1)
    model_file = conf.get('model_file', _DEEP_SORT_REID_MODEL)
    
    gen_features = PublishReIDFeatures(extractor=load_feature_extractor(model_file, normalize=True),
                                       distinct_distance=distinct_distance,
                                       min_crop_size=min_crop_size,
                                       max_iou=max_iou,
                                       logger=logger)
    pipeline.group_event_queue.add_listener(gen_features)
    image_processor.add_clean_frame_reader(gen_features)
    pipeline.services['reid_features'] = (gen_features, False)


def load_kafka_publisher(conf:DictConfig,
                         pipeline:NodeTrackEventPipeline,
                         *,
                         logger:Optional[logging.Logger]=None) -> None:
    publish_tracks_conf = config.get(conf, 'publish_node_tracks')
    if publish_tracks_conf:
        load_plugin_publish_tracks(publish_tracks_conf, pipeline,
                                   logger=sub_logger(logger, 'tracks'))
            
    # 'PublishReIDFeatures' plugin은 ImageProcessor가 지정된 경우에만 등록시킴
    publish_features_conf = config.get(conf, "publish_track_features")
    if publish_features_conf:
        load_plugin_publish_features(publish_features_conf, pipeline,
                                     logger=sub_logger(logger, 'features'))


def load_plugin_publish_tracks(conf:DictConfig, pipeline:NodeTrackEventPipeline, 
                               *,
                               logger:Optional[logging.Logger]=None):
    from dna.event import KafkaEventPublisher
    
    kafka_brokers = config.get(conf, 'kafka_brokers')
    topic = config.get(conf, 'topic', default='node-tracks')
    publisher = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger)
    pipeline.add_listener(publisher)
    pipeline.services['publish_node_tracks'] = (publisher, False)


def load_plugin_publish_features(conf:DictConfig, pipeline:NodeTrackEventPipeline,
                                 *,
                                 logger:Optional[logging.Logger]=None):
    if 'reid_features' in pipeline.services:
        from dna.event import KafkaEventPublisher
        kafka_brokers = config.get(conf, 'kafka_brokers')
        topic = config.get(conf, 'topic', default='track-features')
        publisher = KafkaEventPublisher(kafka_brokers=kafka_brokers, topic=topic, logger=logger)
        reid_feature_gen, _ = pipeline.services['reid_features']
        reid_feature_gen.add_listener(publisher)
        pipeline.services['publish_track_features'] = (publisher, False)
    else:
        raise ValueError(f'ReIDFeatures are not generated')


def load_output_writer(output_file:str, pipeline:NodeTrackEventPipeline):
    from .utils import NodeEventWriter
    writer = NodeEventWriter(output_file)
    pipeline.group_event_queue.add_listener(writer)
    
    if 'reid_features' in pipeline.services:
        reid_feature_gen, _ = pipeline.services['reid_features']
        reid_feature_gen.add_listener(writer)
    
    # file writer는 pipeline이 종료될 때 close가 호출되도록 한다.
    pipeline.services['output_writer'] = (writer, False)