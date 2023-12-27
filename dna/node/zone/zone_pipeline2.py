from __future__ import annotations

from typing import Optional
import logging

from omegaconf.dictconfig import DictConfig

from dna import config, sub_logger
from dna.camera import ImageProcessor
from dna.event import EvenNodePipeline


class ZonePipeline2(EvenNodePipeline):
    def __init__(self, conf:DictConfig,
                 *,
                 image_processor:ImageProcessor,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        from .to_line_transform import ToLineTransform
        to_line = ToLineTransform(logger=sub_logger(logger, 'line'))
        self.append(to_line)
        
        from .zone_event_generator import ZoneEventGenerator
        named_zones = config.get(conf, "zones", default=[])
        zone_detector = ZoneEventGenerator(named_zones, logger=sub_logger(logger, 'zone_gen'))
        self.append(zone_detector)
        
        from .zone_event_refiner import ZoneEventRefiner
        event_refiner = ZoneEventRefiner(logger=sub_logger(logger, 'zone_refine'))
        self.append(event_refiner)
        
        # zone sequence 수집 여부를 결정한다.
        # 수집 여부는 zone sequence의 출력이 필요하거나 zone sequence의 로깅 여부에 따른다.
        # 둘 중 하나라도 필요한 경우넨 zone sequence collector를 추가시킨다.
        
        # ZoneSequence 요약 정보 출력 여부 확인
        draw_zone_seqs = config.get(conf, 'draw', default=False)
        # ZoneSequence 로깅 여부 확인
        zone_log_path = config.get(conf, 'zone_seq_log')
        if (image_processor.is_drawing and draw_zone_seqs) or zone_log_path is not None:
            # ZoneSequence collector를 생성시킨다.
            from .zone_sequence_collector import ZoneSequenceCollector
            collector = ZoneSequenceCollector()
            self.add_listener(collector)
            
            if zone_log_path is not None:
                from .zone_sequence_collector import ZoneSequenceWriter
                collector.add_listener(ZoneSequenceWriter(zone_log_path))
        
            if image_processor.is_drawing and draw_zone_seqs:
                from .zone_sequences_display import ZoneSequenceDisplay
                display = ZoneSequenceDisplay()
                collector.add_listener(display)
                image_processor.add_frame_updater(display)