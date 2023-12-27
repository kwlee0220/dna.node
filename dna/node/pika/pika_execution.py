from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

from argparse import Namespace
from omegaconf import OmegaConf
import pika
from pathlib import Path
import logging

from dna import config
from dna.camera import Frame, ImageProcessor, FrameProcessor
from .pika_execution_context import PikaExecutionContext

LOGGER = logging.getLogger('dna.node.pika')


_DEFAULT_RABBITMQ_HOST = 'localhost'
_DEFAULT_RABBITMQ_PORT = 5672
_DEFAULT_RABBITMQ_USER = 'admin'
_DEFAULT_RABBITMQ_PASSWORD = 'admin'
_DEFAULT_RABBITMQ_QNAME = '/track_requests'

@dataclass(frozen=True) # slots=True
class PikaConnector:
    host:str
    port:int
    user_id:str
    password:str
    
    def blocking_connection(self) -> pika.BlockingConnection:
        try:
            credentials = pika.PlainCredentials(username=self.user_id, password=self.password)
            conn_params = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
            return pika.BlockingConnection(conn_params)
        except pika.exceptions.AMQPConnectionError as e:
            import sys
            print(f"fails to connect RabbitMQ broker: host={self.host}, port={self.port}, "
                  f"username={self.user_id}, password={self.password}", file=sys.stderr)
            raise e
        
    @classmethod
    def parse_url(cls, url:str) -> tuple[PikaConnector,str]:
        import urllib
        
        result = urllib.parse.urlparse(url)
        if result.scheme != 'rabbitmq':
            import sys
            raise ValueError(f"invalid RabbitMQ URL: {url}")
            
        connector = cls(host=result.hostname if result.hostname else _DEFAULT_RABBITMQ_HOST,
                        port=result.port if result.port else _DEFAULT_RABBITMQ_PORT,
                        user_id=result.username if result.username else _DEFAULT_RABBITMQ_USER,
                        password=result.password if result.password else _DEFAULT_RABBITMQ_PASSWORD)
        request_qname = result.path if result.path else _DEFAULT_RABBITMQ_QNAME
        request_qname = request_qname[1:]
        
        return connector, request_qname


class PikaExecutionFactory:
    def __init__(self, args:Namespace) -> None:
        super().__init__()
        self.args = args

    def create(self, pika_ctx:PikaExecutionContext) -> ImageProcessor:
        from dna.node.node_processor import build_node_processor

        request = OmegaConf.create(pika_ctx.request)
        if config.exists(request, 'node'):
            conf_root = Path(self.args.conf_root)
            node_conf_fname = request.node.replace(":", "_") + '.yaml'
            path = conf_root / node_conf_fname
            try:
                conf = config.load(path)
            except Exception as e:
                LOGGER.error(f"fails to load node configuration file: "
                             f"conf_root={self.args.conf_root}, node={request.node}, path='{path}'")
                raise e
                
        # args에 포함된 ImageProcess 설정 정보를 추가한다.
        config.update_values(conf, self.args, 'show', 'show_progress')
        if self.args.kafka_brokers:
            # 'kafka_brokers'가 설정된 경우 publishing 작업에서 이 broker로 접속하도록 설정한다.
            config.update(conf, 'publishing.plugins.kafka_brokers', self.args.kafka_brokers)
        config.update_values(conf, request, "show")
            
        config.update(conf, "camera", request.camera)
        camera = create_camera_from_conf(conf.camera)
        
        options = config.filter(conf, 'show', 'show_progress')
        img_proc = ImageProcessor(camera.open(), context=pika_ctx, **options)
        
        build_node_processor(img_proc, conf)
        
        report_interval = config.get(request, 'report_frame_interval', default=-1)
        if report_interval > 0:
            img_proc.add_frame_processor(PikaExecutionProgressReporter(pika_ctx, report_interval))

        return img_proc


class PikaExecutionProgressReporter(FrameProcessor):
    def __init__(self, context:PikaExecutionContext, report_frame_interval:int) -> None:
        super().__init__()
        self.ctx = context
        self.report_frame_interval = report_frame_interval

    def on_started(self, proc:ImageProcessor) -> None:
        self.next_report_frame = self.report_frame_interval

    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if frame.index >= self.next_report_frame:
            progress = {
                'frame_index': frame.index
            }
            self.ctx.report_progress(progress)
            self.next_report_frame += self.report_frame_interval
        return frame