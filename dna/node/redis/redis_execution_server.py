from __future__ import annotations

from typing import Optional, Any
import logging

import dataclasses
from argparse import Namespace
import threading
import redis
from redis.exceptions import ConnectionError
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path

from dna import config, utils, camera
from dna.camera import FrameProcessor, ImageProcessor
from dna.node.node_processor import NodeTrackEventPipeline
from dna.execution import ExecutionContext, AbstractExecution, ExecutionState
from dna.support import redis as dna_redis
from . import Serde, JsonSerde
from .redis_event_publisher import RedisEventPublisher


class RedisExecutionContext(ExecutionContext):
    def __init__(self, id:str, redis:redis.Redis, progress_channel:str, serde:Serde) -> None:
        self.id = id
        self.redis = redis
        self.progress_channel = progress_channel
        self.serde = serde
            
    def reply(self, json_obj:Any) -> None:
        json_str = self.serde.serialize(json_obj)
        self.redis.publish(self.progress_channel, json_str)

    def started(self) -> None:
        started = {
            'id': self.id,
            'state': 'STARTED',
            'timestamp': utils.utc_now_millis()
        }
        self.reply(started)
        
    def report_progress(self, progress:Any) -> None:
        progress = {
            'id': self.id,
            'state': ExecutionState.RUNNING.name,
            'timestamp': utils.utc_now_millis(),
            'progress': progress
        }
        self.reply(progress)

    def completed(self, result:Any) -> None:
        completed = {
            'id': self.id,
            'state': ExecutionState.COMPLETED.name,
            'timestamp': utils.utc_now_millis(),
            'result': dataclasses.asdict(result)
        }
        self.reply(completed)

    def stopped(self, details:str) -> None:
        stopped = {
            'id': self.id,
            'state': ExecutionState.STOPPED.name,
            'timestamp': utils.utc_now_millis(),
            'cause': details
        }
        self.reply(stopped)

    def failed(self, cause:str|Exception) -> None:
        failed = {
            'id': self.id,
            'state': ExecutionState.FAILED.name,
            'timestamp': utils.utc_now_millis(),
            'cause': repr(cause)
        }
        self.reply(failed)
    

class RedisExecutionServer:
    def __init__(self, redis_url:str, request_channel:str, args:Namespace, logger:logging.Logger) -> None:
        self.redis_url = redis_url
        self.req_channel = request_channel
        self.args = args
        self.serde = JsonSerde()
        self.logger = logger

    def run(self) -> None:
        redis = dna_redis.connect(self.redis_url)
        with redis.pubsub() as pubsub:
            while True:
                try:
                    pubsub.subscribe([self.req_channel])
                    pubsub.get_message(timeout=1)
                    
                    while True:
                        req:dict[str,Any] = pubsub.get_message(timeout=None)  # type: ignore
                        request:DictConfig = config.to_conf(self.serde.deserialize(req['data']))
                        self.on_request(redis=redis, req=request)
                except ConnectionError as e:
                    import sys
                    print(f"retry to connect Redis server: url: '{self.redis_url}'", file=sys.stderr)

    def on_request(self, redis:redis.Redis, req:DictConfig) -> None:
        context = None
        try:
            id:str = config.get(req, 'id')
            if id is None:
                raise ValueError(f"'id' is not specified. msg={req}")
            
            if config.get(req, 'action') is None:
                raise ValueError(f"'action' is not specified. msg={req}")
                
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"received a request: req={req}")
                
            if req.action != "track":
                if self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f"unknown action: action={req.action}, req={req}")
                return
            
            channels = req.channels
            reports_channel = config.get(channels, "progress_reports")
            context = RedisExecutionContext(id=id, redis=redis, progress_channel=reports_channel, serde=self.serde)
            
            img_proc, _ = self.create_execution(context, request=req)
            
            controls_channel = config.get(channels, "controls")
            control_processor = ControlProcessor(conn=redis, control_channel=controls_channel,
                                                 image_processor=img_proc, logger=self.logger)
                
            control_thread = threading.Thread(target=control_processor.run)
            control_thread.start()
            img_proc.run()
        except Exception as e:
            self.logger.error(f'fails to create execution: cause={e}')
            if context is not None:
                context.failed(e)
            
    def create_execution(self, context:RedisExecutionContext, request:DictConfig) \
        -> tuple[ImageProcessor, NodeTrackEventPipeline]:
        from dna.node.node_processor import build_node_processor

        if config.exists(request, 'node'):
            node_conf_fname = request.node.replace(":", "_") + '.yaml'
            path = Path(self.args.conf_root) / node_conf_fname
            try:
                conf = config.load(path)
            except Exception as e:
                self.logger.error(f"fails to load node configuration file: "
                                  f"conf_root={self.args.conf_root}, node={request.node}, path='{path}'")
                raise e
        else:
            conf = request
                
        # args에 포함된 ImageProcess 설정 정보를 추가한다.
        config.update_values(conf, self.args, 'show', 'output_video', 'progress', 'title')
        
        # configuration에 포함된 kafka event publishing관련 설정을 제거한다.
        conf = config.remove(conf, 'publishing.plugins.kafka_brokers')
        conf = config.remove(conf, 'publishing.plugins.publish_tracks')
        conf = config.remove(conf, 'publishing.plugins.publish_features')
        
        config.update_values(conf, request, "show")
        
        config.update(conf, "camera", request.camera)
        cam = camera.load_camera(**dict(conf.camera))
        
        options = config.filter(conf, 'show', 'output_video', 'progress', 'title')
        img_proc = camera.create_image_processor(camera=cam, context=context, **options)
        
        _, node_track_pipeline = build_node_processor(img_proc, conf)
        
        tracks_channel = config.get(request, "channels.node-tracks")
        node_track_pipeline.add_listener(RedisEventPublisher(redis=context.redis, channel=tracks_channel))
        
        report_interval = config.get(request, 'report_frame_interval', default=-1)
        if report_interval > 0:
            img_proc.add_frame_processor(RedisExecutionProgressReporter(context, report_interval))

        return img_proc, node_track_pipeline
            

class ControlProcessor(AbstractExecution):
    def __init__(self, conn:redis.Redis, control_channel:str, image_processor:ImageProcessor,
                 *, logger:Optional[logging.Logger]=None):
        super().__init__()
        
        self.conn = conn
        self.control_channel = control_channel
        self.image_processor = image_processor
        self.serde = JsonSerde()
        self.logger = logger
        
        self.last_frame = None
        
    def run_work(self) -> None:
        with self.conn.pubsub() as pubsub:
            pubsub.subscribe([self.control_channel])
            pubsub.get_message(timeout=1)
            
            while True:
                self.check_stopped()
                
                control_req = pubsub.get_message(timeout=1.0)
                if control_req:
                    control_req = self.serde.deserialize(control_req['data'])
                    control_req = OmegaConf.create(control_req)
                    if config.exists(control_req, 'action') and control_req.action == 'stop':
                        details = config.get(control_req, 'details', default='client requests')
                        self.image_processor.stop(details=details, nowait=True)
                    else:
                        if self.logger:
                            self.logger.warn(f'unknown control request: {control_req}')
    def finalize(self) -> None: pass
    def stop_work(self) -> None: pass


class RedisExecutionProgressReporter(FrameProcessor):
    def __init__(self, context:RedisExecutionContext, report_frame_interval:int) -> None:
        super().__init__()
        self.ctx = context
        self.report_frame_interval = report_frame_interval

    def on_started(self, proc:ImageProcessor) -> None:
        self.next_report_frame = self.report_frame_interval

    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if frame.index >= self.next_report_frame:
            progress = {
                'frame_index': frame.index,
                # 'ts': round(frame.ts * 1000)
                'ts': frame.ts
            }
            self.ctx.report_progress(progress)
            self.next_report_frame += self.report_frame_interval
        return frame