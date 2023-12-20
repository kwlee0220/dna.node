from __future__ import annotations

from typing import Optional

import logging
import threading

import pika

from dna import config
from dna.camera import Frame, ImageProcessor, FrameProcessor
from dna.execution import AbstractExecution
from .pika_execution import PikaExecutionFactory, PikaExecutionContext, PikaConnector
from .pika_rpc import JSON_SERDE


class PikaExecutionServer:
    def __init__(self, connector:PikaConnector,
                 execution_factory:PikaExecutionFactory,
                 request_qname:str,
                 logger:logging.Logger) -> None:
        self.connector = connector
        self.req_qname = request_qname
        self.exec_factory = execution_factory
        self.serde = JSON_SERDE
        self.logger = logger

    def run(self) -> None:
        self.conn = self.connector.blocking_connection()
        self.channel = self.conn.channel()
        self.channel.queue_declare(queue=self.req_qname)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.req_qname, on_message_callback=self.on_request)
        self.channel.start_consuming()

    def on_request(self, channel, method, props:pika.BasicProperties, body:object) -> None:
        pika_ctx = PikaExecutionContext(body, channel, method, props.reply_to, props)
        try:
            req = self.serde.deserialize(body)
            req = config.to_conf(req)
            
            if config.get(req, 'action') is None:
                raise ValueError(f"'action' is not specified. msg={req}")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"received a request: node={req.node}, action={req.action}")
                
            if req.action != "start":
                return
            
            pika_ctx.request = req
            if config.get(req, 'id') is None:
                raise ValueError(f"'id' is not specified. msg={req}")
            
            img_proc = self.exec_factory.create(pika_ctx)
            
            handler = ControlRequestHandler(self.connector, pika_ctx, img_proc, logger=self.logger)
            img_proc.add_frame_processor(handler)
            
            control_thread = threading.Thread(target=handler.run)
            control_thread.start()
            
            img_proc.run()
        except Exception as e:
            self.logger.error(f'fails to create execution: cause={e}')
            pika_ctx.failed(e)

class ControlRequestHandler(AbstractExecution, FrameProcessor):
    def __init__(self, connector:PikaConnector, pika_ctx:PikaExecutionContext, image_processor:ImageProcessor,
                 *, logger:Optional[logging.Logger]=None):
        super(ControlRequestHandler, self).__init__()
        
        conn = connector.blocking_connection()
        self.channel = conn.channel()
        self.qname = self.channel.queue_declare(queue='', exclusive=True).method.queue
        pika_ctx.control_qname = self.qname
        self.image_processor = image_processor
        self.serde = JSON_SERDE
        self.logger = logger
        
    def run_work(self) -> None:
        for _, _, body in self.channel.consume(queue=self.qname, auto_ack=True):
            try:
                req = self.serde.deserialize(body)
                req = config.to_conf(req)
                if config.exists(req, 'action') and req.action == 'stop':
                    details = config.get(req, 'details', default='client requests')
                    self.image_processor.stop(details=details, nowait=True)
            except Exception as e:
                if self.logger:
                    self.logger.error(f'fails to handle control message: {body}, cause={e}')
            break
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'ControlRequestHandler jumps out-out loop.')
            
    def finalize(self) -> None:
        self.channel.queue_delete(queue=self.qname)
        self.channel.connection.close()
        
    def stop_work(self) -> None:
        self.channel.cancel()

    def on_started(self, proc:ImageProcessor) -> None: pass

    def on_stopped(self) -> None:
        self.stop("cancel at server-side", nowait=True)
        pass
        
    def process_frame(self, frame:Frame) -> Optional[Frame]: return frame