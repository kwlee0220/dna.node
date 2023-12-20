from __future__ import annotations

from omegaconf import OmegaConf
import pika

from dna import utils, ByteString
from dna.execution import ExecutionContext, ExecutionState
from .pika_rpc import JSON_SERDE


class PikaExecutionContext(ExecutionContext):
    def __init__(self, request:ByteString|OmegaConf, channel, method, reply_to:str,
                 properties:pika.BasicProperties) -> None:
        self.request = request
        self.channel = channel
        self.tag = method.delivery_tag
        self.reply_to = reply_to
        self.response_props = pika.BasicProperties(correlation_id=properties.correlation_id)
        self.serde = JSON_SERDE

        self.control_qname = None
        self.acked = False
        
    @property
    def id(self) -> str:
        if isinstance(self.request, bytes):
            return 'unknown'
        else:
            return self.request.id

    def started(self) -> None:
        started = {
            'id': self.id,
            'state': 'STARTED',
            'timestamp': utils.utc_now_millis(),
            'control_queue': self.control_qname
        }
        self.reply(started)
        self.ack()
        

    def report_progress(self, progress:object) -> None:
        progress = {
            'id': self.id,
            'state': ExecutionState.RUNNING.name,
            'timestamp': utils.utc_now_millis(),
            'progress': progress
        }
        self.reply(progress)

    def completed(self, result:object) -> None:
        completed = {
            'id': self.id,
            'state': ExecutionState.COMPLETED.name,
            'timestamp': utils.utc_now_millis(),
            'result': result
        }
        self.reply(completed)
        self.ack()

    def stopped(self, details:str) -> None:
        failed = {
            'id': self.id,
            'state': ExecutionState.STOPPED.name,
            'timestamp': utils.utc_now_millis(),
            'cause': details
        }
        self.reply(failed)
        self.ack()

    def failed(self, cause:str|Exception) -> None:
        failed = {
            'id': self.id,
            'state': ExecutionState.FAILED.name,
            'timestamp': utils.utc_now_millis(),
            'cause': repr(cause)
        }
        self.reply(failed)
        self.ack()
            
    def ack(self) -> None:
        if not self.acked:
            self.channel.basic_ack(delivery_tag=self.tag)
            self.acked = True

    def reply(self, data:object) -> None:
        if self.reply_to is not None:
            self.channel.basic_publish(exchange='', routing_key=self.reply_to,
                                       properties=self.response_props,
                                       body=self.serde.serialize(data))