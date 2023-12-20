from __future__ import annotations

from collections.abc import Generator

import pika
import uuid

from dna.execution import ExecutionState, CancellationError
from .pika_execution import PikaConnector
from .pika_rpc import RpcCallError, JSON_SERDE


class PikaExecutionClient:
    def __init__(self, connector:PikaConnector, request_qname:str) -> None:
        self.req_qname = request_qname
        self.serde = JSON_SERDE

        self.conn = connector.blocking_connection()
        self.channel = self.conn.channel()

        self.callback_qname = self.channel.queue_declare(queue='', exclusive=True).method.queue
        self.channel.basic_consume(queue=self.callback_qname, on_message_callback=self.on_response,
                                   auto_ack=True)
        self.channel.queue_declare(queue=self.req_qname)

        self.response = None
        self.corr_id = None
        self.control_qname = None

    def close(self):
        try:
            self.channel.queue_delete(queue=self.callback_qname)
            self.conn.close()
        finally:
            self.conn = None
            
    @classmethod
    def from_url(cls, url:str) -> PikaExecutionClient:
        connector, request_qname = PikaConnector.parse_url(url)
        return cls(connector, request_qname)

    def start(self, request:object) -> None:
        if self.conn is None:
            raise AssertionError(f"{__name__} is closed already")
        
        self.corr_id = str(uuid.uuid4())
        props = pika.BasicProperties(reply_to=self.callback_qname, correlation_id=self.corr_id)
        self.channel.basic_publish(exchange='', routing_key=self.req_qname, properties=props,
                                   body=self.serde.serialize(request))
        
    def stop(self, id:str, details:str='client request') -> None:
        if self.conn is None:
            raise AssertionError(f"{__name__} is closed already")
        if self.control_qname is None:
            raise AssertionError(f"Control queue name is not defined.")
        
        props = pika.BasicProperties()
        request = {
            "id": id,
            "action": "stop",
            "details": details
        }
        self.channel.basic_publish(exchange='', routing_key=self.control_qname, properties=props,
                                   body=self.serde.serialize(request))
        
    def run(self, request:object) -> dict[str,object]:
        self.start(request)
        
        *_, last_report = self.report_progress()
        if last_report['state'] == ExecutionState.COMPLETED.name:
            return last_report['result']
        elif last_report['state'] == ExecutionState.STOPPED.name:
            raise CancellationError(last_report['cause'])
        elif last_report['state'] == ExecutionState.FAILED.name:
            raise RpcCallError(last_report['cause'])
        else:
            raise AssertionError(f"unexpected status message: {last_report['state']}")
            
    def report_progress(self) -> Generator[dict,None,None]:
        while True:
            self.conn.process_data_events(time_limit=None)
            state = self.response['state']
            if state == 'STARTED':
                self.control_qname = self.response.get('control_queue', None)
                yield self.response
            elif state == ExecutionState.RUNNING.name:
                yield self.response
            elif state == ExecutionState.COMPLETED.name \
                or state == ExecutionState.STOPPED.name \
                or state == ExecutionState.FAILED.name:
                yield self.response
                return
            else:
                raise AssertionError(f'unexpected progress message: {self.response}')

    def on_response(self, channel, method, props:pika.BasicProperties, body:object):
        if props.correlation_id in self.corr_id:
            self.response = self.serde.deserialize(body)