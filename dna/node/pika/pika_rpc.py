from abc import ABCMeta, abstractmethod

import json
import uuid
import pika

class Serde(metaclass=ABCMeta):
    @abstractmethod
    def serialize(self, data:object) -> bytes: pass

    @abstractmethod
    def deserialize(self, body:bytes) -> object: pass


class NoOpSerde(Serde):
    def serialize(self, data:object) -> bytes:
        return data

    def deserialize(self, body:bytes) -> object:
        return body


class JsonSerde(Serde):
    def deserialize(self, body:bytes) -> object:
        json_str = body.decode('utf-8')
        return json.loads(json_str)

    def serialize(self, resp:object) -> bytes:
        if isinstance(resp, str):
            resp = resp.encode('utf-8')
        elif not isinstance(resp, bytes):
            resp = json.dumps(resp, default=lambda o: o.__dict__).encode('utf-8')
        return resp


NO_OP_SERDE = NoOpSerde()
JSON_SERDE = JsonSerde()

_REQ_QUEUE = 'rpc_requests'

class RpcCallError(Exception):
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(message)


def ConnectionParameters(host:str='localhost', port:int=5672, user_id='dna', password='') -> pika.ConnectionParameters:
    credentials = pika.PlainCredentials(username=user_id, password=password)
    return pika.ConnectionParameters(host=host, port=port, credentials=credentials)


class RpcClient:
    def __init__(self, conn_params, request_queue:str=_REQ_QUEUE, serde:Serde=JSON_SERDE) -> None:
        self.request_queue = request_queue
        self.serde = serde

        self.conn = pika.BlockingConnection(conn_params)
        self.channel = self.conn.channel()

        self.callback_queue = self.channel.queue_declare(queue='', exclusive=True).method.queue
        self.channel.basic_consume(queue=self.callback_queue, on_message_callback=self.on_response, auto_ack=True)
        self.req_queue = self.channel.queue_declare(queue=self.request_queue)

        self.response = None
        self.corr_id = None

    def close(self):
        try:
            self.conn.close()
        finally:
            self.conn = None

    def call(self, request:object, timeout:int=None) -> object:
        if self.conn is None:
            raise AssertionError(f"{__name__} is closed already")
        
        self.corr_id = str(uuid.uuid4())
        props = pika.BasicProperties(reply_to=self.callback_queue, correlation_id=self.corr_id)

        self.response = None
        self.channel.basic_publish(exchange='', routing_key=self.request_queue, properties=props,
                                    body=self.serde.serialize(request))
        self.conn.process_data_events(time_limit=timeout)

        return self.response

    def on_response(self, channel, method, props:pika.BasicProperties, body:object) -> object:
        if self.corr_id ==  props.correlation_id:
            self.response = self.serde.deserialize(body)


class RpcServer(metaclass=ABCMeta):
    def __init__(self, conn_params:pika.ConnectionParameters, request_queue:str=_REQ_QUEUE,
                serde:Serde=JSON_SERDE) -> None:
        self.conn_params = conn_params
        self.request_queue = request_queue
        self.serde = serde

    def run(self) -> None:
        self.conn = pika.BlockingConnection(self.conn_params)
        self.channel = self.conn.channel()
        self.req_queue = self.channel.queue_declare(queue=self.request_queue)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.request_queue, on_message_callback=self.on_request)
        self.channel.start_consuming()

    @abstractmethod
    def call(self, body:object) -> object:
        pass

    def on_request(self, channel, method, props:pika.BasicProperties, body:object) -> None:
        req = self.serde.deserialize(body)
        response = self.call(req)
        resposne = self.serde.serialize(response)

        resp_props = pika.BasicProperties(correlation_id=props.correlation_id)
        channel.basic_publish(exchange='', routing_key=props.reply_to, properties=resp_props,  body=response)
        channel.basic_ack(delivery_tag=method.delivery_tag)