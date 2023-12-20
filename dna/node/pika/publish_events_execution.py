from __future__ import annotations
import heapq
import time

import pika
from kafka import KafkaProducer

import dna
from dna.event import NodeTrack
from dna.execution import AbstractExecution, Execution, ExecutionContext, ExecutionFactory
from dna.pika_execution import PikaExecutionContext, PikaExecutionFactory, PikaExecutionContext


class TrackEventPublishingExecution(AbstractExecution):
    def __init__(self, context: ExecutionContext, topic:str, kafka_brokers:list[str], sync:bool=True) -> None:
        super().__init__()

        self.ctx = context
        self.log_path = context.request.rtsp_uri
        self.topic = topic
        self.kafka_brokers = kafka_brokers
        self.sync = sync
        
    def run_work(self) -> object:
        interval = 60 * 60
        if dna.conf.exists_config(self.ctx.request, 'progress_report.interval_seconds'):
            interval = int(self.ctx.request.progress_report.interval_seconds)
        started = time.time()
        next_report_time = started + interval
        
        producer = KafkaProducer(bootstrap_servers=self.kafka_brokers)
        try:
            heap = []
            heapq.heapify(heap)
            record_count = 0

            with open(self.log_path, 'r') as fp:
                last_ts = 0
                while True:
                    line = fp.readline().rstrip()
                    if len(line) > 0:
                        te = NodeTrack.from_json(line)
                        heapq.heappush(heap, te)
                    elif len(heap) == 0:
                        break
                    
                    if len(heap) >= 32 or len(line) == 0:
                        track: NodeTrack = heapq.heappop(heap)
                        if self.sync and last_ts > 0:
                            remains = track.ts - last_ts
                            if remains > 30:
                                time.sleep(remains / 1000.0)
                        producer.send(self.topic, value=track.serialize(), key=track.key())
                        producer.flush()
                        last_ts = track.ts

                        record_count += 1
                        now = time.time()
                        if now >= next_report_time:
                            self.ctx.report_progress({"frame_index": track.frame_index})
                            next_report_time += interval
        finally:
            producer.close()


from omegaconf import OmegaConf
class PikaEventPublisherFactory(PikaExecutionFactory):
    def __init__(self, topic:str, kafka_brokers:list[str], sync:bool=True) -> None:
        super().__init__()

        self.topic = topic
        self.kafka_brokers = kafka_brokers
        self.sync = sync

    def create(self, pika_ctx: PikaExecutionContext) -> TrackEventPublishingExecution:
        conf = OmegaConf.create(pika_ctx.request)
        
        return TrackEventPublishingExecution(context=pika_ctx, topic=self.topic,
                                             kafka_brokers=self.kafka_brokers, sync=self.sync)