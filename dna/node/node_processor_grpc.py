from __future__ import annotations
from typing import Optional
from abc import ABCMeta, abstractmethod
import threading

from concurrent import futures
import logging
from omegaconf import OmegaConf
import uuid

import grpc
from .proto import node_processor_pb2
from .proto import node_processor_pb2_grpc

from dna import config, utils, camera
from dna.camera import FrameProcessor
from dna.node.node_processor import build_node_processor


class StatusReporter(FrameProcessor):
    __slots__ = ('proc_id', 'report_interval', 'remains', 'lock', 'cond', 'status')
    
    def __init__(self, proc_id:str, report_interval:int=30) -> None:
        super().__init__()
        self.proc_id = proc_id
        self.report_interval = report_interval
        self.remains = report_interval
        
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.status = None
        
    def on_started(self, proc:ImageProcessor) -> None:
        with self.lock:
            started = node_processor_pb2.StartedStatus(ts=utils.utc_now_millis())
            self.status = node_processor_pb2.StatusReport(proc_id=self.proc_id,
                                                          status=node_processor_pb2.Status.STARTED,
                                                          started=started)
            self.cond.notify_all()
            self.remains = self.report_interval
        
    def on_stopped(self) -> None: 
        with self.lock:
            stopped = node_processor_pb2.StoppedStatus(ts=utils.utc_now_millis())
            self.status = node_processor_pb2.StatusReport(proc_id=self.proc_id,
                                                          status=node_processor_pb2.Status.STOPPED,
                                                          stopped=stopped)
            self.cond.notify_all()
            self.remains = self.report_interval

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        self.remains -= 1
        if self.remains == 0:
            running = node_processor_pb2.RunningStatus(frame_index=frame.index, ts=round(frame.ts*1000))
            with self.lock:
                self.status = node_processor_pb2.StatusReport(proc_id=self.proc_id,
                                                              status=node_processor_pb2.Status.RUNNING,
                                                              running=running)
                self.cond.notify_all()
            self.remains = self.report_interval
        
        return frame
    
    def wait_for_change(self, last_status:node_processor_pb2.StatusReport) -> node_processor_pb2.StatusReport:
        with self.lock:
            self.cond.wait_for(lambda: self.status != last_status)
            return self.status
            
    
class NodeProcessorServicer(node_processor_pb2_grpc.NodeProcessorServicer):
    def __init__(self) -> None:
        super().__init__()
        
    def Run(self, request, context):
        conf = OmegaConf.create()
        if request.WhichOneof('conf') == 'node_id':
            conf = config.load(f'conf/etri_testbed/{request.node_id}.yaml')
        elif request.WhichOneof('conf') == 'conf_path':
            conf = config.load(request.conf_path)
                
        camera_conf = conf.camera
        camera_conf.sync = request.sync if request.HasField('sync') else False
        if request.HasField('camera_uri'):
            camera_conf.uri = request.camera_uri
        
        cam = camera.load_camera(**dict(camera_conf))
        
        proc_id = str(uuid.uuid4())
        status_reporter = StatusReporter(proc_id=proc_id)
        last_status = status_reporter.status
        
        kwargs = {'frame_processor':status_reporter, 'show':True}
        thread = threading.Thread(target=camera.process_images, kwargs=kwargs)
        thread.start()
        
        while True:
            last_status = status_reporter.wait_for_change(last_status)
            yield last_status
            
            if last_status.status == node_processor_pb2.Status.FINISHED \
                or last_status.status == node_processor_pb2.Status.STOPPED:
                break
        thread.join()
    
    def Stop(self, request, context):
        pass
    
    
def main():
    service = NodeProcessorServicer()
    
    request = node_processor_pb2.RunNodeProcessRequest(node_id='etri_04')
    
    service.Run()
    
    listener = threading.Thread(listen, args=(reporter,))
    listener.start()
    
    reporter.on_started(proc=None)
    for idx in range(1, 11):
        reporter.process_frame(Frame(None, idx, idx))
    reporter.on_stopped()
    
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    # node_processor_pb2_grpc.add_NodeProcessorServicer_to_server(NodeProcessorServicer(), server)
    # server.add_insecure_port('[::]:50051')
    # server.start()
    # server.wait_for_termination()
    
def listen(reporter:StatusReporter) -> None:
    last_status = None
    while True:
        last_status = reporter.wait_for_change(last_status=last_status)
        if last_status.status == node_processor_pb2.Status.RUNNING:
            print(f'status changed: {last_status.running}')
        elif last_status.status == node_processor_pb2.Status.STARTED:
            print(f'process started: {last_status.started}')
        elif last_status.status == node_processor_pb2.Status.FINISHED \
            or last_status.status == node_processor_pb2.Status.STOPPED:
            print(f'process done: {last_status.stopped}')
            break
    
    
if __name__ == '__main__':
    logging.basicConfig()
    main()