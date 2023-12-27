from __future__ import annotations

from typing import Optional

from .event_processor import EventNodeImpl, EventListener
from .event_processors import EventRelay


class MultiStagePipeline(EventNodeImpl):
    def __init__(self) -> None:
        super().__init__()
        
        self.stages:dict[str,EventNodeImpl] = dict()
        self.output = EventRelay(self)

    def close(self) -> None:
        for stage in self.stages.values():
            stage.close()
        super().close()

    def add_stage(self, name:str, proc:EventNodeImpl) -> None:
        if len(self.stages) > 0:
            # 이전 마지막 stage에서 본 listener를 떼고, 새로 추가된 processor에 등록시킨다.
            last_stage_name = list(self.stages)[-1]
            last_stage = self.stages[last_stage_name]
            last_stage.remove_listener(self.output)
            last_stage.add_listener(proc)
            
        self.stages[name] = proc
        proc.add_listener(self.output)
            
    def handle_event(self, ev:object) -> None:
        # 첫번째 stage에만 event를 발송하면 pipeline 전체에 event가 전파된다.
        if len(self.stages) > 0:
            first_stage_name = list(self.stages)[0]
            self.stages[first_stage_name].handle_event(ev)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}[' + " -> ".join(repr(stage) for stage in self.stages) + ']'