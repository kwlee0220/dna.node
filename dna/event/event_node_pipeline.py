from __future__ import annotations

from typing import Any, overload
from collections.abc import Sequence

from .event_processor import EventListener, EventNode, EventNodeImpl


class EvenNodePipeline(EventNode, Sequence[EventNode]):
    __slots__ = ( '__nodes', '__gateway' )
    
    def __init__(self):
        super().__init__()
        
        self.__nodes:list[EventNode] = []
        self.__gateway = EventNodeImpl()
        
    # @override EventListener.handle_event
    def handle_event(self, ev:Any) -> None:
        if len(self.__nodes) > 0:
            self.__nodes[0].handle_event(ev)
    
    # @override EventListener.on_completed
    def on_completed(self) -> None:
        # 각 구성 EventNode의 'on_complete()'를 호출하는 과정에서
        # 이벤트가 publish될 수 있기 때문에 gateway에 대한 on_complete() 함수를
        # 호출하기 전에 구성 EventNode에 대한 on_complete()를 먼저 호출한다.
        for node in self.__nodes:
            node.on_completed()
        self.__gateway.on_completed()
        
    # @override EventPublisher.add_listener
    def add_listener(self, listener:EventListener) -> None:
        self.__gateway.add_listener(listener)

    # @override EventPublisher.remove_listener
    def remove_listener(self, listener:EventListener) -> bool:
        return self.__gateway.remove_listener(listener)
        
    # @override Sequence.__len__
    def __len__(self) -> int:
        return len(self.__nodes)
    
    @overload
    def __getitem__(self, idx:int) -> EventNode: pass
    @overload
    def __getitem__(self, idx:slice) -> list[EventNode]: pass
    @overload
    def __getitem__(self, idx:type[EventNode]) -> EventNode: pass
    # @override Sequence.__getitem__
    def __getitem__(self, idx:int|slice|type[EventNode]) -> EventNode|list[EventNode]:
        if isinstance(idx, int):
            return self.__nodes[idx]
        elif isinstance(idx, EventNode):
            return [node for node in self.__nodes if isinstance(node, idx)]
        elif isinstance(idx, slice):
            return self.__nodes[idx]
        else:
            raise ValueError(f"unknown item key: {idx}")
        
    def append(self, node:EventNode) -> None:
        if len(self.__nodes) > 0:
            last = self.__nodes[-1]
            last.remove_listener(self.__gateway)
            last.add_listener(node)
            
        node.add_listener(self.__gateway)
        self.__nodes.append(node)