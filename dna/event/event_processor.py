from __future__ import annotations

from typing import Optional, Any, TypeVar
from collections.abc import Iterable
from abc import ABC, abstractmethod
from contextlib import suppress


class EventListener(ABC):
    @abstractmethod
    def handle_event(self, ev:Any) -> None:
        pass
    
    @abstractmethod
    def on_completed(self) -> None:
        pass
    

class EventPublisher(ABC):
    @abstractmethod
    def add_listener(self, listener:EventListener) -> None:
        pass

    @abstractmethod
    def remove_listener(self, listener:EventListener) -> bool:
        pass


class EventProcessor(ABC):
    @abstractmethod
    def process(self, ev:Any) -> Iterable[Any]:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass


class EventQueue(EventPublisher):
    __slots__ = ( '__listeners', '__closed' )
    
    def __init__(self) -> None:
        self.__listeners:list[EventListener] = []
        self.__closed = False

    def close(self) -> None:
        if not self.__closed:
            for sub in self.__listeners:
                with suppress(Exception): sub.on_completed()
            self.__closed = True
            
    @property
    def listeners(self) -> list[EventListener]:
        return self.__listeners

    def publish_event(self, ev:Any) -> None:
        for sub in self.__listeners:
            sub.handle_event(ev)

    def add_listener(self, listener:EventListener) -> None:
        self.__listeners.append(listener)

    def remove_listener(self, listener:EventListener) -> bool:
        if listener in self.listeners:
            self.__listeners.remove(listener)
            return True
        else:
            return False
        
    def clear_listeners(self) -> None:
        self.__listeners.clear()


class EventNode(EventListener, EventPublisher):
    pass

EventNodeT = TypeVar("EventNodeT", bound=EventNode)


class EventNodeImpl(EventNode):
    __slots__ = ( '__queue', )
    
    def __init__(self) -> None:
        self.__queue = EventQueue()
        
    def handle_event(self, ev:Any) -> None:
        self.publish_event(ev)

    def publish_event(self, ev:Any) -> None:
        self.__queue.publish_event(ev)
    
    def on_completed(self) -> None:
        self.__queue.close()
        
    def add_listener(self, listener:EventListener) -> None:
        self.__queue.add_listener(listener)

    def remove_listener(self, listener:EventListener) -> bool:
        return self.__queue.remove_listener(listener)

    def clear_listeners(self) -> None:
        self.__queue.clear_listeners()