from __future__ import annotations

import threading
from datetime import timedelta

from .types import TimeElapsed
from .event_processor import EventProcessor, EventListener, EventQueue


class PrintEvent(EventListener):
    def __init__(self, header:str='') -> None:
        super().__init__()
        self.header = header
        
    def close(self) -> None: pass
    def handle_event(self, ev: object) -> None:
        print(f"[{self.header}] {ev}")           

class EventRelay(EventListener):
    def __init__(self, target:EventQueue) -> None:
        self.target = target
        
    def close(self) -> None:
        pass
    
    def handle_event(self, ev:object) -> None:
        self.target._publish_event(ev)


class TimeElapsedGenerator(threading.Thread, EventQueue):
    def __init__(self, interval:timedelta):
        threading.Thread.__init__(self)
        EventQueue.__init__(self)
        
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        
    def stop(self):
        self.stopped.set()
        # self.join()
        
    def run(self):
        while not self.stopped.wait(self.interval.total_seconds()):
            self._publish_event(TimeElapsed())


class DropEventByType(EventProcessor):
    def __init__(self, event_types:list[type]) -> None:
        super().__init__()
        self.drop_list = event_types

    def handle_event(self, ev:object) -> None:
        if not any(ev_type for ev_type in self.drop_list if isinstance(ev, ev_type)):
            self._publish_event(ev)
    
    def __repr__(self) -> str:
        types_str = ",".join(ev_type.__name__ for ev_type in self.drop_list)
        return f"DropEventByType(types={types_str})"



class FilterEventByType(EventProcessor):
    def __init__(self, event_types:list[type]) -> None:
        super().__init__()
        self.keep_list = event_types

    def handle_event(self, ev:object) -> None:
        if any(ev_type for ev_type in self.keep_list if isinstance(ev, ev_type)):
            self._publish_event(ev)