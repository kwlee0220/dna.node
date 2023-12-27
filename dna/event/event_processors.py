from __future__ import annotations


from .event_processor import EventNodeImpl, EventListener, EventQueue


class PrintEvent(EventListener):
    def __init__(self, header:str='') -> None:
        super().__init__()
        self.header = header
        
    def on_completed(self) -> None: pass
    def handle_event(self, ev: object) -> None:
        print(f"[{self.header}] {ev}")           


class EventRelay(EventListener):
    def __init__(self, target:EventQueue) -> None:
        self.target = target
    
    def handle_event(self, ev:object) -> None:
        self.target.publish_event(ev)
        
    def on_completed(self) -> None:
        self.target.close()


class DropEventByType(EventNodeImpl):
    def __init__(self, event_types:list[type]) -> None:
        super().__init__()
        self.drop_list = event_types

    def handle_event(self, ev:object) -> None:
        if not any(ev_type for ev_type in self.drop_list if isinstance(ev, ev_type)):
            self.publish_event(ev)
    
    def __repr__(self) -> str:
        types_str = ",".join(ev_type.__name__ for ev_type in self.drop_list)
        return f"DropEventByType(types={types_str})"