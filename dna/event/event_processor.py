from __future__ import annotations

from abc import ABCMeta, abstractmethod


class EventListener(metaclass=ABCMeta):
    @abstractmethod
    def handle_event(self, ev:object) -> None:
        pass

    def close(self) -> None:
        pass


class EventQueue:
    def __init__(self) -> None:
        self.listeners:list[EventListener] = []

    def close(self) -> None:
        for sub in self.listeners:
            sub.close()

    def add_listener(self, listener:EventListener) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener:EventListener) -> bool:
        if listener in self.listeners:
            self.listeners.remove(listener)
            return True
        else:
            return False

    def _publish_event(self, ev:object) -> None:
        for sub in self.listeners:
            sub.handle_event(ev)


class EventProcessor(EventListener, EventQueue):
    def __init__(self) -> None:
        EventListener.__init__(self)
        EventQueue.__init__(self)

    def close(self) -> None:
        EventQueue.close(self)
        EventListener.close(self)