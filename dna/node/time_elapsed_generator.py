from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import threading

from dna.event import EventQueue
from dna.utils import utc_now_millis


@dataclass(frozen=True, slots=True, unsafe_hash=True)
class TimeElapsed:
    ts: int = field(default_factory=utc_now_millis)


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
            self.publish_event(TimeElapsed())