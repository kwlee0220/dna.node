from __future__ import annotations

from typing import Union, Optional
from collections.abc import Iterable, Generator
import logging

from dna import NodeId, TrackletId
from dna.support import iterables
from dna.event import EventProcessor, NodeTrack, TimeElapsed


class Window:
    __slots__ = ('begin_millis', 'end_millis', 'events')
    
    def __init__(self, begin_millis:int, end_millis:int) -> None:
        self.begin_millis = begin_millis
        self.end_millis = end_millis
        self.events:list[NodeTrack] = []
        
    def range_contains(self, ev:NodeTrack) -> bool:
        return ev.ts >= self.begin_millis and ev.ts < self.end_millis
        
    def append(self, te:NodeTrack) -> None:
        self.events.append(te)
            
    def remove(self, trk_id:TrackletId) -> None:
        """주어진 tracklet id를 갖는 모든 NodeTrackEvent를 삭제한다.

        Args:
            trk_id (TrackletId): 삭제 대상 tracklet의 식별자.
        """
        self.events = [te for te in self.events if te.tracklet_id != trk_id]
        
    def __len__(self) -> int:
        return len(self.events)
        
    def __bool__(self) -> bool:
        return bool(self.events)
    
    def __iter__(self):
        return (ev for ev in self.events)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: time={self.begin_millis}-{self.end_millis}, "
                f"length={len(self.events)}")
        

class TumblingWindowAssigner(EventProcessor):
    __slots__ = ( 'schema', 'max_ts', 'windows', 'dirty', 'logger' )
    
    def __init__(self, nodes:Iterable[NodeId],
                 window_interval_ms:int,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.max_ts:dict[NodeId,int] = {node_id:0 for node_id in nodes}
        self.window_interval_ms = window_interval_ms
        self.windows:list[Window] = []
        self.dirty = True
        self.logger = logger
        
    def close(self) -> None:
        # 잔여 window에 포함된 NodeTrackEvent에 대해 association을 생성한다.
        for window in self.windows:
            self._publish_event((window, None))
        self.windows.clear()
            
        super().close()
        
    def handle_event(self, ev:Union[NodeTrack,TimeElapsed]) -> None:
        if isinstance(ev, NodeTrack):
            self.dirty = True
            
            # 너무 늦게 도착한 NodeTrackEvent인 경우는 무시한다.
            if ev.node_id not in self.max_ts:
                return
            
            if ev.is_confirmed() or ev.is_tentative():
                if ev.ts > self.max_ts[ev.node_id]:
                    self.max_ts[ev.node_id] = ev.ts
                
                # 본 NodeTrackEvent를 해당 window에 추가하고, close된 window들에 대해 association들을 생성한다.
                self.fill_window(ev)
                for window in self._list_closed_windows():
                    self._publish_event((window, None))
            elif ev.is_deleted():
                # 대상 NodeTrackEvent 생성 시점보다 빠른 영역을 갖는 모든 window에 대해
                # 종료된 tracklet과 관련된 association을 생성하고, 관련 window들에서
                # 종료된 tracklet에 의해 생성된 NodeTrackEvent를 모두 삭제한다.
                for window in self.windows:
                    if ev.ts < window.begin_millis:
                        break
                    self._publish_event((window, ev.tracklet_id))
                    window.remove(ev.tracklet_id)
                self._publish_event(ev)
        elif isinstance(ev, TimeElapsed):
            # 바로 전 TimeElapsed 이벤트를 받은 후부터 지금까지 한번도 NodeTrackEvent를 받지 않은 경우에는
            # 가장 오래된 window를 통한 association을 생성하고, 해당 window를 제거한다.
            if not self.dirty and self.windows:
                window = self.windows.pop(0)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"publish idle window: {window}")
                self._publish_event((window, None))
            self.dirty = False
                        
    def fill_window(self, te:NodeTrack) -> None:
        if self.windows:
            if te.ts < self.windows[0].begin_millis:
                # 가장 오래된 (첫번째) window의 시간 구간보다도 일찍 생성된 event는 누락시킨다.
                if self.logger and self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f"drop too-late event: {te}")
                return
            
            while True:
                # NodeTrackEvent의 timestamp를 기준으로 이벤트가 포함될 window를 검색한다.
                window = iterables.first((w for w in self.windows if w.range_contains(te)))
                if window is not None:
                    window.append(te)
                    return
                
                # 해당 window가 존재하지 않는 경우는 다음 time slot의 window를 생성하여 추가함.
                last_window = self.windows[-1]
                next_window = Window(begin_millis=last_window.end_millis,
                                     end_millis=last_window.end_millis+self.window_interval_ms)
                self.windows.append(next_window)
        else:
            # 첫번째 호출이어서 첫 window를 생성하여 이벤트를 추가함.
            window = Window(begin_millis=te.ts, end_millis=te.ts + self.window_interval_ms)
            window.append(te)
            self.windows.append(window)
        
    MARGIN_DURATION_MS = 2 * 1000
    def _list_closed_windows(self) -> Generator[Window, None, None]:
        min_ts = min(self.max_ts.values(), default=0)
        max_ts = max(self.max_ts.values())  # 가장 최근에 추가된 시각
        while self.windows:
            first_window = self.windows[0]
            if first_window.end_millis < min_ts:
                # 첫번재 window의 시간 구간에 포함된 이벤트가 없는 경우
                # 첫 window를 close 시킨다.
                self.windows.pop(0)
                yield first_window
            elif first_window.end_millis < max_ts - TumblingWindowAssigner.MARGIN_DURATION_MS:
                # 가장 최근에 생성된 이벤트가 첫번째 window의 시간 구간(MARGIN)보다 일정 기간보다 더 최근인 경우.
                # 이후에 추가될 이벤트가 첫번째 window에 할당될 경우가 거의 없기 때문에 해당 window를
                # 종료시키고 window에 포함된 이벤트들을 반환한다.
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"flush frozen window: {first_window}, max_ts={max_ts}")
                self.windows.pop(0)
                yield first_window
            else:
                return