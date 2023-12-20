from __future__ import annotations

from typing import Union, Optional
import sys
from dataclasses import dataclass, field, replace
import logging

from dna import NodeId, TrackId, Box
from dna.event import TimeElapsed, SilentFrame, NodeTrack, EventProcessor
from dna.track import TrackState


@dataclass(eq=True) # slots=True
class Session:
    node_id: NodeId = field(hash=True)
    track_id: TrackId = field(hash=True)
    '''본 세션에 해당하는 track id.'''
    state: TrackState = field(hash=False, compare=False)
    '''본 track session의 상태.'''
    first_ts: int = field(hash=True)
    pendings: list[NodeTrack] = field(hash=False, compare=False)
    '''TrackEvent refinement를 위해 track별로 보류되고 있는 TrackEvent 리스트.'''
    
    @property
    def first_frame_index(self) -> int:
        return self.pendings[0].frame_index if self.pendings else None
        
    def index_of(self, frame_index:int) -> int:
        npendings = len(self.pendings)
        if npendings == 0:
            return -1
        else:
            index = frame_index - self.pendings[0].frame_index
            if index < npendings:
                return index
            elif index == npendings:
                return -1
            else:
                raise ValueError(f'invalid frame_index: {frame_index}, '
                                 f'pendings=[{self.pendings[0].frame_index}-{self.pendings[-1].frame_index}]')

    def trim_right_to(self, frame_index:int) -> None:
        end_index = self.index_of(frame_index)
        if end_index > 0 and end_index < len(self.pendings):
            self.pendings = self.pendings[:end_index]
            
    def __repr__(self) -> str:
        interval_str = ""
        if self.pendings:
            interval_str = f':{self.pendings[0].frame_index}-{self.pendings[-1].frame_index}'
        return f'{self.track_id}({self.state.abbr})[{len(self.pendings)}{interval_str}]'


class RefineTrackEvent(EventProcessor):
    __slots__ = ('sessions', 'buffer_size', 'timeout', 'timeout_millis', 'oldest_pending_session',
                 'max_frame_index', 'max_ts', 'logger')

    def __init__(self, buffer_size:int=30, buffer_timeout:float=1.0,
                 *,
                 delete_on_close: bool=True,
                 logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)

        self.sessions: dict[str, Session] = {}
        self.buffer_size = buffer_size
        self.timeout = buffer_timeout
        self.timeout_millis = round(buffer_timeout * 1000)
        self.delete_on_close = delete_on_close
        self.logger = logger
        self.oldest_pending_session:Session = None
        self.max_frame_index = 0
        self.max_ts = 0

    def close(self) -> None:
        if self.delete_on_close:
            non_closed_sessions = list(self.sessions.values())
            for session in non_closed_sessions:
                deleted = NodeTrack(node_id=session.node_id, track_id=session.track_id,
                                     state=TrackState.Deleted,
                                     bbox=Box([0, 0, 0, 0]),
                                     first_ts=session.first_ts,
                                     frame_index=self.max_frame_index,
                                     ts=self.max_ts)
                self.handle_event(deleted)
        else:
            self.sessions.clear()
        super().close()
        
    def min_frame_index(self) -> Optional[int]:
        """Pending된 TravkEvent들 중에서 가장 작은 frame index를 반환한다.

        Returns:
            Optional[int]: Pending된 TravkEvent들 중에서 가장 작은 frame index.
            만일 pending된 TrackEvent가 없는 경우에는 None.
        """
        if not self.oldest_pending_session:
            pending_sessions = [session for session in self.sessions.values() if session.pendings]
            self.oldest_pending_session = min(pending_sessions, key=lambda s: s.first_frame_index) if pending_sessions else None
        
        return self.oldest_pending_session.first_frame_index if self.oldest_pending_session else None

    def handle_event(self, ev:Union[NodeTrack,TimeElapsed|SilentFrame]) -> None:
        if isinstance(ev, NodeTrack):
            self.max_frame_index = max(self.max_frame_index, ev.frame_index)
            self.max_ts = max(self.max_ts, ev.ts)
            
            self.handle_track_event(ev)
            pass
        elif isinstance(ev, TimeElapsed):
            self.handle_time_elapsed(ev)
        elif isinstance(ev, SilentFrame):
            self.handle_silent_frame(ev)

    def handle_track_event(self, ev:NodeTrack) -> None:
        session:Session = self.sessions.get(ev.track_id)
        if session is None: # TrackState.Null or TrackState.Deleted
            self.__on_initial(ev)
        elif session.state == TrackState.Confirmed:
            self.__on_confirmed(session, ev)
        elif session.state == TrackState.Tentative:
            self.__on_tentative(session, ev)
        elif session.state == TrackState.TemporarilyLost:
            self.__on_temporarily_lost(session, ev)

    def handle_time_elapsed(self, ev:TimeElapsed) -> None:
        for session in self.sessions.values():
            self._publish_old_events(session, ev.ts)
        self._publish_event(ev)

    def handle_silent_frame(self, ev:SilentFrame) -> None:
        for session in self.sessions.values():
            self._publish_old_events(session, ev.ts)
        self._publish_event(ev)
        
    def _remove_session(self, id:TrackId) -> None:
        session = self.sessions.pop(id, None)
        self._unset_oldest_pending_session(session)

    def __on_initial(self, ev:NodeTrack) -> None:
        # track과 관련된 session 정보가 없다는 것은 이 track event가 한 물체의 첫번째 track event라는 것을 
        # 의미하기 때문에 session을 새로 생성한다.
        self.sessions[ev.track_id] = session = Session(node_id=ev.node_id, track_id=ev.track_id, state=ev.state,
                                                       first_ts=ev.first_ts, pendings=[])
        if ev.state == TrackState.Tentative:
            self._append_track_event(session, ev)
        elif ev.state == TrackState.Confirmed:
            self._publish_event(ev)
        elif ev.state == TrackState.Deleted:
            self._remove_session(ev.track_id)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): {ev}")

    def __on_confirmed(self, session:Session, ev:NodeTrack) -> None:
        if ev.state == TrackState.Confirmed:
            self._publish_event(ev)
        elif ev.state == TrackState.TemporarilyLost:
            self._append_track_event(session, ev)
            session.state = TrackState.TemporarilyLost
        elif ev.state == TrackState.Deleted:
            self._remove_session(ev.track_id)
            self._publish_event(ev)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): "
                                f"state={session.state}, event={ev.track}")

    def __on_tentative(self, session:Session, ev:NodeTrack) -> None:
        if ev.state == TrackState.Confirmed:
            # 본 trail을 임시 상태에서 정식의 trail 상태로 변환시키고,
            # 지금까지 pending된 모든 tentative event를 trail에 포함시킨다
            self._publish_all_pended_events(session)
            self._unset_oldest_pending_session(session)
            self._publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.Tentative:
            self._append_track_event(session, ev)
        elif ev.state == TrackState.Deleted:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"discard tentative track: "
                                    f"track_id={ev.track_id}, count={len(session.pendings)}")
            # track 전체를 제거하기 때문에, 'delete' event로 publish하지 않는다.
            self._remove_session(ev.track_id)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): "
                                    f"state={session.state}, event={ev.track}")

    def __on_temporarily_lost(self, session:Session, ev:NodeTrack) -> None:
        if ev.state == TrackState.Confirmed:
            session.trim_right_to(ev.frame_index)
            self._publish_all_pended_events(session)
            self._unset_oldest_pending_session(session)
            self._publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.TemporarilyLost:
            self._append_track_event(session, ev)
            # event buffer가 overflow가 발생하면, overflow되는
            # event 갯수만큼 oldest event를 publish시킨다.
            n_overflows = len(session.pendings) - self.buffer_size
            if n_overflows > 0:
                if self.logger and self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f'overflow TrackEvent refine buffer: length={self.buffer_size}, track={session.track_id}, '
                                     f'range={session.pendings[0].frame_index}-{session.pendings[n_overflows-1].frame_index}')
                for tev in session.pendings[:n_overflows]:
                    self._publish_event(tev)
                session.pendings = session.pendings[n_overflows:]
                self._unset_oldest_pending_session(session)
        elif ev.state == TrackState.Deleted:
            # 기존에 pending 중이던 모든 'temporarily-lost' track event들을 제거한다.
            # 또한 인자로 주어진 'delete' track의 frame_index와 ts를 가장 오랫동안
            # pending되었던 track event의 그것들로 대체시킨다.
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"discard all pending lost track events: "
                                f"track_id={ev.track_id}, count={len(session.pendings)}")
            oldest_pended = session.pendings[0]
            ev = replace(ev, frame_index=oldest_pended.frame_index, ts=oldest_pended.ts)
            self._remove_session(ev.track_id)
            self._publish_event(ev)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): "
                                    f"state={session.state}, event={ev.track}")
        
    def _unset_oldest_pending_session(self, session) -> None:
        if self.oldest_pending_session == session:
            self.oldest_pending_session = None
        
    def _append_track_event(self, session:Session, te:NodeTrack) -> None:
        session.pendings.append(te)
        if self.oldest_pending_session and self.oldest_pending_session != session:
            if te.frame_index < self.oldest_pending_session.first_frame_index:
                self.oldest_pending_session = None

    def _publish_all_pended_events(self, session:Session):
        for pended in session.pendings:
            self._publish_event(pended)
        session.pendings.clear()

    def _publish_old_events(self, session:Session, ts:int) -> int:
        from dna.support import iterables
        
        end_idx = iterables.first((idx for idx, tev in enumerate(session.pendings)
                                   if (ts - tev.ts) < self.timeout_millis), default=-1)
        if end_idx > 0:
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                longest = (ts - session.pendings[0].ts) / 1000
                self.logger.info(f'flush too old {end_idx} TrackEvents: track_id={session.track_id}, '
                                 f'range={session.pendings[0].frame_index}-{session.pendings[end_idx-1].frame_index}, '
                                 f'longest={longest:.3f}s, '
                                 f'timeout={self.timeout:.3f}s')
            self._unset_oldest_pending_session(session)
            for pending in session.pendings[:end_idx]:
                self._publish_event(pending)
            session.pendings = session.pendings[end_idx:]
            return end_idx
        else:
            return 0
    
    def __repr__(self) -> str:
        return f"RefineTrackEvent(nbuffers={self.buffer_size}, timeout={self.timeout})"