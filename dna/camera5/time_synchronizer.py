from __future__ import annotations
from enum import Enum

from typing import Optional
import time

from dna.utils import datetime2utc, utc_now_millis


class TimestampType(Enum):
    FROM_ZERO = 1
    TS_ON_OPEN = 2
    FROM_GIVEN_TS = 3
    REALTIME = 4

class TimeSynchronizer:
    __slots__ = ('type', 'sync', '__init_ts', '__adjust_ts', '__frame_interval', '__last_ts')

    def __init__(self, type:TimestampType, sync:bool, *, init_ts:Optional[int]=None):
        """TimeSynchronizer 객체를 생성한다.

        Args:
            type (TimestampType): Timestamp 종류.
            sync (bool): time 동기화 여부.
            init_ts (Optional[int], optional): 첫번째 frame에 부여할 timestamp. Defaults to None.
        """
        self.type = type
        self.sync = sync
        self.__init_ts = init_ts
        self.__adjust_ts = None
        self.__frame_interval = None
        self.__last_ts = 0

    @classmethod
    def parse(cls, init_ts_expr:str, sync:bool) -> TimeSynchronizer:
        match init_ts_expr:
            case '0' | 'zero':
                return TimeSynchronizer(TimestampType.FROM_ZERO, sync=sync, init_ts=0)
            case 'open':
                return TimeSynchronizer(TimestampType.TS_ON_OPEN, sync=sync)
            case 'realtime':
                return TimeSynchronizer(TimestampType.REALTIME, sync=sync)
            case _:
                try:
                    import dateutil.parser as dt_parser
                    # 별도의 timezone 지정없이 'parse'를 호출하면 localzone을 기준으로 datetime을 반환함.
                    dt = dt_parser.parse(init_ts_expr)
                    return TimeSynchronizer(TimestampType.FROM_GIVEN_TS, sync=sync, init_ts=datetime2utc(dt))
                except ValueError:
                    return TimeSynchronizer(TimestampType.FROM_GIVEN_TS, sync=sync, init_ts=round(eval(init_ts_expr)))

    def start(self, fps:int) -> None:
        """Synchronizer를 시작시킨다.

        Synchronizer가 시작되면 초기 timestamp가 설정되고, 이 후 wait() 함수 호출때마다
        timestamp을 계산할 수 있도록 초기화시킨다.

        Args:
            fps (int): 초당 프레임 갯수.
        """
        now = utc_now_millis()
        if self.type == TimestampType.TS_ON_OPEN or self.type == TimestampType.REALTIME:
            self.__init_ts = now
        # 매 frame마다 ts를 생성할 때 보정용으로 사용함.
        self.__adjust_ts = now - self.__init_ts
        self.__frame_interval = 1000.0 / fps

    def wait(self, frame_index:int) -> int:
        """주어진 frame 번호에 해당하는 frame이 준비될 때까지 대기하고,
        해당 frame의 timestamp를 반환한다.

        Args:
            frame_index (int): 대기 대상 프레임 번호.

        Raises:
            ValueError: Valid하지 않는 timestamp type인 경우. 현 구현에서는 발생되지 않음.

        Returns:
            int: 주어진 frame 번호에 해당하는 timestamp 값.
        """
        ts = round(self.__init_ts + (frame_index*self.__frame_interval))
        if self.sync:
            now = utc_now_millis()
            if self.type == TimestampType.REALTIME:
                remains_ms = self.__frame_interval - (now - self.__last_ts)
            else:
                remains_ms = (ts + self.__adjust_ts) - now
            # print(f'frame_index={frame_index}, ts={ts}, remain_ms={remains_ms}')
            if remains_ms > 20:
                time.sleep((remains_ms-5) / 1000.0)

        match self.type:
            case TimestampType.FROM_ZERO | TimestampType.TS_ON_OPEN | TimestampType.FROM_GIVEN_TS:
                return ts
            case TimestampType.REALTIME:
                self.__last_ts = utc_now_millis()
                return self.__last_ts
            case _:
                raise ValueError(f'invalid initial-timestamp: {self.type}')

    def __repr__(self):
        init_ts_str = f", init_ts={self.init_ts}" if self.init_ts is not None else ''
        intvl_ms_str = f", interval={self.interval_ms}ms" if self.interval_ms is not None else ''
        return f'{self.type}{self.init_ts_str}{self.intvl_ms_str}'