from __future__ import annotations

from enum import Enum


_ABBR_TO_STATE:dict[str,TrackState] = dict()
class TrackState(Enum):
    Null = (0, 'N')
    Tentative = (1, 'T')
    Confirmed = (2, 'C')
    TemporarilyLost = (3, 'L')
    Deleted = (4, 'D')

    def __init__(self, code, abbr) -> None:
        super().__init__()
        self.code = code
        self.abbr = abbr

    @staticmethod
    def from_abbr(abbr:str) -> TrackState:
        global _ABBR_TO_STATE
        if not _ABBR_TO_STATE:
            _ABBR_TO_STATE = { state.value[1]:state for state in TrackState }
        return _ABBR_TO_STATE[abbr]