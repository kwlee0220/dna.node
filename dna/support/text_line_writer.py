from __future__ import annotations
import sys

from contextlib import suppress
from collections import defaultdict
from pathlib import Path

_STDOUT_STDERR = set(('stdout', 'stderr'))


class TextLineWriter:
    __slots__ = 'file_path', 'fp'
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        if self.file_path == 'stdout':
            self.fp = sys.stdout
        elif self.file_path == 'stderr':
            self.fp = sys.stderr
        else:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            self.fp = open(self.file_path, 'w')
    
    def close(self) -> None:
        if self.file_path not in _STDOUT_STDERR and self.fp:
            self.fp.close()
            self.fp = None

    def write(self, line:str) -> None:
        if not self.fp:
            raise ValueError(f'not opened, supposed to write: {line}')
        self.fp.write(line)