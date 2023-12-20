from typing import TypeVar, Protocol, runtime_checkable

from dna import Point


@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...
    
    
@runtime_checkable
class PointSupplier(Protocol):
    def getPoint(self) -> Point: ...