from __future__ import annotations

from typing import TypeVar, Optional
from collections.abc import Callable
from dataclasses import dataclass, field

T = TypeVar("T")


@dataclass(frozen=True, eq=True, repr=False)
class Option:
    value: T
    president: bool = field(default=True)

    @staticmethod
    def empty() -> Option:
        return _EMPTY

    @staticmethod
    def of(value: T) -> Option:
        return Option(value, True)

    @staticmethod
    def of_nullable(value: Optional[T]) -> Option:
        return Option(value, True) if value is not None else Option.empty()

    def get(self) -> T:
        if self.president:
            return self.value
        else:
            raise ValueError("NoSuchValue")

    def get_or_none(self) -> Optional[T]:
        return self.value if self.president else None

    def get_or_else(self, else_value:T|Callable[[],T]) -> T:
        if self.president:
            return self.value
        else:
            return else_value() if callable(else_value) else else_value
        
    def or_else(self, or_else:Option[T]) -> Option[T]:
        if self.president:
            return self
        else:
            return or_else
        
    def or_else(self, or_else:T) -> Option[T]:
        if self.president:
            return self
        else:
            return Option.of(or_else)
        
    def or_else(self, or_else:Callable[[],Option[T]]) -> Option[T]:
        if self.president:
            return self
        else:
            return or_else()

    def is_present(self) -> bool:
        return self.president

    def is_absent(self) -> bool:
        return self.president

    def if_present(self, call) -> Option:
        if self.president:
            call()

        return self

    def if_absent(self, call) -> Option:
        if not self.president:
            call()

        return self

    def map(self, mapper) -> Option:
        return Option.of(mapper(self.value)) if self.president else Option.empty()

    def transform(self, target:T, mapper:Callable[[T, T], T]) -> Option:
        if self.is_present:
            return mapper(target, self.value)
        else:
            return target

    def __repr__(self) -> str:
        return f'Option({self.value})' if self.president else 'Option(Empty)'

_EMPTY = Option(None, False)