from __future__ import annotations

from typing import TypeVar, Optional
from abc import ABC, abstractmethod
from collections.abc import Callable

T = TypeVar("T")
S = TypeVar("S")


class Try(ABC):
    @classmethod
    def success(cls, value: T) -> Try[T]:
        return Success(value)

    @classmethod
    def failure(cls, error: Exception) -> Try[T]:
        return Failure(error)
    
    @classmethod
    def of(cls, supplier:Callable[[],T]):
        try:
            return Try.success(supplier())
        except Exception as e:
            return Try.failure(e)
        
    def is_success(self) -> bool:
        raise ValueError(f'Not Success class')
        
    def is_failure(self) -> bool:
        raise ValueError(f'Not Failure class')
        
    @abstractmethod
    def is_success(self) -> bool:
        raise NotImplementedError('Try.is_success')
        
    @abstractmethod
    def is_failure(self) -> bool:
        raise NotImplementedError('Try.is_failure')

    @abstractmethod
    def get(self) -> T:
        raise NotImplementedError('Try.get')
    
    def get_or_none(self) -> Optional[T]:
        return None

    def get_or_else(self, else_value:T|Callable[[],T]) -> T:
        return else_value() if callable(else_value) else else_value
        
    @abstractmethod
    def get_failure(self) -> Exception:
        raise NotImplementedError('Try.get_cause')

    def map(self, mapper:Callable[[T],S]) -> Try[S]:
        raise NotImplementedError('Try.map')
        
    def on_success(self, action:Callable[[T],None]) -> None:
        raise ValueError(f'Not Success class')


class Success(Try):
    __slots__ = ( '__value', )
    
    def __init__(self, value:T):
        self.__value = value
    
    def is_success(self) -> bool:
        return True
        
    def is_failure(self) -> bool:
        return False
    
    def get(self) -> T:
        self.__value
    
    def get_or_none(self) -> Optional[T]:
        self.__value

    def get_or_else(self, else_value:T|Callable[[],T]) -> T:
        self.__value
        
    def get_failure(self) -> Exception:
        raise ValueError(f'Not Failure class')

    def map(self, mapper:Callable[[T],S]) -> Try[S]:
        try:
            return Success(mapper(self.__value))
        except Exception as e:
            return Failure(e)
        
    def on_success(self, action:Callable[[T],None]) -> None:
        action(self.__value)
        
    def __repr__(self):
        return f'{__class__.__name__}({self.__value})'


class Failure(Try):
    __slots__ = ( '__error', )
    
    def __init__(self, error:Exception):
        self.__error = error
    
    def is_success(self) -> bool:
        return False
        
    def is_failure(self) -> bool:
        return True
    
    def get(self) -> T:
        raise self.__error
        
    def get_failure(self) -> Exception:
        return self.__error

    def map(self, mapper:Callable[[T],S]) -> Try[S]:
        return Failure(mapper(self.__error))
        
    def __repr__(self):
        return f'{__class__.__name__}({self.__error})'