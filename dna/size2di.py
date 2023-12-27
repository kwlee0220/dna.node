from __future__ import annotations

from typing import TypeAlias, overload
from collections.abc import Sequence


class Size2di(Sequence[int]):
    __slots__ = ( '__width', '__height' )
    
    def __init__(self, width:int, height:int) -> None:
        self.__width = width
        self.__height = height

    @staticmethod
    def from_expr(expr:Size2di|str|Sequence[int]) -> Size2di:
        """인자 값을 'Size2di' 객체로 형 변화시킨다.
        - 인자가 Size2di인 경우는 별도의 변환없이 인자를 복사하여 반환한다.
        - 인자가 문자열인 경우에는 '<width> x <height>' 형식으로 파싱하여 Size2di를 생성함.
        - 그렇지 않은 경우는 numpy.array() 함수를 통해 numpy array로 변환하고 이를 다시 Size2di로 생성함.

        Args:
            expr (object): 형 변환시킬 대상 객체.

        Returns:
            Size2di: 형 변환된 Size2di 객체.
        """
        if isinstance(expr, Size2di):
            return Size2di(width=expr.width, height=expr.height)
        elif isinstance(expr, str):
            return Size2di.parse_string(expr)
        elif isinstance(expr, Sequence):
            return Size2di(*expr[:2])
        else:
            raise ValueError(f"invalid Size2di expression: {expr}")

    @staticmethod
    def parse_string(expr:str) -> Size2di:
        """'<width> x <height>' 형식으로 표기된 문자열을 파싱하여 Size2di[N] 객체를 생성한다.

        Args:
            expr (str): '<width> x <height>' 형식의 문자열.

        Raises:
            ValueError: '<width> x <height>' 형식의 문자열이 아닌 경우.

        Returns:
            Size2di[N]: Size2di[N] 객체.
        """
        parts: list[int] = [int(p) for p in expr.split("x")]
        if len(parts) == 2:
            return Size2di.from_expr(parts)
        raise ValueError(f"invalid Size2di string: {expr}")
        
    @property
    def width(self) -> int:
        return self.__width
        
    @property
    def height(self) -> int:
        return self.__height
    
    def __getitem__(self, index:int) -> int:
        if index == 0:
            return self.__width
        elif index == 1:
            return self.__height
        else:
            raise ValueError(f"invalid index: {index}")
        
    def __len__(self) -> int:
        return 2
    
    def __iter__(self):
        return self

    def aspect_ratio(self) -> float:
        """본 Size2di의 aspect ratio (=w/h)를 반환한다.

        Returns:
            float: aspect ratio
        """
        return self.width / self.height

    def area(self) -> int:
        """본 Size2di에 해당하는 영역을 반환한다.

        Returns:
            int: 영역.
        """
        return self.width * self.height

    def __add__(self, rhs:Size2di|Sequence[int]|int) -> Size2di:
        if isinstance(rhs, Sequence):
            return Size2di(width=self.width+int(rhs[0]), height=self.height+int(rhs[1]))
        elif isinstance(rhs, int):
            return Size2di(width=self.width+rhs, height=self.height+rhs)
        else:
            raise ValueError(f"incompatible for __add__: {rhs}")

    def __sub__(self, rhs:Size2di|Sequence[int]|int) -> Size2di:
        if isinstance(rhs, Sequence):
            return Size2di(width=self.width-int(rhs[0]), height=self.height-int(rhs[1]))
        elif isinstance(rhs, int):
            return Size2di(width=self.width-rhs, height=self.height-rhs)
        else:
            raise ValueError(f"incompatible for __sub__: {rhs}")

    def __mul__(self, rhs:Size2di|Sequence[int]|int) -> Size2di:
        if isinstance(rhs, Sequence):
            return Size2di(width=self.width*int(rhs[0]), height=self.height*int(rhs[1]))    # type: ignore
        elif isinstance(rhs, int):
            return Size2di(width=self.width*rhs, height=self.height*rhs)
        else:
            raise ValueError(f"incompatible for __mul__: {rhs}")

    def __truediv__(self, rhs:Size2di|Sequence[int]|int) -> 'Size2d':   # type: ignore
        from .size2d import Size2d
        if isinstance(rhs, Sequence):
            return Size2d((self.width/int(rhs[0]), self.height/int(rhs[1])))    # type: ignore
        elif isinstance(rhs, int):
            return Size2d((self.width/rhs, self.height/rhs))
        else:
            raise ValueError(f"incompatible for __truediv__: {rhs}")

    def __eq__(self, other:Size2di):
        if isinstance(other, Size2di):
            return self.width == other.width and self.height == other.height
        else:
            raise NotImplemented
        
    def __hash__(self):
        return hash(self)
    
    def __repr__(self) -> str:
        return f'{self.width}x{self.height}' 