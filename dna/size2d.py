from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt


class Size2d:
    """A size object in 2d plane.

    Attributes:
        wh (numpy.ndarray): (width, height) as a numpy array.
    """
    __slots__ = ('wh',)

    def __init__(self, wh:npt.ArrayLike) -> None:
        """width와 height를 구성된 Size2d 객체를 반환한다.

        Args:
            wh (npt.ArrayLike): 2차원 크기의 넓이(w)와 높이(h) 값의 배열
        """
        self.wh = np.array(wh)

    @staticmethod
    def from_expr(expr:Size2d|str|npt.ArrayLike) -> Size2d:
        """인자 값을 'Size2d' 객체로 형 변화시킨다.
        - 인자가 Size2d인 경우는 별도의 변환없이 인자를 복사하여 반환한다.
        - 인자가 문자열인 경우에는 '<width> x <height>' 형식으로 파싱하여 Size2d를 생성함.
        - 그렇지 않은 경우는 numpy.array() 함수를 통해 numpy array로 변환하고 이를 다시 Size2d로 생성함.

        Args:
            expr (object): 형 변환시킬 대상 객체.

        Returns:
            Size2d: 형 변환된 Size2d 객체.
        """
        if isinstance(expr, Size2d):
            return Size2d(expr.wh)
        elif isinstance(expr, str):
            return Size2d.parse_string(expr)
        else:
            return Size2d(expr)

    @staticmethod
    def parse_string(expr:str) -> Size2d:
        """'<width> x <height>' 형식으로 표기된 문자열을 파싱하여 Size2d 객체를 생성한다.

        Args:
            expr (str): '<width> x <height>' 형식의 문자열.

        Raises:
            ValueError: '<width> x <height>' 형식의 문자열이 아닌 경우.

        Returns:
            Size2d: Size2d 객체.
        """
        parts: list[float] = [float(p) for p in expr.split("x")]
        if len(parts) == 2:
            return Size2d(parts)
        raise ValueError(f"invalid Size2d string: {expr}")

    def is_valid(self) -> bool:
        """Size2d 객체의 유효성 여부를 반환한다.

        Returns:
            bool: 유효성 여부. 넓이와 높이가 모두 0보다 크거나 같은지 여부.
        """
        return self.wh[0] >= 0 and self.wh[1] >= 0
    
    @staticmethod
    def from_image(img:np.ndarray) -> Size2d:
        """주어진 OpenCV 이미지의 크기를 반환한다.

        Args:
            img (np.ndarray): OpenCV 이미지 객체.

        Returns:
            Size2d: 이미지 크기 (width, height)
        """
        h, w, _ = img.shape
        return Size2d([w, h])

    @property
    def width(self) -> Union[int,float]:
        """본 Size2d의 넓이 값.

        Returns:
            Union[int,float]: 본 Size2d의 넓이 값.
        """
        return self.wh[0]
    
    @property
    def height(self) -> Union[int,float]:
        """본 Size2d의 높이 값.

        Returns:
            Union[int,float]: 본 Size2d의 높이 값.
        """
        return self.wh[1]
    
    def __iter__(self):
        return (c for c in self.wh)
        
    def __array__(self, dtype=None):
        if not dtype or dtype == self.wh.dtype:
            return self.wh
        else:
            return self.wh.astype(dtype)

    def aspect_ratio(self) -> float:
        """본 Size2d의 aspect ratio (=w/h)를 반환한다.

        Returns:
            float: aspect ratio
        """
        return self.wh[0] / self.wh[1]

    def area(self) -> float:
        """본 Size2d에 해당하는 영역을 반환한다.

        Returns:
            float: 영역.
        """
        return self.wh[0] * self.wh[1]

    def to_rint(self) -> 'Size2di':
        """본 Size2d 크기 값을 int형식으로 반올림한 값을 갖는 Size2d 객체를 반환한다.

        Returns:
            Size2d: int형식으로 반올림한 크기를 갖는 Size2d 객체.
        """
        from .size2di import Size2di
        
        tup = tuple(self.wh)
        return Size2di.from_expr((round(tup[0]), round(tup[1])))
    
    def __hash__(self):
        return hash(tuple(self))

    def __add__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d(self.wh + rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d(self.wh + rhs)
        else:
            return Size2d(self.wh + np.array(rhs))

    def __sub__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d(self.wh - rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d(self.wh - rhs)
        else:
            return Size2d(self.wh - np.array(rhs))

    def __mul__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d(self.wh * rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d(self.wh * rhs)
        else:
            return Size2d(self.wh * np.array(rhs))

    def __truediv__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d(self.wh / rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d(self.wh / rhs)
        else:
            return Size2d(self.wh / np.array(rhs))

    def __eq__(self, other):
        if isinstance(other, Size2d):
            return np.array_equal(self.wh, other.wh)
        else:
            return False

    def __gt__(self, other:Size2d):
        if isinstance(other, Size2d):
            return self.wh[0] > other.wh[0] and self.wh[1] > other.wh[1]
        else:
            raise ValueError(f'invalid Size2d object: {self}')

    def __ge__(self, other:Size2d):
        if isinstance(other, Size2d):
            return self.wh[0] >= other.wh[0] and self.wh[1] >= other.wh[1]
        else:
            raise ValueError(f'invalid Size2d object: {self}')

    def __lt__(self, other:Size2d):
        if isinstance(other, Size2d):
            return self.wh[0] < other.wh[0] and self.wh[1] < other.wh[1]
        else:
            raise ValueError(f'invalid Size2d object: {self}')

    def __le__(self, other:Size2d):
        if isinstance(other, Size2d):
            return self.wh[0] <= other.wh[0] and self.wh[1] <= other.wh[1]
        else:
            raise ValueError(f'invalid Size2d object: {self}')
    
    def __repr__(self) -> str:
        if isinstance(self.wh[0], np.int32):
            return '{}x{}'.format(*self.wh)
        else:
            return '{:.1f}x{:.1f}'.format(*self.wh)
INVALID_SIZE2D: Size2d = Size2d([-1, -1])