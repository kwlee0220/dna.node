from __future__ import annotations

from typing import TypeAlias, TypeVar, Optional, Callable, overload, Generic, Any
from collections.abc import Iterable, Sequence, MutableSequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import cv2

from .size2d import Size2d, INVALID_SIZE2D
from .color import BGR, RED, BLUE
from .camera import Image


class InvalidStateError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

NodeId:TypeAlias = str
TrackId:TypeAlias = str


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

K = TypeVar("K")

@dataclass(frozen=True, unsafe_hash=True, slots=True)
class Named(Generic[T]):
    name: str
    value: T
    
    def __iter__(self):
        return (self.name, self.value)


@dataclass(frozen=True, unsafe_hash=True, slots=True)
class KeyValue(Generic[K,T]):
    key: K
    value: T
    
    def __iter__(self):
        return iter((self.key, self.value))


Serializer:TypeAlias = Callable[[T], bytes]
Deserializer:TypeAlias = Callable[[bytes], T]
JsonSerializer:TypeAlias = Callable[[T], str]
JsonDeserializer:TypeAlias = Callable[[str], T]
JsonObjectSerializer:TypeAlias = Callable[[dict[str,Any]], str]
JsonObjectDeserializer:TypeAlias = Callable[[str], dict[str,Any]]


class Serializable(ABC,Generic[T]):
    @abstractmethod
    def serialize(self) -> bytes: pass
    
class Deserializable(ABC,Generic[T]):
    @staticmethod
    @abstractmethod
    def deserialize(serialized:bytes) -> T: pass
    
class SerDeable(Serializable[T],Deserializable[T]):
    pass


class JsonSerializable(ABC,Generic[T]):
    @abstractmethod
    def to_json() -> str: pass
    
class JsonDeserializable(ABC,Generic[T]):
    @staticmethod
    @abstractmethod
    def from_json(serialized:str) -> T: pass
    
class JsonSerDeable(JsonSerializable[T],JsonDeserializable[T]):
    pass
    

@dataclass(frozen=True, order=True, eq=True) # slots=True
class TrackletId:
    node_id: NodeId
    track_id: TrackId

    def __iter__(self):
        return iter((self.node_id, self.track_id))
    
    @classmethod
    def from_string(cls, id:str) -> TrackletId:
        idx = id.rindex('[')
        return TrackletId(id[:idx], id[idx+1:-1])

    @staticmethod
    def from_json_object(json_obj:dict[str,object]) -> TrackletId:
        return TrackletId(node_id=str(json_obj['node']), track_id=str(json_obj['track']))

    def to_json_object(self) -> dict[str,str]:
        return {'node':self.node_id, 'track':self.track_id}

    def __repr__(self) -> str:
        return f'{self.node_id}[{self.track_id}]'


class Point(Sequence[float]):
    """A point coordinate in 2d plane.

    Attributes:
        xy (numpy.ndarray): (x,y) coordinate as a numpy array.
    """
    __slots__ = ('xy', )

    def __init__(self, xy:npt.ArrayLike) -> None:
        """(x,y) 좌표를 갖는 Point 객체를 반환한다.

        Args:
            xy (npt.ArrayLike): (x,y) 좌표
        """
        self.xy = np.array(xy, dtype=np.float32)

    @property
    def x(self) -> float:
        """Point 객체 좌표의 x축 값.

        Returns:
            float: 좌표의 x축 값.
        """
        return self.xy[0]

    @property
    def y(self) -> float:
        """Point 객체 좌표의 y축 값.

        Returns:
            float: 좌표의 y축 값.
        """
        return self.xy[1]

    def distance_to(self, pt:Point) -> float:
        """Returns an Euclidean distance to the point pt.

        Args:
            pt (Point): target Point object to calculate distance to.

        Returns:
            float: distance.
        """
        return float(np.linalg.norm(self.xy - pt.xy))

    def angle_between(self, pt:Point) -> float:
        """본 Point 객체 벡터와 인자 Point 객체 벡터 사이의 각(radian)을 반환한다.

        Args:
            pt (Point): 각을 계산할 대상 Point 객체.

        Returns:
            float: 두 벡터 사이의 각 (단위: radian)
        """
        return float(np.arctan2(np.cross(self.xy, pt.xy), np.dot(self.xy, pt.xy)))

    def line_function_to(self, pt2:Point) -> Callable[[float],float]:
        """본 Point 객체와 인자로 주어진 Point까지를 잇는 1차원 함수를 반환한다.

        Args:
            pt2 (Point): 목표 Point 객체.

        Raises:
            ValueError: 목표 Point 객체의 위치를 잇는 1차원 함수를 구할 수 없는 경우.
                        예를들어 두 Point의 x좌표가 동일한 경우.

        Returns:
            Callable[[float],float]: 1차원 함수.
        """
        delta = self.xy - pt2.xy
        if delta[0] == 0:
            raise ValueError(f"Cannot find a line function: {self} - {pt2}")
        slope = delta[1] / delta[0]
        y_int = pt2.y - (slope * pt2.x)

        def func(x):
            return (slope * x) + y_int
        return func

    def split_points_to(self, pt2:Point, npoints:int) -> list[Point]:
        func = self.line_function_to(pt2)
        step_x = (pt2.x - self.x) / (npoints+1)
        xs = [self.x + (idx * step_x) for idx in range(1, npoints+1)]
        return [Point([x, func(x)]) for x in xs]

    def round(self) -> tuple[int,int]:
        """본 Point 객체의 좌표값을 int형식으로 반올림한 좌표를 갖는 Point 객체를 반환한다.

        Returns:
            Point: int형식으로 반올림한 좌표를 갖는 Point 객체.
        """
        return (round(self.x), round(self.y))
    
    def __iter__(self) -> Iterable[float]:
        return iter((float(self.x), float(self.y)))
    
    def __len__(self) -> int:
        return 2
    
    def __getitem__(self, idx:int) -> int:
        return self.xy[idx]
        
    def __array__(self, dtype=None) -> np.ndarray:
        if not dtype or dtype == self.xy.dtype:
            return self.xy
        else:
            return self.xy.astype(dtype)

    def __add__(self, rhs) -> Point:
        if isinstance(rhs, Point):
            return Point(self.xy + rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point(self.xy + rhs.wh)
        elif isinstance(rhs, Sequence) and len(rhs) == 2:
            return Point(self.xy + np.array(rhs))
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.xy + rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    @overload
    def __sub__(self, rhs:Size2d|float|Sequence[float]) -> Point: pass
    @overload
    def __sub__(self, rhs:Point) -> Size2d: pass
    def __sub__(self, rhs) -> Point|Size2d:
        if isinstance(rhs, Point):
            return Size2d(self.xy - rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point(self.xy - rhs.wh)
        elif isinstance(rhs, Sequence) and len(rhs) == 2:
            return Point(self.xy - np.array(rhs))
        elif isinstance(rhs, float):
            return Point(self.xy - rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __mul__(self, rhs) -> Point:
        if isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.xy * rhs)
        elif isinstance(rhs, Size2d):
            return Point(self.xy * rhs.wh)
        elif isinstance(rhs, Sequence) and len(rhs) == 2:
            return Point(self.xy * np.array(rhs))
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __truediv__(self, rhs) -> Point:
        if isinstance(rhs, Size2d):
            return Point(self.xy / rhs.wh)
        elif isinstance(rhs, float):
            return Point(self.xy / rhs)
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        if isinstance(self.xy[0], int):
            return '({},{})'.format(*self.xy)
        else:
            return '({:.1f},{:.1f})'.format(*self.xy)

class Box:
    """A box object in 2d plane.

    Attributes:
        tlbr (numpy.ndarray): (x1, y1, x2, y2), where (x1, y1) is the coordinate of the top-left corner
                                and (x2, y2) is the coordinate of the bottom-right corner as a numpy arrays.
    """
    __slots__ = ('tlbr', )

    def __init__(self, tlbr:npt.ArrayLike|Iterable[float]) -> None:
        """두 개의 좌표 (x1,y2), (x2, y2) 로 구성된 Box 객체를 반환한다.

        Args:
            tlbr (npt.ArrayLike): (l,t), (r, b) 좌표
        """
        tlbr = np.array(tlbr)
        if tlbr.shape == (2,2):
            tlbr = tlbr.flatten()
        self.tlbr = tlbr

    @staticmethod
    def from_points(tl:Point, br:Point) -> Box:
        """두 개의 Point tl, br로 구성된 Box 객체를 반환한다.

        Args:
            tl (Point): 왼쪽 위 꼭지점 좌표.
            br (Point): 오른쪽 아래 꼭지점 좌표.

        Returns:
            Box: Box 객체
        """
        return Box(np.hstack([tl.xy, br.xy]))

    @staticmethod
    def  from_tlwh(tlwh:npt.ArrayLike) -> Box:
        """Box의 좌상단 꼭지점의 좌표와 box의 넓이와 높이 정보를 이용하여 Box 객체를 생성한다.

        Args:
            tlwh (npt.ArrayLike): 좌상단 꼭지점의 좌표 (tl)와 넓이(w)와 높이(h)

        Returns:
            Box: Box 객체
        """
        tlwh = np.array(tlwh)
        tlbr = tlwh.copy()
        tlbr[2:] = tlwh[:2] + tlwh[2:]
        return Box(tlbr)

    @staticmethod
    def from_size(size:Size2d|npt.ArrayLike) -> Box:
        """
        Create a box object of the given size.
        The top-left corner of the create box will be (0, 0).

        Args:
            size (Union[Size2d,npt.ArrayLike]): the size of the created box.

        Returns:
            Box: a Box object.
        """
        w, h = tuple(size.wh) if isinstance(size, Size2d) else tuple(np.array(size))
        return Box([0, 0, w, h])
    
    @staticmethod
    def from_image(img:np.ndarray) -> Box:
        h, w, _ = img.shape
        return Box([0, 0, w, h])

    def translate(self, delta:Size2d|npt.ArrayLike) -> Box:
        """본 Box 객체를 주어진 거리만큼 평행 이동시킨다.

        Args:
            delta (Union[Size2d,npt.ArrayLike]): 평행 이동 거리.

        Returns:
            Box: 평행 이동된 Box 객체.
        """
        w, h = tuple(delta.wh) if isinstance(delta, Size2d) else tuple(np.array(delta))
        delta = np.array([w, h, w, h])
        return Box(self.tlbr + delta)

    def is_valid(self) -> bool:
        """본 Box의 유효성 여부를 반환한다.

        Returns:
            bool: 유효성 여부.  l <= r and t <= b
        """
        return self.tlbr[0] <= self.tlbr[2] and self.tlbr[1] <= self.tlbr[3]
    
    def __iter__(self):
        return iter(tuple(self.tlbr))
        
    def __array__(self, dtype=None):
        if not dtype or dtype == self.tlbr.dtype:
            return self.tlbr
        else:
            return self.tlbr.astype(dtype)

    @property
    def tlwh(self) -> np.ndarray:
        """``tlwh`` (top-left corner coordinates, width, and height) of this box object.

        Returns:
            np.ndarray: ``tlwh`` (top-left corner coordinates, width, and height)
        """
        tlwh = self.tlbr.copy()
        tlwh[2:] = self.br - self.tl
        return tlwh

    @property
    def xyah(self) -> np.ndarray:
        """``xyah`` (center-coordinate, aspect ratio, and height) of this box object.

        Returns:
            np.ndarray: ``xyah`` (center-coordinate, aspect ratio, and height)
        """
        ret = self.tlwh
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def xywh(self) -> np.ndarray:
        """``xyah`` (center-coordinate, width, and height) of this box object.

        Returns:
            np.ndarray: ``xyah`` (center-coordinate, width, and height)
        """
        ret = self.tlwh
        ret[:2] += ret[2:] / 2
        return ret

    def round(self) -> Box:
        """Returns a box object whose coordindates are round to integers.

        Returns:
            Box: ``Box`` object of integer coordinates.
        """
        return Box(np.rint(self.tlbr).astype(int))

    @property
    def tl(self) -> np.ndarray:
        '''Returns the coordinate of top-left corner of this box object.'''
        return self.tlbr[:2]

    @property
    def br(self) -> np.ndarray:
        '''Returns the coordinate of bottom-right corner of this box object.'''
        return self.tlbr[2:]

    @property
    def wh(self) -> np.ndarray:
        '''Returns width and height pair of this box object.'''
        return self.br - self.tl

    @property
    def width(self) -> float:
        '''Returns width of this box object.'''
        return self.wh[0]

    @property
    def height(self) -> float:
        '''Returns height of this box object.'''
        return self.wh[1]
    
    @property
    def coords(self) -> list[tuple[float,float]]:
        """Returns a list of four corners of this box object.
        The order of corners are top-left, top-right, bottom-right, bottom-left.

        Returns:
            list[tuple[float,float]]: a list of corner coordinates.
        """
        return [(self.tlbr[0], self.tlbr[1]),
                (self.tlbr[2], self.tlbr[1]),
                (self.tlbr[2], self.tlbr[3]),
                (self.tlbr[0], self.tlbr[3])]

    def top_left(self) -> Point:
        '''Returns the ``Point`` object of top-left corner of this box object.'''
        return Point(self.tl)

    def bottom_right(self) -> Point:
        '''Returns the ``Point`` object of bottom-right corner of this box object.'''
        return Point(self.br)

    def center(self) -> Point:
        '''Returns the ``Point`` object of the center of this box object.'''
        return Point(self.tl + (self.wh / 2.))

    def size(self) -> Size2d:
        return Size2d(self.wh) if self.is_valid() else INVALID_SIZE2D

    def area(self) -> float:
        """Returns the area of this box.
        If the box is invalid, zero will be returned.

        Returns:
            float: area
        """
        return self.size().area() if self.is_valid() else 0

    def distance_to(self, box:Box) -> float:
        """Returns the distance to the given box.

        Args:
            box (Box): target box which the distance is calculated.

        Returns:
            float: distance.
        """
        tlbr1 = self.tlbr
        tlbr2 = box.tlbr

        delta1 = tlbr1[[0,3]] - tlbr2[[2,1]]
        delta2 = tlbr2[[0,3]] - tlbr2[[2,1]]
        u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
        v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
        dist = np.linalg.norm(np.concatenate([u, v]))
        return float(dist)

    def contains_point(self, pt:Point) -> bool:
        """Returns whether this box contains the given point or not.

        Args:
            pt (Point): a Point object for containment test.

        Returns:
            bool: True if this box contains the point object, otherwise False.
        """
        x, y = tuple(pt.xy)
        return x >= self.tlbr[0] and y >= self.tlbr[1] and x < self.tlbr[2] and y < self.tlbr[3]

    def contains(self, box:Box) -> bool:
        return self.tlbr[0] <= box.tlbr[0] and self.tlbr[1] <= box.tlbr[1] \
                and self.tlbr[2] >= box.tlbr[2] and self.tlbr[3] >= box.tlbr[3]

    def intersection(self, bbox:Box) -> Box:
        """Returns the intersection box of this box and the box given by the argument.

        Args:
            bbox (Box): a box object to take intersection with.

        Returns:
            Box: intersection box
        """
        x1 = max(self.tlbr[0], bbox.tlbr[0])
        y1 = max(self.tlbr[1], bbox.tlbr[1])
        x2 = min(self.tlbr[2], bbox.tlbr[2])
        y2 = min(self.tlbr[3], bbox.tlbr[3])
        
        return Box([x1, y1, x2, y2])

    def iou(self, box:Box) -> float:
        inter_area = self.intersection(box).area()
        area1, area2 = self.area(), box.area()
        return inter_area / (area1 + area2 - inter_area)

    def overlap_ratios(self, other:Box) -> tuple[float,float,float]:
        if self.is_valid() and other.is_valid():
            inter_area = self.intersection(other).area()
            r1 = inter_area / self.area()
            r2 = inter_area / other.area()
            return (r1, r2, self.iou(other))
        elif not self.is_valid():
            raise ValueError(f'invalid "Box" object: {self}')
        else:
            raise ValueError(f'invalid "Box" object: {other}')

    def draw(self, convas:Image, color:BGR, line_thickness=2) -> Image:
        """Draw this box on the given convas.

        Args:
            convas (Image): the convas image on which this box is drawn.
            color (BGR): the color to draw the box with.
            line_thickness (int, optional): line thickness. Defaults to 2.

        Returns:
            Image: the image the box is drawn on.
        """
        box_int = self.round()
        return cv2.rectangle(convas, box_int.tl, box_int.br, color,
                            thickness=line_thickness, lineType=cv2.LINE_AA)

    def crop(self, image:Image) -> Image:
        """Crops the image of the box is located out of the given image.

        Args:
            img (Image): the source image from which the crop is taken.

        Returns:
            Image: cropped image.
        """
        x1, y1, x2, y2 = tuple(self.tlbr)
        return image[y1:y2, x1:x2]
    
    def expand(self, margin:float|npt.ArrayLike) -> Box:
        """Expand this box with the amount of the given margin.

        Args:
            margin (Union[float,npt.ArrayLike]): the margin

        Returns:
            Box: the expanded box object.
        """
        if isinstance(margin, float|int):
            tlbr = self.tlbr + [-margin, -margin, margin, margin]
            return Box(tlbr)
        else:
            w, h = tuple(np.array(margin))
            return Box(self.tlbr + [-w, -h, w, h])
        
    def update_roi(self, to_image:Image, from_image:Image) -> None:
        """Read data from ``from_image`` and write it onto ``to_image`` of this box is located.

        Args:
            to_image (Image): the target convas where the image is written to
            from_image (Image): the source image which data is from.
        """
        x1, y1, x2, y2 = tuple(self.tlbr)
        to_image[y1:y2, x1:x2] = from_image
    
    def __repr__(self):
        return '{}:{}'.format(Point(self.tl), self.size())

EMPTY_BOX:Box = Box(np.array([0,0,-1,-1]))
    

from collections import abc
class Trajectory(abc.MutableSequence):
    @dataclass(frozen=True,slots=True)
    class Sample:
        point: Point
        ts: int = field(default=-1)
    
    def __init__(self, initial_samples:Iterable[Sample]=[]) -> None:
        self._samples = list(initial_samples)
        
    def __len__(self) -> int:
        return len(self._samples)
        
    def __getitem__(self, pos:int) -> Sample:
        return self._samples[pos]
        
    def __setitem__(self, pos:int, sample:Sample) -> None:
        assert isinstance(sample, Trajectory.Sample)
        self._samples[pos] = sample
        
    def __delitem__(self, pos:int) -> None:
        del self._samples[pos]
        
    def insert(self, pos:int, sample:Sample) -> None:
        assert isinstance(sample, Trajectory.Sample)
        self._samples.insert(pos, sample)

    def draw(self, convas:Image,
             *,
             head_radius:int=9,
             head_color:BGR=RED,
             line_color:BGR=BLUE,
             line_thickness=2) -> Image:
        """Draw this trajectory on the given convas.

        Args:
            convas (Image): the convas image on which this box is drawn.
            color (BGR): the color to draw the box with.
            line_thickness (int, optional): line thickness. Defaults to 2.

        Returns:
            Image: the image the box is drawn on.
        """
        if len(self._samples) == 0:
            return convas
        
        from dna.support import plot_utils
        pts = reversed([s.point for s in self._samples])
        convas = plot_utils.draw_line_string(convas, pts, line_color, line_thickness)
        
        last_pt = self._samples[-1].point
        convas = cv2.circle(convas, last_pt, radius=head_radius, color=head_color,
                            thickness=-1, lineType=cv2.LINE_AA)
        
        return convas
