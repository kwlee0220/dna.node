from __future__ import annotations
from typing import Union

import numbers
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry
import shapely.geometry as geometry

from dna import Point, BGR
from dna.camera import Image
from dna.support import plot_utils


class Zone:
    def __init__(self, geom:BaseGeometry) -> None:
        self.geom = geom

    @staticmethod
    def from_coords(coords:list, as_line_string:bool=False) -> Zone:
        if isinstance(coords[0], numbers.Number):
            return Zone(geometry.box(*coords))
        elif as_line_string:
            npoints = len(coords)
            if npoints < 3:
                return Zone(geometry.LineString([tuple(c) for c in coords]))
            else:
                if coords[0] == coords[-1]:
                    return Zone(geometry.Polygon([tuple(c) for c in coords[:-1]]))
                else:
                    return Zone(geometry.LineString([tuple(c) for c in coords]))
            return Zone(line_string)
        else:
            npoints = len(coords)
            if npoints < 3:
                return Zone(geometry.LineString([tuple(c) for c in coords]))
            else:
                return Zone(geometry.Polygon([tuple(c) for c in coords]))
    
    @property
    def coords(self):
        return self.geom.coords

    def covers(self, geom:BaseGeometry) -> bool:
        return self.geom.covers(geom)

    def covers_point(self, pt:Union[Point,tuple[float,float],npt.ArrayLike]) -> bool:
        xy = tuple(pt.xy) if isinstance(pt, Point) else tuple(pt)
        return self.geom.covers(geometry.Point(xy))
    
    def intersects(self, geom:BaseGeometry):
        return self.geom.intersects(geom) 
    
    def intersection(self, geom:BaseGeometry) -> BaseGeometry:
        return self.geom.intersection(geom) 
    
    def distance(self, geom:BaseGeometry) -> float:
        return self.geom.distance(geom)

    def draw(self, convas:Image, color:BGR, line_thickness=2) -> Image:
        if isinstance(self.geom, geometry.LineString):
            pts = [Point(coord) for coord in self.geom.coords]
            return plot_utils.draw_line_string(convas, pts, color=color, line_thickness=line_thickness)
        else:
            return plot_utils.draw_polygon(convas, list(self.geom.exterior.coords), color=color, line_thickness=line_thickness)

    def __repr__(self) -> str:
        return repr(self.geom)

    @staticmethod
    def find_covering_zone(pt:Point, zones:list[Zone]):
        for idx, zone in enumerate(zones):
            if zone.covers_point(pt):
                return idx
        return -1