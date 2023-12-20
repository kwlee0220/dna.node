from __future__ import annotations

import cv2
import numpy as np
from shapely import geometry

from dna import color

_RADIUS = 4

class PolygonDrawer:
    def __init__(self, image: np.ndarray, coords: list[list[float]]=[]) -> None:
        self.image = image
        self.drawing = False
        self.coords:list[list[float]] = coords
        self.index = -1
        self.selected_corner = -1
        self.selected_line = -1

        self.draw(-1, 0, 0, cv2.EVENT_LBUTTONUP, None)

    def run(self) -> list[list[float]]:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)

        self.convas = self.image.copy()
        while ( True ):
            cv2.imshow('image', self.convas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("output.png", self.convas)

        cv2.destroyWindow('image')
        return self.coords

    def is_on_corner(self, coord: list[float], radius=_RADIUS) -> int:
        pt = geometry.Point(coord).buffer(radius)
        for idx, coord in enumerate(self.coords):
            if geometry.Point(coord).intersects(pt):
                return idx
        return -1

    def is_on_line(self, coord: list[float], radius=_RADIUS) -> int:
        if len(self.coords) > 1:
            pt = geometry.Point(coord).buffer(radius)
            extended = self.coords + [self.coords[0]]
            for idx in list(range(len(extended)-1)):
                line = geometry.LineString([extended[idx], extended[idx+1]])
                if line.intersects(pt):
                    return idx
        return -1

    def draw(self, event, x, y, flags, param):
        cursor = [x,y]
        # print(f"coords={self.coords}, cursor={cursor}, index={self.index}, event={event}")
        n_coords = len(self.coords)
        if event == cv2.EVENT_LBUTTONDOWN:
            if n_coords == 0:
                self.coords = [cursor]
                self.index = 0
            elif n_coords == 1:
                self.coords.append(cursor)
                self.index = 0
            else:
                idx = self.is_on_corner(cursor)
                if idx >= 0:
                    del self.coords[idx]
                    self.coords.insert(idx, cursor)
                    self.index = idx
                else:
                    pt = geometry.Point(cursor).buffer(_RADIUS)
                    for idx in list(range(n_coords-1)):
                        line = geometry.LineString([self.coords[idx], self.coords[idx+1]])
                        if line.intersects(pt):
                            self.coords.insert(idx+1, cursor)
                            self.index = idx+1
                            break
                    if self.index < 0:
                        self.coords.append(cursor)
                        self.index = len(self.coords) - 1
        elif event == cv2.EVENT_LBUTTONUP:
            self.index = -1
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags == 1:
                if self.index >= 0:
                    del self.coords[self.index]
                    self.coords.insert(self.index, cursor)
            else:
                self.selected_corner = -1
                self.selected_line = -1
                idx = self.is_on_corner(cursor)
                if idx >= 0:
                    self.selected_corner = idx
                else:
                    idx = self.is_on_line(cursor)
                    if idx >= 0:
                        self.selected_line = idx
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            idx = self.is_on_corner(cursor)
            if idx >= 0:
                del self.coords[idx]

        self.convas = self.image.copy()

        self.draw_polygon(self.coords, color.GREEN, 1)
        self.draw_corners(self.coords, color.RED, 3)

        if self.selected_line >= 0:
            self.highlight_polygon(self.coords, self.selected_line, color.GREEN, 5)
        if self.selected_corner >= 0:
            cv2.circle(self.convas, self.coords[self.selected_corner], 8, color.RED, -1)

    def _is_on_boundary(self, pt: tuple[float,float]):
        return geometry.LineString(self.points).contains(geometry.Point(pt))

    def draw_polygon(self, coords:list[list[float]], color, thickness):
        if len(coords) > 2:
            coords = np.array(self.coords)
            cv2.polylines(self.convas, [coords], True, color, thickness, cv2.LINE_AA)
        elif len(coords) == 2:
            cv2.line(self.convas, coords[0], coords[1], color, thickness, cv2.LINE_AA)
        elif len(coords) == 1:
            cv2.circle(self.convas, coords[0], 3, color, -1)

    def draw_corners(self, coords:list[list[float]], color, radius):
        for coord in coords:
            cv2.circle(self.convas, coord, radius, color, -1)

    def highlight_polygon(self, coords:list[list[float]], index: int, color, thickness):
        n_coords = len(coords)
        if n_coords >= 2:
            if index < n_coords-1:
                cv2.line(self.convas, coords[index], coords[index+1], color, thickness, cv2.LINE_AA)
            else:
                cv2.line(self.convas, coords[index], coords[0], color, thickness, cv2.LINE_AA)

    def _to_int(self, pt: tuple[float,float]):
        return round(pt[0]), round(pt[1])

    def _to_ints(self, pts: list[tuple[float,float]]):
        return [self._to_int(pt) for pt in pts]