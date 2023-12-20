from __future__ import annotations

import cv2
import numpy as np
from shapely import geometry

from dna import Box


_RADIUS = 4
class RectangleDrawer:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.drawing = False
        self.coords = [0, 0, 0, 0]
        # self.bx, self.by, self.ex, self.ey = 0, 0, 0, 0

    def run(self) -> tuple[np.ndarray, Box]:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw)

        self.convas = self.image.copy()
        while ( True ):
            cv2.imshow('image', self.convas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow('image')

        return self.convas, Box(self.coords)

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.coords[0], self.coords[1] = x, y
            # self.bx, self.by = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            print(f'({x}),({y})')
            if self.drawing == True:
                self.convas = self.image.copy()
                cv2.rectangle(self.convas, tuple(self.coords[:2]), (x,y), (0,255,0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.coords[2], self.coords[3] = x, y
            # self.ex, self.ey = x, y
            cv2.rectangle(self.convas, tuple(self.coords[:2]), (x,y), (0,255,0), 2)

    def is_on_corner(self, coord: list[float], radius=_RADIUS) -> int:
        pt = geometry.Point(coord).buffer(radius)
        for idx, coord in enumerate(self.coords):
            if geometry.Point(coord).intersects(pt):
                return idx
        return -1

    def is_on_the_line(self, coord: list[float], radius=_RADIUS) -> int:
        if len(self.coords) > 1:
            pt = geometry.Point(coord).buffer(radius)
            extended = self.coords + [self.coords[0]]
            for idx in list(range(len(extended)-1)):
                line = geometry.LineString([extended[idx], extended[idx+1]])
                if line.intersects(pt):
                    return idx
        return -1