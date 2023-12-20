from __future__ import annotations

from typing import Optional

from dna import Point
from . import stabilizer


def _x(pt):
    return pt.x if isinstance(pt, Point) else pt[0]
def _y(pt):
    return pt.y if isinstance(pt, Point) else pt[1]


class RunningStabilizer:
    def __init__(self, look_ahead:int, smoothing_factor:float=1) -> None:
        self.look_ahead = look_ahead
        self.smoothing_factor = smoothing_factor
        self.current, self.upper = 0, 0
        self.pending_xs: list[float] = []
        self.pending_ys: list[float] = []

    def transform(self, pt:Point) -> Optional[Point]:
        self.pending_xs.append(_x(pt))
        self.pending_ys.append(_y(pt))
        self.upper += 1

        if self.upper - self.current > self.look_ahead:
            xs = stabilizer.stabilization_location(self.pending_xs, self.look_ahead, self.smoothing_factor)
            ys = stabilizer.stabilization_location(self.pending_ys, self.look_ahead, self.smoothing_factor)
            xy = [xs[self.current], ys[self.current]]
            stabilized = Point(xy)

            self.current += 1
            if self.current > self.look_ahead:
                self.pending_xs.pop(0)
                self.pending_ys.pop(0)
                self.current -= 1
                self.upper -= 1
            return stabilized
        else:
            return None

    def get_tail(self) -> list[Point]:
        xs = stabilizer.stabilization_location(self.pending_xs, self.look_ahead, self.smoothing_factor)
        ys = stabilizer.stabilization_location(self.pending_ys, self.look_ahead, self.smoothing_factor)
        return [Point([x,y]) for x, y in zip(xs[self.current:], ys[self.current:])]

    def reset(self) -> None:
        self.current, self.upper = 0, 0
        self.pending_xs: list[float] = []
        self.pending_ys: list[float] = []