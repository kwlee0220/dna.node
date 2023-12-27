from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from dna import BGR, Box, Size2d, Point
from dna.camera import Image
from dna.support import plot_utils


class Detection:
    __slots__ = 'bbox', 'label', 'score', 'feature', 'exit_zone'

    def __init__(self, bbox:Box, label:Optional[str]=None, score:float=-1) -> None:
        self.bbox = bbox
        self.label = label
        self.score = score
        self.feature = None
        self.exit_zone = -1

    def draw(self, convas: Image, color:BGR,
            label:Optional[str]=None,
            label_color:Optional[BGR]=None,
            label_tl:Optional[Point]=None,
            line_thickness:int=2) -> Image:
        loc = self.bbox
        
        convas = loc.draw(convas, color=color, line_thickness=line_thickness)
        if label_color:
            if not label:
                label = f"{self.label}({self.score:.3f})"
            if not label_tl:
                label_tl = Point(loc.tl.astype(int))
            convas = plot_utils.draw_label2(convas=convas, label=label, pos=label_tl,
                                            font_face=cv2.FONT_HERSHEY_SIMPLEX, color=label_color, fill_color=color,
                                            line_thickness=1, font_scale=0.4, bg_margin=Size2d((0,0)))
        return convas

    def __truediv__(self, rhs) -> Detection:
        if isinstance(rhs, Size2d):
            return Detection(bbox=self.bbox/rhs, label=self.label, score=self.score)
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        return f'{self.label}:{self.bbox},{self.score:.3f}'