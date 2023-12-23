from __future__ import annotations

from collections import namedtuple

BGR = namedtuple('BGR', 'blue green red')
BLACK = BGR(0,0,0)
WHITE = BGR(255,255,255)
YELLOW = BGR(0,255,255)
RED = BGR(0, 0, 255)
PURPLE = BGR(128,0,128)
MAGENTA = BGR(255,0,255)
GREEN = BGR(0,255,0)
BLUE = BGR(255,0,0)
LAVENDER = BGR(250,230,230)
SKYBLUE = BGR(235,206,135)
INDIGO = BGR(130,0,75)
LIGHT_GREY = BGR(211, 211, 211)
ORANGE = BGR(0, 165, 255)
OLIVE = BGR(0, 128, 128)
DARK_OLIVE_GREEN = BGR(47, 107, 85)
GOLD = BGR(0, 215, 255)
SILVER = BGR(192, 192, 192)
KHAKI = BGR(140, 230, 240)
DARK_KHAKI = BGR(107, 183, 189)
INDIAN_RED = BGR(92, 92, 205)
TEAL = BGR(128, 0, 0)
CYAN = BGR(255, 255, 0)
GREY = BGR(128, 128, 128)
DIM_GREY = BGR(105, 105, 105)

import sys
def name_to_color(name: str):
    current_module = sys.modules[__name__]
    return current_module.__dict__[name]