from __future__ import annotations

import datetime
from typing import NamedTuple

import numpy as np


class Position(NamedTuple):
    """A position on the osu! screen.

    Parameters
    ----------
    x : int or float
        The x coordinate in the range.
    y : int or float
        The y coordinate in the range.

    Notes
    -----
    The visible region of the osu! standard playfield is [0, 512] by [0, 384].
    Positions may fall outside of this range for slider curve control points.
    """

    x: float
    y: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return self.x == other.x and self.y == other.y


class Point(NamedTuple):
    """A position and time on the osu! screen.

    Parameters
    ----------
    x : int or float
        The x coordinate in the range.
    y : int or float
        The y coordinate in the range.
    offset : timedelta
        The time

    Notes
    -----
    The visible region of the osu! standard playfield is [0, 512] by [0, 384].
    Positions may fall outside of this range for slider curve control points.
    """

    x: float
    y: float
    offset: datetime.timedelta


def distance(start: Position, end: Position) -> float:
    return float(np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2))
