"""Geometry primitives for the counting layer."""
from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]


def line_signed_distance(point: Point, p1: Point, p2: Point) -> float:
    """Signed perpendicular distance from ``point`` to the segment line p1->p2.

    Positive on the right side of the p1->p2 direction, negative on the left,
    in pixels. For a vertical line drawn top->bottom (p1 above p2) the right
    side in image coordinates is the region to the right of the line, so a
    person walking rightwards crosses from negative to positive.
    """
    x, y = float(point[0]), float(point[1])
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return 0.0
    cross = (x - x1) * dy - (y - y1) * dx
    return cross / length


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test (works for convex and concave)."""
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        # strict / non-strict half-open test keeps boundary crossings stable
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon: Sequence[Point]) -> float:
    """Shoelace area (px^2) — used for sanity checks of ROI configs."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    j = n - 1
    for i in range(n):
        s += (float(polygon[j][0]) + float(polygon[i][0])) * \
             (float(polygon[j][1]) - float(polygon[i][1]))
        j = i
    return abs(s) / 2.0
