"""Geometry in StatsBomb's 120-by-80 pitch coordinates."""

import numpy as np
from matplotlib.path import Path
from numpy.typing import ArrayLike, NDArray

from setpiece.plots.pitch import PITCH_WIDTH

_EDGE_TOLERANCE = 1e-9


def polygon_from_flat(flat: ArrayLike) -> NDArray[np.float64]:
    """Convert StatsBomb's alternating x/y list to an array of polygon vertices."""
    coordinates = np.asarray(flat, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 6 or coordinates.size % 2:
        raise ValueError("a visible-area polygon needs at least three x/y pairs")
    if not np.isfinite(coordinates).all():
        raise ValueError("visible-area coordinates must be finite")
    return coordinates.reshape(-1, 2)


def mirror_xy(xy: ArrayLike, side: str) -> NDArray[np.float64]:
    """Copy x/y coordinates into the left-corner reference frame."""
    if side not in ("left", "right"):
        raise ValueError(f"unknown corner side {side!r}")
    points = np.asarray(xy, dtype=float).copy()
    if side == "right":
        points[..., 1] = PITCH_WIDTH - points[..., 1]
    return points


def polygon_area(polygon: ArrayLike) -> float:
    """Compute the unsigned shoelace area of vertices in perimeter order."""
    x, y = np.asarray(polygon, dtype=float).T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def points_in_polygon(
    points: ArrayLike, poly: ArrayLike, tol: float = _EDGE_TOLERANCE
) -> NDArray[np.bool_]:
    """Return one membership flag per point, including points on the boundary."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty(0, dtype=bool)
    vertices = np.asarray(poly, dtype=float)
    closed = np.vstack([vertices, vertices[0]])
    inside = Path(closed, closed=True).contains_points(points)

    # Matplotlib leaves boundary membership undefined. Check point-to-segment distance
    # explicitly, including repeated vertices and polygons with reversed winding.
    segments = closed[1:] - closed[:-1]
    length_squared = np.sum(segments**2, axis=1)
    offsets = points[:, None, :] - vertices[None, :, :]
    projection = np.einsum("ijk,jk->ij", offsets, segments)
    fraction = np.divide(
        projection,
        length_squared,
        out=np.zeros_like(projection),
        where=length_squared > 0,
    ).clip(0, 1)
    distances = np.sum((offsets - fraction[..., None] * segments) ** 2, axis=2)
    return inside | (distances <= tol**2).any(axis=1)


def zone_grid(x0: float, x1: float, y0: float, y1: float, step: float = 1.0) -> NDArray[np.float64]:
    """Make a rectangular grid including both endpoints on each axis."""
    if not np.isfinite([x0, x1, y0, y1, step]).all():
        raise ValueError("zone bounds and grid step must be finite")
    if step <= 0 or x0 > x1 or y0 > y1:
        raise ValueError("zone bounds must be ordered and grid step positive")

    def axis(low: float, high: float) -> NDArray[np.float64]:
        ticks = np.arange(low, high, step)
        return np.append(ticks[ticks < high], high)

    x, y = np.meshgrid(axis(x0, x1), axis(y0, y1))
    return np.column_stack([x.ravel(), y.ravel()])
