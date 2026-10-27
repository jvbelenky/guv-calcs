"""Recover a vertical plane's placement on a room wall."""

from typing import NamedTuple
import numpy as np
from .polygon import Polygon2D
from .room_dims import WallFace


class WallFrame(NamedTuple):
    """A plane's placement on a wall: the inverse of SurfaceGrid.from_wall_region.

    ``polygon`` is in wall coordinates (along from the wall's first vertex,
    up from the floor).
    """
    wall_id: str
    normal_offset: float
    flip: bool
    polygon: Polygon2D


def wall_frame_of(grid, dims, tol: float = 1e-6) -> WallFrame | None:
    """Find the wall a vertical plane sits on and its place on that wall.

    Returns None for a plane that is not vertical or is parallel to no
    wall. Among parallel walls, the nearest by perpendicular distance from
    the plane's centroid wins.
    """
    n = np.asarray(grid.normal, float)
    if abs(n[2]) > tol:
        return None
    pts = np.asarray(grid.boundary_vertices, float)
    centroid = pts.mean(axis=0)

    best = None
    for wall_id, face in dims.faces.items():
        if not isinstance(face, WallFace):
            continue
        p1 = np.array([face.x1, face.y1, 0.0])
        d = np.array([face.x2 - face.x1, face.y2 - face.y1, 0.0])
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            continue
        u_hat = d / length
        n_room = np.array([-u_hat[1], u_hat[0], 0.0])
        alignment = float(np.dot(n, n_room))
        if abs(alignment) < 1 - tol:
            continue
        dist = float(np.dot(centroid - p1, n_room))
        if best is None or abs(dist) < abs(best[1]):
            best = (wall_id, dist, alignment < 0, p1, u_hat)

    if best is None:
        return None
    wall_id, dist, flip, p1, u_hat = best
    along = (pts - p1) @ u_hat
    verts = tuple((float(a), float(z)) for a, z in zip(along, pts[:, 2]))
    return WallFrame(wall_id, dist, bool(flip), Polygon2D(vertices=verts))
