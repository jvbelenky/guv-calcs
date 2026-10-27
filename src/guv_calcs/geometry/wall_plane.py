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
    wall. Among parallel walls the plane's centroid stands in front of
    (on the room side, within the wall's length), the nearest wins; with
    none, the nearest by absolute perpendicular distance.
    """
    n = np.asarray(grid.normal, float)
    if abs(n[2]) > tol:
        return None
    pts = np.asarray(grid.boundary_vertices, float)
    centroid = pts.mean(axis=0)

    candidates = []
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
        along_c = float(np.dot(centroid - p1, u_hat))
        eps = tol * max(length, face.z_height)
        in_front = dist >= -eps and -eps <= along_c <= length + eps
        candidates.append((in_front, dist, wall_id, alignment < 0, p1, u_hat))

    if not candidates:
        return None
    in_front = [c for c in candidates if c[0]]
    if in_front:
        # the wall the plane stands in front of, nearest first
        best = min(in_front, key=lambda c: c[1])
    else:
        best = min(candidates, key=lambda c: abs(c[1]))
    _, dist, wall_id, flip, p1, u_hat = best
    along = (pts - p1) @ u_hat
    verts = tuple((float(a), float(z)) for a, z in zip(along, pts[:, 2]))
    return WallFrame(wall_id, dist, bool(flip), Polygon2D(vertices=verts))
