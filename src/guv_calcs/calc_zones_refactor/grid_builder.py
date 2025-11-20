import numpy as np
from dataclasses import dataclass


@dataclass
class RectGrid:
    xp: np.ndarray
    yp: np.ndarray
    zp: np.ndarray | None
    points: list
    coords: np.ndarray
    num_points: np.ndarray


def build_rect_grid(
    mins,
    maxs,
    num_points,
    spacing,
    offset,
):
    # mins, maxs are tuples of length 2 or 3
    # num_points and spacing are tuples of same length

    dims = len(mins)

    axes = []
    for i in range(dims):
        pt1, pt2 = mins[i], maxs[i]
        n = max(num_points[i], 1)
        s = spacing[i]
        offset_val = min(pt1, pt2)

        axis = np.array([j * s + offset_val for j in range(n)])
        if offset:
            total = abs(pt2 - pt1)
            span = abs(axis[-1] - axis[0]) if n > 1 else 0
            axis += (total - span) / 2

        axes.append(axis)

    # build mesh & coords
    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.vstack([m.reshape(-1) for m in mesh]).T
    coords = np.unique(coords, axis=0)

    # pad zp to None if 2D
    zp = axes[2] if dims == 3 else None

    return RectGrid(
        xp=axes[0],
        yp=axes[1],
        zp=zp,
        points=axes,
        coords=coords,
        num_points=np.array([len(a) for a in axes]),
    )
