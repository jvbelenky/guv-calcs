import numpy as np


def get_lamp_positions(num_lamps, x, y, num_divisions=100):
    """
    generate a list of (x,y) positions for a lamp given room dimensions and
    the number of lamps desired
    """
    lst = [new_lamp_position(i + 1, x, y) for i in range(num_lamps)]
    return np.array(lst).T


def new_lamp_position(lamp_idx, x, y, num_divisions=100):
    """
    get the default position for an additional new lamp
    x and y are the room dimensions
    first index is 1, not 0.
    """
    xp = np.linspace(0, x, num_divisions + 1)
    yp = np.linspace(0, y, num_divisions + 1)
    xidx, yidx = _get_idx(lamp_idx, num_divisions=num_divisions)
    return xp[xidx], yp[yidx]


def _get_idx(num_points, num_divisions=100):
    grid_size = (num_divisions, num_divisions)
    return _place_points(grid_size, num_points)[-1]


def _place_points(grid_size, num_points):
    M, N = grid_size
    grid = np.zeros(grid_size)
    points = []

    # Place the first point in the center
    center = (M // 2, N // 2)
    points.append(center)
    grid[center] = 1  # Marking the grid cell as occupied

    for _ in range(1, num_points):
        max_dist = -1
        best_point = None

        for x in range(M):
            for y in range(N):
                if grid[x, y] == 0:
                    # Calculate the minimum distance to all existing points
                    min_point_dist = min(
                        [np.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in points]
                    )
                    # Calculate the distance to the nearest boundary
                    min_boundary_dist = min(x, M - 1 - x, y, N - 1 - y)
                    # Find the point where the minimum of these distances is maximized
                    min_dist = min(min_point_dist, min_boundary_dist)

                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_point = (x, y)

        if best_point:
            points.append(best_point)
            grid[best_point] = 1  # Marking the grid cell as occupied
    return points


def new_lamp_position_perimeter(lamp_idx, x, y, num_divisions=100):
    """
    Place lamps as far apart as possible *on the perimeter* of the rectangle.
    Order: corner, opposite corner, remaining two corners, then farthest-point
    sampling along the edges.

    lamp_idx is 1-based.
    """
    xp = np.linspace(0, x, num_divisions + 1)
    yp = np.linspace(0, y, num_divisions + 1)

    xidx, yidx = _get_perimeter_idx(lamp_idx, num_divisions=num_divisions)
    return xp[xidx], yp[yidx]


def _get_perimeter_idx(num_points, num_divisions=100):
    # indices run 0..num_divisions inclusive (same as xp/yp)
    return _place_points_on_perimeter(num_divisions, num_points)[-1]


def _place_points_on_perimeter(num_divisions, num_points):
    """
    Return a list of (xidx, yidx) integer indices on the perimeter of a
    (num_divisions+1) x (num_divisions+1) lattice.
    """
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    M = N = num_divisions + 1
    # perimeter candidates
    candidates = []
    for xi in range(M):
        candidates.append((xi, 0))
        candidates.append((xi, N - 1))
    for yi in range(1, N - 1):
        candidates.append((0, yi))
        candidates.append((M - 1, yi))
    # de-dupe while preserving order
    candidates = list(dict.fromkeys(candidates))

    # seed corners in the order you described
    corners = [(0, 0), (M - 1, N - 1), (0, N - 1), (M - 1, 0)]
    points = []
    chosen = set()

    for c in corners:
        if len(points) >= num_points:
            return points
        points.append(c)
        chosen.add(c)

    # greedy farthest-point sampling on the perimeter
    for _ in range(len(points), num_points):
        best = None
        best_d = -1.0

        for p in candidates:
            if p in chosen:
                continue
            # maximize distance to nearest already-chosen point
            d = min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in points)
            if d > best_d:
                best_d = d
                best = p

        points.append(best)
        chosen.add(best)

    return points
