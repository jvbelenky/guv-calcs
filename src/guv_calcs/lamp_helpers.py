import numpy as np
from .polygon import Polygon2D


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


# ============== Polygon Room Lamp Placement ==============


def new_lamp_position_polygon(lamp_idx: int, polygon: "Polygon2D", num_divisions: int = 100):
    """
    Get the position for lamp number `lamp_idx` inside a polygon.
    Uses farthest-point sampling: maximizes distance to existing lamps and polygon edges.

    lamp_idx is 1-based.
    """
    x_min, y_min, x_max, y_max = polygon.bounding_box

    # Create candidate grid inside polygon
    xp = np.linspace(x_min, x_max, num_divisions + 1)
    yp = np.linspace(y_min, y_max, num_divisions + 1)
    xx, yy = np.meshgrid(xp, yp, indexing="ij")
    all_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Filter to points inside polygon
    inside_mask = polygon.contains_points(all_points)
    inside_points = all_points[inside_mask]

    if len(inside_points) == 0:
        # Fallback to centroid
        return polygon.centroid

    # Get indices for this lamp using farthest-point sampling
    chosen_idx = _place_points_in_polygon(inside_points, polygon, lamp_idx)
    return tuple(inside_points[chosen_idx])


def _place_points_in_polygon(candidates: np.ndarray, polygon: "Polygon2D", num_points: int):
    """
    Place points inside polygon using farthest-point sampling.
    Returns the index of the last placed point.
    """
    # Start with centroid (or closest candidate to centroid)
    cx, cy = polygon.centroid
    dists_to_centroid = np.sqrt((candidates[:, 0] - cx) ** 2 + (candidates[:, 1] - cy) ** 2)
    first_idx = np.argmin(dists_to_centroid)

    chosen_indices = [first_idx]
    chosen_set = {first_idx}

    for _ in range(1, num_points):
        best_idx = None
        best_min_dist = -1.0

        for i, pt in enumerate(candidates):
            if i in chosen_set:
                continue

            # Distance to nearest chosen point
            min_dist_to_chosen = min(
                np.sqrt((pt[0] - candidates[j, 0]) ** 2 + (pt[1] - candidates[j, 1]) ** 2)
                for j in chosen_indices
            )

            # Distance to nearest polygon edge
            min_dist_to_edge = _distance_to_polygon_boundary(pt, polygon)

            # Take minimum of both constraints
            min_dist = min(min_dist_to_chosen, min_dist_to_edge)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx is not None:
            chosen_indices.append(best_idx)
            chosen_set.add(best_idx)

    return chosen_indices[-1]


def _distance_to_polygon_boundary(point: np.ndarray, polygon: "Polygon2D") -> float:
    """Calculate minimum distance from point to any polygon edge."""
    px, py = point
    min_dist = float("inf")

    for (x1, y1), (x2, y2) in polygon.edges:
        # Distance from point to line segment
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)

    return min_dist


def _point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
    """Calculate distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        # Segment is a point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Project point onto line, clamping to segment
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def new_lamp_position_polygon_perimeter(
    lamp_idx: int, polygon: "Polygon2D", num_divisions: int = 100
):
    """
    Place lamps along the perimeter of a polygon, as far apart as possible.
    Starts with vertices, then fills in along edges.

    lamp_idx is 1-based.
    """
    # Generate candidate points along perimeter
    candidates = _get_polygon_perimeter_points(polygon, num_divisions)

    if lamp_idx == 1:
        # First lamp at first vertex
        return candidates[0]

    # Use farthest-point sampling
    chosen_idx = _place_points_on_polygon_perimeter(candidates, lamp_idx)
    return candidates[chosen_idx]


def _get_polygon_perimeter_points(polygon: "Polygon2D", num_divisions: int) -> list:
    """Generate evenly-spaced points along polygon perimeter."""
    total_perimeter = sum(polygon.edge_lengths)
    points_per_unit = num_divisions / total_perimeter

    candidates = []
    for (x1, y1), (x2, y2) in polygon.edges:
        edge_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_edge_points = max(2, int(edge_length * points_per_unit))

        for i in range(num_edge_points):
            t = i / num_edge_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            candidates.append((x, y))

    return candidates


def _place_points_on_polygon_perimeter(candidates: list, num_points: int) -> int:
    """
    Place points on polygon perimeter using farthest-point sampling.
    Returns the index of the last placed point.
    """
    # Start with first vertex
    chosen_indices = [0]
    chosen_set = {0}

    for _ in range(1, num_points):
        best_idx = None
        best_min_dist = -1.0

        for i, pt in enumerate(candidates):
            if i in chosen_set:
                continue

            # Distance to nearest chosen point
            min_dist = min(
                np.sqrt((pt[0] - candidates[j][0]) ** 2 + (pt[1] - candidates[j][1]) ** 2)
                for j in chosen_indices
            )

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx is not None:
            chosen_indices.append(best_idx)
            chosen_set.add(best_idx)

    return chosen_indices[-1]
