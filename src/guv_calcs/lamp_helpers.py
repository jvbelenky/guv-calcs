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
    Place points inside polygon using rectangular decomposition + farthest-point sampling.
    Returns the index of the last placed point.
    """
    # Get rectangle centroids (sorted by area, largest first)
    rect_centroids = _get_rectangle_centroids(polygon)

    chosen_indices = []
    chosen_set = set()

    for lamp_num in range(num_points):
        if lamp_num < len(rect_centroids):
            # Use rectangle centroid for first N lamps (N = number of rectangles)
            best_idx = _find_closest_candidate(candidates, rect_centroids[lamp_num])
            # Avoid duplicates if centroids map to same candidate
            while best_idx in chosen_set and lamp_num < len(rect_centroids):
                lamp_num += 1
                if lamp_num < len(rect_centroids):
                    best_idx = _find_closest_candidate(candidates, rect_centroids[lamp_num])
                else:
                    best_idx = None
                    break

        if lamp_num >= len(rect_centroids) or best_idx is None or best_idx in chosen_set:
            # Fallback: farthest-point sampling
            best_idx = None
            best_min_dist = -1.0

            for i, pt in enumerate(candidates):
                if i in chosen_set:
                    continue

                if not chosen_indices:
                    # First lamp (no centroids available): maximize distance to edges
                    min_dist = _distance_to_polygon_boundary(pt, polygon)
                else:
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

    return chosen_indices[-1] if chosen_indices else 0


def _distance_to_polygon_boundary(point: np.ndarray, polygon: "Polygon2D") -> float:
    """Calculate minimum distance from point to any polygon edge."""
    px, py = point
    min_dist = float("inf")

    for (x1, y1), (x2, y2) in polygon.edges:
        # Distance from point to line segment
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)

    return min_dist


def _median_distance_to_edges(point: np.ndarray, polygon: "Polygon2D") -> float:
    """Calculate median distance from point to all polygon edges."""
    px, py = point
    dists = []

    for (x1, y1), (x2, y2) in polygon.edges:
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        dists.append(dist)

    return float(np.median(dists))


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


def _get_rectangle_centroids(polygon: "Polygon2D") -> list[tuple[float, float]]:
    """
    Decompose an axis-aligned rectilinear polygon into rectangles and return their centroids.
    Returns centroids sorted by rectangle area (largest first).
    """
    # Find reflex (concave) vertices - where interior angle > 180°
    reflex_vertices = _find_reflex_vertices(polygon)

    if not reflex_vertices:
        # Convex polygon (or simple rectangle) - just return centroid
        return [polygon.centroid]

    # For rectilinear polygons, decompose by extending cuts from reflex vertices
    rectangles = _decompose_rectilinear(polygon, reflex_vertices)

    if not rectangles:
        return [polygon.centroid]

    # Compute centroids and sort by area (largest first)
    centroids_with_area = []
    for x1, y1, x2, y2 in rectangles:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = abs((x2 - x1) * (y2 - y1))
        centroids_with_area.append((area, (cx, cy)))

    centroids_with_area.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in centroids_with_area]


def _find_reflex_vertices(polygon: "Polygon2D") -> list[tuple[float, float]]:
    """Find vertices where the interior angle is > 180° (concave/reflex vertices)."""
    reflex = []
    n = len(polygon.vertices)

    for i in range(n):
        prev_v = polygon.vertices[(i - 1) % n]
        curr_v = polygon.vertices[i]
        next_v = polygon.vertices[(i + 1) % n]

        # Cross product to determine turn direction (CCW polygon)
        cross = (curr_v[0] - prev_v[0]) * (next_v[1] - curr_v[1]) - \
                (curr_v[1] - prev_v[1]) * (next_v[0] - curr_v[0])

        # For CCW polygon, negative cross product means reflex vertex
        if cross < 0:
            reflex.append(curr_v)

    return reflex


def _decompose_rectilinear(
    polygon: "Polygon2D", reflex_vertices: list[tuple[float, float]]
) -> list[tuple[float, float, float, float]]:
    """
    Decompose a rectilinear polygon into rectangles using cuts from reflex vertices.
    Returns list of (x1, y1, x2, y2) tuples for each rectangle.
    """
    x_min, y_min, x_max, y_max = polygon.bounding_box

    # Collect all x and y coordinates that define region boundaries
    x_coords = {x_min, x_max}
    y_coords = {y_min, y_max}

    for vx, vy in polygon.vertices:
        x_coords.add(vx)
        y_coords.add(vy)

    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    # Generate candidate rectangles from the grid
    rectangles = []
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            x1, x2 = x_coords[i], x_coords[i + 1]
            y1, y2 = y_coords[j], y_coords[j + 1]

            # Check if rectangle center is inside polygon
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if polygon.contains_point(cx, cy):
                rectangles.append((x1, y1, x2, y2))

    return rectangles


def _find_closest_candidate(
    candidates: np.ndarray, target: tuple[float, float]
) -> int:
    """Find the index of the candidate point closest to the target."""
    tx, ty = target
    dists = (candidates[:, 0] - tx) ** 2 + (candidates[:, 1] - ty) ** 2
    return int(np.argmin(dists))


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


# ============== Visibility and Aim Point Helpers ==============


def _line_intersects_polygon_edge(
    p1: tuple[float, float],
    p2: tuple[float, float],
    polygon: "Polygon2D",
    origin_edge_idx: int = -1,
) -> bool:
    """
    Check if line segment p1-p2 intersects any polygon edge.

    Args:
        p1: Start point of line segment
        p2: End point of line segment
        polygon: The polygon to check against
        origin_edge_idx: Index of edge to skip (the edge the origin sits on)

    Returns:
        True if the line intersects any edge (excluding origin_edge_idx)
    """
    for i, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
        if i == origin_edge_idx:
            continue

        # Standard line segment intersection test
        # Check if segments (p1, p2) and ((x1, y1), (x2, y2)) intersect
        if _segments_cross(p1, p2, (x1, y1), (x2, y2)):
            return True

    return False


def _segments_cross(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """
    Check if line segment a1-a2 properly crosses segment b1-b2.
    Returns True only for proper crossing (not endpoint touching).
    """

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross_product(b1, b2, a1)
    d2 = cross_product(b1, b2, a2)
    d3 = cross_product(a1, a2, b1)
    d4 = cross_product(a1, a2, b2)

    # Check if segments straddle each other (proper intersection)
    # Use small epsilon for numerical stability
    eps = 1e-10
    if d1 * d2 < -eps and d3 * d4 < -eps:
        return True

    return False


def _find_origin_edge(
    origin: tuple[float, float], polygon: "Polygon2D", tolerance: float = 1e-6
) -> int:
    """Find which edge the origin point lies on (or near), returns -1 if none."""
    ox, oy = origin
    for i, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
        dist = _point_to_segment_distance(ox, oy, x1, y1, x2, y2)
        if dist < tolerance:
            return i
    return -1


def farthest_visible_point(
    origin: tuple[float, float], polygon: "Polygon2D", num_divisions: int = 50
) -> tuple[float, float]:
    """
    Find the farthest point inside the polygon visible from origin.

    The returned point has a clear line-of-sight from origin (doesn't cross any
    polygon edges). Used for aiming tilted lamps toward the farthest reachable point.

    Args:
        origin: The (x, y) position of the lamp
        polygon: The polygon boundary
        num_divisions: Resolution for internal grid sampling

    Returns:
        The (x, y) coordinates of the farthest visible point
    """
    # Find which edge the origin sits on (to exclude from intersection tests)
    origin_edge_idx = _find_origin_edge(origin, polygon)

    # Generate candidate points
    candidates = []

    # 1. Polygon vertices
    for v in polygon.vertices:
        candidates.append(v)

    # 2. Edge midpoints
    for (x1, y1), (x2, y2) in polygon.edges:
        candidates.append(((x1 + x2) / 2, (y1 + y2) / 2))

    # 3. Grid points inside polygon
    x_min, y_min, x_max, y_max = polygon.bounding_box
    xp = np.linspace(x_min, x_max, num_divisions + 1)
    yp = np.linspace(y_min, y_max, num_divisions + 1)
    for x in xp:
        for y in yp:
            if polygon.contains_point(x, y):
                candidates.append((x, y))

    # 4. Centroid as fallback candidate
    candidates.append(polygon.centroid)

    # Find farthest visible point
    ox, oy = origin
    best_point = polygon.centroid
    best_dist = -1.0

    for cx, cy in candidates:
        # Skip points too close to origin
        dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
        if dist < 1e-6:
            continue

        # Check visibility (line doesn't cross any edge)
        if not _line_intersects_polygon_edge(origin, (cx, cy), polygon, origin_edge_idx):
            if dist > best_dist:
                best_dist = dist
                best_point = (cx, cy)

    return best_point
