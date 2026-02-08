"""Lamp placement algorithms for various room geometries and placement modes."""

import warnings
from dataclasses import dataclass
from math import atan, degrees, hypot
import numpy as np
from ..polygon import Polygon2D
from .lamp_configs import resolve_keyword


# =============================================================================
# SECTION 1: Shared Utilities
# =============================================================================


def _normalize(dx: float, dy: float) -> tuple[float, float]:
    """Normalize a 2D vector, returning (0, 0) for zero-length."""
    length = hypot(dx, dy)
    return (dx / length, dy / length) if length > 1e-10 else (0.0, 0.0)


def _offset_inward(
    point: tuple[float, float], inward_dir: tuple[float, float], offset: float = 0.05
) -> tuple[float, float]:
    """Move a point inward by offset distance."""
    return (point[0] + offset * inward_dir[0], point[1] + offset * inward_dir[1])


# =============================================================================
# SECTION 2: LampPlacer Class
# =============================================================================


@dataclass
class PlacementResult:
    """Result of lamp placement calculation."""

    position: tuple[float, float]
    aim: tuple[float, float]


class LampPlacer:
    """
    Coordinates lamp placement within a room polygon.

    Encapsulates polygon geometry and tracks existing lamp positions
    to calculate optimal placements for different modes.

    Example usage:
        # Simple usage - just get positions
        placer = LampPlacer.for_room(x=4, y=4, z=3)
        pos, aim = placer.place("corner", lamp_idx=1)

        # Full usage with lamp objects
        placer = LampPlacer.for_room(x=4, y=4, z=3)
        placer.place_lamp(lamp, mode="corner", tilt=45)
    """

    def __init__(
        self,
        polygon: Polygon2D,
        z: float = None,
        existing_positions: list[tuple[float, float]] = None,
    ):
        self.polygon = polygon
        self.z = z
        self._existing = list(existing_positions) if existing_positions else []

    @classmethod
    def for_room(
        cls,
        x: float = None,
        y: float = None,
        z: float = None,
        polygon: Polygon2D = None,
        existing: list[tuple[float, float]] = None,
    ) -> "LampPlacer":
        """Create placer from room dimensions or polygon."""
        if polygon is None:
            if x is None or y is None:
                raise ValueError("Must provide either polygon or both x and y")
            polygon = Polygon2D.rectangle(x, y)
        return cls(polygon, z=z, existing_positions=existing)

    @classmethod
    def for_dims(cls, dims, existing: list[tuple[float, float]] = None) -> "LampPlacer":
        """Create placer from a RoomDimensions object."""
        return cls(dims.polygon, z=dims.z, existing_positions=existing)

    def place(self, mode: str, lamp_idx: int, **kwargs) -> PlacementResult:
        """
        Calculate position and aim for a lamp.

        Args:
            mode: "downlight", "corner", "edge", or "horizontal"
            lamp_idx: Index of lamp being placed (1-based)
            **kwargs: Mode-specific options (e.g., beam_angle for corner mode)
        """
        handlers = {
            "downlight": self._place_downlight,
            "corner": self._place_corner,
            "edge": self._place_edge,
            "horizontal": self._place_horizontal,
        }
        if mode not in handlers:
            raise ValueError(f"invalid lamp placement mode {mode}")
        return handlers[mode](lamp_idx, **kwargs)

    def record(self, x: float, y: float):
        """Record a placed lamp position for future spacing calculations."""
        self._existing.append((x, y))

    def _place_downlight(self, idx: int, **kwargs) -> PlacementResult:
        x, y = new_lamp_position_polygon(idx, self.polygon)
        return PlacementResult(position=(x, y), aim=(x, y))

    def _place_corner(self, idx: int, **kwargs) -> PlacementResult:
        beam_angle = kwargs.get("beam_angle", 30.0)
        wall_offset = kwargs.get("wall_offset", 0.05)
        (x, y), (aim_x, aim_y) = new_lamp_position_corner(
            idx, self.polygon, self._existing, beam_angle=beam_angle, wall_offset=wall_offset
        )
        return PlacementResult(position=(x, y), aim=(aim_x, aim_y))

    def _place_edge(self, idx: int, **kwargs) -> PlacementResult:
        wall_offset = kwargs.get("wall_offset", 0.05)
        (x, y), (aim_x, aim_y) = new_lamp_position_edge(
            idx, self.polygon, self._existing, wall_offset=wall_offset
        )
        return PlacementResult(position=(x, y), aim=(aim_x, aim_y))

    def _place_horizontal(self, idx: int, **kwargs) -> PlacementResult:
        # Same as edge but caller handles aim z-coordinate
        return self._place_edge(idx, **kwargs)

    def place_lamp(
        self,
        lamp,
        mode: str = None,
        tilt: float = None,
        max_tilt: float = None,
        offset: float = None,
        wall_clearance: float = None,
        angle: float = None,
    ):
        """
        Position and aim a lamp, returning it for chaining.

        Args:
            lamp: Lamp object to position
            mode: Placement mode ("downlight", "corner", "edge", "horizontal").
                If None, uses the lamp's config default or "downlight".
            tilt: Force exact tilt angle in degrees (0=down, 90=horizontal)
            max_tilt: Maximum allowed tilt angle in degrees. If None, uses the
                lamp's config default.
            offset: Distance below ceiling to place lamp. If None, calculated from
                fixture.housing_height + 0.02 margin (minimum 0.05).
            wall_clearance: Distance from walls for corner/edge modes. If None,
                calculated from fixture diagonal to account for rotation when aiming
                (minimum 0.05).

        Returns:
            The lamp object for method chaining
        """
        if self.z is None:
            raise ValueError("z must be set to use place_lamp (use for_room or for_dims)")

        # Get placement defaults from lamp config if available
        config_angle = 0
        if mode is None or max_tilt is None or angle is None:
            try:
                _, config = resolve_keyword(lamp.lamp_id)
                placement = config.get("placement", {})
                if mode is None:
                    mode = placement.get("mode", "downlight")
                if max_tilt is None:
                    max_tilt = placement.get("max_tilt")
                config_angle = placement.get("angle", 0)
            except KeyError:
                if mode is None:
                    mode = "downlight"
        fixture_angle = angle if angle is not None else config_angle

        # Calculate ceiling offset from fixture if not specified
        if offset is None:
            fixture = getattr(lamp, "fixture", None)
            if fixture is not None and fixture.housing_height > 0:
                offset = fixture.housing_height + 0.02
                offset = max(offset, 0.05)
            else:
                offset = 0.1  # Default when no fixture dimensions

        # Calculate wall clearance from fixture dimensions
        if wall_clearance is None:
            fixture = getattr(lamp, "fixture", None)
            if fixture is not None and fixture.has_dimensions:
                # 2D footprint diagonal plus housing height for tilt contribution.
                # Post-placement nudge corrects any remaining overshoot.
                w, l, h = fixture.housing_width, fixture.housing_length, fixture.housing_height
                diagonal_2d = (w**2 + l**2) ** 0.5
                wall_clearance = diagonal_2d / 2 + h / 2
                wall_clearance = max(wall_clearance, 0.05)
            else:
                wall_clearance = 0.05  # Default when no fixture dimensions

        idx = len(self._existing) + 1
        beam_angle = self._get_beam_angle(lamp)
        mode_lower = mode.lower()

        result = self.place(mode_lower, idx, beam_angle=beam_angle, wall_offset=wall_clearance)

        lamp_z = self.z - offset
        lamp.move(result.position[0], result.position[1], lamp_z)

        if mode_lower == "horizontal":
            aim_z = lamp_z
        else:
            aim_z = 0.0

        if mode_lower in ("corner", "edge"):
            aim_xy = self._apply_tilt(lamp.position, result.aim, tilt, max_tilt)
        else:
            aim_xy = result.aim

        lamp.aim(aim_xy[0], aim_xy[1], aim_z)
        if fixture_angle:
            lamp.rotate(fixture_angle)
        self._nudge_into_bounds(lamp)
        self.record(lamp.x, lamp.y)
        return lamp

    def _get_beam_angle(self, lamp, default: float = 30.0) -> float:
        """Extract beam angle from lamp photometry."""
        try:
            if lamp.ies and lamp.ies.photometry:
                return lamp.ies.photometry.beam_angle
        except (AttributeError, TypeError):
            pass
        return default

    def _apply_tilt(
        self,
        lamp_pos: tuple[float, float, float],
        aim_point: tuple[float, float],
        tilt: float,
        max_tilt: float,
    ) -> tuple[float, float]:
        """Apply tilt constraints to aim point."""
        if tilt is not None:
            return apply_tilt(lamp_pos, aim_point, self.polygon, tilt)
        elif max_tilt is not None:
            return clamp_aim_to_max_tilt(lamp_pos, aim_point, max_tilt, self.polygon)
        return aim_point

    def _nudge_into_bounds(self, lamp, max_iterations: int = 3):
        """Nudge lamp so its bounding box stays within the room polygon and z bounds."""
        fixture = getattr(lamp, "fixture", None)
        if fixture is None or not fixture.has_dimensions:
            return

        margin = 1e-4
        for _ in range(max_iterations):
            corners = lamp.geometry.get_bounding_box_corners()  # (8, 3)
            dx = 0.0
            dy = 0.0
            dz = 0.0

            for corner in corners:
                cx, cy, cz = corner
                # XY check
                if not self.polygon.contains_point_inclusive(cx, cy):
                    nearest = _nearest_point_on_polygon_boundary((cx, cy), self.polygon)
                    nudge_x = nearest[0] - cx
                    nudge_y = nearest[1] - cy
                    # Keep the largest-magnitude nudge per axis
                    if abs(nudge_x) > abs(dx):
                        dx = nudge_x
                    if abs(nudge_y) > abs(dy):
                        dy = nudge_y
                # Z check
                if cz > self.z:
                    shift = self.z - cz
                    if shift < dz:
                        dz = shift
                if cz < 0:
                    shift = -cz
                    if shift > dz:
                        dz = shift

            if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
                return  # Already in bounds

            # Apply nudge with small inward margin
            if dx != 0:
                dx += margin if dx > 0 else -margin
            if dy != 0:
                dy += margin if dy > 0 else -margin
            if dz != 0:
                dz += margin if dz > 0 else -margin
            lamp.move(lamp.x + dx, lamp.y + dy, lamp.z + dz)

        # After max iterations, warn if still out of bounds
        corners = lamp.geometry.get_bounding_box_corners()
        for corner in corners:
            cx, cy, cz = corner
            if not self.polygon.contains_point_inclusive(cx, cy) or cz > self.z or cz < 0:
                warnings.warn(
                    "Fixture bounding box still extends past room boundaries after nudging"
                )
                return


# =============================================================================
# SECTION 3: Low-level Geometry Helpers
# =============================================================================


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

        if _segments_cross(p1, p2, (x1, y1), (x2, y2)):
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


def _distance_to_polygon_boundary(point: np.ndarray, polygon: "Polygon2D") -> float:
    """Calculate minimum distance from point to any polygon edge."""
    px, py = point
    min_dist = float("inf")

    for (x1, y1), (x2, y2) in polygon.edges:
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)

    return min_dist


def _nearest_point_on_polygon_boundary(
    point: tuple[float, float], polygon: "Polygon2D"
) -> tuple[float, float]:
    """Return the closest point on the polygon boundary to the given point."""
    px, py = point
    best_dist = float("inf")
    best_point = (px, py)

    for (x1, y1), (x2, y2) in polygon.edges:
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            proj_x, proj_y = x1, y1
        else:
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
        dist = hypot(px - proj_x, py - proj_y)
        if dist < best_dist:
            best_dist = dist
            best_point = (proj_x, proj_y)

    return best_point


def _ray_polygon_intersection(
    origin: tuple[float, float],
    direction: tuple[float, float],
    polygon: "Polygon2D",
    origin_edge_idx: int = -1,
) -> tuple[float, float] | None:
    """
    Find where a ray from origin in given direction intersects the polygon boundary.

    Args:
        origin: Starting point of ray
        direction: Direction vector (will be normalized)
        polygon: The polygon boundary
        origin_edge_idx: Edge to skip (if ray starts on an edge)

    Returns:
        Intersection point, or None if no intersection found
    """
    ox, oy = origin
    dx, dy = _normalize(*direction)
    if dx == 0.0 and dy == 0.0:
        return None

    best_t = float("inf")
    best_point = None

    for i, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
        if i == origin_edge_idx:
            continue

        # Edge direction
        ex, ey = x2 - x1, y2 - y1

        # Solve for intersection: origin + t*dir = edge_start + s*edge_dir
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-10:
            continue  # Parallel

        t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
        s = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

        # Check if intersection is valid (t > 0 for ray, 0 <= s <= 1 for segment)
        if t > 1e-6 and 0 <= s <= 1:
            if t < best_t:
                best_t = t
                best_point = (ox + t * dx, oy + t * dy)

    return best_point


# =============================================================================
# SECTION 4: Corner Placement Algorithms
# =============================================================================


def get_corners(polygon: Polygon2D) -> list[tuple[float, float]]:
    """
    Return convex vertices (interior angle < 180°) as potential corner lamp positions.

    Excludes reflex vertices (interior angle > 180°) because they have poor visibility
    and are tucked into concave regions of the room.
    """
    reflex = set(_find_reflex_vertices(polygon))
    return [v for v in polygon.vertices if v not in reflex]


def _calculate_visible_floor_area(
    corner: tuple[float, float], polygon: Polygon2D, num_rays: int = 36
) -> float:
    """
    Estimate floor area visible from corner position by casting rays.

    Uses a simplified approach: cast rays in all directions and sum the distances
    to polygon boundaries (proportional to visible area in that direction).
    """
    cx, cy = corner
    total_dist = 0.0

    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        dx, dy = np.cos(angle), np.sin(angle)

        # Find intersection with polygon boundary
        intersection = _ray_polygon_intersection(corner, (dx, dy), polygon)
        if intersection:
            ix, iy = intersection
            total_dist += hypot(ix - cx, iy - cy)

    return total_dist


def _rank_corners_by_visibility(
    polygon: Polygon2D, num_rays: int = 36
) -> list[int]:
    """
    Return corner indices sorted by visible floor area (most visible first).
    Uses angle from centroid as tiebreaker for consistent ordering on symmetric shapes.
    """
    corners = get_corners(polygon)
    cx, cy = polygon.centroid
    visibility_scores = []

    for i, corner in enumerate(corners):
        # Offset corner slightly inward to avoid edge-case issues
        dx, dy = cx - corner[0], cy - corner[1]
        length = hypot(dx, dy)
        if length > 0:
            offset = 0.01  # Small inward offset
            test_point = (corner[0] + offset * dx / length, corner[1] + offset * dy / length)
        else:
            test_point = corner

        score = _calculate_visible_floor_area(test_point, polygon, num_rays)
        # Angle from centroid as tiebreaker (for consistent ordering on symmetric shapes)
        angle = np.arctan2(corner[1] - cy, corner[0] - cx)
        visibility_scores.append((i, score, angle))

    # Sort by score descending, then by angle for consistent tiebreaking
    visibility_scores.sort(key=lambda x: (-x[1], x[2]))
    return [idx for idx, _, _ in visibility_scores]


def _get_corner_edge_directions(
    corner: tuple[float, float],
    polygon: Polygon2D,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get the two edge directions at a corner (pointing away from corner)."""
    vertices = list(polygon.vertices)
    try:
        idx = vertices.index(corner)
    except ValueError:
        return ((1, 0), (0, 1))  # Fallback

    n = len(vertices)
    prev_v = vertices[(idx - 1) % n]
    next_v = vertices[(idx + 1) % n]

    # Directions along edges (away from corner), normalized
    d1 = _normalize(prev_v[0] - corner[0], prev_v[1] - corner[1])
    d2 = _normalize(next_v[0] - corner[0], next_v[1] - corner[1])

    return (d1, d2)


def _angle_to_wall(
    direction: tuple[float, float],
    wall_dir: tuple[float, float],
) -> float:
    """Calculate angle between a direction and a wall (0-90 degrees)."""
    # Dot product gives cos of angle
    dot = abs(direction[0] * wall_dir[0] + direction[1] * wall_dir[1])
    # Clamp to valid range for acos
    dot = min(1.0, max(-1.0, dot))
    angle = np.arccos(dot)
    # Convert to degrees and ensure 0-90 range
    return min(degrees(angle), 90.0)


def _calculate_corner_aim(
    corner: tuple[float, float],
    polygon: Polygon2D,
    beam_angle: float = 30.0,
    num_candidates: int = 72,
    min_wall_angle: float = 15.0,
) -> tuple[float, float]:
    """
    Calculate optimal aim point for a corner-mounted lamp.

    Finds the farthest visible point that isn't too parallel to adjacent walls.
    This maximizes coverage while avoiding losing cone output to nearby walls.
    Samples angles relative to the corner bisector for symmetric results.

    Args:
        corner: The corner position
        polygon: Room polygon
        beam_angle: Lamp beam angle in degrees
        num_candidates: Number of directions to sample
        min_wall_angle: Minimum angle (degrees) from wall direction to consider
    """
    edge_dirs = _get_corner_edge_directions(corner, polygon)

    # Get bisector as reference angle for symmetric sampling
    inward = _get_corner_inward_direction(corner, polygon)
    bisector_angle = np.arctan2(inward[1], inward[0])

    best_point = polygon.centroid
    best_score = -1.0

    # Sample angles symmetrically around the bisector
    for i in range(num_candidates):
        # Offset from bisector: -pi to +pi, centered on bisector
        offset = (i / num_candidates - 0.5) * 2 * np.pi
        angle = bisector_angle + offset
        dx, dy = np.cos(angle), np.sin(angle)

        intersection = _ray_polygon_intersection(corner, (dx, dy), polygon)
        if intersection is None:
            continue

        # Check angle to both adjacent walls
        angle_to_wall1 = _angle_to_wall((dx, dy), edge_dirs[0])
        angle_to_wall2 = _angle_to_wall((dx, dy), edge_dirs[1])
        min_angle = min(angle_to_wall1, angle_to_wall2)

        # Skip directions too parallel to walls
        if min_angle < min_wall_angle:
            continue

        ix, iy = intersection
        dist = hypot(ix - corner[0], iy - corner[1])

        # Score by distance, with bonus for being far from walls
        wall_bonus = min_angle / 90.0  # 0 to 1 based on angle from walls
        score = dist * (0.5 + 0.5 * wall_bonus)

        if score > best_score:
            best_score = score
            best_point = intersection

    # If no valid candidates found (all too close to walls), use bisector
    if best_score < 0:
        aim = _ray_polygon_intersection(corner, inward, polygon)
        if aim is not None:
            return aim

    return best_point


def _get_corner_inward_direction(
    corner: tuple[float, float],
    polygon: Polygon2D,
) -> tuple[float, float]:
    """
    Calculate the inward-pointing bisector direction at a corner.

    Returns a unit vector pointing into the room interior.
    """
    # Find the vertex index
    vertices = list(polygon.vertices)
    try:
        idx = vertices.index(corner)
    except ValueError:
        # Corner not in vertices, fall back to centroid direction
        cx, cy = polygon.centroid
        return _normalize(cx - corner[0], cy - corner[1])

    n = len(vertices)
    prev_v = vertices[(idx - 1) % n]
    next_v = vertices[(idx + 1) % n]

    # Vectors along edges from corner, normalized
    v1 = _normalize(prev_v[0] - corner[0], prev_v[1] - corner[1])
    v2 = _normalize(next_v[0] - corner[0], next_v[1] - corner[1])

    # Bisector is sum of unit vectors, normalized
    bx, by = _normalize(v1[0] + v2[0], v1[1] + v2[1])
    if bx == 0.0 and by == 0.0:
        # Edges are parallel, use perpendicular
        bx, by = -v1[1], v1[0]

    # Test if bisector points inward (into polygon)
    test_point = (corner[0] + bx * 0.01, corner[1] + by * 0.01)
    if not polygon.contains_point(*test_point):
        # Flip direction
        bx, by = -bx, -by

    return (bx, by)


def new_lamp_position_corner(
    lamp_idx: int,
    polygon: Polygon2D,
    existing_positions: list[tuple[float, float]] | None = None,
    beam_angle: float = 30.0,
    wall_offset: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get position and aim point for corner placement.

    Places lamps at corners ranked by visibility. When all corners are occupied,
    delegates to edge placement.

    Args:
        lamp_idx: 1-based lamp index
        polygon: Room polygon
        existing_positions: Positions of already-placed lamps
        beam_angle: Lamp beam angle in degrees
        wall_offset: Distance to offset from corner (default 0.05)

    Returns:
        (position, aim_point) tuple
    """
    if existing_positions is None:
        existing_positions = []

    corners = get_corners(polygon)
    ranked_indices = _rank_corners_by_visibility(polygon)

    # Find available corners (not already occupied)
    # Tolerance must account for the wall_offset (lamp is offset from corner)
    # Use wall_offset * sqrt(2) since offset is along the diagonal, plus margin
    tolerance = max(0.2, wall_offset * 1.5 + 0.1)
    occupied_corners = set()
    for ex, ey in existing_positions:
        for i, (cx, cy) in enumerate(corners):
            if hypot(ex - cx, ey - cy) < tolerance:
                occupied_corners.add(i)

    available = [i for i in ranked_indices if i not in occupied_corners]

    if available:
        # Place at best available corner
        corner_idx = available[0]
        corner = corners[corner_idx]

        # Get inward direction using corner bisector
        inward = _get_corner_inward_direction(corner, polygon)
        position = _offset_inward(corner, inward, offset=wall_offset)

        # Verify position is inside polygon; if not, reduce offset
        if not polygon.contains_point(*position):
            position = _offset_inward(corner, inward, offset=min(wall_offset, 0.01))

        # Final check - if still outside, use corner directly
        if not polygon.contains_point(*position):
            position = corner

        aim = _calculate_corner_aim(position, polygon, beam_angle)
        return position, aim
    else:
        # All corners occupied, delegate to edge placement
        return new_lamp_position_edge(lamp_idx, polygon, existing_positions, wall_offset=wall_offset)


# =============================================================================
# SECTION 5: Edge Placement Algorithms
# =============================================================================


def get_edge_centers(polygon: Polygon2D) -> list[tuple[float, float, int]]:
    """
    Return (x, y, edge_index) for each edge midpoint.
    """
    centers = []
    for i, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        centers.append((mx, my, i))
    return centers


def _calculate_edge_perpendicular_aim(
    lamp_pos: tuple[float, float],
    edge_index: int,
    polygon: Polygon2D,
) -> tuple[float, float]:
    """
    Find where perpendicular ray from edge intersects opposite boundary.

    The perpendicular direction points inward (opposite of outward normal).
    """
    # Get inward normal (opposite of outward normal)
    outward = polygon.edge_normals[edge_index]
    inward = (-outward[0], -outward[1])

    # Cast ray from lamp position in inward direction
    intersection = _ray_polygon_intersection(lamp_pos, inward, polygon, edge_index)

    if intersection:
        return intersection
    else:
        # Fallback to centroid if no intersection found
        return polygon.centroid


def _calculate_sightline_distance(
    position: tuple[float, float],
    edge_idx: int,
    polygon: Polygon2D,
) -> float:
    """Calculate the perpendicular sightline distance from edge position into room."""
    aim = _calculate_edge_perpendicular_aim(position, edge_idx, polygon)
    return hypot(aim[0] - position[0], aim[1] - position[1])


def _best_position_on_edge(
    edge_idx: int,
    polygon: Polygon2D,
    num_samples: int = 11,
) -> tuple[tuple[float, float], float]:
    """Find the position on an edge with the longest perpendicular sightline.

    Samples evenly-spaced points along the edge, finds the best sightline,
    then returns the midpoint of the region that achieves it.

    Returns:
        (best_position, best_sightline)
    """
    (x1, y1), (x2, y2) = polygon.edges[edge_idx]

    # Sample sightlines along the edge
    t_values = [i / (num_samples - 1) for i in range(num_samples)]
    sightlines = []
    for t in t_values:
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        sightlines.append(_calculate_sightline_distance((px, py), edge_idx, polygon))

    best_sightline = max(sightlines)

    # Collect all t values that achieve the best sightline, pick their center
    best_ts = [t for t, s in zip(t_values, sightlines) if abs(s - best_sightline) <= 1e-6]
    center_t = (min(best_ts) + max(best_ts)) / 2

    best_pos = (x1 + center_t * (x2 - x1), y1 + center_t * (y2 - y1))
    return best_pos, best_sightline


def new_lamp_position_edge(
    lamp_idx: int,
    polygon: Polygon2D,
    existing_positions: list[tuple[float, float]] | None = None,
    wall_offset: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get position and aim point for edge placement.

    Places lamps at edge centers, prioritizing long sightlines (perpendicular distance
    across the room). When all edge centers are occupied, uses farthest-point sampling.

    Args:
        lamp_idx: 1-based lamp index
        polygon: Room polygon
        existing_positions: Positions of already-placed lamps
        wall_offset: Distance to offset from edge (default 0.05)

    Returns:
        (position, aim_point) tuple
    """
    if existing_positions is None:
        existing_positions = []

    edge_centers = get_edge_centers(polygon)

    # Find available edges (not already occupied by a nearby lamp)
    # Tolerance must exceed wall_offset since lamps are offset inward from the edge
    tolerance = max(0.2, wall_offset + 0.1)
    occupied_edges = set()
    for ex, ey in existing_positions:
        for i, (_, _, edge_idx) in enumerate(edge_centers):
            (x1, y1), (x2, y2) = polygon.edges[edge_idx]
            if _point_to_segment_distance(ex, ey, x1, y1, x2, y2) < tolerance:
                occupied_edges.add(i)

    available = [i for i in range(len(edge_centers)) if i not in occupied_edges]

    if available:
        # Score each edge: midpoint sightline primary, best-position sightline tiebreaker.
        # Midpoint sightline preserves ranking for simple rooms; the tiebreaker ensures
        # edges with good non-midpoint positions (e.g. outer walls of concave rooms)
        # rank above edges where every position is poor.
        best_center_idx = None
        best_mid_score = -1.0
        best_edge_score = -1.0

        for i in available:
            mx, my, edge_idx = edge_centers[i]
            mid_sightline = _calculate_sightline_distance((mx, my), edge_idx, polygon)
            _, edge_sightline = _best_position_on_edge(edge_idx, polygon)

            if (mid_sightline > best_mid_score + 1e-6) or (
                abs(mid_sightline - best_mid_score) <= 1e-6
                and edge_sightline > best_edge_score + 1e-6
            ):
                best_mid_score = mid_sightline
                best_edge_score = edge_sightline
                best_center_idx = i

        _, _, edge_idx = edge_centers[best_center_idx]

        # Find best position on the chosen edge (may differ from midpoint in concave rooms)
        best_pos, _ = _best_position_on_edge(edge_idx, polygon)

        # Offset slightly inward from edge
        inward = (-polygon.edge_normals[edge_idx][0], -polygon.edge_normals[edge_idx][1])
        position = _offset_inward(best_pos, inward, offset=wall_offset)

        aim = _calculate_edge_perpendicular_aim(position, edge_idx, polygon)
        return position, aim
    else:
        # All edge centers occupied, use farthest-point sampling along perimeter
        return _farthest_perimeter_position(polygon, existing_positions, wall_offset=wall_offset)


def _farthest_perimeter_position(
    polygon: Polygon2D,
    existing_positions: list[tuple[float, float]],
    num_samples_per_edge: int = 20,
    wall_offset: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Find the perimeter position farthest from existing lamps."""
    best_pos = None
    best_edge_idx = 0
    best_score = -1.0

    for edge_idx, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
        for i in range(num_samples_per_edge):
            t = (i + 0.5) / num_samples_per_edge
            px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

            if existing_positions:
                min_dist = min(hypot(ex - px, ey - py) for ex, ey in existing_positions)
            else:
                min_dist = float("inf")

            if min_dist > best_score:
                best_score = min_dist
                best_pos = (px, py)
                best_edge_idx = edge_idx

    # Offset slightly inward
    inward = (-polygon.edge_normals[best_edge_idx][0], -polygon.edge_normals[best_edge_idx][1])
    position = _offset_inward(best_pos, inward, offset=wall_offset)

    aim = _calculate_edge_perpendicular_aim(position, best_edge_idx, polygon)
    return position, aim


# =============================================================================
# SECTION 6: Visibility / Aiming
# =============================================================================


def farthest_visible_point(
    origin: tuple[float, float], polygon: "Polygon2D", num_divisions: int = 50
) -> tuple[float, float]:
    """
    Find the farthest point inside the polygon visible from origin.

    The returned point has a clear line-of-sight from origin (doesn't cross any
    polygon edges). Used for aiming tilted lamps toward the farthest reachable point.
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


def _calculate_cone_coverage(
    origin: tuple[float, float],
    target: tuple[float, float],
    polygon: Polygon2D,
    beam_half_angle: float,
    num_rays: int = 9,
) -> float:
    """
    Returns 0.0-1.0 indicating what fraction of cone is unobstructed.

    Samples rays across cone width, counts how many reach target distance without
    hitting walls first.
    """
    ox, oy = origin
    tx, ty = target
    target_dist = hypot(tx - ox, ty - oy)

    if target_dist < 1e-6:
        return 1.0

    # Angle to target
    center_angle = np.arctan2(ty - oy, tx - ox)
    half_angle_rad = np.radians(beam_half_angle)

    origin_edge_idx = _find_origin_edge(origin, polygon)
    unobstructed = 0

    for i in range(num_rays):
        # Spread rays across cone
        t = (i / (num_rays - 1)) - 0.5 if num_rays > 1 else 0
        ray_angle = center_angle + t * 2 * half_angle_rad

        dx, dy = np.cos(ray_angle), np.sin(ray_angle)
        intersection = _ray_polygon_intersection(origin, (dx, dy), polygon, origin_edge_idx)

        if intersection:
            ix, iy = intersection
            intersect_dist = hypot(ix - ox, iy - oy)
            # Ray is unobstructed if it reaches at least as far as target
            if intersect_dist >= target_dist * 0.95:
                unobstructed += 1
        else:
            unobstructed += 1

    return unobstructed / num_rays


# =============================================================================
# SECTION 7: Tilt Utilities
# =============================================================================


def calculate_tilt(lamp_z: float, lamp_xy: tuple[float, float], aim_xy: tuple[float, float]) -> float:
    """
    Calculate tilt angle from vertical.

    Args:
        lamp_z: Height of lamp above floor
        lamp_xy: (x, y) position of lamp
        aim_xy: (x, y) position of aim point on floor

    Returns:
        Tilt angle in degrees. 0 = straight down, 90 = horizontal.
    """
    h_dist = hypot(aim_xy[0] - lamp_xy[0], aim_xy[1] - lamp_xy[1])
    v_dist = lamp_z  # aiming at floor (z=0)

    if v_dist <= 0:
        return 90.0
    return degrees(atan(h_dist / v_dist))


def clamp_aim_to_max_tilt(
    lamp_pos: tuple[float, float, float],
    aim_point: tuple[float, float],
    max_tilt: float,
    polygon: Polygon2D,
) -> tuple[float, float]:
    """
    If calculated tilt > max_tilt, recalculate aim point at max_tilt angle.

    Ensures the adjusted aim point stays inside the polygon.

    Args:
        lamp_pos: (x, y, z) lamp position
        aim_point: Current (x, y) aim point on floor
        max_tilt: Maximum allowed tilt angle in degrees
        polygon: Room polygon

    Returns:
        Adjusted (x, y) aim point
    """
    lx, ly, lz = lamp_pos
    ax, ay = aim_point

    current_tilt = calculate_tilt(lz, (lx, ly), (ax, ay))

    if current_tilt <= max_tilt:
        return aim_point

    # Calculate new aim point at max_tilt
    # horizontal distance = z * tan(max_tilt)
    max_h_dist = lz * np.tan(np.radians(max_tilt))

    # Direction from lamp to original aim
    dx, dy = ax - lx, ay - ly
    current_h_dist = hypot(dx, dy)

    if current_h_dist < 1e-6:
        return aim_point

    # Scale direction to max_h_dist
    scale = max_h_dist / current_h_dist
    new_ax = lx + dx * scale
    new_ay = ly + dy * scale

    # Ensure aim is inside polygon
    if not polygon.contains_point(new_ax, new_ay):
        # Find intersection with polygon boundary along this direction
        intersection = _ray_polygon_intersection(
            (lx, ly), (dx, dy), polygon, _find_origin_edge((lx, ly), polygon)
        )
        if intersection:
            ix, iy = intersection
            # Use point just inside boundary
            new_ax = lx + (ix - lx) * 0.95
            new_ay = ly + (iy - ly) * 0.95

    return (new_ax, new_ay)


def apply_tilt(
    lamp_pos: tuple[float, float, float],
    aim_point: tuple[float, float],
    polygon: Polygon2D,
    tilt: float,
) -> tuple[float, float]:
    """
    Force exact tilt angle, keeping the aim direction the same.

    Args:
        lamp_pos: (x, y, z) lamp position
        aim_point: Current (x, y) aim point
        polygon: Room polygon
        tilt: Exact tilt angle in degrees

    Returns:
        New (x, y) aim point at specified tilt
    """
    lx, ly, lz = lamp_pos
    ax, ay = aim_point

    # Direction from lamp to original aim
    dx, dy = ax - lx, ay - ly
    current_h_dist = hypot(dx, dy)

    # Normalize direction (use arbitrary direction if aim is directly below lamp)
    dx, dy = _normalize(dx, dy)
    if dx == 0.0 and dy == 0.0:
        dx, dy = 1.0, 0.0

    # Calculate horizontal distance for exact tilt
    h_dist = lz * np.tan(np.radians(tilt))

    new_ax = lx + dx * h_dist
    new_ay = ly + dy * h_dist

    # Ensure aim is inside polygon
    if not polygon.contains_point(new_ax, new_ay):
        intersection = _ray_polygon_intersection(
            (lx, ly), (dx, dy), polygon, _find_origin_edge((lx, ly), polygon)
        )
        if intersection:
            ix, iy = intersection
            new_ax = lx + (ix - lx) * 0.95
            new_ay = ly + (iy - ly) * 0.95

    return (new_ax, new_ay)


# =============================================================================
# SECTION 8: Downlight Placement
# =============================================================================


def get_lamp_positions(num_lamps, x, y, num_divisions=100):
    """
    Generate a list of (x,y) positions for lamps given room dimensions and
    the number of lamps desired.
    """
    lst = [new_lamp_position(i + 1, x, y) for i in range(num_lamps)]
    return np.array(lst).T


def new_lamp_position(lamp_idx, x, y, num_divisions=100):
    """
    Get the default position for an additional new lamp.
    x and y are the room dimensions. lamp_idx is 1-based.
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
    grid[center] = 1

    for _ in range(1, num_points):
        max_dist = -1
        best_point = None

        for x in range(M):
            for y in range(N):
                if grid[x, y] == 0:
                    min_point_dist = min(
                        [np.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in points]
                    )
                    min_boundary_dist = min(x, M - 1 - x, y, N - 1 - y)
                    min_dist = min(min_point_dist, min_boundary_dist)

                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_point = (x, y)

        if best_point:
            points.append(best_point)
            grid[best_point] = 1
    return points


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
        return polygon.centroid

    # Get indices for this lamp using farthest-point sampling
    chosen_idx = _place_points_in_polygon(inside_points, polygon, lamp_idx)
    return tuple(inside_points[chosen_idx])


def _place_points_in_polygon(candidates: np.ndarray, polygon: "Polygon2D", num_points: int):
    """
    Place points inside polygon using rectangular decomposition + farthest-point sampling.
    Returns the index of the last placed point.
    """
    rect_centroids = _get_rectangle_centroids(polygon)

    chosen_indices = []
    chosen_set = set()

    for lamp_num in range(num_points):
        if lamp_num < len(rect_centroids):
            best_idx = _find_closest_candidate(candidates, rect_centroids[lamp_num])
            while best_idx in chosen_set and lamp_num < len(rect_centroids):
                lamp_num += 1
                if lamp_num < len(rect_centroids):
                    best_idx = _find_closest_candidate(candidates, rect_centroids[lamp_num])
                else:
                    best_idx = None
                    break

        if lamp_num >= len(rect_centroids) or best_idx is None or best_idx in chosen_set:
            best_idx = None
            best_min_dist = -1.0

            for i, pt in enumerate(candidates):
                if i in chosen_set:
                    continue

                if not chosen_indices:
                    min_dist = _distance_to_polygon_boundary(pt, polygon)
                else:
                    min_dist_to_chosen = min(
                        np.sqrt((pt[0] - candidates[j, 0]) ** 2 + (pt[1] - candidates[j, 1]) ** 2)
                        for j in chosen_indices
                    )
                    min_dist_to_edge = _distance_to_polygon_boundary(pt, polygon)
                    min_dist = min(min_dist_to_chosen, min_dist_to_edge)

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

        if best_idx is not None:
            chosen_indices.append(best_idx)
            chosen_set.add(best_idx)

    return chosen_indices[-1] if chosen_indices else 0


def _median_distance_to_edges(point: np.ndarray, polygon: "Polygon2D") -> float:
    """Calculate median distance from point to all polygon edges."""
    px, py = point
    dists = []

    for (x1, y1), (x2, y2) in polygon.edges:
        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        dists.append(dist)

    return float(np.median(dists))


def _get_rectangle_centroids(polygon: "Polygon2D") -> list[tuple[float, float]]:
    """
    Decompose an axis-aligned rectilinear polygon into rectangles and return their centroids.
    Returns centroids sorted by rectangle area (largest first).
    """
    reflex_vertices = _find_reflex_vertices(polygon)

    if not reflex_vertices:
        return [polygon.centroid]

    rectangles = _decompose_rectilinear(polygon, reflex_vertices)

    if not rectangles:
        return [polygon.centroid]

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

        cross = (curr_v[0] - prev_v[0]) * (next_v[1] - curr_v[1]) - \
                (curr_v[1] - prev_v[1]) * (next_v[0] - curr_v[0])

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

    x_coords = {x_min, x_max}
    y_coords = {y_min, y_max}

    for vx, vy in polygon.vertices:
        x_coords.add(vx)
        y_coords.add(vy)

    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    rectangles = []
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            x1, x2 = x_coords[i], x_coords[i + 1]
            y1, y2 = y_coords[j], y_coords[j + 1]

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if polygon.contains_point(cx, cy):
                rectangles.append((x1, y1, x2, y2))

    return rectangles


def _find_closest_candidate(candidates: np.ndarray, target: tuple[float, float]) -> int:
    """Find the index of the candidate point closest to the target."""
    tx, ty = target
    dists = (candidates[:, 0] - tx) ** 2 + (candidates[:, 1] - ty) ** 2
    return int(np.argmin(dists))
