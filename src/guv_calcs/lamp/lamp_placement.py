"""Lamp placement algorithms for various room geometries and placement modes."""

import warnings
from dataclasses import dataclass
from math import atan, degrees, hypot
import numpy as np
from ..geometry import Polygon2D
from .lamp_configs import resolve_keyword


# Placement order: corners most constrained (finite convex vertices), edges next
# (perimeter-bound), downlights most flexible (any interior point).
_MODE_ORDER = ("corner", "edge", "horizontal", "downlight")

_VALID_MODES = set(_MODE_ORDER)

_VALID_AIM_MODES = {"down", "point", "direction", "centroid", "furthest_edge", "furthest_corner"}


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
    """Complete 3D lamp placement result."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    aimx: float = 0.0
    aimy: float = 0.0
    aimz: float = 0.0
    angle: float = 0.0
    index: int = 0
    count: int = 1

    @property
    def position(self) -> tuple[float, float]:
        """2D position (for geometry handlers and backward compat)."""
        return (self.x, self.y)

    @property
    def aim(self) -> tuple[float, float]:
        """2D aim point (for geometry handlers and backward compat)."""
        return (self.aimx, self.aimy)

    @property
    def heading(self) -> float:
        """Azimuth angle in degrees (0-360)."""
        dx, dy = self.aimx - self.x, self.aimy - self.y
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return 0.0
        return float(np.degrees(np.arctan2(dy, dx)) % 360)

    @property
    def bank(self) -> float:
        """Tilt angle from vertical in degrees (0=down, 90=horizontal)."""
        dx = self.aimx - self.x
        dy = self.aimy - self.y
        dz = self.aimz - self.z
        norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        if norm == 0:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(-dz / norm, -1.0, 1.0))))


@dataclass
class AimResult:
    """Result of an aim computation for a single lamp."""

    aimx: float
    aimy: float
    aimz: float


class LampPlacer:
    """
    Coordinates lamp placement within a room polygon.

    Encapsulates polygon geometry and tracks existing lamp positions
    to calculate optimal placements for different modes.

    Example usage:
        placer = LampPlacer.for_room(x=4, y=4, z=3)

        # Query-only — returns PlacementResult without mutating lamp
        result = placer.get_placement(lamp, mode="corner")

        # Mutate lamp — moves, aims, rotates, nudges into bounds
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
        # Lazy caches for batch operations (populated on first use)
        self._corner_cache = None  # list[int] — ranked corner indices
        self._edge_cache = None  # list[int] — ranked edge indices
        self._grid_cache = None  # dict with interior grid + boundary dists

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

    def get_placement(
        self,
        lamp,
        mode: str = None,
        tilt: float = None,
        max_tilt: float = None,
        offset: float = None,
        wall_clearance: float = None,
        angle: float = None,
        position_index: int = None,
    ) -> PlacementResult:
        """
        Compute lamp placement without mutating the lamp.

        When position_index is None (default), computes the best available
        position, skipping occupied spots.

        When position_index is provided, computes the Nth-ranked position
        regardless of occupancy (used for interactive cycling).

        Args:
            lamp: Lamp object (read for config/fixture info, not mutated)
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
            position_index: 0-based rank for cycling (wraps via modulo).
                If None, uses auto-placement.

        Returns:
            PlacementResult with full 3D position, aim, angle, and cycling metadata
        """
        if self.z is None:
            raise ValueError("z must be set (use for_room or for_dims)")

        # Resolve config from lamp config and explicit params
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
        if offset is None:
            offset = self.ceiling_offset(lamp)
        if wall_clearance is None:
            wall_clearance = self.wall_clearance(lamp)
        beam_angle = self._get_beam_angle(lamp)
        mode = mode.lower()

        # Get 2D position and aim
        if position_index is not None:
            if mode == "corner":
                result = self._ranked_corner(position_index, beam_angle=beam_angle, wall_offset=wall_clearance)
            elif mode in ("edge", "horizontal"):
                result = self._ranked_edge(position_index, beam_angle=beam_angle, wall_offset=wall_clearance)
            else:
                raise ValueError(f"Indexed placement not supported for mode '{mode}'")
        else:
            idx = len(self._existing) + 1
            if mode == "downlight":
                result = self._place_downlight(idx)
            elif mode == "corner":
                result = self._place_corner(idx, beam_angle=beam_angle, wall_offset=wall_clearance)
            elif mode in ("edge", "horizontal"):
                result = self._place_edge(idx, wall_offset=wall_clearance)
            else:
                raise ValueError(f"invalid lamp placement mode {mode}")

        # Fill 3D fields
        result.z = self.z - offset
        result.aimz = result.z if mode == "horizontal" else 0.0
        result.angle = fixture_angle

        if mode in ("corner", "edge"):
            aim_xy = self._apply_tilt((result.x, result.y, result.z), result.aim, tilt, max_tilt)
            result.aimx, result.aimy = aim_xy

        return result

    def place_lamp(self, lamp, mode=None, tilt=None, max_tilt=None,
                   offset=None, wall_clearance=None, angle=None):
        """Compute placement, apply it to the lamp, and record the position."""
        result = self.get_placement(
            lamp, mode=mode, tilt=tilt, max_tilt=max_tilt,
            offset=offset, wall_clearance=wall_clearance, angle=angle,
        )
        lamp.move(result.x, result.y, result.z)
        lamp.aim(result.aimx, result.aimy, result.aimz)
        if result.angle:
            lamp.rotate(result.angle)
        self._nudge_into_bounds(lamp)
        self.record(lamp.x, lamp.y)
        return lamp

    def place_lamp_at_index(self, lamp, mode, index, *, max_tilt=None,
                            offset=None, wall_clearance=None, angle=None):
        """Deprecated: use get_placement(position_index=index) instead."""
        return self.get_placement(
            lamp, mode=mode, max_tilt=max_tilt, offset=offset,
            wall_clearance=wall_clearance, angle=angle, position_index=index,
        )

    def record(self, x: float, y: float):
        """Record a placed lamp position for future spacing calculations."""
        self._existing.append((x, y))

    def _place_downlight(self, idx: int, **kwargs) -> PlacementResult:
        x, y = new_lamp_position_downlight(
            self.polygon, self._existing
        )
        return PlacementResult(x=x, y=y, aimx=x, aimy=y)

    def _place_corner(self, idx: int, **kwargs) -> PlacementResult:
        beam_angle = kwargs.get("beam_angle", 30.0)
        wall_offset = kwargs.get("wall_offset", 0.05)
        (x, y), (aim_x, aim_y) = new_lamp_position_corner(
            idx, self.polygon, self._existing, beam_angle=beam_angle, wall_offset=wall_offset
        )
        return PlacementResult(x=x, y=y, aimx=aim_x, aimy=aim_y)

    def _place_edge(self, idx: int, **kwargs) -> PlacementResult:
        wall_offset = kwargs.get("wall_offset", 0.05)
        (x, y), (aim_x, aim_y) = new_lamp_position_edge(
            idx, self.polygon, self._existing, wall_offset=wall_offset
        )
        return PlacementResult(x=x, y=y, aimx=aim_x, aimy=aim_y)

    @staticmethod
    def ceiling_offset(lamp) -> float:
        """Compute ceiling offset from fixture housing dimensions.

        Returns the distance below the ceiling to place a lamp, based on
        its fixture housing height plus a small margin.
        """
        fixture = getattr(lamp, "fixture", None)
        if fixture is not None and fixture.housing_height > 0:
            return max(fixture.housing_height + 0.02, 0.05)
        return 0.1

    @staticmethod
    def wall_clearance(lamp) -> float:
        """Compute wall clearance from fixture dimensions.

        Returns the minimum distance from walls to avoid fixture collision,
        based on the 2D diagonal of the fixture footprint.
        """
        fixture = getattr(lamp, "fixture", None)
        if fixture is None or not fixture.has_dimensions:
            return 0.05
        w = fixture.housing_width or 0
        l = fixture.housing_length or 0
        h = fixture.housing_height or 0
        diagonal_2d = (w**2 + l**2) ** 0.5
        return max(diagonal_2d / 2 + h / 2, 0.05)

    def _ranked_corner(self, index: int, **kwargs) -> PlacementResult:
        """Return placement for the Nth-ranked corner."""
        wall_offset = kwargs.get("wall_offset", 0.05)
        beam_angle = kwargs.get("beam_angle", 30.0)

        corners = get_corners(self.polygon)
        ranked = _rank_corners_by_visibility(self.polygon)
        count = len(ranked)
        idx = index % count

        corner = corners[ranked[idx]]
        inward = _get_corner_inward_direction(corner, self.polygon)
        position = _offset_inward(corner, inward, offset=wall_offset)

        if not self.polygon.contains_point(*position):
            position = _offset_inward(corner, inward, offset=min(wall_offset, 0.01))
        if not self.polygon.contains_point(*position):
            position = corner

        aim = _calculate_corner_aim(position, self.polygon, beam_angle)
        return PlacementResult(x=position[0], y=position[1], aimx=aim[0], aimy=aim[1], index=idx, count=count)

    def _ranked_edge(self, index: int, **kwargs) -> PlacementResult:
        """Return placement for the Nth-ranked edge."""
        wall_offset = kwargs.get("wall_offset", 0.05)

        ranked = _rank_edges_by_sightline(self.polygon)
        count = len(ranked)
        idx = index % count
        chosen_edge_idx = ranked[idx]

        best_pos, _ = _best_position_on_edge(chosen_edge_idx, self.polygon)
        outward = self.polygon.edge_normals[chosen_edge_idx]
        inward = (-outward[0], -outward[1])
        position = _offset_inward(best_pos, inward, offset=wall_offset)

        aim = _calculate_edge_perpendicular_aim(position, chosen_edge_idx, self.polygon)
        return PlacementResult(x=position[0], y=position[1], aimx=aim[0], aimy=aim[1], index=idx, count=count)

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
        from ..geometry import RoomDimensions
        room_dims = RoomDimensions(polygon=self.polygon, z=self.z)
        lamp.nudge_into_bounds(room_dims, max_iterations=max_iterations)

    # ----- Lazy cache helpers -----

    def _get_corner_cache(self) -> list:
        """Return cached ranked corner indices, computing on first call."""
        if self._corner_cache is None:
            self._corner_cache = _rank_corners_by_visibility(self.polygon)
        return self._corner_cache

    def _get_edge_cache(self) -> dict:
        """Return cached edge ranking + best positions, computing on first call."""
        if self._edge_cache is None:
            edge_centers = get_edge_centers(self.polygon)
            scored = []
            best_positions = {}
            for _, (mx, my, edge_idx) in enumerate(edge_centers):
                mid_sight = _calculate_sightline_distance((mx, my), edge_idx, self.polygon)
                best_pos, best_sight = _best_position_on_edge(edge_idx, self.polygon)
                scored.append((edge_idx, mid_sight, best_sight))
                best_positions[edge_idx] = best_pos
            scored.sort(key=lambda x: (-x[1], -x[2]))
            self._edge_cache = {
                "ranked": [s[0] for s in scored],
                "best_positions": best_positions,
                "edge_centers": edge_centers,
            }
        return self._edge_cache

    def _get_grid_cache(self) -> dict:
        """Return cached interior grid + boundary distances, computing on first call."""
        if self._grid_cache is None:
            x_min, y_min, x_max, y_max = self.polygon.bounding_box
            xp = np.linspace(x_min, x_max, 101)
            yp = np.linspace(y_min, y_max, 101)
            xx, yy = np.meshgrid(xp, yp, indexing="ij")
            all_points = np.column_stack([xx.ravel(), yy.ravel()])
            inside_mask = self.polygon.contains_points(all_points)
            candidates = all_points[inside_mask]
            boundary_dists = np.array(
                [_distance_to_polygon_boundary(candidates[i], self.polygon)
                 for i in range(len(candidates))]
            )
            self._grid_cache = {
                "candidates": candidates,
                "boundary_dists": boundary_dists,
            }
        return self._grid_cache

    # ----- Cached placement helpers (used by batch methods) -----

    def _cached_place_downlight(self) -> PlacementResult:
        """Downlight placement using cached grid + boundary distances.

        Uses an incremental running-minimum strategy: rather than recomputing
        distances to all existing lamps each time, it only computes the distance
        to the most recently added lamp and updates the running minimum.
        """
        cache = self._get_grid_cache()
        candidates = cache["candidates"]
        boundary_dists = cache["boundary_dists"]

        if len(candidates) == 0:
            cx, cy = self.polygon.centroid
            return PlacementResult(x=cx, y=cy, aimx=cx, aimy=cy)

        # Lazily initialise or update the running minimum distance to existing lamps
        min_existing = cache.get("_min_existing")
        n_accounted = cache.get("_n_accounted", 0)

        if not self._existing:
            idx = int(np.argmax(boundary_dists))
        else:
            # Bootstrap running min if this is the first call with existing lamps
            if min_existing is None:
                min_existing = np.full(len(candidates), np.inf)
                n_accounted = 0

            # Incrementally add any new positions since last call
            for i in range(n_accounted, len(self._existing)):
                ex, ey = self._existing[i]
                dx = candidates[:, 0] - ex
                dy = candidates[:, 1] - ey
                np.minimum(min_existing, np.sqrt(dx * dx + dy * dy), out=min_existing)

            cache["_min_existing"] = min_existing
            cache["_n_accounted"] = len(self._existing)

            min_dist = np.minimum(min_existing, boundary_dists)
            idx = int(np.argmax(min_dist))

        x, y = float(candidates[idx][0]), float(candidates[idx][1])
        return PlacementResult(x=x, y=y, aimx=x, aimy=y)

    def _cached_place_corner(self, wall_clearance: float, beam_angle: float) -> PlacementResult:
        """Corner placement using cached ranking."""
        ranked = self._get_corner_cache()
        corners = get_corners(self.polygon)

        tolerance = max(0.2, wall_clearance * 1.5 + 0.1)
        occupied = set()
        for ex, ey in self._existing:
            for i, (cx, cy) in enumerate(corners):
                if hypot(ex - cx, ey - cy) < tolerance:
                    occupied.add(i)
        available = [i for i in ranked if i not in occupied]

        if available:
            corner = corners[available[0]]
            inward = _get_corner_inward_direction(corner, self.polygon)
            position = _offset_inward(corner, inward, offset=wall_clearance)
            if not self.polygon.contains_point(*position):
                position = _offset_inward(corner, inward, offset=min(wall_clearance, 0.01))
            if not self.polygon.contains_point(*position):
                position = corner
            aim = _calculate_corner_aim(position, self.polygon, beam_angle)
            return PlacementResult(x=position[0], y=position[1], aimx=aim[0], aimy=aim[1])

        # All corners occupied — delegate to edge
        return self._cached_place_edge(wall_clearance)

    def _cached_place_edge(self, wall_clearance: float) -> PlacementResult:
        """Edge placement using cached ranking + best positions."""
        cache = self._get_edge_cache()
        ranked = cache["ranked"]
        best_positions = cache["best_positions"]
        edge_centers = cache["edge_centers"]

        tolerance = max(0.2, wall_clearance + 0.1)
        occupied_edges = set()
        for ex, ey in self._existing:
            for _, _, edge_idx in edge_centers:
                (x1, y1), (x2, y2) = self.polygon.edges[edge_idx]
                if _point_to_segment_distance(ex, ey, x1, y1, x2, y2) < tolerance:
                    occupied_edges.add(edge_idx)

        for edge_idx in ranked:
            if edge_idx not in occupied_edges:
                best_pos = best_positions[edge_idx]
                inward = (-self.polygon.edge_normals[edge_idx][0],
                          -self.polygon.edge_normals[edge_idx][1])
                position = _offset_inward(best_pos, inward, offset=wall_clearance)
                aim = _calculate_edge_perpendicular_aim(position, edge_idx, self.polygon)
                return PlacementResult(
                    x=position[0], y=position[1], aimx=aim[0], aimy=aim[1],
                )

        # All edges occupied — farthest perimeter point
        position, aim = _farthest_perimeter_position(
            self.polygon, self._existing, wall_offset=wall_clearance,
        )
        return PlacementResult(x=position[0], y=position[1], aimx=aim[0], aimy=aim[1])

    def _compute_batch_placement(self, lamp, mode, *, tilt, max_tilt) -> PlacementResult:
        """Compute a single placement using cached room data."""
        if self.z is None:
            raise ValueError("z must be set (use for_room or for_dims)")

        # Resolve per-lamp config — beam_angle only for corner mode (expensive)
        config_angle = 0
        resolved_max_tilt = max_tilt
        try:
            _, config = resolve_keyword(lamp.lamp_id)
            placement = config.get("placement", {})
            config_angle = placement.get("angle", 0)
            if resolved_max_tilt is None:
                resolved_max_tilt = placement.get("max_tilt")
        except KeyError:
            pass
        offset = self.ceiling_offset(lamp)

        if mode == "downlight":
            result = self._cached_place_downlight()
        elif mode == "corner":
            wall_clearance = self.wall_clearance(lamp)
            beam_angle = self._get_beam_angle(lamp)
            result = self._cached_place_corner(wall_clearance, beam_angle)
        elif mode in ("edge", "horizontal"):
            wall_clearance = self.wall_clearance(lamp)
            result = self._cached_place_edge(wall_clearance)
        else:
            raise ValueError(f"invalid lamp placement mode {mode}")

        result.z = self.z - offset
        result.aimz = result.z if mode == "horizontal" else 0.0
        result.angle = config_angle

        if mode in ("corner", "edge"):
            aim_xy = self._apply_tilt(
                (result.x, result.y, result.z), result.aim, tilt, resolved_max_tilt,
            )
            result.aimx, result.aimy = aim_xy

        return result

    # ----- Batch placement -----

    @staticmethod
    def _validate_lamps_by_mode(lamps_by_mode: dict) -> set:
        """Validate lamps_by_mode dict. Returns set of all lamp IDs."""
        seen_ids = set()
        for mode, lamps in lamps_by_mode.items():
            if mode not in _VALID_MODES:
                raise ValueError(
                    f"Unknown placement mode '{mode}'. "
                    f"Valid modes: {', '.join(_MODE_ORDER)}"
                )
            for lamp in lamps:
                lid = lamp.lamp_id
                if lid in seen_ids:
                    raise ValueError(f"Lamp '{lid}' appears in multiple mode lists")
                seen_ids.add(lid)
        return seen_ids

    def get_layout(
        self,
        lamps_by_mode: dict,
        *,
        tilt: float = None,
        max_tilt: float = None,
    ) -> dict:
        """Compute placements for multiple lamps without mutating them.

        Uses cached room geometry so placement scales well to many lamps.

        Args:
            lamps_by_mode: {mode: [lamps]} dict grouping lamps by placement mode.
                Valid modes: "corner", "edge", "horizontal", "downlight".
            tilt: Force exact tilt angle in degrees (applied to corner/edge modes)
            max_tilt: Maximum allowed tilt angle in degrees

        Returns:
            dict[str, PlacementResult] keyed by lamp.lamp_id
        """
        self._validate_lamps_by_mode(lamps_by_mode)
        results = {}

        for mode in _MODE_ORDER:
            for lamp in lamps_by_mode.get(mode, []):
                result = self._compute_batch_placement(
                    lamp, mode, tilt=tilt, max_tilt=max_tilt,
                )
                results[lamp.lamp_id] = result
                self.record(result.x, result.y)

        return results

    def place_lamps(
        self,
        lamps_by_mode: dict,
        *,
        tilt: float = None,
        max_tilt: float = None,
    ) -> dict:
        """Compute and apply placements for multiple lamps.

        Same as get_layout() but also applies each result: lamp.move(),
        lamp.aim(), lamp.rotate(), _nudge_into_bounds(). Records post-nudge
        positions so subsequent placements respect spacing.
        """
        self._validate_lamps_by_mode(lamps_by_mode)
        results = {}

        for mode in _MODE_ORDER:
            for lamp in lamps_by_mode.get(mode, []):
                result = self._compute_batch_placement(
                    lamp, mode, tilt=tilt, max_tilt=max_tilt,
                )
                lamp.move(result.x, result.y, result.z)
                lamp.aim(result.aimx, result.aimy, result.aimz)
                if result.angle:
                    lamp.rotate(result.angle)
                self._nudge_into_bounds(lamp)
                self.record(lamp.x, lamp.y)
                results[lamp.lamp_id] = result

        return results

    # ----- Batch aiming -----

    def get_aim(
        self,
        lamp,
        aim_mode: str,
        *,
        target: tuple = None,
        direction: tuple = None,
    ) -> "AimResult":
        """Compute aim for a single lamp without mutating it.

        Args:
            lamp: Lamp object (read for position, not mutated)
            aim_mode: One of "down", "point", "direction", "centroid",
                "furthest_edge", "furthest_corner"
            target: (x, y, z) for "point" mode
            direction: (dx, dy, dz) for "direction" mode

        Returns:
            AimResult with computed aim coordinates
        """
        if aim_mode not in _VALID_AIM_MODES:
            raise ValueError(
                f"Unknown aim mode '{aim_mode}'. "
                f"Valid modes: {', '.join(sorted(_VALID_AIM_MODES))}"
            )

        lx, ly = lamp.x, lamp.y

        if aim_mode == "down":
            return AimResult(aimx=lx, aimy=ly, aimz=0.0)

        if aim_mode == "point":
            if target is None:
                raise ValueError("'point' aim mode requires 'target' parameter")
            return AimResult(aimx=target[0], aimy=target[1], aimz=target[2])

        if aim_mode == "direction":
            if direction is None:
                raise ValueError("'direction' aim mode requires 'direction' parameter")
            dx, dy, dz = direction
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length < 1e-10:
                return AimResult(aimx=lx, aimy=ly, aimz=0.0)
            ndx, ndy, ndz = dx / length, dy / length, dz / length
            return AimResult(
                aimx=lx + ndx, aimy=ly + ndy, aimz=lamp.z + ndz,
            )

        if aim_mode == "centroid":
            cx, cy = _visible_centroid((lx, ly), self.polygon)
            return AimResult(aimx=cx, aimy=cy, aimz=0.0)

        if aim_mode == "furthest_edge":
            edges = get_edge_centers(self.polygon)
            best = None
            best_dist = -1.0
            for ex, ey, _ in edges:
                d = hypot(ex - lx, ey - ly)
                if d > best_dist:
                    best_dist = d
                    best = (ex, ey)
            if best is None:
                best = self.polygon.centroid
            return AimResult(aimx=best[0], aimy=best[1], aimz=0.0)

        # aim_mode == "furthest_corner"
        corners = get_corners(self.polygon)
        best = None
        best_dist = -1.0
        for cx, cy in corners:
            d = hypot(cx - lx, cy - ly)
            if d > best_dist:
                best_dist = d
                best = (cx, cy)
        if best is None:
            best = self.polygon.centroid
        return AimResult(aimx=best[0], aimy=best[1], aimz=0.0)

    def get_aims(
        self,
        lamps: list,
        aim_mode: str,
        *,
        target: tuple = None,
        direction: tuple = None,
    ) -> dict:
        """Compute aim for multiple lamps. Returns dict[lamp_id, AimResult]."""
        return {
            lamp.lamp_id: self.get_aim(lamp, aim_mode, target=target, direction=direction)
            for lamp in lamps
        }

    def aim_lamps(
        self,
        lamps: list,
        aim_mode: str,
        *,
        target: tuple = None,
        direction: tuple = None,
    ) -> dict:
        """Compute and apply aim for multiple lamps.

        Calls get_aims(), then lamp.aim() for each.
        """
        results = self.get_aims(lamps, aim_mode, target=target, direction=direction)
        for lamp in lamps:
            r = results[lamp.lamp_id]
            lamp.aim(r.aimx, r.aimy, r.aimz)
        return results


# =============================================================================
# SECTION 3: Low-level Geometry Helpers
# =============================================================================


def _visible_centroid(
    position: tuple[float, float], polygon: Polygon2D, grid_size: int = 30
) -> tuple[float, float]:
    """Centroid of the floor area visible from position within the polygon.

    For convex rooms, every interior point sees every other, so returns the
    polygon centroid directly. For concave rooms, generates an interior grid
    and computes the centroid of only those points with unobstructed line-of-sight
    from the given position.
    """
    if polygon.is_convex:
        return polygon.centroid

    x_min, y_min, x_max, y_max = polygon.bounding_box
    xp = np.linspace(x_min, x_max, grid_size)
    yp = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xp, yp, indexing="ij")
    all_points = np.column_stack([xx.ravel(), yy.ravel()])

    inside_mask = polygon.contains_points(all_points)
    interior = all_points[inside_mask]

    if len(interior) == 0:
        return polygon.centroid

    origin_edge_idx = _find_origin_edge(position, polygon)

    visible = []
    for pt in interior:
        if not _line_intersects_polygon_edge(position, (pt[0], pt[1]), polygon, origin_edge_idx):
            visible.append(pt)

    if not visible:
        return polygon.centroid

    visible_arr = np.array(visible)
    return (float(np.mean(visible_arr[:, 0])), float(np.mean(visible_arr[:, 1])))


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
    return polygon.nearest_boundary_point(point[0], point[1])


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

    # Also consider polygon vertices as candidates (ensures exact corner targeting)
    for vx, vy in polygon.vertices:
        dist_to_corner = hypot(vx - corner[0], vy - corner[1])
        if dist_to_corner < 1e-9:
            continue  # Skip the source corner itself

        # Check direction to this vertex for wall-angle filtering
        dx, dy = _normalize(vx - corner[0], vy - corner[1])
        angle_to_wall1 = _angle_to_wall((dx, dy), edge_dirs[0])
        angle_to_wall2 = _angle_to_wall((dx, dy), edge_dirs[1])
        min_angle = min(angle_to_wall1, angle_to_wall2)

        if min_angle < min_wall_angle:
            continue

        wall_bonus = min_angle / 90.0
        score = dist_to_corner * (0.5 + 0.5 * wall_bonus)

        if score > best_score:
            best_score = score
            best_point = (vx, vy)

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


def _rank_edges_by_sightline(polygon: Polygon2D) -> list[int]:
    """Rank edge indices by sightline distance (longest first).

    Primary key is midpoint sightline; tiebreaker is best-position sightline
    (matching the ranking used in new_lamp_position_edge).
    """
    edges = get_edge_centers(polygon)
    scored = []
    for _, (mx, my, edge_idx) in enumerate(edges):
        mid_sight = _calculate_sightline_distance((mx, my), edge_idx, polygon)
        _, best_sight = _best_position_on_edge(edge_idx, polygon)
        scored.append((edge_idx, mid_sight, best_sight))
    scored.sort(key=lambda x: (-x[1], -x[2]))
    return [s[0] for s in scored]


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


def new_lamp_position_downlight(
    polygon: "Polygon2D",
    existing_positions: list[tuple[float, float]] | None = None,
    num_divisions: int = 100,
) -> tuple[float, float]:
    """
    Get position for downlight placement inside a polygon.

    Finds the point that maximizes min(dist_to_existing, dist_to_boundary)
    in a single vectorized pass over a candidate grid.

    Args:
        polygon: Room polygon
        existing_positions: List of (x, y) positions of already-placed lamps
        num_divisions: Grid resolution (default 100)

    Returns:
        (x, y) position for the new lamp
    """
    existing = existing_positions or []
    x_min, y_min, x_max, y_max = polygon.bounding_box

    xp = np.linspace(x_min, x_max, num_divisions + 1)
    yp = np.linspace(y_min, y_max, num_divisions + 1)
    xx, yy = np.meshgrid(xp, yp, indexing="ij")
    all_points = np.column_stack([xx.ravel(), yy.ravel()])

    inside_mask = polygon.contains_points(all_points)
    candidates = all_points[inside_mask]

    if len(candidates) == 0:
        return polygon.centroid

    # Precompute boundary distances
    boundary_dists = np.array(
        [_distance_to_polygon_boundary(candidates[i], polygon) for i in range(len(candidates))]
    )

    # Among tied candidates, prefer the one closest to the centroid so
    # placement is visually centered rather than biased by ravel order.
    centroid = np.array(polygon.centroid)
    dist_to_centroid = np.sqrt(np.sum((candidates - centroid) ** 2, axis=1))
    # Small tiebreaker: slightly favour centroid-proximity without
    # overriding the primary metric (boundary dist or min-dist).
    eps = np.max(boundary_dists) * 1e-6
    tiebreak = eps * (1.0 - dist_to_centroid / (np.max(dist_to_centroid) + 1e-12))

    if not existing:
        # No existing lamps — place at point farthest from boundary (center-ish)
        return tuple(candidates[int(np.argmax(boundary_dists + tiebreak))])

    # Vectorized distance to all existing lamps
    existing_arr = np.array(existing)  # (K, 2)
    diff = candidates[:, np.newaxis, :] - existing_arr[np.newaxis, :, :]  # (N, K, 2)
    dist_sq_to_existing = np.sum(diff ** 2, axis=2)  # (N, K)
    min_dist_to_existing = np.sqrt(np.min(dist_sq_to_existing, axis=1))  # (N,)

    # Combine with boundary distances — maximize the minimum
    min_dist = np.minimum(min_dist_to_existing, boundary_dists)
    return tuple(candidates[int(np.argmax(min_dist + tiebreak))])


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


# =============================================================================
# SECTION 9: Height Utilities
# =============================================================================


def set_height(lamps: list, *, z: float = None, ceiling_offset: float = None, room_z: float = None):
    """Set the height of multiple lamps.

    Exactly one of ``z`` or ``ceiling_offset`` must be provided.
    When using ``ceiling_offset``, ``room_z`` is also required.

    Calls ``lamp.move(z=z)`` for each lamp, which shifts the aim point by
    the same delta, preserving relative aim.

    Args:
        lamps: List of Lamp objects to modify.
        z: Absolute height to place lamps at.
        ceiling_offset: Distance below the ceiling. Requires ``room_z``.
        room_z: Ceiling height (required when using ``ceiling_offset``).
    """
    if z is not None and ceiling_offset is not None:
        raise ValueError("Provide either 'z' or 'ceiling_offset', not both")
    if z is None and ceiling_offset is None:
        raise ValueError("Must provide either 'z' or 'ceiling_offset'")
    if ceiling_offset is not None:
        if room_z is None:
            raise ValueError("'ceiling_offset' requires 'room_z'")
        z = room_z - ceiling_offset
    for lamp in lamps:
        lamp.move(z=z)
