"""2D polygon representation for non-rectangular room shapes."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Polygon2D:
    """
    A 2D polygon defined by vertices in counter-clockwise order.

    Used to represent non-rectangular floor plans for rooms with uniform ceiling height.
    Supports both convex and concave simple polygons (no self-intersections or holes).
    """

    vertices: tuple[tuple[float, float], ...]
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        # Convert to tuples if needed and validate
        verts = tuple(tuple(float(c) for c in v) for v in self.vertices)
        object.__setattr__(self, "vertices", verts)

        if not self._is_simple():
            raise ValueError("Polygon edges must not self-intersect")

        # Auto-correct winding order to CCW
        if self._signed_area() < 0:
            object.__setattr__(self, "vertices", self.vertices[::-1])

    @classmethod
    def rectangle(cls, x: float, y: float) -> "Polygon2D":
        """Create a rectangular polygon with given dimensions, origin at (0, 0)."""
        return cls(vertices=((0.0, 0.0), (x, 0.0), (x, y), (0.0, y)))

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def edges(self) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
        """Return edges as pairs of consecutive vertices."""
        if "edges" in self._cache:
            return self._cache["edges"]
        n = len(self.vertices)
        edges = tuple((self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n))
        self._cache["edges"] = edges
        return edges

    @property
    def edge_lengths(self) -> tuple[float, ...]:
        """Return the length of each edge."""
        if "edge_lengths" in self._cache:
            return self._cache["edge_lengths"]
        lengths = []
        for (x1, y1), (x2, y2) in self.edges:
            lengths.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        result = tuple(lengths)
        self._cache["edge_lengths"] = result
        return result

    @property
    def edge_normals(self) -> tuple[tuple[float, float], ...]:
        """Return outward-pointing unit normals for each edge (CCW winding)."""
        if "edge_normals" in self._cache:
            return self._cache["edge_normals"]
        normals = []
        for (x1, y1), (x2, y2) in self.edges:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            # For CCW winding, outward normal is (dy, -dx) normalized
            normals.append((dy / length, -dx / length))
        result = tuple(normals)
        self._cache["edge_normals"] = result
        return result

    @property
    def area(self) -> float:
        """Return the area of the polygon using the shoelace formula."""
        return abs(self._signed_area())

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the centroid (center of mass) of the polygon."""
        if "centroid" in self._cache:
            return self._cache["centroid"]
        n = len(self.vertices)
        cx, cy = 0.0, 0.0
        signed_area = self._signed_area()
        for i in range(n):
            x0, y0 = self.vertices[i]
            x1, y1 = self.vertices[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        factor = 1.0 / (6.0 * signed_area)
        result = (cx * factor, cy * factor)
        self._cache["centroid"] = result
        return result

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) of the bounding box."""
        if "bounding_box" in self._cache:
            return self._cache["bounding_box"]
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        result = (min(xs), min(ys), max(xs), max(ys))
        self._cache["bounding_box"] = result
        return result

    @property
    def x(self) -> float:
        """Bounding box width."""
        x_min, _, x_max, _ = self.bounding_box
        return x_max - x_min

    @property
    def y(self) -> float:
        """Bounding box height."""
        _, y_min, _, y_max = self.bounding_box
        return y_max - y_min

    def _signed_area(self) -> float:
        """Compute signed area (positive for CCW, negative for CW)."""
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            x0, y0 = self.vertices[i]
            x1, y1 = self.vertices[(i + 1) % n]
            area += x0 * y1 - x1 * y0
        return area / 2.0

    def _is_simple(self) -> bool:
        """Check that polygon edges don't self-intersect (other than at vertices)."""
        n = len(self.vertices)
        if n < 4:
            return True  # Triangle can't self-intersect

        for i in range(n):
            p1, p2 = self.vertices[i], self.vertices[(i + 1) % n]
            # Check against non-adjacent edges
            for j in range(i + 2, n):
                if j == (i - 1) % n or (i == 0 and j == n - 1):
                    continue  # Skip adjacent edges
                p3, p4 = self.vertices[j], self.vertices[(j + 1) % n]
                if self._segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """Check if line segments (p1,p2) and (p3,p4) intersect (not at endpoints)."""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
            )

        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)

        if d1 != d2 and d3 != d4:
            return True

        # Collinear cases - check for overlap
        # For simplicity, we'll treat touching at endpoints as non-intersecting
        return False

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the polygon using ray casting."""
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def point_on_boundary(self, x: float, y: float, tol: float = 1e-9) -> bool:
        """Check if a point is on the polygon boundary within tolerance."""
        for (x1, y1), (x2, y2) in self.edges:
            # Check if point is within bounding box of edge (with tolerance)
            if not (min(x1, x2) - tol <= x <= max(x1, x2) + tol and
                    min(y1, y2) - tol <= y <= max(y1, y2) + tol):
                continue
            # Calculate distance from point to line segment
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq < tol * tol:
                # Degenerate edge (point), check distance to vertex
                if (x - x1) ** 2 + (y - y1) ** 2 <= tol * tol:
                    return True
                continue
            # Project point onto line and check distance
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
            if dist_sq <= tol * tol:
                return True
        return False

    def contains_point_inclusive(self, x: float, y: float, tol: float = 1e-9) -> bool:
        """Check if a point is inside or on the boundary of the polygon."""
        return self.contains_point(x, y) or self.point_on_boundary(x, y, tol)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """
        Check if multiple points are inside the polygon.

        Args:
            points: Array of shape (N, 2) with x, y coordinates

        Returns:
            Boolean array of shape (N,)
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        n = len(self.vertices)
        verts = np.array(self.vertices)

        # Vectorized ray casting
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), dtype=bool)

        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]

            # Check if point is between y coordinates of edge
            cond1 = (yi > y) != (yj > y)
            # Check if point is to the left of edge
            with np.errstate(divide="ignore", invalid="ignore"):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            cond2 = x < x_intersect

            # Toggle inside flag for points that cross this edge
            inside ^= cond1 & cond2
            j = i

        return inside

    def is_valid(self) -> bool:
        """Check if the polygon is valid (simple, at least 3 vertices)."""
        return len(self.vertices) >= 3 and self._is_simple()

    def to_dict(self) -> dict:
        return {"vertices": list(list(v) for v in self.vertices)}

    @classmethod
    def from_dict(cls, data: dict) -> "Polygon2D":
        return cls(vertices=tuple(tuple(v) for v in data["vertices"]))

    def translate(self, dx: float, dy: float) -> "Polygon2D":
        """Return a new polygon translated by (dx, dy)."""
        new_verts = tuple((x + dx, y + dy) for x, y in self.vertices)
        return Polygon2D(vertices=new_verts)
