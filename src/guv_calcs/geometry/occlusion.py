"""Ray-surface intersection for computing light transmission through surfaces."""

import numpy as np

from .polygon import points_in_polygon


def compute_transmission(sources, targets, surfaces, exclude=None):
    """Compute transmission factors for rays from sources to targets through surfaces.

    Args:
        sources: (M, 3) or (3,) array of source positions
        targets: (N, 3) array of target positions
        surfaces: dict-like of Surface objects (must have .plane.geometry.boundary_vertices
            and .T attributes)
        exclude: optional surface key to skip (for self-occlusion)

    Returns:
        (M, N) or (N,) float array of transmission factors in [0, 1].
        1.0 = unobstructed, 0.0 = fully blocked.
    """
    sources = np.asarray(sources, dtype=float)
    targets = np.asarray(targets, dtype=float)
    squeeze = sources.ndim == 1
    if squeeze:
        sources = sources[None, :]  # (1, 3)

    M, N = len(sources), len(targets)
    transmission = np.ones((M, N), dtype=np.float64)

    for key, surface in surfaces.items():
        if key == exclude:
            continue
        T = surface.T
        if T >= 1.0:
            continue  # fully transparent, skip

        verts = np.asarray(surface.plane.geometry.boundary_vertices, float)
        blocked = _ray_polygon_intersect(sources, targets, verts)
        # blocked: (M, N) bool
        if T == 0.0:
            transmission[blocked] = 0.0
        else:
            transmission[blocked] *= T

    if squeeze:
        return transmission[0]  # (N,)
    return transmission


def _ray_polygon_intersect(sources, targets, verts):
    """Test if rays from sources to targets intersect a planar polygon.

    Args:
        sources: (M, 3)
        targets: (N, 3)
        verts: (K, 3) polygon vertices in order

    Returns:
        (M, N) bool array — True where the ray intersects the polygon
    """
    # Compute plane normal from first 3 vertices
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    normal = np.cross(e1, e2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12 * np.linalg.norm(e1) * np.linalg.norm(e2):
        return np.zeros((len(sources), len(targets)), dtype=bool)
    normal = normal / norm_len

    # Plane equation: normal · (P - verts[0]) = 0
    d = np.dot(normal, verts[0])

    # Ray: P(t) = source + t * (target - source), t ∈ (0, 1)
    # t = (d - normal · source) / (normal · (target - source))
    source_dots = sources @ normal  # (M,)
    target_dots = targets @ normal  # (N,)

    denom = target_dots[None, :] - source_dots[:, None]  # (M, N)
    numer = d - source_dots[:, None]  # (M, 1) broadcast to (M, N)

    # Avoid division by zero (ray parallel to plane); relative to the ray's
    # length so the cutoff is the same in every unit
    ray_len = np.linalg.norm(targets[None, :, :] - sources[:, None, :], axis=-1)
    parallel = np.abs(denom) < 1e-12 * ray_len
    safe_denom = np.where(parallel, 1.0, denom)
    t = numer / safe_denom  # (M, N)

    # Valid intersection: ray parameter in open interval (epsilon, 1-epsilon)
    eps = 1e-9
    valid_t = (~parallel) & (t > eps) & (t < 1.0 - eps)

    if not valid_t.any():
        return np.zeros((len(sources), len(targets)), dtype=bool)

    # Compute intersection points: P = source + t * (target - source)
    directions = targets[None, :, :] - sources[:, None, :]  # (M, N, 3)
    hit_points = sources[:, None, :] + t[:, :, None] * directions  # (M, N, 3)

    # Project onto polygon's local 2D frame
    u_axis = e1 / np.linalg.norm(e1)
    v_axis = np.cross(normal, u_axis)

    verts_rel = verts - verts[0]
    poly_2d = np.column_stack([verts_rel @ u_axis, verts_rel @ v_axis])  # (K, 2)

    hits_rel = hit_points - verts[0]  # (M, N, 3)
    hits_u = np.einsum("ijk,k->ij", hits_rel, u_axis)  # (M, N)
    hits_v = np.einsum("ijk,k->ij", hits_rel, v_axis)  # (M, N)

    # A ray meeting the polygon exactly on an edge is blocked, so ties resolve
    # the same in every unit and light can't slip through where faces meet
    inside = points_in_polygon(hits_u, hits_v, poly_2d, include_boundary=True)  # (M, N)

    return valid_t & inside
