"""Tests for compute_transmission from geometry/occlusion.py."""

import pytest
import numpy as np
from guv_calcs import CalcPlane, SurfaceGrid, Polygon2D
from guv_calcs.reflectance import Surface
from guv_calcs.geometry.occlusion import compute_transmission


def _make_wall(T=0.0, origin=(5.0, 0.0, 0.0), key="wall"):
    """Create a vertical wall surface for testing."""
    poly = Polygon2D.rectangle(10.0, 10.0)
    grid = SurfaceGrid(
        polygon=poly,
        origin=origin,
        u_vec=(0.0, 1.0, 0.0),
        v_vec=(0.0, 0.0, 1.0),
    )
    plane = CalcPlane(zone_id=key, geometry=grid)
    return Surface(R=0.0, T=T, plane=plane)


class TestComputeTransmission:

    def test_unobstructed_returns_ones(self):
        """No surfaces -> transmission is 1.0."""
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[10.0, 5.0, 5.0]])
        result = compute_transmission(sources, targets, {})
        assert np.allclose(result, 1.0)

    def test_fully_blocked(self):
        """Opaque wall between source and target -> transmission 0.0."""
        wall = _make_wall(T=0.0)
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[10.0, 5.0, 5.0]])
        result = compute_transmission(sources, targets, {"wall": wall})
        assert np.allclose(result, 0.0)

    def test_partial_transmittance(self):
        """T=0.5 wall -> transmission 0.5."""
        wall = _make_wall(T=0.5)
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[10.0, 5.0, 5.0]])
        result = compute_transmission(sources, targets, {"wall": wall})
        assert np.allclose(result, 0.5)

    def test_transparent_surface_skipped(self):
        """T=1.0 wall -> transmission 1.0 (fast-path skip)."""
        wall = _make_wall(T=1.0)
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[10.0, 5.0, 5.0]])
        result = compute_transmission(sources, targets, {"wall": wall})
        assert np.allclose(result, 1.0)

    def test_exclude_key(self):
        """Excluded surface doesn't block rays."""
        wall = _make_wall(T=0.0)
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[10.0, 5.0, 5.0]])
        result = compute_transmission(sources, targets, {"wall": wall}, exclude="wall")
        assert np.allclose(result, 1.0)

    def test_miss(self):
        """Source and target on same side of wall -> no intersection."""
        wall = _make_wall(T=0.0)
        sources = np.array([[0.0, 5.0, 5.0]])
        targets = np.array([[3.0, 5.0, 5.0]])  # both on x<5 side
        result = compute_transmission(sources, targets, {"wall": wall})
        assert np.allclose(result, 1.0)

    def test_squeeze(self):
        """Single source (3,) returns (N,) not (1, N)."""
        source = np.array([0.0, 5.0, 5.0])  # 1D
        targets = np.array([[10.0, 5.0, 5.0], [3.0, 5.0, 5.0]])
        wall = _make_wall(T=0.0)
        result = compute_transmission(source, targets, {"wall": wall})
        assert result.shape == (2,)  # (N,) not (1, N)

    def test_multiple_sources(self):
        """(M, 3) sources, (N, 3) targets -> (M, N) output."""
        sources = np.array([[0.0, 5.0, 5.0], [0.0, 5.0, 3.0]])  # M=2
        targets = np.array([[10.0, 5.0, 5.0], [3.0, 5.0, 5.0], [10.0, 5.0, 3.0]])  # N=3
        wall = _make_wall(T=0.0)
        result = compute_transmission(sources, targets, {"wall": wall})
        assert result.shape == (2, 3)
