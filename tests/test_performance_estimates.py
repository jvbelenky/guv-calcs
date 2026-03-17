"""
Comprehensive profiling of estimate_calculation_time() and estimate_memory()
against actual measurements.

Run with: python -m pytest tests/test_performance_estimates.py -v -s
"""
import time
import tracemalloc
import gc
from math import prod

import pytest
import numpy as np

pytestmark = pytest.mark.slow

from guv_calcs import Room, Lamp
from guv_calcs.performance import estimate_calculation_time, estimate_memory


PRESET = "ushio_b1"


def make_room(x=4, y=6, z=2.7, n_lamps=1, R=0.078, enable_refl=True,
              surf_n=10, add_zones=True):
    """Create a room with specified parameters."""
    room = Room(x=x, y=y, z=z, enable_reflectance=enable_refl)
    room.set_reflectance(R)
    room.set_reflectance_num_points(num_x=surf_n, num_y=surf_n)

    for i in range(n_lamps):
        lamp = Lamp.from_keyword(PRESET)
        lamp.move(
            x=x * (i + 1) / (n_lamps + 1),
            y=y / 2,
            z=z,
        )
        lamp.aim(x=lamp.x, y=lamp.y, z=0)
        room.add_lamp(lamp)

    if add_zones:
        room.add_standard_zones()

    return room


def measure_calc_time(room):
    """Measure actual calculation time."""
    start = time.perf_counter()
    room.calculate()
    return time.perf_counter() - start


def measure_memory(room):
    """Measure actual memory usage during calculation."""
    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    room.calculate()

    snap_after = tracemalloc.take_snapshot()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Compute allocated bytes
    stats = snap_after.compare_to(snap_before, 'lineno')
    alloc_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)

    return alloc_bytes, peak


# ============================================================================
# Configurations to test
# ============================================================================

# (label, room_x, room_y, room_z, n_lamps, enable_refl, R, surf_n)
CONFIGS = [
    # --- Vary lamp count (no reflectance) ---
    ("1L no_refl",       8, 14.6, 2.53,  1, False, 0.078, 10),
    ("2L no_refl",       8, 14.6, 2.53,  2, False, 0.078, 10),
    ("4L no_refl",       8, 14.6, 2.53,  4, False, 0.078, 10),
    ("8L no_refl",       8, 14.6, 2.53,  8, False, 0.078, 10),
    ("14L no_refl",      8, 14.6, 2.53, 14, False, 0.078, 10),

    # --- Vary lamp count (with reflectance) ---
    ("1L refl",          8, 14.6, 2.53,  1, True,  0.078, 10),
    ("4L refl",          8, 14.6, 2.53,  4, True,  0.078, 10),
    ("14L refl",         8, 14.6, 2.53, 14, True,  0.078, 10),

    # --- Vary room size (4 lamps, no refl) ---
    ("room 2x3",         2,  3,   2.4,   4, False, 0.078, 10),
    ("room 4x6",         4,  6,   2.7,   4, False, 0.078, 10),
    ("room 8x15",        8, 14.6, 2.53,  4, False, 0.078, 10),
    ("room 15x20",      15, 20,   3.0,   4, False, 0.078, 10),

    # --- Vary room size (4 lamps, with refl) ---
    ("room 2x3 refl",    2,  3,   2.4,   4, True,  0.078, 10),
    ("room 4x6 refl",    4,  6,   2.7,   4, True,  0.078, 10),
    ("room 8x15 refl",   8, 14.6, 2.53,  4, True,  0.078, 10),
    ("room 15x20 refl", 15, 20,   3.0,   4, True,  0.078, 10),

    # --- Vary surface resolution (4 lamps, refl) ---
    ("surf 5x5",         8, 14.6, 2.53,  4, True,  0.078,  5),
    ("surf 10x10",       8, 14.6, 2.53,  4, True,  0.078, 10),
    ("surf 15x15",       8, 14.6, 2.53,  4, True,  0.078, 15),
    ("surf 20x20",       8, 14.6, 2.53,  4, True,  0.078, 20),

    # --- Vary reflectance value (4 lamps, 10x10) ---
    ("R=0.05",           8, 14.6, 2.53,  4, True,  0.05,  10),
    ("R=0.078",          8, 14.6, 2.53,  4, True,  0.078, 10),
    ("R=0.15",           8, 14.6, 2.53,  4, True,  0.15,  10),
    ("R=0.3",            8, 14.6, 2.53,  4, True,  0.3,   10),
    ("R=0.5",            8, 14.6, 2.53,  4, True,  0.5,   10),

    # --- Combinations ---
    ("14L surf 15x15",   8, 14.6, 2.53, 14, True,  0.078, 15),
    ("1L small refl",    2,  2,   2.4,   1, True,  0.078, 10),
    ("1L big no_refl",  15, 20,   3.0,   1, False, 0.078, 10),

    # --- Edge: small room, 1 lamp, no reflectance ---
    ("minimal",          2,  2,   2.4,   1, False, 0.078, 10),
]


@pytest.fixture(scope="module")
def profiling_results():
    """Run all configs and collect results. Module-scoped so it runs once."""
    results = []
    for label, rx, ry, rz, n_lamps, en_refl, R, surf_n in CONFIGS:
        room = make_room(rx, ry, rz, n_lamps, R, en_refl, surf_n)

        # Gather info
        lamp_count = len(room.lamps.valid())
        zone_pts = sum(prod(z.num_points) for z in room.calc_zones.values()
                       if getattr(z, 'enabled', True))
        refl_pts = sum(prod(s.plane.num_points) for s in room.surfaces.values())
        n_surfs = len(room.surfaces)

        # Estimates
        est_time = estimate_calculation_time(room)
        est_mem = estimate_memory(room)

        # Actual time
        actual_time = measure_calc_time(room)

        # Actual memory (separate room to avoid cache effects)
        room2 = make_room(rx, ry, rz, n_lamps, R, en_refl, surf_n)
        actual_alloc, actual_peak = measure_memory(room2)

        results.append({
            'label': label,
            'lamps': lamp_count,
            'zone_pts': zone_pts,
            'refl_pts': refl_pts,
            'n_surfs': n_surfs,
            'R': R,
            'enable_refl': en_refl,
            'est_time': est_time,
            'actual_time': actual_time,
            'time_ratio': est_time / actual_time if actual_time > 0.01 else None,
            'est_peak_mb': est_mem['peak_bytes'] / 1e6,
            'actual_peak_mb': actual_peak / 1e6,
            'peak_ratio': est_mem['peak_bytes'] / actual_peak if actual_peak > 1000 else None,
            'actual_alloc_mb': actual_alloc / 1e6,
        })

    return results


class TestTimeEstimates:
    """Validate calculation time estimates against actual measurements."""

    def test_print_time_table(self, profiling_results):
        """Print full time comparison table."""
        print("\n" + "=" * 110)
        print("TIME ESTIMATES vs ACTUAL")
        print("=" * 110)
        print(f"{'Config':22s} {'Lamps':>5s} {'ZonePts':>8s} {'ReflPts':>8s} "
              f"{'Est(s)':>7s} {'Actual(s)':>9s} {'Ratio':>6s}")
        print("-" * 110)
        for r in profiling_results:
            ratio_str = f"{r['time_ratio']:.2f}" if r['time_ratio'] else "N/A"
            print(f"  {r['label']:20s} {r['lamps']:5d} {r['zone_pts']:8,d} {r['refl_pts']:8,d} "
                  f"{r['est_time']:7.2f} {r['actual_time']:9.2f} {ratio_str:>6s}")

        # Summary
        ratios = [r['time_ratio'] for r in profiling_results
                  if r['time_ratio'] is not None and r['actual_time'] > 0.1]
        if ratios:
            print(f"\nNon-trivial (actual > 0.1s): n={len(ratios)}")
            print(f"  Range: {min(ratios):.2f} - {max(ratios):.2f}")
            print(f"  Mean: {np.mean(ratios):.2f}, Median: {np.median(ratios):.2f}")

    def test_no_time_estimate_exceeds_3x(self, profiling_results):
        """No estimate should be more than 3x the actual (overprediction)."""
        for r in profiling_results:
            if r['time_ratio'] is not None and r['actual_time'] > 0.5:
                assert r['time_ratio'] < 3.0, (
                    f"{r['label']}: time ratio {r['time_ratio']:.2f} exceeds 3x "
                    f"(est={r['est_time']:.2f}, actual={r['actual_time']:.2f})"
                )

    def test_no_time_estimate_below_half(self, profiling_results):
        """No estimate should be less than 0.5x the actual (dangerous underprediction)."""
        for r in profiling_results:
            if r['time_ratio'] is not None and r['actual_time'] > 0.5:
                assert r['time_ratio'] > 0.5, (
                    f"{r['label']}: time ratio {r['time_ratio']:.2f} below 0.5x "
                    f"(est={r['est_time']:.2f}, actual={r['actual_time']:.2f})"
                )


class TestMemoryEstimates:
    """Validate memory estimates against actual measurements."""

    def test_print_memory_table(self, profiling_results):
        """Print full memory comparison table."""
        print("\n" + "=" * 110)
        print("MEMORY ESTIMATES vs ACTUAL")
        print("=" * 110)
        print(f"{'Config':22s} {'Lamps':>5s} {'ZonePts':>8s} {'ReflPts':>8s} "
              f"{'EstPeak':>9s} {'ActPeak':>9s} {'Ratio':>6s}")
        print("-" * 110)
        for r in profiling_results:
            ratio_str = f"{r['peak_ratio']:.2f}" if r['peak_ratio'] else "N/A"
            print(f"  {r['label']:20s} {r['lamps']:5d} {r['zone_pts']:8,d} {r['refl_pts']:8,d} "
                  f"{r['est_peak_mb']:8.1f}M {r['actual_peak_mb']:8.1f}M {ratio_str:>6s}")

        # Summary
        ratios = [r['peak_ratio'] for r in profiling_results
                  if r['peak_ratio'] is not None and r['actual_peak_mb'] > 1]
        if ratios:
            print(f"\nNon-trivial (peak > 1 MB): n={len(ratios)}")
            print(f"  Range: {min(ratios):.2f} - {max(ratios):.2f}")
            print(f"  Mean: {np.mean(ratios):.2f}, Median: {np.median(ratios):.2f}")

    def test_no_memory_underpredict_for_large_configs(self, profiling_results):
        """For configs with peak > 100 MB, estimate should not be below 0.8x actual."""
        for r in profiling_results:
            if r['peak_ratio'] is not None and r['actual_peak_mb'] > 100:
                assert r['peak_ratio'] > 0.8, (
                    f"{r['label']}: memory ratio {r['peak_ratio']:.2f} below 0.8x "
                    f"(est={r['est_peak_mb']:.0f}MB, actual={r['actual_peak_mb']:.0f}MB)"
                )

    def test_no_memory_overpredict_above_5x(self, profiling_results):
        """No estimate should be more than 5x the actual."""
        for r in profiling_results:
            if r['peak_ratio'] is not None and r['actual_peak_mb'] > 1:
                assert r['peak_ratio'] < 5.0, (
                    f"{r['label']}: memory ratio {r['peak_ratio']:.2f} exceeds 5x "
                    f"(est={r['est_peak_mb']:.0f}MB, actual={r['actual_peak_mb']:.0f}MB)"
                )

    def test_reflectance_memory_scales_with_surface_points(self, profiling_results):
        """Reflectance memory should increase with surface resolution."""
        refl_configs = [r for r in profiling_results
                        if r['enable_refl'] and r['label'].startswith('surf')]
        if len(refl_configs) >= 2:
            sorted_by_pts = sorted(refl_configs, key=lambda r: r['refl_pts'])
            for i in range(1, len(sorted_by_pts)):
                assert sorted_by_pts[i]['actual_peak_mb'] > sorted_by_pts[i-1]['actual_peak_mb'], (
                    f"Memory didn't increase from {sorted_by_pts[i-1]['label']} to {sorted_by_pts[i]['label']}"
                )

    def test_reflectance_value_does_not_affect_memory(self, profiling_results):
        """Changing R should not significantly change memory (form factors are constant)."""
        r_configs = [r for r in profiling_results if r['label'].startswith('R=')]
        if len(r_configs) >= 2:
            peaks = [r['actual_peak_mb'] for r in r_configs]
            # All should be within 20% of each other
            ratio = max(peaks) / min(peaks) if min(peaks) > 0 else float('inf')
            assert ratio < 1.2, f"Memory varies too much with R: {peaks}"


class TestRawData:
    """Export raw data for coefficient fitting."""

    def test_print_fitting_data(self, profiling_results):
        """Print data in a format suitable for coefficient fitting."""
        print("\n" + "=" * 110)
        print("RAW DATA FOR COEFFICIENT FITTING")
        print("=" * 110)
        print("# label, lamps, zone_pts, n_surfs, refl_pts, R, enable_refl, "
              "actual_time, actual_peak_mb")
        for r in profiling_results:
            print(f"  ({r['label']!r:22s}, {r['lamps']:2d}, {r['zone_pts']:6d}, "
                  f"{r['n_surfs']}, {r['refl_pts']:5d}, {r['R']:.3f}, "
                  f"{str(r['enable_refl']):5s}, {r['actual_time']:7.3f}, "
                  f"{r['actual_peak_mb']:8.1f}),")
