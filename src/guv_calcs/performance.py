"""
Performance estimation for Room calculations.

Provides empirically calibrated estimates of calculation time and peak memory
usage. Coefficients are fitted from profiling across configurations varying
lamp count (1-14), room size (2×2 to 15×20), surface resolution (5×5 to
20×20), and reflectance value (0-0.5).
"""

from math import prod, log
import numpy as np


# =============================================================================
# Time Estimation Coefficients
# =============================================================================

# Empirically fitted from 29 test configurations.
# Validated: 24 non-trivial tests within [0.78, 1.49] ratio, mean 1.07.

# Base: direct lamp → zone calculation, O(lamps × zone_points)
_TIME_BASE = 0.00000047

# Surface → zone form factors, O(surfaces × refl_points × zone_points)
_TIME_ZONE_REFLECT = 0.000000024

# First-pass surface ↔ surface form factor computation, O(refl_points²)
_TIME_FORM_FACTORS = 0.0000022

# Subsequent interreflection passes reuse cached form factors (~47× cheaper)
_TIME_EXTRA_PASS = 0.000000047


# =============================================================================
# Memory Estimation Coefficients
# =============================================================================

# Fitted (non-negative least squares) to the resident-set peak of
# room.calculate() measured over 20 configurations: rooms 2x2 to 12x15 m,
# 1-3 lamps, 5-40 surface points per axis, 670-24,000 zone points. Resident
# memory is what a server is limited on, so the fit is to RSS rather than
# to tracemalloc. The fitted values were 23 MB, 4.2, 3.5, 412 and 3.7 MB;
# each is rounded up so the estimate lands at 0.95-1.3x the measured peak
# for every case above 50 MB (small cases overpredict, which is harmless).

# Per lamp: IES photometry, spectrum, and the per-zone lamp cache entries.
BYTES_PER_LAMP = 4_000_000

# Per zone point: result arrays, per-lamp base and filtered values, and the
# reflected-value temporaries.
BYTES_PER_ZONE_POINT = 500

# Fixed overhead when reflectance is enabled: the surface managers' deep
# copies and the bounded chunk of float64 temporaries the solver works in
# (calc_manager._CHUNK_BYTES).
REFLECTANCE_OVERHEAD = 30_000_000

# Surface→zone form factor cache: one float32 per (surface point, zone
# point) pair, plus its share of temporaries.
BYTES_PER_FORM_FACTOR_ENTRY = 5

# Surface↔surface form factor cache for the interreflection passes: one
# float32 per (surface point, surface point) pair.
BYTES_PER_SURFACE_PAIR = 4

# Floor: minimum peak estimate regardless of configuration
MIN_PEAK_BYTES = 10_000_000


def estimate_calculation_time(room) -> float:
    """Estimate wall-clock time (seconds) for the next calculate() call.

    Uses empirically fitted coefficients for four cost terms:

    1. Base: direct lamp → zone calculation, O(lamps × zone_points)
    2. Zone reflect: surface → zone form factors, O(surfaces × refl_points × zone_points)
    3. Form factors: first-pass surface ↔ surface coordinate transforms, O(refl_points²)
    4. Extra passes: subsequent interreflection passes reuse cached form factors,
       so each additional pass is ~47× cheaper than the first.  Pass count is
       estimated from average reflectance: passes ≈ log(threshold) / log(R_avg).

    Args:
        room: A Room instance with lamps, calc_zones, surfaces, and ref_manager.

    Returns:
        Estimated seconds for room.calculate().
    """
    lamp_count = len(room.lamps.valid())
    if lamp_count == 0:
        return 0.0

    total_zone_points = sum(
        prod(zone.num_points)
        for zone in room.calc_zones.values()
        if getattr(zone, 'enabled', True)
    )

    # Base cost: direct lamp → zone calculation
    calc_time = lamp_count * total_zone_points * _TIME_BASE

    # Reflectance cost (only if enabled AND needs recalculation)
    if room.ref_manager.enabled and room.recalculate_incidence:
        all_surfs = room.all_surfaces
        R_avg = np.mean([s.R for s in all_surfs.values()]) if all_surfs else 0
        if R_avg <= 0:
            return calc_time

        refl_points = sum(prod(s.plane.num_points) for s in all_surfs.values())
        num_surfaces = len(all_surfs)

        # Estimate interreflection passes from average reflectance
        if R_avg < 1:
            est_passes = min(
                log(room.ref_manager.threshold) / log(R_avg),
                room.ref_manager.max_num_passes,
            )
        else:
            est_passes = float(room.ref_manager.max_num_passes)
        est_passes = max(est_passes, 1.0)
        extra_passes = max(est_passes - 1.0, 0.0)

        # Surface → zone form factors (linear in refl_points)
        zone_reflect_time = (
            num_surfaces * refl_points * total_zone_points * _TIME_ZONE_REFLECT
        )
        # First-pass surface ↔ surface form factor computation (quadratic)
        form_factor_time = refl_points ** 2 * _TIME_FORM_FACTORS
        # Subsequent passes reuse cached form factors (much cheaper per pass)
        extra_pass_time = extra_passes * refl_points ** 2 * _TIME_EXTRA_PASS
        calc_time += zone_reflect_time + form_factor_time + extra_pass_time

    return calc_time


def estimate_memory(room) -> dict:
    """Estimate peak memory usage (bytes) for a Room during calculation.

    Memory model (fitted to the resident-set peak of room.calculate() over
    20 configurations; see the coefficient comments above):

    - Per lamp: 4 MB
    - Per zone point: 500 bytes
    - Reflectance overhead: 30 MB fixed (manager copies, solver chunk)
    - Surface→zone form factors: 5 bytes per (surface point x zone point)
    - Surface↔surface form factors: 4 bytes per (surface point)^2
    - Floor: 10 MB minimum

    Lands at 0.95-1.3x the measured peak for every fitted case above
    50 MB. The reflectance share dominates: form factors scale with
    surface resolution squared (per axis) times zone points.

    Args:
        room: A Room instance with lamps, calc_zones, surfaces, and ref_manager.

    Returns:
        Dictionary with memory breakdown in bytes and session metadata.
    """
    # Count enabled lamps with photometric data
    lamp_count = sum(
        1 for lamp in room.lamps.values()
        if getattr(lamp, 'enabled', True) and getattr(lamp, 'ies', None) is not None
    )

    # Total zone grid points (enabled zones only)
    total_zone_points = sum(
        prod(zone.num_points)
        for zone in room.calc_zones.values()
        if getattr(zone, 'enabled', True)
    )

    # Reflectance surface info
    reflectance_enabled = (
        hasattr(room, 'ref_manager')
        and room.ref_manager.enabled
    )
    # Room walls and object faces alike get form factors to every zone
    all_surfs = room.all_surfaces
    num_surfaces = len(all_surfs)
    reflectance_grid_points = sum(
        prod(s.plane.num_points) for s in all_surfs.values()
    )

    # Memory estimate
    lamp_bytes = lamp_count * BYTES_PER_LAMP
    zone_bytes = total_zone_points * BYTES_PER_ZONE_POINT

    reflectance_bytes = 0
    if reflectance_enabled and reflectance_grid_points > 0:
        reflectance_bytes = (
            REFLECTANCE_OVERHEAD
            + reflectance_grid_points * total_zone_points * BYTES_PER_FORM_FACTOR_ENTRY
            + reflectance_grid_points ** 2 * BYTES_PER_SURFACE_PAIR
        )

    peak_bytes = max(
        lamp_bytes + zone_bytes + reflectance_bytes,
        MIN_PEAK_BYTES,
    )

    return {
        'lamp_bytes': lamp_bytes,
        'zone_bytes': zone_bytes,
        'reflectance_bytes': reflectance_bytes,
        'peak_bytes': peak_bytes,
        'total_zone_points': total_zone_points,
        'lamp_count': lamp_count,
        'reflectance_grid_points': reflectance_grid_points,
        'num_surfaces': num_surfaces,
    }
