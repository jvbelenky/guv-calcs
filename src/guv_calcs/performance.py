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

# Empirically fitted from tracemalloc profiling across 11 configurations.
# Validated: all reflectance cases overpredict by 1.24-1.34× (conservative).

# Per lamp: IES photometry (~66 KB) + spectrum (~112 KB) + metadata
BYTES_PER_LAMP = 1_000_000

# Per zone point: zone result arrays + overhead
BYTES_PER_ZONE_POINT_BASE = 90

# Per zone point per lamp: 2 × float32 in lamp cache
BYTES_PER_ZONE_POINT_PER_LAMP = 8

# Form factors + theta_zone per (surface_point, zone_point) pair: 2 × float32
BYTES_PER_FORM_FACTOR_ENTRY = 8

# Peak multiplier: temporaries during form factor computation use ~2×
PEAK_MULTIPLIER = 2.0


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
    """Estimate memory usage (bytes) for a Room during calculation.

    Memory model (empirically calibrated from tracemalloc profiling):

    - Per lamp: ~1 MB (IES photometry + spectrum + metadata)
    - Per zone point: 90 bytes base + 8 bytes per lamp (cache entries)
    - Reflectance form factors: 8 bytes × surface_pts × zone_pts per surface
      + 8 bytes × total_surface_pts² (surface↔surface interreflection)
    - Peak ≈ 2× allocated (temporaries during form factor computation)

    Args:
        room: A Room instance with lamps, calc_zones, surfaces, and ref_manager.

    Returns:
        Dictionary with memory breakdown in bytes:
        - lamp_bytes: memory for lamp IES/spectrum data
        - zone_bytes: memory for zone result arrays and lamp caches
        - reflectance_bytes: memory for form factor matrices (0 if disabled)
        - stored_bytes: total retained memory after calculation
        - peak_bytes: estimated peak memory during calculation
        - total_zone_points: total grid points across enabled zones
        - lamp_count: number of enabled lamps with IES data
        - reflectance_grid_points: total surface grid points
        - num_surfaces: number of reflective surfaces
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
    num_surfaces = len(room.surfaces)
    reflectance_grid_points = sum(
        prod(s.plane.num_points) for s in room.surfaces.values()
    )

    # Memory estimate
    lamp_bytes = lamp_count * BYTES_PER_LAMP
    zone_bytes = total_zone_points * (
        BYTES_PER_ZONE_POINT_BASE + lamp_count * BYTES_PER_ZONE_POINT_PER_LAMP
    )

    reflectance_bytes = 0
    if reflectance_enabled and reflectance_grid_points > 0:
        avg_pts_per_surface = reflectance_grid_points / max(num_surfaces, 1)
        # Surface → zone form factors
        surf_zone_ff = (
            num_surfaces * avg_pts_per_surface
            * total_zone_points * BYTES_PER_FORM_FACTOR_ENTRY
        )
        # Surface ↔ surface form factors during interreflection
        surf_surf_ff = reflectance_grid_points ** 2 * BYTES_PER_FORM_FACTOR_ENTRY
        reflectance_bytes = surf_zone_ff + surf_surf_ff

    stored_bytes = lamp_bytes + zone_bytes + reflectance_bytes
    peak_bytes = stored_bytes * PEAK_MULTIPLIER

    return {
        'lamp_bytes': lamp_bytes,
        'zone_bytes': zone_bytes,
        'reflectance_bytes': reflectance_bytes,
        'stored_bytes': stored_bytes,
        'peak_bytes': peak_bytes,
        'total_zone_points': total_zone_points,
        'lamp_count': lamp_count,
        'reflectance_grid_points': reflectance_grid_points,
        'num_surfaces': num_surfaces,
    }
