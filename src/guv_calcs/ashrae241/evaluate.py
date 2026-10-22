"""Evaluate a room against ASHRAE 241 equivalent clean airflow requirements."""

from typing import get_args

import numpy as np

from ..efficacy import each_uv_range
from ..safety import ComplianceStatus, PhotStandard
from ..standard_zones import EYE_LIMITS, SKIN_LIMITS, WHOLE_ROOM_FLUENCE
from ..units import LengthUnits, convert_length
from .inputs import Ashrae241Inputs, Organism
from .result import Ashrae241Result, Assumption, SourceResult
from .sources import AirCleaner, OutdoorAir, SourceContext
from .tables import CFM_TO_LPS, get_category, get_space_type

_STATISTIC_INDEX = {"low": 0, "median": 1, "high": 2}

_SOURCE_LABELS = {
    "outdoor_air": "Outdoor air",
    "filtration": "Filtration",
    "air_cleaner": "Air cleaner",
    "natural_ventilation": "Natural ventilation",
}


def evaluate(room, inputs: Ashrae241Inputs) -> Ashrae241Result:
    """Check a room's equivalent clean airflow and UV safety per ASHRAE 241."""
    space = get_space_type(inputs.space_type)
    category = get_category(space.category)
    assumptions = [Assumption("uv_simulated")]

    ft_per_unit = convert_length(room.dim.units, LengthUnits.FEET, 1.0)
    floor_area_ft2 = room.dim.polygon.area * ft_per_unit**2
    volume_m3 = room.dim.cubic_meters

    # occupants
    if inputs.occupants is not None:
        occupants, occupants_origin = float(inputs.occupants), "user"
    elif space.density_per_1000ft2:
        occupants = space.density_per_1000ft2 * floor_area_ft2 / 1000
        occupants_origin = "default_62_1"
        assumptions.append(Assumption("occupants_from_density"))
    else:
        occupants, occupants_origin = None, "default_62_1"

    vocal = inputs.vocalization if inputs.vocalization is not None else space.vocalization
    factor = 2 if vocal else 1
    ecai = category.ecai_cfm * factor * CFM_TO_LPS
    v_ecai = ecai * occupants if occupants is not None else None

    # non-UV clean air sources; without occupants the Std 62.1 minimum counts area only
    ctx = SourceContext(
        floor_area_ft2=floor_area_ft2,
        volume_m3=volume_m3,
        occupants=occupants if occupants is not None else 0.0,
        space_type=space,
        room_air_changes=room.air_changes,
    )
    sources = tuple(_source_result(s, ctx) for s in inputs.sources)
    if any(s.origin == "default_62_1" for s in sources):
        assumptions.append(Assumption("outdoor_air_621_min"))
    if space.std170:
        assumptions.append(Assumption("std170_space"))
    v_non_uv = sum(s.v_acs for s in sources)

    # UV
    v_uv_ranges, missing = _v_uv_ranges(room, volume_m3, inputs.reference.organism)
    if v_uv_ranges is None:
        v_uv = None
    else:
        v_uv = v_uv_ranges[inputs.reference.organism][
            _STATISTIC_INDEX[inputs.reference.statistic]
        ]
    v_supplied = v_non_uv + (v_uv or 0.0)

    if v_ecai is None:
        required_each_uv = None
    else:
        required_each_uv = max(0.0, v_ecai - v_non_uv) * 3.6 / volume_m3
    margin = v_supplied - v_ecai if v_ecai is not None and v_uv is not None else None

    # airflow status
    incomplete_reason = None
    if v_ecai is None:
        airflow_status, incomplete_reason = "incomplete", "occupants_required"
    elif v_uv is None:
        airflow_status, incomplete_reason = "incomplete", "not_calculated"
    elif v_supplied >= v_ecai:
        airflow_status = "compliant"
        assumptions += [Assumption("no_kdata", {"wavelength": float(wv)}) for wv in missing]
    elif missing:
        airflow_status, incomplete_reason = "incomplete", "missing_kdata"
    else:
        airflow_status = "non_compliant"

    uv_safety_status = _uv_safety(room)
    if EYE_LIMITS in room.calc_zones:
        height = room.calc_zones[EYE_LIMITS].height
        height_m = convert_length(room.dim.units, LengthUnits.METERS, height)
        assumptions.append(
            Assumption("safety_planes", {"height_m": float(height_m), "standard": "ACGIH"})
        )

    return Ashrae241Result(
        space_type=space.name,
        category=category.name,
        ecai=ecai,
        vocalization_factor=factor,
        occupants=occupants,
        occupants_origin=occupants_origin,
        v_ecai=v_ecai,
        sources=sources,
        v_uv=v_uv,
        v_uv_ranges=v_uv_ranges,
        v_supplied=v_supplied,
        margin=margin,
        required_each_uv=required_each_uv,
        airflow_status=airflow_status,
        incomplete_reason=incomplete_reason,
        uv_safety_status=uv_safety_status,
        assumptions=tuple(assumptions),
    )


def _source_result(source, ctx: SourceContext) -> SourceResult:
    default = isinstance(source, OutdoorAir) and source.basis == "std621_min"
    label = source.label if isinstance(source, AirCleaner) and source.label else None
    return SourceResult(
        kind=source.kind,
        label=label or _SOURCE_LABELS[source.kind],
        v_acs=float(source.v_acs(ctx)),
        origin="default_62_1" if default else "user",
    )


def _fluence_by_wavelength(room) -> dict | None:
    """Whole-room mean fluence (uW/cm2) per wavelength, or None if not calculated."""
    zone = room.calc_zones.get(WHOLE_ROOM_FLUENCE)
    if zone is None or zone.get_values() is None:
        return None
    lamps = {k: lamp for k, lamp in room.lamps.items() if lamp.enabled}
    return zone.calculator.cache.by_wavelength(lamps, reduce=np.mean)


def _v_uv_ranges(room, volume_m3: float, reference: str):
    """(low, median, high) V_UV in L/s per organism, and the reference's missing wavelengths."""
    fluence = _fluence_by_wavelength(room)
    if fluence is None:
        return None, []
    lps_per_each = volume_m3 * 1000 / 3600
    ranges, missing = {}, []
    for organism in get_args(Organism):
        per_wv, gaps = each_uv_range(fluence, organism)
        ranges[organism] = tuple(
            float(sum(r[i] for r in per_wv.values()) * lps_per_each) for i in range(3)
        )
        if organism == reference:
            missing = gaps
    return ranges, missing


def _uv_safety(room) -> str:
    for zone_id in (SKIN_LIMITS, EYE_LIMITS):
        zone = room.calc_zones.get(zone_id)
        if zone is None or zone.get_values() is None:
            return "not_calculated"
    result = room.check_lamps(standard=PhotStandard.ACGIH)
    return "pass" if result.status == ComplianceStatus.COMPLIANT else "fail"
