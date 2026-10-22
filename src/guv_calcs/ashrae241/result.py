"""Result of an ASHRAE 241 compliance check."""

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from .inputs import Organism

Origin = Literal["default_62_1", "user"]
AirflowStatus = Literal["compliant", "non_compliant", "incomplete"]
IncompleteReason = Literal["not_calculated", "occupants_required", "missing_kdata"]
UVSafetyStatus = Literal["pass", "fail", "not_calculated"]


@dataclass(frozen=True)
class Assumption:
    """A stated assumption behind the result, as a code and its parameters."""

    code: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceResult:
    """One clean-air source's contribution (L/s)."""

    kind: str
    label: str
    v_acs: float
    origin: Origin


@dataclass(frozen=True)
class Ashrae241Result:
    """Airflow and UV-safety verdicts for a room. Flows in L/s, eACH in 1/h."""

    space_type: str
    category: str
    ecai: float
    vocalization_factor: int
    occupants: float | None
    occupants_origin: Origin
    v_ecai: float | None
    sources: tuple[SourceResult, ...]
    v_uv: float | None
    v_uv_ranges: dict[Organism, tuple[float, float, float]] | None
    v_supplied: float
    margin: float | None
    required_each_uv: float | None
    airflow_status: AirflowStatus
    incomplete_reason: IncompleteReason | None
    uv_safety_status: UVSafetyStatus
    assumptions: tuple[Assumption, ...]

    def to_dict(self) -> dict:
        """JSON-safe dict: tuples become lists, numpy scalars plain numbers."""
        return _json_safe(asdict(self))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
