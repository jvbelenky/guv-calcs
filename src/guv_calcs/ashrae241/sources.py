"""Clean-air sources: each contributes an equivalent clean airflow V_acs (L/s)."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from .tables import CFM_TO_LPS, SpaceType, merv_pr


@dataclass(frozen=True)
class SourceContext:
    """Room facts a source may need to compute its V_acs."""

    floor_area_ft2: float
    volume_m3: float
    occupants: float | None
    space_type: SpaceType
    room_air_changes: float


@dataclass
class OutdoorAir:
    """Outdoor air, as the Std 62.1 minimum, a flow (L/s) or air changes/hr."""

    kind: ClassVar[str] = "outdoor_air"
    basis: Literal["std621_min", "flow", "ach"] = "std621_min"
    value: float | None = None
    zf: float = 1.0

    def v_acs(self, ctx: SourceContext) -> float:
        if self.basis == "std621_min":
            st = ctx.space_type
            if st.rp_cfm is None or st.ra_cfm_ft2 is None:
                raise ValueError(f"No Std 62.1 minimum for {st.name!r}")
            if ctx.occupants is None:
                raise ValueError("Occupants are required for the Std 62.1 minimum")
            cfm = st.rp_cfm * ctx.occupants + st.ra_cfm_ft2 * ctx.floor_area_ft2
            return self.zf * cfm * CFM_TO_LPS
        if self.basis == "flow":
            if self.value is None:
                raise ValueError("A flow value is required for the 'flow' basis")
            return self.zf * self.value
        if self.basis == "ach":
            ach = self.value if self.value is not None else ctx.room_air_changes
            return self.zf * ach * ctx.volume_m3 * 1000 / 3600
        raise ValueError(f"Unknown outdoor air basis: {self.basis!r}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "basis": self.basis, "value": self.value, "zf": self.zf}


@dataclass
class Filtration:
    """Recirculating filters in series, each a MERV rating (1-16) or "HEPA"."""

    kind: ClassVar[str] = "filtration"
    ratings: list[int | str]
    recirc_flow: float
    zf: float = 1.0

    @property
    def pr(self) -> float:
        remaining = 1.0
        for r in self.ratings:
            remaining *= 1 - merv_pr(r)
        return 1 - remaining

    def v_acs(self, ctx: SourceContext) -> float:
        return self.zf * self.pr * self.recirc_flow

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ratings": list(self.ratings),
            "recirc_flow": self.recirc_flow,
            "zf": self.zf,
        }


@dataclass
class AirCleaner:
    """A standalone air cleaner with a known equivalent clean airflow (L/s)."""

    kind: ClassVar[str] = "air_cleaner"
    v_acs_lps: float
    label: str = ""

    def v_acs(self, ctx: SourceContext) -> float:
        return self.v_acs_lps

    def to_dict(self) -> dict:
        return {"kind": self.kind, "v_acs_lps": self.v_acs_lps, "label": self.label}


@dataclass
class NaturalVentilation:
    """Natural ventilation flow (L/s)."""

    kind: ClassVar[str] = "natural_ventilation"
    flow: float

    def v_acs(self, ctx: SourceContext) -> float:
        return self.flow

    def to_dict(self) -> dict:
        return {"kind": self.kind, "flow": self.flow}


CleanAirSource = OutdoorAir | Filtration | AirCleaner | NaturalVentilation

_SOURCE_TYPES = {
    cls.kind: cls for cls in (OutdoorAir, Filtration, AirCleaner, NaturalVentilation)
}


def source_from_dict(d: dict) -> CleanAirSource:
    """Rebuild a source from its `to_dict` form; unknown kinds raise ValueError."""
    cls = _SOURCE_TYPES.get(d.get("kind"))
    if cls is None:
        raise ValueError(f"Unknown clean air source kind: {d.get('kind')!r}")
    return cls(**{k: v for k, v in d.items() if k != "kind"})
