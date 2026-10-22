"""User inputs for an ASHRAE 241 compliance check."""

from dataclasses import dataclass, field
from typing import Literal, get_args

from .sources import CleanAirSource, OutdoorAir, source_from_dict
from .tables import get_space_type

Organism = Literal["Phage MS2", "Human coronavirus"]
Statistic = Literal["low", "median", "high"]


@dataclass
class Reference:
    """Reference organism and statistic of its UV susceptibility."""

    organism: Organism = "Phage MS2"
    statistic: Statistic = "median"

    def __post_init__(self):
        if self.organism not in get_args(Organism):
            raise ValueError(f"Unknown reference organism: {self.organism!r}")
        if self.statistic not in get_args(Statistic):
            raise ValueError(f"Unknown reference statistic: {self.statistic!r}")


@dataclass
class Ashrae241Inputs:
    space_type: str
    occupants: float | None = None
    vocalization: bool | None = None
    sources: list[CleanAirSource] = field(default_factory=list)
    reference: Reference = field(default_factory=Reference)

    def to_dict(self) -> dict:
        return {
            "space_type": self.space_type,
            "occupants": self.occupants,
            "vocalization": self.vocalization,
            "sources": [s.to_dict() for s in self.sources],
            "reference": {
                "organism": self.reference.organism,
                "statistic": self.reference.statistic,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ashrae241Inputs":
        """Rebuild from `to_dict` form. The space type is kept unvalidated."""
        return cls(
            space_type=d["space_type"],
            occupants=d.get("occupants"),
            vocalization=d.get("vocalization"),
            sources=[source_from_dict(s) for s in d.get("sources", [])],
            reference=Reference(**d.get("reference", {})),
        )

    @classmethod
    def default(cls, space_type: str) -> "Ashrae241Inputs":
        """Inputs with outdoor air at the Std 62.1 minimum, where the row has one."""
        has_rp = get_space_type(space_type).rp_cfm is not None
        return cls(space_type, sources=[OutdoorAir()] if has_rp else [])
