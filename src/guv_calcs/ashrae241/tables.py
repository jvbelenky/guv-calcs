"""ASHRAE 241 space types (mapped from 62.1 Table 6-1) and Table 5-1 categories."""

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources

CFM_TO_LPS = 0.47194745

STD170_GROUP = "Health Care (Std 170)"

# Table 7-1: particle removal fraction by MERV rating (below MERV 11 counts as 0)
_MERV_PR = {11: 0.60, 12: 0.71, 13: 0.77, 14: 0.88, 15: 0.91, 16: 0.95}
_HEPA_PR = 0.99


@dataclass(frozen=True)
class Category:
    group: str
    name: str
    ecai_cfm: float


@dataclass(frozen=True)
class SpaceType:
    group: str
    name: str
    rp_cfm: float | None
    ra_cfm_ft2: float | None
    density_per_1000ft2: float | None
    category: str
    vocalization: bool

    @property
    def std170(self) -> bool:
        return self.group == STD170_GROUP


def _rows(fname: str) -> list[dict[str, str]]:
    path = resources.files("guv_calcs.data").joinpath(fname)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(value: str) -> float | None:
    return float(value) if value else None


@cache
def categories() -> dict[str, Category]:
    """Table 5-1 categories by name, in table order."""
    return {
        r["category"]: Category(r["group"], r["category"], float(r["ecai_cfm_per_person"]))
        for r in _rows("ashrae241_categories.csv")
    }


@cache
def space_types() -> dict[str, SpaceType]:
    """Space types by name, in CSV order."""
    return {
        r["space_type"]: SpaceType(
            group=r["group"],
            name=r["space_type"],
            rp_cfm=_num(r["rp_cfm_per_person"]),
            ra_cfm_ft2=_num(r["ra_cfm_per_ft2"]),
            density_per_1000ft2=_num(r["density_per_1000ft2"]),
            category=r["category"],
            vocalization=r["vocalization"] == "true",
        )
        for r in _rows("ashrae241_space_types.csv")
    }


def get_space_type(name: str) -> SpaceType:
    try:
        return space_types()[name]
    except KeyError:
        raise ValueError(f"Unknown ASHRAE 62.1 space type: {name}") from None


def get_category(name: str) -> Category:
    try:
        return categories()[name]
    except KeyError:
        raise ValueError(f"Unknown ASHRAE 241 category: {name}") from None


def merv_pr(rating: int | str) -> float:
    """Table 7-1 particle removal fraction for a MERV rating (1-16) or "HEPA"."""
    if rating == "HEPA":
        return _HEPA_PR
    if isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 16:
        return _MERV_PR.get(rating, 0.0)
    raise ValueError(f"Invalid MERV rating: {rating!r}")
