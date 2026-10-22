"""Per-point TLVs for the mix of lamps lighting a zone."""

import numpy as np
import pytest

from guv_calcs import Room, Lamp, EYE_LIMITS
from guv_calcs.safety import PhotStandard, effective_tlvs, get_tlvs


def _room(*wavelengths):
    room = Room(x=6, y=4, z=2.7, units="meters")
    for i, wl in enumerate(wavelengths):
        x = 1 + 4 * i
        lamp = Lamp.from_keyword("aerolamp", lamp_id=f"l{i}").move(x, 2, 2.7).aim(x, 2, 0)
        if wl is not None:
            lamp.set_wavelength(wl)
        room.add_lamp(lamp)
    room.add_standard_zones()
    room.calculate()
    return room


def test_one_lamp_type_has_its_own_limit_wherever_it_reaches():
    room = _room(254)
    tlvs = effective_tlvs(room, room.calc_zones[EYE_LIMITS], PhotStandard.ICNIRP, "eye")
    lit = np.isfinite(tlvs)
    assert lit.any()
    assert tlvs[lit] == pytest.approx(get_tlvs(254, PhotStandard.ICNIRP)[1])
    assert tlvs.shape == room.calc_zones[EYE_LIMITS].get_values().shape


def test_mixed_lamps_have_a_limit_between_theirs_that_follows_the_mix():
    room = _room(None, 254)  # the krypton chloride spectrum, and 254 nm
    zone = room.calc_zones[EYE_LIMITS]
    tlvs = effective_tlvs(room, zone, PhotStandard.ACGIH, "eye")
    kr = room.lamps["l0"].get_tlvs(PhotStandard.ACGIH)[1]
    hg = get_tlvs(254, PhotStandard.ACGIH)[1]
    lit = tlvs[np.isfinite(tlvs)]
    assert lit.min() >= hg - 1e-9 and lit.max() <= kr + 1e-9
    # Nearer the 254 nm lamp, the limit is nearer its own
    assert lit.max() - lit.min() > 1
    # Each lamp's dose over its own limit, summed: the dose over the mixed limit
    e0 = zone.lamp_cache["l0"].values
    e1 = zone.lamp_cache["l1"].values
    expected = e0 / kr + e1 / hg
    ok = np.isfinite(tlvs)
    assert ((e0 + e1)[ok] / tlvs[ok]) == pytest.approx(expected[ok])


def test_no_limit_without_a_lamp_that_has_one():
    room = Room(x=6, y=4, z=2.7, units="meters")
    room.add_standard_zones()
    room.calculate()
    assert effective_tlvs(room, room.calc_zones[EYE_LIMITS]) is None
