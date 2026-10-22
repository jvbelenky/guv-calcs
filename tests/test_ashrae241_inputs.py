from dataclasses import replace

import pytest

from guv_calcs.ashrae241 import (
    CFM_TO_LPS,
    AirCleaner,
    Ashrae241Inputs,
    Filtration,
    NaturalVentilation,
    OutdoorAir,
    Reference,
    SourceContext,
    get_space_type,
    source_from_dict,
    space_types,
)

CLASSROOM = get_space_type("Classrooms (age 9 plus)")
CTX = SourceContext(
    floor_area_ft2=260,
    volume_m3=2340 * 0.0283168,
    occupants=9.1,
    space_type=CLASSROOM,
    room_air_changes=2.0,
)


def test_outdoor_air_621_min():
    assert OutdoorAir().v_acs(CTX) == pytest.approx(122.2 * CFM_TO_LPS)


def test_outdoor_air_621_min_zero_rp_row():
    zero = next(s for s in space_types().values() if s.rp_cfm == 0)
    ctx = replace(CTX, space_type=zero)
    assert OutdoorAir().v_acs(ctx) == pytest.approx(zero.ra_cfm_ft2 * 260 * CFM_TO_LPS)


def test_outdoor_air_needs_occupants():
    with pytest.raises(ValueError):
        OutdoorAir().v_acs(replace(CTX, occupants=None))


def test_outdoor_air_ach_defaults_to_room_air_changes():
    assert OutdoorAir(basis="ach").v_acs(CTX) == pytest.approx(2.0 * CTX.volume_m3 / 3.6)


def test_outdoor_air_zf():
    assert OutdoorAir(basis="flow", value=100, zf=0.5).v_acs(CTX) == 50


def test_no_621_min_for_std170():
    ctx = replace(CTX, space_type=get_space_type("Waiting room"))
    with pytest.raises(ValueError):
        OutdoorAir().v_acs(ctx)


def test_filters_in_series():
    f = Filtration(ratings=[13, 8, "HEPA"], recirc_flow=200)
    assert f.pr == pytest.approx(1 - (1 - 0.77) * (1 - 0.0) * (1 - 0.99))
    assert f.v_acs(CTX) == pytest.approx(200 * f.pr)


def test_merv_below_11_counts_zero():
    assert Filtration(ratings=[10], recirc_flow=500).v_acs(CTX) == 0


def test_air_cleaner_and_natural_ventilation():
    assert AirCleaner(50, "x").v_acs(CTX) == 50
    assert NaturalVentilation(20).v_acs(CTX) == 20


def test_source_dict_fields():
    assert OutdoorAir("flow", 80, 0.5).to_dict() == {
        "kind": "outdoor_air", "basis": "flow", "value": 80, "zf": 0.5,
    }
    assert Filtration([13], 400, 0.8).to_dict() == {
        "kind": "filtration", "ratings": [13], "recirc_flow": 400, "zf": 0.8,
    }
    assert AirCleaner(50, "u").to_dict() == {"kind": "air_cleaner", "v_acs_lps": 50, "label": "u"}
    assert NaturalVentilation(20).to_dict() == {"kind": "natural_ventilation", "flow": 20}


def test_unknown_source_kind():
    with pytest.raises(ValueError):
        source_from_dict({"kind": "magic"})


def test_inputs_round_trip():
    inp = Ashrae241Inputs(
        "Classrooms (age 9 plus)",
        occupants=12,
        vocalization=True,
        sources=[
            OutdoorAir("flow", 80),
            Filtration([13], 400, 0.8),
            AirCleaner(50, "HEPA unit"),
            NaturalVentilation(20),
        ],
        reference=Reference("Human coronavirus", "high"),
    )
    assert Ashrae241Inputs.from_dict(inp.to_dict()) == inp
    assert inp.to_dict()["reference"] == {"organism": "Human coronavirus", "statistic": "high"}


def test_from_dict_keeps_unknown_space_type():
    assert Ashrae241Inputs.from_dict({"space_type": "Moon base"}).space_type == "Moon base"


@pytest.mark.parametrize("ref", [
    {"organism": "Ebola", "statistic": "median"},
    {"organism": "Phage MS2", "statistic": "max"},
])
def test_bad_reference_rejected(ref):
    with pytest.raises(ValueError):
        Ashrae241Inputs.from_dict({"space_type": "Office", "reference": ref})


def test_default_sources():
    assert Ashrae241Inputs.default("Classrooms (age 9 plus)").sources == [OutdoorAir()]
    assert Ashrae241Inputs.default("Waiting room").sources == []
    assert Ashrae241Inputs.default("Office").sources == [OutdoorAir()]
