import json
from dataclasses import replace

import pytest

from guv_calcs import Lamp, Room
from guv_calcs.ashrae241 import (
    CFM_TO_LPS,
    Ashrae241Inputs,
    Ashrae241Result,
    OutdoorAir,
    Reference,
    evaluate,
)
from guv_calcs.safety import ComplianceStatus, PhotStandard


def _room_222():
    """Small calculated room with two 222 nm lamps at the ceiling."""
    room = Room(x=3, y=3, z=2.7, units="meters").add_standard_zones()
    for x in (1, 2):
        room.add_lamp(Lamp.from_keyword("aerolamp").move(x, 1.5, 2.7).aim(x, 1.5, 0))
    room.calculate()
    return room


def _room_222_and_405():
    """Calculated room with a 222 nm lamp and a 405 nm lamp (no MS2 data)."""
    room = Room(x=3, y=3, z=2.7, units="meters").add_standard_zones()
    lamp = Lamp.from_keyword("aerolamp")
    room.add_lamp(lamp.move(1.5, 1.5, 2.7).aim(1.5, 1.5, 0))
    room.add_lamp(
        Lamp(filedata=lamp.ies, guv_type="other", wavelength=405)
        .move(1, 1, 2.7)
        .aim(1, 1, 0)
    )
    room.calculate()
    return room


@pytest.fixture
def classroom_room():
    return Room(x=13, y=20, z=9, units="feet").add_standard_zones()


@pytest.fixture(scope="module")
def calculated_222_room():
    return _room_222()


@pytest.fixture(scope="module")
def calculated_room_with_unknown_wavelength():
    return _room_222_and_405()


def test_classroom_before_calculating(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs.default("Classrooms (age 9 plus)"))
    assert r.occupants == pytest.approx(9.1)
    assert r.occupants_origin == "default_62_1"
    assert r.v_ecai / CFM_TO_LPS == pytest.approx(364)
    assert r.sources[0].v_acs / CFM_TO_LPS == pytest.approx(122.2)
    assert r.sources[0].origin == "default_62_1"
    assert r.required_each_uv == pytest.approx(6.2, abs=0.05)
    assert r.v_uv is None and r.v_uv_ranges is None and r.margin is None
    assert (r.airflow_status, r.incomplete_reason, r.uv_safety_status) == (
        "incomplete",
        "not_calculated",
        "not_calculated",
    )
    assert {a.code for a in r.assumptions} >= {
        "uv_simulated",
        "occupants_from_density",
        "outdoor_air_621_min",
    }


def test_music_room_vocalization(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs("Music/theater/dance", occupants=25))
    assert r.vocalization_factor == 2 and r.v_ecai / CFM_TO_LPS == pytest.approx(2500)
    assert r.occupants_origin == "user"
    assert "occupants_from_density" not in {a.code for a in r.assumptions}


def test_vocalization_override_off(classroom_room):
    inputs = Ashrae241Inputs("Music/theater/dance", occupants=25, vocalization=False)
    assert evaluate(classroom_room, inputs).vocalization_factor == 1


def test_density_zero_requires_occupants(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs("Warehouses"))
    assert (r.airflow_status, r.incomplete_reason, r.v_ecai) == (
        "incomplete",
        "occupants_required",
        None,
    )
    assert r.occupants is None and r.margin is None and r.required_each_uv is None


@pytest.mark.parametrize("occupants", [0, -3])
def test_nonpositive_occupants_are_incomplete(calculated_222_room, occupants):
    r = evaluate(calculated_222_room, Ashrae241Inputs("Office", occupants=occupants))
    assert (r.airflow_status, r.incomplete_reason) == ("incomplete", "occupants_required")
    assert r.occupants is None and r.v_ecai is None


def test_occupants_required_with_default_outdoor_air(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs.default("Warehouses"))
    assert (r.airflow_status, r.incomplete_reason) == ("incomplete", "occupants_required")


def test_occupants_required_wins_over_not_calculated(calculated_222_room):
    r = evaluate(calculated_222_room, Ashrae241Inputs("Warehouses"))
    assert r.v_uv is not None
    assert r.incomplete_reason == "occupants_required"


def test_calculated_meets_with_strong_lamp(calculated_222_room):
    r = evaluate(calculated_222_room, Ashrae241Inputs("Office", occupants=1))
    assert r.airflow_status == "compliant" and r.incomplete_reason is None
    assert r.margin > 0
    low, median, high = r.v_uv_ranges["Phage MS2"]
    assert low <= r.v_uv <= high and r.v_uv == pytest.approx(median)
    assert set(r.v_uv_ranges) == {"Phage MS2", "Human coronavirus"}
    assert r.v_supplied == pytest.approx(r.v_uv)
    assert r.margin == pytest.approx(r.v_supplied - r.v_ecai)
    assert not any(a.code == "no_kdata" for a in r.assumptions)


def test_reference_statistic_and_organism(calculated_222_room):
    base = Ashrae241Inputs("Office", occupants=1)
    low = evaluate(calculated_222_room, replace(base, reference=Reference(statistic="low")))
    assert low.v_uv == pytest.approx(low.v_uv_ranges["Phage MS2"][0])
    cov = evaluate(
        calculated_222_room,
        replace(base, reference=Reference("Human coronavirus", "high")),
    )
    assert cov.v_uv == pytest.approx(cov.v_uv_ranges["Human coronavirus"][2])


def test_gap_and_meets_is_compliant(calculated_room_with_unknown_wavelength):
    r = evaluate(calculated_room_with_unknown_wavelength, Ashrae241Inputs("Office", occupants=0.01))
    assert r.airflow_status == "compliant"
    gaps = [a for a in r.assumptions if a.code == "no_kdata"]
    assert [a.params["wavelength"] for a in gaps] == [405]


def test_gap_and_fails_is_incomplete(calculated_room_with_unknown_wavelength):
    r = evaluate(calculated_room_with_unknown_wavelength, Ashrae241Inputs("Office", occupants=10_000))
    assert (r.airflow_status, r.incomplete_reason) == ("incomplete", "missing_kdata")
    gaps = [a for a in r.assumptions if a.code == "no_kdata"]
    assert [a.params for a in gaps] == [{"wavelength": 405.0}]


def test_fails_without_gap(calculated_222_room):
    r = evaluate(calculated_222_room, Ashrae241Inputs("Office", occupants=10_000))
    assert r.airflow_status == "non_compliant" and r.margin < 0


def test_non_uv_sources_count_toward_supplied(calculated_222_room):
    inputs = Ashrae241Inputs("Office", occupants=10, sources=[OutdoorAir("flow", 50.0)])
    r = evaluate(calculated_222_room, inputs)
    assert r.v_supplied == pytest.approx(50.0 + r.v_uv)
    assert r.sources[0].origin == "user"
    assert r.required_each_uv == pytest.approx(
        max(0, r.v_ecai - 50.0) * 3.6 / calculated_222_room.dim.cubic_meters
    )


@pytest.mark.filterwarnings("ignore:No valid lamps:UserWarning")
def test_all_lamps_disabled_is_zero_uv():
    room = _room_222()
    for lamp in room.lamps.values():
        lamp.enabled = False
    room.calculate()
    r = evaluate(room, Ashrae241Inputs("Office", occupants=1))
    assert r.v_uv == 0.0 and r.airflow_status in ("compliant", "non_compliant")


def test_resize_follows_defaults_not_user_values(classroom_room):
    default = Ashrae241Inputs("Classrooms (age 9 plus)", sources=[OutdoorAir("ach", None)])
    a = evaluate(classroom_room, default)
    classroom_room.set_dimensions(x=26)
    b = evaluate(classroom_room, default)
    assert b.occupants == pytest.approx(2 * a.occupants)
    assert b.sources[0].v_acs == pytest.approx(2 * a.sources[0].v_acs)
    assert evaluate(classroom_room, replace(default, occupants=9)).occupants == 9


def test_polygon_room_uses_polygon_area():
    room = Room(
        polygon=[(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)],
        z=9,
        units="feet",
    )
    assert evaluate(room, Ashrae241Inputs("Office")).occupants == pytest.approx(300 * 5 / 1000)


def test_metric_room_same_requirement(classroom_room):
    metric = Room(x=13 * 0.3048, y=20 * 0.3048, z=9 * 0.3048, units="meters")
    inputs = Ashrae241Inputs.default("Classrooms (age 9 plus)")
    assert evaluate(metric, inputs).v_ecai == pytest.approx(
        evaluate(classroom_room, inputs).v_ecai, rel=1e-6
    )


def test_uv_safety_uses_acgih_even_on_icnirp():
    room = _room_222()
    room.set_standard("IEC 62471-6:2022 (ICNIRP Limits)")
    room.calculate()
    r = evaluate(room, Ashrae241Inputs("Office", occupants=1))
    expected = room.check_lamps(standard=PhotStandard.ACGIH).status == ComplianceStatus.COMPLIANT
    assert r.uv_safety_status == ("pass" if expected else "fail")
    planes = [a for a in r.assumptions if a.code == "safety_planes"]
    assert planes and planes[0].params["standard"] == "ACGIH"


def test_check_lamps_standard_override():
    room = _room_222()
    lamp_id = next(iter(room.lamps))
    acgih = room.check_lamps().lamp_results[lamp_id]
    room.set_standard("IEC 62471-6:2022 (ICNIRP Limits)")
    room.calculate()
    icnirp = room.check_lamps().lamp_results[lamp_id]
    forced = room.check_lamps(standard=PhotStandard.ACGIH).lamp_results[lamp_id]
    assert icnirp.eye_tlv != acgih.eye_tlv
    assert forced.eye_tlv == pytest.approx(acgih.eye_tlv)
    assert forced.skin_tlv == pytest.approx(acgih.skin_tlv)


def test_uv_safety_pass_and_planes(calculated_222_room):
    r = evaluate(calculated_222_room, Ashrae241Inputs("Office", occupants=1))
    expected = (
        "pass"
        if calculated_222_room.check_lamps(standard=PhotStandard.ACGIH).status
        == ComplianceStatus.COMPLIANT
        else "fail"
    )
    assert r.uv_safety_status == expected
    planes = [a for a in r.assumptions if a.code == "safety_planes"]
    assert planes[0].params == {"height_m": pytest.approx(1.8), "standard": "ACGIH"}


def test_uv_safety_not_calculated_without_zones():
    room = Room(x=3, y=3, z=2.7, units="meters")
    r = evaluate(room, Ashrae241Inputs("Office", occupants=1))
    assert (r.uv_safety_status, r.incomplete_reason) == ("not_calculated", "not_calculated")


def test_std170_assumption(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs("Waiting room", occupants=4))
    assert "std170_space" in {a.code for a in r.assumptions}


def test_unknown_space_type_raises(classroom_room):
    with pytest.raises(ValueError):
        evaluate(classroom_room, Ashrae241Inputs("Moon base"))


def test_result_to_dict_is_json_safe(classroom_room):
    r = evaluate(classroom_room, Ashrae241Inputs.default("Classrooms (age 9 plus)"))
    assert isinstance(r, Ashrae241Result)
    json.dumps(r.to_dict())


def test_calculated_result_to_dict_is_json_safe(calculated_room_with_unknown_wavelength):
    r = evaluate(calculated_room_with_unknown_wavelength, Ashrae241Inputs("Office", occupants=1))
    d = json.loads(json.dumps(r.to_dict()))
    assert set(d["v_uv_ranges"]) == {"Phage MS2", "Human coronavirus"}
    assert isinstance(d["v_uv_ranges"]["Phage MS2"], list)
    assert isinstance(d["sources"], list) and isinstance(d["assumptions"], list)
    assert d["v_uv"] == pytest.approx(r.v_uv)
