import pytest

from guv_calcs import Project, Room
from guv_calcs.ashrae241 import Ashrae241Inputs, Ashrae241Result, Filtration


def _inputs():
    return Ashrae241Inputs(
        "Classrooms (age 9 plus)", occupants=12, sources=[Filtration([13], 400)]
    )


def test_guv_round_trip(tmp_path):
    room = Room(x=13, y=20, z=9, units="feet", ashrae241=_inputs())
    path = str(tmp_path / "r.guv")
    room.save(path)
    loaded = Room.load(path)
    assert loaded.ashrae241 == room.ashrae241


def test_dict_is_accepted():
    room = Room(x=4, y=6, z=2.7, ashrae241=_inputs().to_dict())
    assert room.ashrae241 == _inputs()


def test_default_is_none_and_serialized_as_none():
    room = Room(x=4, y=6, z=2.7)
    assert room.ashrae241 is None
    assert room.to_dict()["ashrae241"] is None


def test_old_file_without_key_loads_as_none():
    d = Room(x=4, y=6, z=2.7).to_dict()
    d.pop("ashrae241")
    assert Room.from_dict(d).ashrae241 is None


def test_unknown_space_type_still_loads():
    d = Room(x=4, y=6, z=2.7).to_dict()
    d["ashrae241"] = {"space_type": "Moon base"}
    room = Room.from_dict(d)
    with pytest.raises(ValueError):
        room.check_ashrae241()


def test_check_without_inputs_raises():
    with pytest.raises(ValueError, match="no ASHRAE 241 inputs"):
        Room(x=4, y=6, z=2.7).check_ashrae241()


def test_check_returns_result():
    room = Room(x=13, y=20, z=9, units="feet", ashrae241=_inputs())
    assert isinstance(room.check_ashrae241(), Ashrae241Result)


def test_241_edits_do_not_change_state():
    room = Room(x=4, y=6, z=2.7)
    calc, update = room.get_calc_state(), room.get_update_state()
    room.ashrae241 = Ashrae241Inputs("Office space")
    assert room.get_calc_state() == calc
    assert room.get_update_state() == update


def test_project_round_trip():
    project = Project()
    project.create_room(room_id="a", x=4, y=6, z=2.7, ashrae241=_inputs())
    loaded = Project.from_dict(project.to_dict())
    assert loaded.room("a").ashrae241 == _inputs()
