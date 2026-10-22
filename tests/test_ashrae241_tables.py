import pytest

from guv_calcs.ashrae241 import (
    categories,
    get_category,
    get_space_type,
    merv_pr,
    space_types,
)


def test_every_space_type_maps_to_a_category():
    cats = categories()
    assert all(st.category in cats for st in space_types().values())


def test_counts():
    assert len(categories()) == 25
    assert len(space_types()) == 108


def test_classroom_row():
    st = get_space_type("Classrooms (age 9 plus)")
    assert (st.rp_cfm, st.ra_cfm_ft2, st.density_per_1000ft2, st.category) == (
        10,
        0.12,
        35,
        "Classroom",
    )
    assert get_category("Classroom").ecai_cfm == 40


def test_vocalization_rows():
    assert {n for n, s in space_types().items() if s.vocalization} == {
        "Music/theater/dance",
        "Stages, studios",
    }


def test_std170_rows_have_no_621_data():
    for name in ("Waiting room", "Resident room"):
        st = get_space_type(name)
        assert st.std170 and st.rp_cfm is None and st.density_per_1000ft2 is None


def test_zero_density_is_none():
    assert get_space_type("Warehouses").density_per_1000ft2 is None


@pytest.mark.parametrize(
    "name,rp,ra,density",
    [
        # ASHRAE 62.1-2022 Normative Appendix D, Table D-1
        ("Birthing room", 10, 0.18, 15),
        ("General examination room", 7.5, 0.12, 20),
        ("Psychiatric group room", 5, 0.06, 50),
    ],
)
def test_outpatient_rows_match_table_d1(name, rp, ra, density):
    st = get_space_type(name)
    assert "Outpatient" in st.group
    assert (st.rp_cfm, st.ra_cfm_ft2, st.density_per_1000ft2) == (rp, ra, density)


@pytest.mark.parametrize(
    "rating,pr",
    [(8, 0.0), (10, 0.0), (11, 0.60), (13, 0.77), (16, 0.95), ("HEPA", 0.99)],
)
def test_merv_pr(rating, pr):
    assert merv_pr(rating) == pr


@pytest.mark.parametrize("rating", [0, 17, "MERV 8", None])
def test_merv_pr_rejects_out_of_range(rating):
    with pytest.raises(ValueError):
        merv_pr(rating)


def test_unknown_space_type():
    with pytest.raises(ValueError, match="Unknown ASHRAE 62.1 space type"):
        get_space_type("Moon base")
