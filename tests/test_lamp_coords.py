import numpy as np

from guv_calcs import Lamp


def test_lamp_get_cartesian_and_polar():
    lamp = Lamp.from_keyword("aerolamp")

    cart = lamp.get_cartesian()
    polar = lamp.get_polar()

    assert isinstance(cart, np.ndarray)
    assert isinstance(polar, np.ndarray)
    assert cart.shape[1] == 3
    assert polar.shape[0] == 3
