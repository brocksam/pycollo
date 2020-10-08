"""Test for correct numerical calculation of quadrature.

Quadrature schemes computed are: Radau (LGR), Lobatto (LGL) and Gauss (LG). 

"""


import numpy as np
import pytest

import pycollo


class BackendMock:

    def __init__(self, ocp):
        self.ocp = ocp


@pytest.fixture
def lobatto_quadrature_fixture():
    ocp = pycollo.OptimalControlProblem("Dummy OCP")
    ocp.settings.quadrature_method = "lobatto"
    backend = BackendMock(ocp)
    quadrature = pycollo.quadrature.Quadrature(backend)
    return quadrature


@pytest.fixture
def radau_quadrature_fixture():
    ocp = pycollo.OptimalControlProblem("Dummy OCP")
    ocp.settings.quadrature_method = "radau"
    backend = BackendMock(ocp)
    quadrature = pycollo.quadrature.Quadrature(backend)
    return quadrature


def test_lobatto_backend_name(lobatto_quadrature_fixture):
    quadrature = lobatto_quadrature_fixture
    assert quadrature.settings.quadrature_method == "lobatto"


def test_radau_backend_name(radau_quadrature_fixture):
    quadrature = radau_quadrature_fixture
    assert quadrature.settings.quadrature_method == "radau"


def test_lobatto_weights(lobatto_quadrature_fixture):
    quadrature = lobatto_quadrature_fixture
    weights_2 = np.array([0.5, 0.5])
    weights_3 = np.array([0.16666666666666666,
                          0.66666666666666666,
                          0.16666666666666666])
    np.testing.assert_array_equal(quadrature.quadrature_weight(2), weights_2)
    np.testing.assert_array_equal(quadrature.quadrature_weight(3), weights_3)
