"""Test creation and initialisation of Iteration objects."""


import numpy as np
import pytest

import pycollo

from .iteration_scaling_test_data import (EXPECT_V,
                                          EXPECT_R,
                                          EXPECT_V_INV,
                                          EXPECT_X,
                                          EXPECT_X_TILDE,
                                          )


@pytest.fixture
def double_pendulum_initialised_fixture(double_pendulum_fixture):
    ocp, _ = double_pendulum_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()
    ocp._initialise_quadrature()
    ocp._postprocess_backend()
    ocp._initialise_initial_mesh()
    ocp._check_initial_guess()
    ocp._backend.mesh_iterations = []
    iteration = object.__new__(pycollo.iteration.Iteration)
    iteration.backend = ocp._backend
    iteration.ocp = ocp
    iteration.index = 0
    iteration.number = 1
    iteration._mesh = ocp._backend.initial_mesh
    iteration.prev_guess = ocp._backend.initial_guess
    iteration.interpolate_guess_to_mesh(iteration.prev_guess)
    iteration.create_variable_constraint_counts_slices()
    scaling = object.__new__(pycollo.scaling.CasadiIterationScaling)
    return ocp, iteration, scaling


def test_initialise_scaling_init(double_pendulum_initialised_fixture):
    """Initialisation successful."""
    ocp, iteration, scaling = double_pendulum_initialised_fixture
    scaling.__init__(iteration)

    assert isinstance(scaling, pycollo.scaling.CasadiIterationScaling)
    assert hasattr(scaling, "iteration")
    assert scaling.iteration is iteration
    assert hasattr(scaling, "backend")
    assert scaling.backend is ocp._backend


def test_initialise_scaling(double_pendulum_initialised_fixture):
    """Has correct attrs and methods."""
    ocp, iteration, scaling = double_pendulum_initialised_fixture
    scaling.__init__(iteration)

    assert hasattr(scaling, "_V")
    assert isinstance(scaling._V, np.ndarray)
    assert scaling._V.shape == (iteration.num_x, )
    assert hasattr(scaling, "_r")
    assert isinstance(scaling._r, np.ndarray)
    assert scaling._r.shape == (iteration.num_x, )
    assert hasattr(scaling, "_V_inv")
    assert isinstance(scaling._V_inv, np.ndarray)
    assert scaling._V_inv.shape == (iteration.num_x, )

    np.testing.assert_allclose(scaling._V, EXPECT_V)
    np.testing.assert_allclose(scaling._r, EXPECT_R)
    np.testing.assert_allclose(scaling._V_inv, EXPECT_V_INV)

    assert hasattr(scaling, "scale_x")
    assert callable(scaling.scale_x)
    assert hasattr(scaling, "unscale_x")
    assert callable(scaling.unscale_x)

    scale_unscale = scaling.unscale_x(scaling.scale_x(EXPECT_X))
    unscale_scale = scaling.scale_x(scaling.unscale_x(EXPECT_X_TILDE))
    np.testing.assert_allclose(scaling.scale_x(EXPECT_X), EXPECT_X_TILDE)
    np.testing.assert_allclose(scaling.unscale_x(EXPECT_X_TILDE), EXPECT_X)
    np.testing.assert_allclose(scale_unscale, EXPECT_X)
    np.testing.assert_allclose(unscale_scale, EXPECT_X_TILDE)
    assert scaling.scale_x(EXPECT_X).all() >= -0.5
    assert scaling.scale_x(EXPECT_X).all() <= 0.5
