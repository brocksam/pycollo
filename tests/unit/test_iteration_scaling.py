"""Test creation and initialisation of Iteration objects."""


import numpy as np
import pytest

import pycollo

from .iteration_scaling_test_data_double_pendulum import (EXPECT_V_DP,
                                                          EXPECT_R_DP,
                                                          EXPECT_V_INV_DP,
                                                          EXPECT_X_DP,
                                                          EXPECT_X_TILDE_DP,
                                                          )
from .iteration_scaling_test_data_brachistochrone import (EXPECT_V_BR,
                                                          EXPECT_R_BR,
                                                          EXPECT_V_INV_BR,
                                                          EXPECT_X_BR,
                                                          EXPECT_X_TILDE_BR,
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


@pytest.fixture
def brachistochrone_initialised_fixture(brachistochrone_fixture):
    ocp, _ = brachistochrone_fixture
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


def test_initialise_scaling_init_dp(double_pendulum_initialised_fixture):
    """Initialisation successful."""
    ocp, iteration, scaling = double_pendulum_initialised_fixture
    scaling.__init__(iteration)

    assert isinstance(scaling, pycollo.scaling.CasadiIterationScaling)
    assert hasattr(scaling, "iteration")
    assert scaling.iteration is iteration
    assert hasattr(scaling, "backend")
    assert scaling.backend is ocp._backend


def test_initialise_scaling_init_br(brachistochrone_initialised_fixture):
    """Initialisation successful."""
    ocp, iteration, scaling = brachistochrone_initialised_fixture
    scaling.__init__(iteration)

    assert isinstance(scaling, pycollo.scaling.CasadiIterationScaling)
    assert hasattr(scaling, "iteration")
    assert scaling.iteration is iteration
    assert hasattr(scaling, "backend")
    assert scaling.backend is ocp._backend


def test_initialise_scaling_dp(double_pendulum_initialised_fixture):
    """Has correct attrs and methods."""
    ocp, iteration, scaling = double_pendulum_initialised_fixture
    scaling.__init__(iteration)

    assert hasattr(scaling, "V")
    assert isinstance(scaling.V, np.ndarray)
    assert scaling.V.shape == (iteration.num_x, )
    assert hasattr(scaling, "r")
    assert isinstance(scaling.r, np.ndarray)
    assert scaling.r.shape == (iteration.num_x, )
    assert hasattr(scaling, "V_inv")
    assert isinstance(scaling.V_inv, np.ndarray)
    assert scaling.V_inv.shape == (iteration.num_x, )

    np.testing.assert_allclose(scaling.V, EXPECT_V_DP)
    np.testing.assert_allclose(scaling.r, EXPECT_R_DP)
    np.testing.assert_allclose(scaling.V_inv, EXPECT_V_INV_DP)

    assert hasattr(scaling, "scale_x")
    assert callable(scaling.scale_x)
    assert hasattr(scaling, "unscale_x")
    assert callable(scaling.unscale_x)

    scale_unscale = scaling.unscale_x(scaling.scale_x(EXPECT_X_DP))
    unscale_scale = scaling.scale_x(scaling.unscale_x(EXPECT_X_TILDE_DP))
    np.testing.assert_allclose(scaling.scale_x(EXPECT_X_DP), EXPECT_X_TILDE_DP)
    np.testing.assert_allclose(scaling.unscale_x(EXPECT_X_TILDE_DP),
                               EXPECT_X_DP)
    np.testing.assert_allclose(scale_unscale, EXPECT_X_DP)
    np.testing.assert_allclose(unscale_scale, EXPECT_X_TILDE_DP)
    assert scaling.scale_x(EXPECT_X_DP).all() >= -0.5
    assert scaling.scale_x(EXPECT_X_DP).all() <= 0.5


def test_initialise_scaling_br(brachistochrone_initialised_fixture):
    """Has correct attrs and methods."""
    ocp, iteration, scaling = brachistochrone_initialised_fixture
    scaling.__init__(iteration)

    assert hasattr(scaling, "V")
    assert isinstance(scaling.V, np.ndarray)
    assert scaling.V.shape == (iteration.num_x, )
    assert hasattr(scaling, "r")
    assert isinstance(scaling.r, np.ndarray)
    assert scaling.r.shape == (iteration.num_x, )
    assert hasattr(scaling, "V_inv")
    assert isinstance(scaling.V_inv, np.ndarray)
    assert scaling.V_inv.shape == (iteration.num_x, )

    np.testing.assert_allclose(scaling.V, EXPECT_V_BR)
    np.testing.assert_allclose(scaling.r, EXPECT_R_BR)
    np.testing.assert_allclose(scaling.V_inv, EXPECT_V_INV_BR)

    assert hasattr(scaling, "scale_x")
    assert callable(scaling.scale_x)
    assert hasattr(scaling, "unscale_x")
    assert callable(scaling.unscale_x)

    scale_unscale = scaling.unscale_x(scaling.scale_x(EXPECT_X_BR))
    unscale_scale = scaling.scale_x(scaling.unscale_x(EXPECT_X_TILDE_BR))
    np.testing.assert_allclose(scaling.scale_x(EXPECT_X_BR), EXPECT_X_TILDE_BR)
    np.testing.assert_allclose(scaling.unscale_x(EXPECT_X_TILDE_BR),
                               EXPECT_X_BR, rtol=10e-5)
    np.testing.assert_allclose(scale_unscale, EXPECT_X_BR)
    np.testing.assert_allclose(unscale_scale, EXPECT_X_TILDE_BR)
    assert scaling.scale_x(EXPECT_X_BR).all() >= -0.5
    assert scaling.scale_x(EXPECT_X_BR).all() <= 0.5
