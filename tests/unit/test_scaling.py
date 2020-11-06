"""Test scaling."""


import numpy as np


def test_none_scaling_init_correct_dp_specific(double_pendulum_fixture):
    """Check OCP scaling initialised correctly using none scaling method."""
    ocp, user_syms = double_pendulum_fixture
    ocp.settings.scaling_method = None
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()

    backend = ocp._backend
    scaling = backend.scaling

    np.testing.assert_array_equal(scaling.x_scales, np.ones(10))
    np.testing.assert_array_equal(scaling.x_shifts, np.zeros(10))
    np.testing.assert_array_equal(scaling.c_scales, np.ones(5))


def test_none_scaling_init_correct_br_specific(brachistochrone_fixture):
    """Check OCP scaling initialised correctly using none scaling method."""
    ocp, user_syms = brachistochrone_fixture
    ocp.settings.scaling_method = None
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()

    backend = ocp._backend
    scaling = backend.scaling

    np.testing.assert_array_equal(scaling.x_scales, np.ones(5))
    np.testing.assert_array_equal(scaling.x_shifts, np.zeros(5))
    np.testing.assert_array_equal(scaling.c_scales, np.ones(3))


def test_bounds_scaling_init_correct_dp_specific(double_pendulum_fixture):
    """Check OCP scaling initialised correctly using bounds scaling method."""
    ocp, user_syms = double_pendulum_fixture
    ocp.settings.scaling_method = "bounds"
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()

    backend = ocp._backend
    scaling = backend.scaling

    expect_x_scales = np.array([0006.283185307179586,
                                0006.283185307179586,
                                0020.000000000000000,
                                0020.000000000000000,
                                0030.000000000000000,
                                0030.000000000000000,
                                1000.000000000000000,
                                0002.000000000000000,
                                0001.000000000000000,
                                0001.000000000000000,
                                ])
    expect_x_shifts = np.array([000.00000000000000000,
                                000.00000000000000000,
                                000.00000000000000000,
                                000.00000000000000000,
                                000.00000000000000000,
                                000.00000000000000000,
                                500.00000000000000000,
                                002.00000000000000000,
                                001.00000000000000000,
                                001.00000000000000000,
                                ])
    expect_c_scales = np.array([0006.283185307179586,
                                0006.283185307179586,
                                0020.000000000000000,
                                0020.000000000000000,
                                1000.000000000000000,
                                ])

    np.testing.assert_allclose(scaling.x_scales, expect_x_scales)
    np.testing.assert_allclose(scaling.x_shifts, expect_x_shifts)
    np.testing.assert_allclose(scaling.c_scales, expect_c_scales)


def test_bounds_scaling_init_correct_br_specific(brachistochrone_fixture):
    """Check OCP scaling initialised correctly using bounds scaling method."""
    ocp, user_syms = brachistochrone_fixture
    ocp.settings.scaling_method = "bounds"
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()

    backend = ocp._backend
    scaling = backend.scaling

    expect_x_scales = np.array([010.000000000000000,
                                010.000000000000000,
                                100.000000000000000,
                                003.141592653589793,
                                010.000000000000000,
                                ])
    expect_x_shifts = np.array([5.00000000000000000,
                                5.00000000000000000,
                                0.00000000000000000,
                                0.00000000000000000,
                                5.00000000000000000,
                                ])
    expect_c_scales = np.array([010.000000000000000,
                                010.000000000000000,
                                100.000000000000000,
                                ])

    np.testing.assert_allclose(scaling.x_scales, expect_x_scales)
    np.testing.assert_allclose(scaling.x_shifts, expect_x_shifts)
    np.testing.assert_allclose(scaling.c_scales, expect_c_scales)
