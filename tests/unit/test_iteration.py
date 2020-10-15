"""Test creation and initialisation of Iteration objects."""


import numpy as np
import pytest

import pycollo


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
    return ocp, iteration


def test_iterpolate_guess_to_new_mesh(double_pendulum_initialised_fixture):
    ocp, iteration = double_pendulum_initialised_fixture
    iteration.interpolate_guess_to_mesh(iteration.prev_guess)

    assert isinstance(iteration.guess_tau, list)
    assert len(iteration.guess_tau) == 1
    assert isinstance(iteration.guess_tau[0], np.ndarray)
    assert iteration.guess_tau[0].shape == (31, )

    assert isinstance(iteration.guess_t0, list)
    assert len(iteration.guess_t0) == 1
    assert isinstance(iteration.guess_t0[0], np.float64)
    assert iteration.guess_t0[0] == 0.0

    assert isinstance(iteration.guess_tF, list)
    assert len(iteration.guess_tF) == 1
    assert isinstance(iteration.guess_tF[0], np.float64)
    assert iteration.guess_tF[0] == 2.0

    assert isinstance(iteration.guess_stretch, list)
    assert len(iteration.guess_stretch) == 1
    assert isinstance(iteration.guess_stretch[0], np.float64)
    assert iteration.guess_stretch[0] == 1.0

    assert isinstance(iteration.guess_shift, list)
    assert len(iteration.guess_shift) == 1
    assert isinstance(iteration.guess_shift[0], np.float64)
    assert iteration.guess_shift[0] == 1.0

    assert isinstance(iteration.guess_time, list)
    assert len(iteration.guess_time) == 1
    assert isinstance(iteration.guess_time[0], np.ndarray)
    assert iteration.guess_time[0].shape == (31, )

    assert isinstance(iteration.guess_y, list)
    assert len(iteration.guess_y) == 1
    assert isinstance(iteration.guess_y[0], np.ndarray)
    assert iteration.guess_y[0].shape == (4, 31)

    assert isinstance(iteration.guess_u, list)
    assert len(iteration.guess_u) == 1
    assert isinstance(iteration.guess_u[0], np.ndarray)
    assert iteration.guess_u[0].shape == (2, 31)

    assert isinstance(iteration.guess_q, list)
    assert len(iteration.guess_q) == 1
    assert isinstance(iteration.guess_q[0], np.ndarray)
    assert iteration.guess_q[0].shape == (1, )

    assert isinstance(iteration.guess_q, list)
    assert len(iteration.guess_q) == 1
    assert isinstance(iteration.guess_q[0], np.ndarray)
    assert iteration.guess_q[0].shape == (1, )

    assert isinstance(iteration.guess_t, list)
    assert len(iteration.guess_t) == 1
    assert isinstance(iteration.guess_t[0], np.ndarray)
    assert iteration.guess_t[0].shape == (1, )

    assert isinstance(iteration.guess_s, np.ndarray)
    assert iteration.guess_s.shape == (2, )

    assert isinstance(iteration.guess_x, np.ndarray)
    assert iteration.guess_x.shape == (190, )


def test_create_var_con_counts_slices(double_pendulum_initialised_fixture):
    ocp, iteration = double_pendulum_initialised_fixture
    iteration.interpolate_guess_to_mesh(iteration.prev_guess)

    iteration.create_variable_counts_per_phase_and_total()
    assert iteration.num_y_per_phase == [124]
    assert iteration.num_u_per_phase == [62]
    assert iteration.num_q_per_phase == [1]
    assert iteration.num_t_per_phase == [1]
    assert iteration.num_x_per_phase == [188]
    assert iteration.num_y == 124
    assert iteration.num_u == 62
    assert iteration.num_q == 1
    assert iteration.num_t == 1
    assert iteration.num_s == 2
    assert iteration.num_x == 190

    iteration.create_variable_slices_per_phase()
    assert iteration.y_slices == [slice(0, 124)]
    assert iteration.u_slices == [slice(124, 186)]
    assert iteration.q_slices == [slice(186, 187)]
    assert iteration.t_slices == [slice(187, 188)]
    assert iteration.x_slices == [slice(0, 188)]
    assert iteration.s_slice == slice(188, 190)

    iteration.create_constraint_counts_per_phase_and_total()
    assert iteration.num_c_defect_per_phase == [120]
    assert iteration.num_c_path_per_phase == [0]
    assert iteration.num_c_integral_per_phase == [1]
    assert iteration.num_c_endpoint == 0
    assert iteration.num_c == 121

    iteration.create_constraint_slices_per_phase()
    assert iteration.c_defect_slices == [slice(0, 120)]
    assert iteration.c_path_slices == [slice(120, 120)]
    assert iteration.c_integral_slices == [slice(120, 121)]
    assert iteration.c_slices == [slice(0, 121)]
    assert iteration.c_endpoint_slice == slice(121, 121)

    iteration.create_constraint_component_function_slices_per_phase()
    assert iteration.y_eqn_slices == [slice(0, 124)]
    assert iteration.p_con_slices == [slice(124, 124)]
    assert iteration.q_fnc_slices == [slice(124, 155)]


def test_initialise_scaling(double_pendulum_initialised_fixture):
    """"""
    ocp, iteration = double_pendulum_initialised_fixture
    iteration.interpolate_guess_to_mesh(iteration.prev_guess)
    iteration.create_variable_constraint_counts_slices()
    iteration.initialise_scaling()