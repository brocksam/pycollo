"""Tests for Pycollo initialisation bounds processing and checking.

Attributes
----------
BR_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the Brachistochrone problem fixture.
    Note that BR is shorthand for 'Brachistochrone' here.
DP_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the double pendulum swing-up fixture.
    Note that DP is shorthand for 'double pendulum' here.
HS_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the Hypersensitive problem fixture.
    Note that HS is shorthand for 'Hypersensitive problem' here.
ALL_FIXTURE_REFS : list
    A list of all problem-specific fixtures in this module so that they can be
    used to easily parametrize the test problems.

"""

import re
from collections import namedtuple

import numpy as np
import pytest
import pytest_cases
import sympy as sym

import pycollo


HS_FIXTURE_REF = pytest_cases.fixture_ref("hypersensitive_problem_fixture")
DP_FIXTURE_REF = pytest_cases.fixture_ref("double_pendulum_fixture")
BR_FIXTURE_REF = pytest_cases.fixture_ref("brachistochrone_fixture")
ALL_FIXTURE_REFS = [HS_FIXTURE_REF, DP_FIXTURE_REF, BR_FIXTURE_REF]

FixtureData = namedtuple("FixtureData", ["ocp", "syms"])


@pytest.fixture
def double_pendulum_init_fixture(double_pendulum_fixture):
    """Double pendulum fixture with backend initialised."""
    ocp, user_syms = double_pendulum_fixture
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    fixture_data = FixtureData(ocp, user_syms)
    return fixture_data


def test_BoundsABC():
    """Test the bounds abstract base class.

    See: https://clamytoe.github.io/articles/2020/Mar/12/
        testing-abcs-with-abstract-methods-with-pytest/

    """
    pycollo.bounds.BoundsABC.__abstractmethods__ = set()
    dummy = pycollo.bounds.BoundsABC()
    _ = dummy.optimal_control_problem()
    _ = dummy._process_and_check_user_values()
    _ = dummy._required_variable_bounds()


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REFS)
def test_init_to_bounds_check(ocp_fixture):
    """Check OCP initialisation is sucessful to just before bounds checking."""
    ocp, user_syms = ocp_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REFS)
def test_bounds_check_sucessful(ocp_fixture):
    """Check OCP initialisation completes bounds checking successfully."""
    ocp, user_syms = ocp_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()


def test_bounds_check_correct_dp_specific(double_pendulum_fixture):
    """Check OCP initialisation completes bounds checking successfully."""
    ocp, user_syms = double_pendulum_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()

    backend = ocp._backend
    phase_backend = backend.p[0]

    assert phase_backend.num_y_eqn == 4
    assert phase_backend.num_p_con == 0
    assert phase_backend.num_q_fnc == 1
    assert backend.num_y_con == 8
    assert backend.num_b_con == 0


def test_bounds_check_correct_br_specific(brachistochrone_fixture):
    """Check OCP initialisation completes bounds checking successfully."""
    ocp, user_syms = brachistochrone_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()

    backend = ocp._backend
    phase_backend = backend.p[0]

    assert phase_backend.num_y_eqn == 3
    assert phase_backend.num_p_con == 0
    assert phase_backend.num_q_fnc == 0
    assert backend.num_y_con == 5
    assert backend.num_b_con == 0


"""
Algorithm:
1. User creates OCP - OCP has `EndpointBounds`, each phase has `PhaseBounds`
2. Backend creates a `Bounds` object
3. User bounds are checked and processed
4. `Bounds` collects all bounds
5. Unrequired bounds are added to the OCP auxiliary data

To do:
* Factor settings to `pycollo/settings.py`
* Test state/control bounds as `dict`
* Test bounds as iterable
* Test cast single value bounds to pair
* Test `None` bounds/assume inf bounds
* Test remove constant variables
* Test override endpoints
* Test bounds clashing

"""


DEFAULT_DP_BOUNDS_T0 = 0
DEFAULT_DP_BOUNDS_TF = [1, 3]
DEFAULT_DP_BOUNDS_Y = [[-np.pi, np.pi],
                       [-np.pi, np.pi],
                       [-10, 10],
                       [-10, 10]]
DEFAULT_DP_BOUNDS_U = [[-15, 15], [-15, 15]]
DEFAULT_DP_BOUNDS_Q = [0, 1000]
DEFAULT_DP_BOUNDS_Y0 = [[-0.5 * np.pi, -0.5 * np.pi],
                        [-0.5 * np.pi, -0.5 * np.pi],
                        [0, 0],
                        [0, 0]]
DEFAULT_DP_BOUNDS_YF = [[0.5 * np.pi, 0.5 * np.pi],
                        [0.5 * np.pi, 0.5 * np.pi],
                        [0, 0],
                        [0, 0]]
DEFAULT_DP_BOUNDS_S = [[0.5, 1.5], [0.5, 1.5]]


def test_initial_user_bounds(double_pendulum_init_fixture):
    """Assert initial values for user bounds are as expected."""
    ocp, user_syms = double_pendulum_init_fixture
    phase = ocp.phases.A
    assert ocp.bounds.parameter_variables == DEFAULT_DP_BOUNDS_S
    assert phase.bounds.initial_time == DEFAULT_DP_BOUNDS_T0
    assert phase.bounds.final_time == DEFAULT_DP_BOUNDS_TF
    assert phase.bounds.state_variables == DEFAULT_DP_BOUNDS_Y
    assert phase.bounds.control_variables == DEFAULT_DP_BOUNDS_U
    assert phase.bounds.integral_variables == DEFAULT_DP_BOUNDS_Q
    assert phase.bounds.initial_state_constraints == DEFAULT_DP_BOUNDS_Y0
    assert phase.bounds.final_state_constraints == DEFAULT_DP_BOUNDS_YF


def test_all_processed_user_bounds(double_pendulum_init_fixture):
    """Assert all processed user bounds are as expected."""
    ocp, user_syms = double_pendulum_init_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()

    np.testing.assert_allclose(ocp.bounds._s_bnd,
                               np.array(DEFAULT_DP_BOUNDS_S))
    np.testing.assert_equal(ocp.bounds._s_needed, np.array([True, True]))


# valid_bounds = [DEFAULT_DP_BOUNDS_Q,
#                 np.array(DEFAULT_DP_BOUNDS_Q),
#                 tuple(DEFAULT_DP_BOUNDS_Q),
#                 {sym.Symbol("_q0_P0"): DEFAULT_DP_BOUNDS_Q},
#                 {sym.Symbol("_q0_P0"): np.array(DEFAULT_DP_BOUNDS_Q)},
#                 {sym.Symbol("_q0_P0"): tuple(DEFAULT_DP_BOUNDS_Q)},
#                 ]


# @pytest.mark.parametrize("bounds", valid_bounds)
# def test_valid_forms_of_one_variable(double_pendulum_init_fixture, bounds):
#     """Single variables """


m0 = sym.Symbol("m0")
p0 = sym.Symbol("p0")
valid_bounds = [DEFAULT_DP_BOUNDS_S,
                np.array(DEFAULT_DP_BOUNDS_S),
                tuple(DEFAULT_DP_BOUNDS_S),
                {m0: DEFAULT_DP_BOUNDS_S[0],
                 p0: DEFAULT_DP_BOUNDS_S[1]},
                {m0: np.array(DEFAULT_DP_BOUNDS_S)[0, :],
                 p0: np.array(DEFAULT_DP_BOUNDS_S)[1, :]},
                {m0: tuple(DEFAULT_DP_BOUNDS_S)[0],
                 p0: tuple(DEFAULT_DP_BOUNDS_S)[1]},
                ]


@pytest.mark.parametrize("bounds", valid_bounds)
def test_valid_forms_of_two_variable(double_pendulum_init_fixture, bounds):
    """Variable types with length two have correct bounds correctly parsed."""
    ocp, user_syms = double_pendulum_init_fixture
    ocp.bounds.parameter_variables = bounds
    ocp.bounds._backend = ocp._backend
    ocp.bounds._INF = ocp.settings.numerical_inf
    ocp.bounds._process_parameter_vars()
    np.testing.assert_allclose(ocp.bounds._s_bnd,
                               np.array(DEFAULT_DP_BOUNDS_S))
    np.testing.assert_array_equal(ocp.bounds._s_needed,
                                  np.array([True, True]))


@pytest.mark.parametrize("bounds", [{m0: DEFAULT_DP_BOUNDS_S[0]}])
def test_var_bound_mapping_var_missing(double_pendulum_init_fixture, bounds):
    """If bounds passeed as dict, raise error or replace None with Inf."""
    ocp, user_syms = double_pendulum_init_fixture
    ocp.bounds.parameter_variables = bounds
    ocp.bounds._backend = ocp._backend
    ocp.bounds._INF = ocp.settings.numerical_inf
    ocp.bounds._process_parameter_vars()
    np.testing.assert_allclose(ocp.bounds._s_bnd,
                               np.array([[0.5, 1.5], [-10e19, 10e19]]))
    np.testing.assert_array_equal(ocp.bounds._s_needed,
                                  np.array([True, True]))

    ocp.settings.assume_inf_bounds = False
    expected_error_msg = re.escape("No bounds have been supplied for the "
                                   "parameter variable 'p0' (index #1).")
    with pytest.raises(ValueError, match=expected_error_msg):
        ocp.bounds._process_parameter_vars()


@pytest.mark.parametrize("bounds", [{m0: [0.0, -1e-7]}])
def test_var_bounds_almost_equal(double_pendulum_init_fixture, bounds):
    """Almost equal bounds handled correctly."""
    ocp, user_syms = double_pendulum_init_fixture
    ocp.bounds.parameter_variables = bounds
    ocp.bounds._backend = ocp._backend
    ocp.bounds._INF = ocp.settings.numerical_inf
    ocp.bounds._process_parameter_vars()
    np.testing.assert_allclose(ocp.bounds._s_bnd,
                               np.array([[-5e-8, -5e-8], [-10e19, 10e19]]))
    np.testing.assert_array_equal(ocp.bounds._s_needed,
                                  np.array([False, True]))


@pytest.mark.parametrize("bounds", [{m0: [0.0, -1e-3]}])
def test_var_bounds_not_almost_equal(double_pendulum_init_fixture, bounds):
    """Almost equal bounds handled correctly."""
    ocp, user_syms = double_pendulum_init_fixture
    ocp.bounds.parameter_variables = bounds
    ocp.bounds._backend = ocp._backend
    ocp.bounds._INF = ocp.settings.numerical_inf
    expected_error_msg = re.escape("The user-supplied upper bound for the "
                                   "parameter variable 'm0' (index #0) of "
                                   "'-0.001' cannot be less than the "
                                   "user-supplied lower bound of '0.0'.")
    with pytest.raises(ValueError, match=expected_error_msg):
        ocp.bounds._process_parameter_vars()


@pytest.mark.parametrize("t0_bounds, tF_bounds", [([1.0, 5.0], [0.0, 10.0])])
def test_t0_lower_less_than_tF_lower(double_pendulum_init_fixture,
                                     t0_bounds,
                                     tF_bounds):
    """Initial time lower bound must be less than final time lower bounds."""
    ocp, user_syms = double_pendulum_init_fixture
    phase = ocp.phases.A
    phase.bounds.initial_time = t0_bounds
    phase.bounds.final_time = tF_bounds
    expected_error_msg = re.escape("The lower bound for the final time "
                                   "('0.0') must be greater than the lower "
                                   "bound for the initial time ('1.0') in "
                                   "phase A (index #0).")
    with pytest.raises(ValueError, match=expected_error_msg):
        ocp._check_problem_and_phase_bounds()


@pytest.mark.parametrize("t0_bounds, tF_bounds", [([0.0, 10.0], [1.0, 5.0])])
def test_t0_upper_less_than_tF_upper(double_pendulum_init_fixture,
                                     t0_bounds,
                                     tF_bounds):
    """Initial time upper bound must be less than final time upper bounds."""
    ocp, user_syms = double_pendulum_init_fixture
    phase = ocp.phases.A
    phase.bounds.initial_time = t0_bounds
    phase.bounds.final_time = tF_bounds
    expected_error_msg = re.escape("The upper bound for the final time "
                                   "('5.0') must be greater than the upper "
                                   "bound for the initial time ('10.0') in "
                                   "phase A (index #0).")
    with pytest.raises(ValueError, match=expected_error_msg):
        ocp._check_problem_and_phase_bounds()





























