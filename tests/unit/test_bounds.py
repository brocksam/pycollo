"""Tests for Pycollo initialisation bounds processing and checking.

Attributes
----------
DP_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the double pendulum swing-up fixture.
    Note that DP is shorthand for 'double pendulum' here.
HS_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the Hypersensitive problem fixture.
    Note that HS is shorthand for 'Hypersensitive problem' here.

"""

import itertools

import casadi as ca
import pytest
import pytest_cases
import sympy as sym

import pycollo


HS_FIXTURE_REF = pytest_cases.fixture_ref("hypersensitive_problem_fixture")
DP_FIXTURE_REF = pytest_cases.fixture_ref("double_pendulum_fixture")


@pytest_cases.parametrize("ocp_fixture", [HS_FIXTURE_REF, DP_FIXTURE_REF])
def test_init_to_bounds_check(ocp_fixture):
    """Check OCP initialisation is sucessful to just before bounds checking."""
    ocp, user_syms = ocp_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()


@pytest_cases.parametrize("ocp_fixture", [HS_FIXTURE_REF, DP_FIXTURE_REF])
def test_bounds_check_sucessful(ocp_fixture):
    """Check OCP initialisation completes bounds checking successfully."""
    ocp, user_syms = ocp_fixture
    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()