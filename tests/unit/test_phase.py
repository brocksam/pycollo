"""Tests for the :class:`Phase` class."""

import pytest

import pycollo


@pytest.fixture
def problem_with_phase_fixture():
    """Simple OCP object with phase fixture."""
    problem = pycollo.OptimalControlProblem(name="Fixture")
    phase = problem.new_phase(name="A")
    return problem


def test_name(problem_with_phase_fixture):
    """Phase name should be set correctly and accessible from OCP."""
    problem = problem_with_phase_fixture
    phase = problem.phases.A
    assert hasattr(problem.phases, "A")
    assert problem.phases[0].name == "A"
    assert problem.phases.A.name == "A"


def test_var_attrs(problem_with_phase_fixture):
    """Protected attrs should exist after instantiation."""
    problem = problem_with_phase_fixture
    phase = problem.phases.A
    assert hasattr(phase, "_y_var_user")
    assert hasattr(phase, "_u_var_user")
    assert hasattr(phase, "_q_var_user")
    assert hasattr(phase, "_t_var_user")
    assert hasattr(phase, "_y_eqn_user")
    assert hasattr(phase, "_c_con_user")
    assert hasattr(phase, "_q_fnc_user")