"""Tests for the :class:`OptimalControlProblem` class."""

import pytest

import pycollo


@pytest.fixture
def ocp_fixture():
    """Simple OCP object fixture."""
    return pycollo.OptimalControlProblem(name="Fixture")


def test_name(ocp_fixture):
    """OCP name should be set correctly."""
    assert ocp_fixture.name == "Fixture"


def test_var_attrs(ocp_fixture):
    """Protected attrs should exist after instantiation."""
    assert hasattr(ocp_fixture, "_s_var_user")
    assert hasattr(ocp_fixture, "_b_con_user")
    assert hasattr(ocp_fixture, "_J_user")
    assert hasattr(ocp_fixture, "_aux_data_user")
