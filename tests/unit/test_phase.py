"""Tests for the :class:`Phase` class."""

import pytest
import sympy as sm
import sympy.physics.mechanics as me

import pycollo


@pytest.fixture
def problem_with_phase_fixture():
    """Simple OCP object with phase fixture."""
    problem = pycollo.OptimalControlProblem(name="Fixture")
    problem.new_phase(name="A")
    return problem


def test_name(problem_with_phase_fixture):
    """Phase name should be set correctly and accessible from OCP."""
    problem = problem_with_phase_fixture
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


class TestPhaseStateVariables:
    """Tests for the `Phase.state_variables` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_v, self.sm_u = sm.symbols("x, y, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.state_variables = self.sm_x
        assert len(self.phase.state_variables) == 1
        assert hasattr(self.phase.state_variables, "x")
        assert self.phase.state_variables.x == self.sm_x

    def test_many_sympy_symbols(self):
        """Multiple `sm.Symbol`s can be set."""
        self.phase.state_variables = [self.sm_x, self.sm_y, self.sm_v]
        assert len(self.phase.state_variables) == 3
        assert hasattr(self.phase.state_variables, "x")
        assert hasattr(self.phase.state_variables, "y")
        assert hasattr(self.phase.state_variables, "v")
        assert self.phase.state_variables.x == self.sm_x
        assert self.phase.state_variables.y == self.sm_y
        assert self.phase.state_variables.v == self.sm_v

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.state_variables = self.me_x
        assert len(self.phase.state_variables) == 1
        assert hasattr(self.phase.state_variables, "x")
        assert self.phase.state_variables.x == self.me_x

    def test_many_sympy_dynamics_symbols(self):
        """Multiple `me.dynamicsymbols` can be set."""
        self.phase.state_variables = [self.me_x, self.me_y, self.me_v]
        assert len(self.phase.state_variables) == 3
        assert hasattr(self.phase.state_variables, "x")
        assert hasattr(self.phase.state_variables, "y")
        assert hasattr(self.phase.state_variables, "v")
        assert self.phase.state_variables.x == self.me_x
        assert self.phase.state_variables.y == self.me_y
        assert self.phase.state_variables.v == self.me_v

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.state_variables = [self.me_x, self.sm_y, self.me_v]
        assert len(self.phase.state_variables) == 3
        assert hasattr(self.phase.state_variables, "x")
        assert hasattr(self.phase.state_variables, "y")
        assert hasattr(self.phase.state_variables, "v")
        assert self.phase.state_variables.x == self.me_x
        assert self.phase.state_variables.y == self.sm_y
        assert self.phase.state_variables.v == self.me_v

    def test_none(self):
        """`None` causes the property attribute to be set as an empty `tuple`."""
        self.phase.state_variables = None
        assert self.phase.state_variables == ()

    @pytest.mark.parametrize(
        "state_variables",
        [
            True,
            [sm.Symbol("x"), True, False, None],
            [True, False, 1]
        ]
    )
    def test_invalid_value(self, state_variables):
        """A `ValueError` is raised for a invalid state variables."""
        with pytest.raises(ValueError):
            self.phase.state_variables = state_variables

    @pytest.mark.parametrize(
        "state_variables",
        [
            [sm.Symbol("x"), sm.Symbol("x")],
            [me.dynamicsymbols("x"), me.dynamicsymbols("x")],
            [sm.Symbol("x"), me.dynamicsymbols("x")],
        ]
    )
    def test_repeated(self, state_variables):
        """A `ValueError` is raised for repeated state variables."""
        with pytest.raises(ValueError):
            self.phase.state_variables = state_variables


class TestPhaseControlVariables:
    """Tests for the `Phase.control_variables` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_v, self.sm_u = sm.symbols("x, y, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.control_variables = self.sm_x
        assert len(self.phase.control_variables) == 1
        assert hasattr(self.phase.control_variables, "x")
        assert self.phase.control_variables.x == self.sm_x

    def test_many_sympy_symbols(self):
        """Multiple `sm.Symbol`s can be set."""
        self.phase.control_variables = [self.sm_x, self.sm_y, self.sm_v]
        assert len(self.phase.control_variables) == 3
        assert hasattr(self.phase.control_variables, "x")
        assert hasattr(self.phase.control_variables, "y")
        assert hasattr(self.phase.control_variables, "v")
        assert self.phase.control_variables.x == self.sm_x
        assert self.phase.control_variables.y == self.sm_y
        assert self.phase.control_variables.v == self.sm_v

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.control_variables = self.me_x
        assert len(self.phase.control_variables) == 1
        assert hasattr(self.phase.control_variables, "x")
        assert self.phase.control_variables.x == self.me_x

    def test_many_sympy_dynamics_symbols(self):
        """Multiple `me.dynamicsymbols` can be set."""
        self.phase.control_variables = [self.me_x, self.me_y, self.me_v]
        assert len(self.phase.control_variables) == 3
        assert hasattr(self.phase.control_variables, "x")
        assert hasattr(self.phase.control_variables, "y")
        assert hasattr(self.phase.control_variables, "v")
        assert self.phase.control_variables.x == self.me_x
        assert self.phase.control_variables.y == self.me_y
        assert self.phase.control_variables.v == self.me_v

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.control_variables = [self.me_x, self.sm_y, self.me_v]
        assert len(self.phase.control_variables) == 3
        assert hasattr(self.phase.control_variables, "x")
        assert hasattr(self.phase.control_variables, "y")
        assert hasattr(self.phase.control_variables, "v")
        assert self.phase.control_variables.x == self.me_x
        assert self.phase.control_variables.y == self.sm_y
        assert self.phase.control_variables.v == self.me_v

    def test_none(self):
        """`None` causes the property attribute to be set as an empty `tuple`."""
        self.phase.control_variables = None
        assert self.phase.control_variables == ()

    @pytest.mark.parametrize(
        "control_variables",
        [
            True,
            [sm.Symbol("x"), True, False, None],
            [True, False, 1]
        ]
    )
    def test_invalid_value(self, control_variables):
        """A `ValueError` is raised for a invalid control variables."""
        with pytest.raises(ValueError):
            self.phase.control_variables = control_variables

    @pytest.mark.parametrize(
        "control_variables",
        [
            [sm.Symbol("x"), sm.Symbol("x")],
            [me.dynamicsymbols("x"), me.dynamicsymbols("x")],
            [sm.Symbol("x"), me.dynamicsymbols("x")],
        ]
    )
    def test_repeated(self, control_variables):
        """A `ValueError` is raised for repeated control variables."""
        with pytest.raises(ValueError):
            self.phase.control_variables = control_variables
