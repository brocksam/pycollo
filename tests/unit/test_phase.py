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
        self.me_dx, self.me_dy, self.me_dv, self.me_du = me.dynamicsymbols("x, y, v, u", 1)
        self.me_ddx, self.me_ddy, self.me_ddv, self.me_ddu = me.dynamicsymbols("x, y, v, u", 2)

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.state_variables = self.sm_x
        assert len(self.phase.state_variables) == 1
        assert hasattr(self.phase.state_variables, "x")
        assert self.phase.state_variables.x == self.sm_x
        assert self.phase.state_variables[0] == self.sm_x
        assert self.phase.state_variables[self.sm_x] == self.sm_x

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
        assert self.phase.state_variables[0] == self.sm_x
        assert self.phase.state_variables[1] == self.sm_y
        assert self.phase.state_variables[2] == self.sm_v
        assert self.phase.state_variables[self.sm_x] == self.sm_x
        assert self.phase.state_variables[self.sm_y] == self.sm_y
        assert self.phase.state_variables[self.sm_v] == self.sm_v

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.state_variables = self.me_x
        assert len(self.phase.state_variables) == 1
        assert hasattr(self.phase.state_variables, "x")
        assert self.phase.state_variables.x == self.me_x
        assert self.phase.state_variables[0] == self.me_x
        assert self.phase.state_variables[self.me_x] == self.me_x

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
        assert self.phase.state_variables[0] == self.me_x
        assert self.phase.state_variables[1] == self.me_y
        assert self.phase.state_variables[2] == self.me_v
        assert self.phase.state_variables[self.me_x] == self.me_x
        assert self.phase.state_variables[self.me_y] == self.me_y
        assert self.phase.state_variables[self.me_v] == self.me_v

    def test_single_sympy_dynamics_symbol_derivative(self):
        """A single `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = self.me_dx
        assert len(self.phase.state_variables) == 1
        assert hasattr(self.phase.state_variables, "dx")
        assert self.phase.state_variables.dx == self.me_dx
        assert self.phase.state_variables[0] == self.me_dx
        assert self.phase.state_variables[self.me_dx] == self.me_dx

    def test_many_sympy_dynamics_symbol_derivative(self):
        """Multiple `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = [self.me_dx, self.me_dy, self.me_dv]
        assert len(self.phase.state_variables) == 3
        assert hasattr(self.phase.state_variables, "dx")
        assert hasattr(self.phase.state_variables, "dy")
        assert hasattr(self.phase.state_variables, "dv")
        assert self.phase.state_variables.dx == self.me_dx
        assert self.phase.state_variables.dy == self.me_dy
        assert self.phase.state_variables.dv == self.me_dv
        assert self.phase.state_variables[0] == self.me_dx
        assert self.phase.state_variables[1] == self.me_dy
        assert self.phase.state_variables[2] == self.me_dv
        assert self.phase.state_variables[self.me_dx] == self.me_dx
        assert self.phase.state_variables[self.me_dy] == self.me_dy
        assert self.phase.state_variables[self.me_dv] == self.me_dv

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.state_variables = [self.me_dx, self.sm_y, self.me_v]
        assert len(self.phase.state_variables) == 3
        assert hasattr(self.phase.state_variables, "dx")
        assert hasattr(self.phase.state_variables, "y")
        assert hasattr(self.phase.state_variables, "v")
        assert self.phase.state_variables.dx == self.me_dx
        assert self.phase.state_variables.y == self.sm_y
        assert self.phase.state_variables.v == self.me_v
        assert self.phase.state_variables[0] == self.me_dx
        assert self.phase.state_variables[1] == self.sm_y
        assert self.phase.state_variables[2] == self.me_v
        assert self.phase.state_variables[self.me_dx] == self.me_dx
        assert self.phase.state_variables[self.sm_y] == self.sm_y
        assert self.phase.state_variables[self.me_v] == self.me_v

    def test_none(self):
        """`None` causes the property attribute to be set as empty."""
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


class TestPhaseInitialStateVariables:
    """Tests for the `Phase.initial_state_variables` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_v, self.sm_u = sm.symbols("x, y, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")
        self.x_t0, self.y_t0, self.v_t0, self.u_t0 = sm.symbols("x_P0(t0), y_P0(t0), v_P0(t0), u_P0(t0)")
        self.me_dx, self.me_dy, self.me_dv, self.me_du = me.dynamicsymbols("x, y, v, u", 1)
        self.dx_t0, self.dy_t0, self.dv_t0, self.du_t0 = sm.symbols('dx_P0(t0), dy_P0(t0), dv_P0(t0), du_P0(t0)')

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.state_variables = self.sm_x
        assert len(self.phase.initial_state_variables) == 1
        assert hasattr(self.phase.initial_state_variables, "x")
        assert self.phase.initial_state_variables.x == self.x_t0
        assert self.phase.initial_state_variables[0] == self.x_t0
        assert self.phase.initial_state_variables[self.sm_x] == self.x_t0

    def test_many_sympy_symbols(self):
        """Multiple `sm.Symbol`s can be set."""
        self.phase.state_variables = [self.sm_x, self.sm_y, self.sm_v]
        assert len(self.phase.initial_state_variables) == 3
        assert hasattr(self.phase.initial_state_variables, "x")
        assert hasattr(self.phase.initial_state_variables, "y")
        assert hasattr(self.phase.initial_state_variables, "v")
        assert self.phase.initial_state_variables.x == self.x_t0
        assert self.phase.initial_state_variables.y == self.y_t0
        assert self.phase.initial_state_variables.v == self.v_t0
        assert self.phase.initial_state_variables[0] == self.x_t0
        assert self.phase.initial_state_variables[1] == self.y_t0
        assert self.phase.initial_state_variables[2] == self.v_t0
        assert self.phase.initial_state_variables[self.sm_x] == self.x_t0
        assert self.phase.initial_state_variables[self.sm_y] == self.y_t0
        assert self.phase.initial_state_variables[self.sm_v] == self.v_t0

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.state_variables = self.me_x
        assert len(self.phase.initial_state_variables) == 1
        assert hasattr(self.phase.initial_state_variables, "x")
        assert self.phase.initial_state_variables.x == self.x_t0
        assert self.phase.initial_state_variables[0] == self.x_t0
        assert self.phase.initial_state_variables[self.me_x] == self.x_t0

    def test_many_sympy_dynamics_symbols(self):
        """Multiple `me.dynamicsymbols` can be set."""
        self.phase.state_variables = [self.me_x, self.me_y, self.me_v]
        assert len(self.phase.initial_state_variables) == 3
        assert hasattr(self.phase.initial_state_variables, "x")
        assert hasattr(self.phase.initial_state_variables, "y")
        assert hasattr(self.phase.initial_state_variables, "v")
        assert self.phase.initial_state_variables.x == self.x_t0
        assert self.phase.initial_state_variables.y == self.y_t0
        assert self.phase.initial_state_variables.v == self.v_t0
        assert self.phase.initial_state_variables[0] == self.x_t0
        assert self.phase.initial_state_variables[1] == self.y_t0
        assert self.phase.initial_state_variables[2] == self.v_t0
        assert self.phase.initial_state_variables[self.me_x] == self.x_t0
        assert self.phase.initial_state_variables[self.me_y] == self.y_t0
        assert self.phase.initial_state_variables[self.me_v] == self.v_t0

    def test_single_sympy_dynamics_symbol_derivative(self):
        """A single `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = self.me_dx
        assert len(self.phase.initial_state_variables) == 1
        assert hasattr(self.phase.initial_state_variables, "dx")
        assert self.phase.initial_state_variables.dx == self.dx_t0
        assert self.phase.initial_state_variables[0] == self.dx_t0
        assert self.phase.initial_state_variables[self.me_dx] == self.dx_t0

    def test_many_sympy_dynamics_symbol_derivative(self):
        """Multiple `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = [self.me_dx, self.me_dy, self.me_dv]
        assert len(self.phase.initial_state_variables) == 3
        assert hasattr(self.phase.initial_state_variables, "dx")
        assert hasattr(self.phase.initial_state_variables, "dy")
        assert hasattr(self.phase.initial_state_variables, "dv")
        assert self.phase.initial_state_variables.dx == self.dx_t0
        assert self.phase.initial_state_variables.dy == self.dy_t0
        assert self.phase.initial_state_variables.dv == self.dv_t0
        assert self.phase.initial_state_variables[0] == self.dx_t0
        assert self.phase.initial_state_variables[1] == self.dy_t0
        assert self.phase.initial_state_variables[2] == self.dv_t0
        assert self.phase.initial_state_variables[self.me_dx] == self.dx_t0
        assert self.phase.initial_state_variables[self.me_dy] == self.dy_t0
        assert self.phase.initial_state_variables[self.me_dv] == self.dv_t0

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.state_variables = [self.me_dx, self.sm_y, self.me_v]
        assert len(self.phase.initial_state_variables) == 3
        assert hasattr(self.phase.initial_state_variables, "dx")
        assert hasattr(self.phase.initial_state_variables, "y")
        assert hasattr(self.phase.initial_state_variables, "v")
        assert self.phase.initial_state_variables.dx == self.dx_t0
        assert self.phase.initial_state_variables.y == self.y_t0
        assert self.phase.initial_state_variables.v == self.v_t0
        assert self.phase.initial_state_variables[0] == self.dx_t0
        assert self.phase.initial_state_variables[1] == self.y_t0
        assert self.phase.initial_state_variables[2] == self.v_t0
        assert self.phase.initial_state_variables[self.me_dx] == self.dx_t0
        assert self.phase.initial_state_variables[self.sm_y] == self.y_t0
        assert self.phase.initial_state_variables[self.me_v] == self.v_t0

    def test_none(self):
        """`None` causes the property attribute to be set as empty."""
        self.phase.state_variables = None
        assert self.phase.initial_state_variables == ()


class TestPhaseFinalStateVariables:
    """Tests for the `Phase.final_state_variables` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_v, self.sm_u = sm.symbols("x, y, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")
        self.x_tF, self.y_tF, self.v_tF, self.u_tF = sm.symbols("x_P0(tF), y_P0(tF), v_P0(tF), u_P0(tF)")
        self.me_dx, self.me_dy, self.me_dv, self.me_du = me.dynamicsymbols("x, y, v, u", 1)
        self.dx_tF, self.dy_tF, self.dv_tF, self.du_tF = sm.symbols('dx_P0(tF), dy_P0(tF), dv_P0(tF), du_P0(tF)')

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.state_variables = self.sm_x
        assert len(self.phase.final_state_variables) == 1
        assert hasattr(self.phase.final_state_variables, "x")
        assert self.phase.final_state_variables.x == self.x_tF
        assert self.phase.final_state_variables[0] == self.x_tF
        assert self.phase.final_state_variables[self.sm_x] == self.x_tF

    def test_many_sympy_symbols(self):
        """Multiple `sm.Symbol`s can be set."""
        self.phase.state_variables = [self.sm_x, self.sm_y, self.sm_v]
        assert len(self.phase.final_state_variables) == 3
        assert hasattr(self.phase.final_state_variables, "x")
        assert hasattr(self.phase.final_state_variables, "y")
        assert hasattr(self.phase.final_state_variables, "v")
        assert self.phase.final_state_variables.x == self.x_tF
        assert self.phase.final_state_variables.y == self.y_tF
        assert self.phase.final_state_variables.v == self.v_tF
        assert self.phase.final_state_variables[0] == self.x_tF
        assert self.phase.final_state_variables[1] == self.y_tF
        assert self.phase.final_state_variables[2] == self.v_tF
        assert self.phase.final_state_variables[self.sm_x] == self.x_tF
        assert self.phase.final_state_variables[self.sm_y] == self.y_tF
        assert self.phase.final_state_variables[self.sm_v] == self.v_tF

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.state_variables = self.me_x
        assert len(self.phase.final_state_variables) == 1
        assert hasattr(self.phase.final_state_variables, "x")
        assert self.phase.final_state_variables.x == self.x_tF
        assert self.phase.final_state_variables[0] == self.x_tF
        assert self.phase.final_state_variables[self.me_x] == self.x_tF

    def test_many_sympy_dynamics_symbols(self):
        """Multiple `me.dynamicsymbols` can be set."""
        self.phase.state_variables = [self.me_x, self.me_y, self.me_v]
        assert len(self.phase.final_state_variables) == 3
        assert hasattr(self.phase.final_state_variables, "x")
        assert hasattr(self.phase.final_state_variables, "y")
        assert hasattr(self.phase.final_state_variables, "v")
        assert self.phase.final_state_variables.x == self.x_tF
        assert self.phase.final_state_variables.y == self.y_tF
        assert self.phase.final_state_variables.v == self.v_tF
        assert self.phase.final_state_variables[0] == self.x_tF
        assert self.phase.final_state_variables[1] == self.y_tF
        assert self.phase.final_state_variables[2] == self.v_tF
        assert self.phase.final_state_variables[self.me_x] == self.x_tF
        assert self.phase.final_state_variables[self.me_y] == self.y_tF
        assert self.phase.final_state_variables[self.me_v] == self.v_tF

    def test_single_sympy_dynamics_symbol_derivative(self):
        """A single `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = self.me_dx
        assert len(self.phase.final_state_variables) == 1
        assert hasattr(self.phase.final_state_variables, "dx")
        assert self.phase.final_state_variables.dx == self.dx_tF
        assert self.phase.final_state_variables[0] == self.dx_tF
        assert self.phase.final_state_variables[self.me_dx] == self.dx_tF

    def test_many_sympy_dynamics_symbol_derivative(self):
        """Multiple `me.dynamicsymbols` derivative can be set."""
        self.phase.state_variables = [self.me_dx, self.me_dy, self.me_dv]
        assert len(self.phase.final_state_variables) == 3
        assert hasattr(self.phase.final_state_variables, "dx")
        assert hasattr(self.phase.final_state_variables, "dy")
        assert hasattr(self.phase.final_state_variables, "dv")
        assert self.phase.final_state_variables.dx == self.dx_tF
        assert self.phase.final_state_variables.dy == self.dy_tF
        assert self.phase.final_state_variables.dv == self.dv_tF
        assert self.phase.final_state_variables[0] == self.dx_tF
        assert self.phase.final_state_variables[1] == self.dy_tF
        assert self.phase.final_state_variables[2] == self.dv_tF
        assert self.phase.final_state_variables[self.me_dx] == self.dx_tF
        assert self.phase.final_state_variables[self.me_dy] == self.dy_tF
        assert self.phase.final_state_variables[self.me_dv] == self.dv_tF

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.state_variables = [self.me_dx, self.sm_y, self.me_v]
        assert len(self.phase.final_state_variables) == 3
        assert hasattr(self.phase.final_state_variables, "dx")
        assert hasattr(self.phase.final_state_variables, "y")
        assert hasattr(self.phase.final_state_variables, "v")
        assert self.phase.final_state_variables.dx == self.dx_tF
        assert self.phase.final_state_variables.y == self.y_tF
        assert self.phase.final_state_variables.v == self.v_tF
        assert self.phase.final_state_variables[0] == self.dx_tF
        assert self.phase.final_state_variables[1] == self.y_tF
        assert self.phase.final_state_variables[2] == self.v_tF
        assert self.phase.final_state_variables[self.me_dx] == self.dx_tF
        assert self.phase.final_state_variables[self.sm_y] == self.y_tF
        assert self.phase.final_state_variables[self.me_v] == self.v_tF

    def test_none(self):
        """`None` causes the property attribute to be set as empty."""
        self.phase.state_variables = None
        assert self.phase.final_state_variables == ()


class TestPhaseControlVariables:
    """Tests for the `Phase.control_variables` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_v, self.sm_u = sm.symbols("x, y, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")
        self.me_dx, self.me_dy, self.me_dv, self.me_du = me.dynamicsymbols("x, y, v, u", 1)
        self.me_ddx, self.me_ddy, self.me_ddv, self.me_ddu = me.dynamicsymbols("x, y, v, u", 2)

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.control_variables = self.sm_x
        assert len(self.phase.control_variables) == 1
        assert hasattr(self.phase.control_variables, "x")
        assert self.phase.control_variables.x == self.sm_x
        assert self.phase.control_variables[0] == self.sm_x
        assert self.phase.control_variables[self.sm_x] == self.sm_x

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
        assert self.phase.control_variables[0] == self.sm_x
        assert self.phase.control_variables[1] == self.sm_y
        assert self.phase.control_variables[2] == self.sm_v
        assert self.phase.control_variables[self.sm_x] == self.sm_x
        assert self.phase.control_variables[self.sm_y] == self.sm_y
        assert self.phase.control_variables[self.sm_v] == self.sm_v

    def test_single_sympy_dynamics_symbol(self):
        """A single `me.dynamicsymbols` can be set."""
        self.phase.control_variables = self.me_x
        assert len(self.phase.control_variables) == 1
        assert hasattr(self.phase.control_variables, "x")
        assert self.phase.control_variables.x == self.me_x
        assert self.phase.control_variables[0] == self.me_x
        assert self.phase.control_variables[self.me_x] == self.me_x

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
        assert self.phase.control_variables[0] == self.me_x
        assert self.phase.control_variables[1] == self.me_y
        assert self.phase.control_variables[2] == self.me_v
        assert self.phase.control_variables[self.me_x] == self.me_x
        assert self.phase.control_variables[self.me_y] == self.me_y
        assert self.phase.control_variables[self.me_v] == self.me_v

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.control_variables = [self.me_dx, self.sm_y, self.me_v]
        assert len(self.phase.control_variables) == 3
        assert hasattr(self.phase.control_variables, "dx")
        assert hasattr(self.phase.control_variables, "y")
        assert hasattr(self.phase.control_variables, "v")
        assert self.phase.control_variables.dx == self.me_dx
        assert self.phase.control_variables.y == self.sm_y
        assert self.phase.control_variables.v == self.me_v
        assert self.phase.control_variables[0] == self.me_dx
        assert self.phase.control_variables[1] == self.sm_y
        assert self.phase.control_variables[2] == self.me_v
        assert self.phase.control_variables[self.me_dx] == self.me_dx
        assert self.phase.control_variables[self.sm_y] == self.sm_y
        assert self.phase.control_variables[self.me_v] == self.me_v

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


class TestStateEquations:
    """Tests for the `Phase.state_equations` property attribute."""

    @pytest.fixture(autouse=True)
    def _ocp_with_phase_fixture(self) -> None:
        """Simple fixture setting up an :obj:`OptimalControlProblem`."""
        self.problem = pycollo.OptimalControlProblem(name="Fixture")
        self.phase = self.problem.new_phase(name="A")
        self.sm_x, self.sm_y, self.sm_z, self.sm_v, self.sm_u = sm.symbols("x, y, z, v, u")
        self.me_x, self.me_y, self.me_v, self.me_u = me.dynamicsymbols("x, y, v, u")
        self.me_dx, self.me_dy, self.me_dv, self.me_du = me.dynamicsymbols("x, y, v, u", 1)
        self.me_ddx, self.me_ddy, self.me_ddv, self.me_ddu = me.dynamicsymbols("x, y, v, u", 2)

    def test_single_sympy_symbol(self):
        """A single `sm.Symbol` can be set."""
        self.phase.state_equations = {self.sm_x: self.sm_u}
        assert len(self.phase.state_equations) == 1
        assert hasattr(self.phase.state_equations, "x")
        assert self.phase.state_equations.x == self.sm_u
        assert self.phase.state_equations[0] == self.sm_u
        assert self.phase.state_equations[self.sm_x] == self.sm_u

    def test_many_mixed_sympy_symbols_and_sympy_dynamics_symbols_mapping(self):
        """`sm.Symbol`s and `me.dynamicsymbols` can be mixed."""
        self.phase.state_equations = {
            self.sm_y: self.sm_v - self.sm_z,
            self.me_x: self.me_dx,
            self.me_dx: self.me_ddx,
            self.me_ddx: self.sm_v - self.me_u,
        }
        assert len(self.phase.state_equations) == 4
        assert hasattr(self.phase.state_equations, "y")
        assert hasattr(self.phase.state_equations, "x")
        assert hasattr(self.phase.state_equations, "dx")
        assert hasattr(self.phase.state_equations, "ddx")
        assert self.phase.state_equations.y == self.sm_v - self.sm_z
        assert self.phase.state_equations.x == self.me_dx
        assert self.phase.state_equations.dx == self.me_ddx
        assert self.phase.state_equations.ddx == self.sm_v - self.me_u
        assert self.phase.state_equations[0] == self.sm_v - self.sm_z
        assert self.phase.state_equations[1] == self.me_dx
        assert self.phase.state_equations[2] == self.me_ddx
        assert self.phase.state_equations[3] == self.sm_v - self.me_u
        assert self.phase.state_equations[self.sm_y] == self.sm_v - self.sm_z
        assert self.phase.state_equations[self.me_x] == self.me_dx
        assert self.phase.state_equations[self.me_dx] == self.me_ddx
        assert self.phase.state_equations[self.me_ddx] == self.sm_v - self.me_u
