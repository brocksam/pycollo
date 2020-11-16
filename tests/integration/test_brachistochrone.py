"""Integration test based on the brachistochrone OCP.

See the example `examples/brachistochrone.py` for a description of and
reference for this optimal control problem.

"""


import numpy as np
import pytest
import sympy as sym

import pycollo


@pytest.mark.incremental
@pytest.mark.usefixtures("state")
class TestHypersensitiveProblem:
    """Test the Hypersensitive problem."""

    @pytest.fixture(autouse=True)
    def ocp_fixture(self):
        """Instantiate the required symbols."""
        self.x = sym.Symbol("x")
        self.y = sym.Symbol("y")
        self.v = sym.Symbol("v")
        self.u = sym.Symbol("u")

        self.g = 9.81
        self.t0 = 0
        self.tfmin = 0
        self.tfmax = 10
        self.x0 = 0
        self.y0 = 0
        self.v0 = 0
        self.xf = 2
        self.yf = 2
        self.xmin = 0
        self.xmax = 10
        self.ymin = 0
        self.ymax = 10
        self.vmin = -50
        self.vmax = 50
        self.umin = -np.pi / 2
        self.umax = np.pi / 2

    def test_instantiate_ocp(self, state):
        """OCP object is instantiated correctly."""
        ocp_name = "Brachistochrone"
        state.ocp = pycollo.OptimalControlProblem(name=ocp_name)
        assert state.ocp.name == ocp_name
        assert isinstance(state.ocp, pycollo.OptimalControlProblem)

    def test_add_phase(self, state):
        """Phase can be correctly added to the OCP."""
        phase_name = "A"
        phase = state.ocp.new_phase(name=phase_name)
        assert phase.name == phase_name
        assert state.ocp.phases.A is phase

    def test_set_phase_data(self, state):
        """Phase data (equations etc.) can be added correctly."""
        x_eqn = self.v * sym.sin(self.u)
        y_eqn = self.v * sym.cos(self.u)
        v_eqn = self.g * sym.cos(self.u)
        state.ocp.phases.A.state_variables = [self.x, self.y, self.v]
        state.ocp.phases.A.control_variables = self.u
        state.ocp.phases.A.state_equations = [x_eqn, y_eqn, v_eqn]
        state.ocp.phases.A.auxiliary_data = {}
        assert state.ocp.phases.A.state_variables == (self.x, self.y, self.v)
        assert state.ocp.phases.A.state_variables.x is self.x
        assert state.ocp.phases.A.state_variables.y is self.y
        assert state.ocp.phases.A.state_variables.v is self.v
        assert state.ocp.phases.A.control_variables == (self.u, )
        assert state.ocp.phases.A.control_variables.u is self.u
        assert state.ocp.phases.A.state_equations == (x_eqn, y_eqn, v_eqn)
        assert state.ocp.phases.A.state_equations.x == x_eqn
        assert state.ocp.phases.A.state_equations.y == y_eqn
        assert state.ocp.phases.A.state_equations.v == v_eqn

    def test_set_phase_bounds(self, state):
        """Phase bounds can be set and are reformatted correctly."""
        state.ocp.phases.A.bounds.initial_time = 0.0
        state.ocp.phases.A.bounds.final_time = [self.tfmin, self.tfmax]
        state.ocp.phases.A.bounds.state_variables = [[self.xmin, self.xmax],
                                                     [self.ymin, self.ymax],
                                                     [self.vmin, self.vmax]]
        state.ocp.phases.A.bounds.control_variables = [[self.umin, self.umax]]
        state.ocp.phases.A.bounds.initial_state_constraints = {self.x: self.x0,
                                                               self.y: self.y0,
                                                               self.v: self.v0}
        state.ocp.phases.A.bounds.final_state_constraints = {self.x: self.xf,
                                                             self.y: self.yf}

        assert state.ocp.phases.A.bounds.initial_time == 0.0
        assert state.ocp.phases.A.bounds.final_time == [self.tfmin, self.tfmax]
        np.testing.assert_array_equal(
            state.ocp.phases.A.bounds.state_variables,
            np.array([[self.xmin, self.xmax],
                      [self.ymin, self.ymax],
                      [self.vmin, self.vmax]]))
        np.testing.assert_array_equal(
            state.ocp.phases.A.bounds.control_variables,
            np.array([[self.umin, self.umax]]))
        assert state.ocp.phases.A.bounds.initial_state_constraints == {self.x: self.x0,
                                                                       self.y: self.y0,
                                                                       self.v: self.v0}
        assert state.ocp.phases.A.bounds.final_state_constraints == {self.x: self.xf,
                                                                     self.y: self.yf}

    def test_set_phase_guess(self, state):
        """Phase guesses can be set and are reformatted correctly."""
        state.ocp.phases.A.guess.time = [self.t0, self.tfmax]
        state.ocp.phases.A.guess.state_variables = [[self.x0, self.xf],
                                                    [self.y0, self.yf],
                                                    [self.v0, self.v0]]
        state.ocp.phases.A.guess.control_variables = [[0, self.umax]]
        np.testing.assert_array_equal(
            state.ocp.phases.A.guess.time, np.array([self.t0, self.tfmax]))
        np.testing.assert_array_equal(
            state.ocp.phases.A.guess.state_variables,
            np.array([[self.x0, self.xf],
                      [self.y0, self.yf],
                      [self.v0, self.v0]]))
        np.testing.assert_array_equal(
            state.ocp.phases.A.guess.control_variables,
            np.array([[0, self.umax]]))

    def test_set_objective_function(self, state):
        """Objective function can be set."""
        tF = state.ocp.phases.A.final_time_variable
        state.ocp.objective_function = tF
        assert state.ocp.objective_function == tF

    def test_set_ocp_settings(self, state):
        """Problem settings can be manipulated sucessfully."""
        state.ocp.settings.display_mesh_result_graph = False
        state.ocp.settings.derivative_level = 2
        state.ocp.settings.quadrature_method = "lobatto"
        state.ocp.settings.max_mesh_iterations = 10
        state.ocp.settings.scaling_method = "bounds"
        assert state.ocp.settings.display_mesh_result_graph is False
        assert state.ocp.settings.derivative_level == 2
        assert state.ocp.settings.quadrature_method == "lobatto"
        assert state.ocp.settings.max_mesh_iterations == 10
        assert state.ocp.settings.scaling_method == "bounds"

    def test_ocp_initialise(self, state):
        """OCP can be initialised sucessfully."""
        state.ocp.initialise()

    def test_ocp_solve(self, state):
        """OCP can be solved sucessfully."""
        state.ocp.solve()

    def test_ocp_solution(self, state):
        """OCP solution is correct."""
        GPOPS_II_SOLUTION = 0.82434
        rtol = 1e-4
        atol = 0.0
        assert np.isclose(state.ocp.solution.objective,
                          GPOPS_II_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert state.ocp.mesh_tolerance_met is True
