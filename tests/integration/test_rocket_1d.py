"""Integration test based on the 1D rocket problem in the CasADi docs.

See the example `examples/rocket_1d.py` for a description of and
reference for this optimal control problem.

"""


import numpy as np
import pytest
import sympy as sym

import pycollo


@pytest.fixture(scope="module")
def state():
    """Fixture instance to hold the (incremental) test state."""
    class State:
        pass
    return State()


@pytest.mark.incremental
@pytest.mark.usefixtures("state")
class TestRocket1D:
    """Test the Rocket problem."""

    @pytest.fixture(autouse=True)
    def ocp_fixture(self):
        """Instantiate the required symbols."""
        self.h = sym.Symbol("h")
        self.v = sym.Symbol("v")
        self.m = sym.Symbol("m")
        self.T = sym.Symbol("T")
        self.g = sym.Symbol("g")
        self.alpha = sym.Symbol("alpha")

    def test_instantiate_ocp(self, state):
        """OCP object is instantiated correctly."""
        ocp_name = "Rocket"
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
        state.ocp.phases.A.state_variables = (self.h, self.v, self.m)
        state.ocp.phases.A.control_variables = self.T
        state.ocp.phases.A.state_equations = (self.v,
                                              self.T / self.m - self.g,
                                              -self.alpha * self.T)
        state.ocp.phases.A.auxiliary_data = {}
        assert state.ocp.phases.A.state_variables == (self.h, self.v, self.m)
        assert state.ocp.phases.A.state_variables.h is self.h
        assert state.ocp.phases.A.state_variables.v is self.v
        assert state.ocp.phases.A.state_variables.m is self.m
        assert state.ocp.phases.A.control_variables == (self.T, )
        assert state.ocp.phases.A.control_variables.T is self.T

    def test_set_auxiliary_data(self, state):
        """Auxiliary data can be set."""
        state.ocp.auxiliary_data = {self.g: 9.81,
                                    self.alpha: 1 / (300 * self.g)}

    def test_set_phase_bounds(self, state):
        """Phase bounds can be set and are reformatted correctly."""
        state.ocp.phases.A.bounds.initial_time = 0.0
        state.ocp.phases.A.bounds.final_time = 100.0
        state.ocp.phases.A.bounds.state_variables = [[0.0, 100000.0],
                                                     [0.0, 10000.0],
                                                     [0.0, 500000.0]]
        state.ocp.phases.A.bounds.control_variables = [[0.0, 10.0e8]]
        state.ocp.phases.A.bounds.initial_state_constraints = {self.h: 0.0,
                                                               self.v: 0.0,
                                                               self.m: 500000.0}
        state.ocp.phases.A.bounds.final_state_constraints = {self.h: 100000.0}
        assert state.ocp.phases.A.bounds.initial_time == 0.0
        assert state.ocp.phases.A.bounds.final_time == 100.0

    def test_set_phase_guess(self, state):
        """Phase guesses can be set and are reformatted correctly."""
        state.ocp.phases.A.guess.time = [0.0, 100.0]
        state.ocp.phases.A.guess.state_variables = [[0.0, 100000.0],
                                                    [0.0, 100.0],
                                                    [500000.0, 250000.0]]
        state.ocp.phases.A.guess.control_variables = [[0.0, 0.0]]

    def test_set_objective_function(self, state):
        """Objective function can be set."""
        state.ocp.objective_function = state.ocp.phases.A.initial_state_variables.m - state.ocp.phases.A.final_state_variables.m

    def test_set_ocp_settings(self, state):
        """Problem settings can be manipulated sucessfully."""
        state.ocp.settings.display_mesh_result_graph = True
        state.ocp.settings.derivative_level = 2
        state.ocp.settings.quadrature_method = "lobatto"
        state.ocp.settings.max_mesh_iterations = 10
        state.ocp.settings.scaling_method = "bounds"
        state.ocp.settings.collocation_points_min = 2

    def test_set_ocp_mesh(self, state):
        """OCP mesh can be manipulated successfully."""
        state.ocp.phases.A.mesh.number_mesh_section_nodes = 2
        state.ocp.phases.A.mesh.number_mesh_sections = 10

    def test_ocp_initialise(self, state):
        """OCP can be initialised sucessfully."""
        state.ocp.initialise()

    # @pytest.mark.xfail(reason="Solving not working yet")
    # def test_ocp_solve(self, state):
    #     """OCP can be solved sucessfully."""
    #     pass

    # @pytest.mark.xfail(reason="Solving not working yet")
    # def test_ocp_solution(self, state):
    #     """OCP solution is correct."""
    #     pass