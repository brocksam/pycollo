"""Integration test based on the hypersensitive problem.

See the example `examples/hypersenstive_problem.py` for a description of and
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
class TestHypersensitiveProblem:
    """Test the Hypersensitive problem."""

    @pytest.fixture(autouse=True)
    def ocp_fixture(self):
        """Instantiate the required symbols."""
        self.y = sym.Symbol("y")
        self.u = sym.Symbol("u")

    def test_instantiate_ocp(self, state):
        """OCP object is instantiated correctly."""
        ocp_name = "Hypersensitive problem"
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
        state.ocp.phases.A.state_variables = self.y
        state.ocp.phases.A.control_variables = self.u
        state.ocp.phases.A.state_equations = -self.y**3 + self.u
        state.ocp.phases.A.integrand_functions = 0.5 * (self.y**2 + self.u**2)
        state.ocp.phases.A.auxiliary_data = {}
        assert state.ocp.phases.A.state_variables == (self.y, )
        assert state.ocp.phases.A.state_variables.y is self.y
        assert state.ocp.phases.A.control_variables == (self.u, )
        assert state.ocp.phases.A.control_variables.u is self.u

    def test_set_phase_bounds(self, state):
        """Phase bounds can be set and are reformatted correctly."""
        state.ocp.phases.A.bounds.initial_time = 0.0
        state.ocp.phases.A.bounds.final_time = 10000.0
        state.ocp.phases.A.bounds.state_variables = [[0, 2]]
        state.ocp.phases.A.bounds.control_variables = [[-1, 8]]
        state.ocp.phases.A.bounds.integral_variables = [[0, 2000]]
        state.ocp.phases.A.bounds.initial_state_constraints = [[1.0, 1.0]]
        state.ocp.phases.A.bounds.final_state_constraints = [[1.5, 1.5]]

    def test_set_phase_guess(self, state):
        """Phase guesses can be set and are reformatted correctly."""
        state.ocp.phases.A.guess.time = np.array([0.0, 10000.0])
        state.ocp.phases.A.guess.state_variables = np.array([[1.0, 1.5]])
        state.ocp.phases.A.guess.control_variables = np.array([[0.0, 0.0]])
        state.ocp.phases.A.guess.integral_variables = np.array([4])

    def test_set_objective_function(self, state):
        """Objective function can be set."""
        state.ocp.objective_function = state.ocp.phases.A.integral_variables[0]

    def test_set_ocp_settings(self, state):
        """Problem settings can be manipulated sucessfully."""
        state.ocp.settings.display_mesh_result_graph = True
        state.ocp.settings.derivative_level = 2
        state.ocp.settings.quadrature_method = "lobatto"
        state.ocp.settings.max_mesh_iterations = 10
        assert state.ocp.settings.quadrature_method == "lobatto"

    @pytest.mark.xfail(reason="Initialisation not working yet")
    def test_ocp_initialise(self, state):
        """OCP can be initialised sucessfully."""
        state.ocp.initialise()

    @pytest.mark.xfail(reason="Solving not working yet")
    def test_ocp_solve(self, state):
        """OCP can be solved sucessfully."""
        pass

    @pytest.mark.xfail(reason="Solving not working yet")
    def test_ocp_solution(self, state):
        """OCP solution is correct."""
        pass