"""Integration test based on the tumour anti-angiogenesis problem.

See the example `examples/optimal_control_problems/tumour_anti_angiogenesis
tumour_anti_angiogenesis.py` for a description of and reference for this
optimal control problem.

"""

import numpy as np
import pytest
import sympy as sym

import pycollo


@pytest.mark.incremental
@pytest.mark.usefixtures("state")
class TestTumourAntiAngiogenesis:
    """Test the Tumour Anti-Angiogenesis problem."""

    @pytest.fixture(autouse=True)
    def ocp_fixture(self):
        """Instantiate the required symbols and variables."""

        # Symbol creation
        self.p = sym.Symbol("p")
        self.q = sym.Symbol("q")
        self.u = sym.Symbol("u")

        self.xi = sym.Symbol("xi")
        self.b = sym.Symbol("b")
        self.d = sym.Symbol("d")
        self.G = sym.Symbol("G")
        self.mu = sym.Symbol("mu")
        self.a = sym.Symbol("a")
        self.A = sym.Symbol("A")

        self.p_max = sym.Symbol("p_max")
        self.p_min = sym.Symbol("p_min")
        self.q_max = sym.Symbol("q_max")
        self.q_min = sym.Symbol("q_min")
        self.y_max = sym.Symbol("y_max")
        self.y_min = sym.Symbol("y_min")
        self.u_max = sym.Symbol("u_max")
        self.u_min = sym.Symbol("u_min")
        self.p_t0 = sym.Symbol("p_t0")
        self.q_t0 = sym.Symbol("q_t0")
        self.y_t0 = sym.Symbol("y_t0")

        # Auxiliary information
        self.t0 = 0.0
        self.tF_max = 5.0
        self.tF_min = 0.1

    def test_ocp_setup(self, state):
        """Set up the OCP."""

        # Set up the Pycollo OCP
        ocp_name = "Tumour Anti-Angiogenesis"
        state.ocp = pycollo.OptimalControlProblem(name=ocp_name)
        state.phase = state.ocp.new_phase(name="A",
                                          state_variables=[self.p, self.q],
                                          control_variables=self.u)

        # Phase information
        p_eqn = -self.xi * self.p * sym.log(self.p / self.q)
        q_eqn_1 = (self.mu + (self.d * self.p**(2 / 3)) + (self.G * self.u))
        q_eqn = self.q * (self.b - q_eqn_1)
        state.phase.state_equations = {self.p: p_eqn,
                                       self.q: q_eqn}
        state.phase.integrand_functions = [self.u]
        state.phase.initial_state_constraints = {self.p: self.p_t0,
                                                 self.q: self.q_t0}

        # Problem information
        state.ocp.objective_function = state.phase.final_state_variables.p
        p_max_eqn = ((self.b - self.mu) / self.d)**(3 / 2)
        state.ocp.auxiliary_data = {self.xi: 0.084,
                                    self.b: 5.85,
                                    self.d: 0.00873,
                                    self.G: 0.15,
                                    self.mu: 0.02,
                                    self.a: 75,
                                    self.A: 15,
                                    self.p_max: p_max_eqn,
                                    self.p_min: 0.1,
                                    self.q_max: self.p_max,
                                    self.q_min: self.p_min,
                                    self.u_max: self.a,
                                    self.u_min: 0,
                                    self.p_t0: self.p_max / 2,
                                    self.q_t0: self.q_max / 4}

        # Bounds
        state.phase.bounds.initial_time = self.t0
        state.phase.bounds.final_time = [self.tF_min, self.tF_max]
        state.phase.bounds.state_variables = {self.p: [self.p_min, self.p_max],
                                              self.q: [self.q_min, self.q_max]}
        state.phase.bounds.control_variables = [[self.u_min, self.u_max]]
        state.phase.bounds.integral_variables = [[0, self.A]]
        state.phase.bounds.initial_state_constraints = {self.p: self.p_t0,
                                                        self.q: self.q_t0}

        # Guess
        state.phase.guess.time = [0, 1]
        state.phase.guess.state_variables = [[self.p_t0, self.p_max],
                                             [self.q_t0, self.q_max]]
        state.phase.guess.control_variables = [[self.u_max, self.u_max]]
        state.phase.guess.integral_variables = [7.5]

    def test_ocp_initialisation(self, state):
        """Initialise the OCP."""
        state.ocp.initialise()

    def test_ocp_solve(self, state):
        """Solve the OCP."""
        state.ocp.solve()

    def test_ocp_solution(self, state):
        """OCP solution is correct.

        The relative tolerance `rtol` is chosen because the GPOPS-II and SOS
        solutions differ at the third decimal place.

        """
        GPOPS_II_SOLUTION = 7.57166986e+03
        SOS_SOLUTION = 7.5716831e+03
        rtol = 1e-5
        atol = 0.0
        assert np.isclose(state.ocp.solution.objective,
                          GPOPS_II_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert np.isclose(state.ocp.solution.objective, SOS_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert state.ocp.mesh_tolerance_met is True
