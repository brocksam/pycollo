"""Integration test based on the free-flying robot problem.

See the example `examples/optimal_control_problems/free_flying_robot
free_flying_robot.py` for a description of and reference for this
optimal control problem.

"""

import numpy as np
import pytest
import sympy as sym

import pycollo


@pytest.mark.incremental
@pytest.mark.usefixtures("state")
class TestFreeFlyingRobot:
    """Test the Free-Flying Robot problem."""

    @pytest.fixture(autouse=True)
    def ocp_fixture(self):
        """Instantiate the required symbols and variables."""

        # Symbol creation
        self.r_x = sym.Symbol("r_x")
        self.r_y = sym.Symbol("r_y")
        self.theta = sym.Symbol("theta")
        self.v_x = sym.Symbol("v_x")
        self.v_y = sym.Symbol("v_y")
        self.omega = sym.Symbol("omega")

        self.u_x_pos = sym.Symbol("u_x_pos")
        self.u_x_neg = sym.Symbol("u_x_neg")
        self.u_y_pos = sym.Symbol("u_y_pos")
        self.u_y_neg = sym.Symbol("u_y_neg")

        self.T_x = sym.Symbol("T_x")
        self.T_y = sym.Symbol("T_y")

        self.I_xx = sym.Symbol("I_xx")
        self.I_yy = sym.Symbol("I_yy")

        # Auxiliary information
        self.u_x_pos_min = 0
        self.u_x_pos_max = 1000
        self.u_x_neg_min = 0
        self.u_x_neg_max = 1000
        self.u_y_pos_min = 0
        self.u_y_pos_max = 1000
        self.u_y_neg_min = 0
        self.u_y_neg_max = 1000

        self.t0 = 0.0
        self.tF = 12.0

        self.r_x_t0 = -10
        self.r_x_tF = 0
        self.r_y_t0 = -10
        self.r_y_tF = 0
        self.theta_t0 = np.pi / 2
        self.theta_tF = 0
        self.v_x_t0 = 0
        self.v_x_tF = 0
        self.v_y_t0 = 0
        self.v_y_tF = 0
        self.omega_t0 = 0
        self.omega_tF = 0

        self.r_x_min = -10
        self.r_x_max = 10
        self.r_y_min = -10
        self.r_y_max = 10
        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.v_x_min = -2
        self.v_x_max = 2
        self.v_y_min = -2
        self.v_y_max = 2
        self.omega_min = -1
        self.omega_max = 1

        self.u_x_pos_min = 0
        self.u_x_pos_max = 1000
        self.u_x_neg_min = 0
        self.u_x_neg_max = 1000
        self.u_y_pos_min = 0
        self.u_y_pos_max = 1000
        self.u_y_neg_min = 0
        self.u_y_neg_max = 1000

    def test_ocp_setup(self, state):
        """Set up the OCP."""

        # Set up the Pycollo OCP
        ocp_name = "Free-Flying Robot"
        state.ocp = pycollo.OptimalControlProblem(name=ocp_name)
        y_vars = [self.r_x,
                  self.r_y,
                  self.theta,
                  self.v_x,
                  self.v_y,
                  self.omega]
        u_vars = [self.u_x_pos, self.u_x_neg, self.u_y_pos, self.u_y_neg]
        state.phase = state.ocp.new_phase(name="A",
                                          state_variables=y_vars,
                                          control_variables=u_vars)

        # Phase information
        v_x_dot = (self.T_x + self.T_y) * sym.cos(self.theta)
        v_y_dot = (self.T_x + self.T_y) * sym.sin(self.theta)
        omega_dot = (self.I_xx * self.T_x) - (self.I_yy * self.T_y)
        state.phase.state_equations = {self.r_x: self.v_x,
                                       self.r_y: self.v_y,
                                       self.theta: self.omega,
                                       self.v_x: v_x_dot,
                                       self.v_y: v_y_dot,
                                       self.omega: omega_dot}
        q = self.u_x_pos + self.u_x_neg + self.u_y_pos + self.u_y_neg
        state.phase.integrand_functions = [q]
        state.phase.path_constraints = [(self.u_x_pos + self.u_x_neg),
                                        (self.u_y_pos + self.u_y_neg)]

        # Problem information
        state.ocp.objective_function = state.phase.integral_variables[0]
        state.ocp.auxiliary_data = {self.I_xx: 0.2,
                                    self.I_yy: 0.2,
                                    self.T_x: self.u_x_pos - self.u_x_neg,
                                    self.T_y: self.u_y_pos - self.u_y_neg,
                                    }

        # Bounds
        state.phase.bounds.initial_time = self.t0
        state.phase.bounds.final_time = self.tF
        y_bnds = {self.r_x: [self.r_x_min, self.r_x_max],
                  self.r_y: [self.r_y_min, self.r_y_max],
                  self.theta: [self.theta_min, self.theta_max],
                  self.v_x: [self.v_x_min, self.v_x_max],
                  self.v_y: [self.v_y_min, self.v_y_max],
                  self.omega: [self.omega_min, self.omega_max]}
        state.phase.bounds.state_variables = y_bnds
        y_t0_bnds = {self.r_x: [self.r_x_t0, self.r_x_t0],
                     self.r_y: [self.r_y_t0, self.r_y_t0],
                     self.theta: [self.theta_t0, self.theta_t0],
                     self.v_x: [self.v_x_t0, self.v_x_t0],
                     self.v_y: [self.v_y_t0, self.v_y_t0],
                     self.omega: [self.omega_t0, self.omega_t0]}
        state.phase.bounds.initial_state_constraints = y_t0_bnds
        y_tF_bnds = {self.r_x: [self.r_x_tF, self.r_x_tF],
                     self.r_y: [self.r_y_tF, self.r_y_tF],
                     self.theta: [self.theta_tF, self.theta_tF],
                     self.v_x: [self.v_x_tF, self.v_x_tF],
                     self.v_y: [self.v_y_tF, self.v_y_tF],
                     self.omega: [self.omega_tF, self.omega_tF]}
        state.phase.bounds.final_state_constraints = y_tF_bnds
        u_bnds = {self.u_x_pos: [self.u_x_pos_min, self.u_x_pos_max],
                  self.u_x_neg: [self.u_x_neg_min, self.u_x_neg_max],
                  self.u_y_pos: [self.u_y_pos_min, self.u_y_pos_max],
                  self.u_y_neg: [self.u_y_neg_min, self.u_y_neg_max]}
        state.phase.bounds.control_variables = u_bnds
        state.phase.bounds.integral_variables = [[0, 100]]
        state.phase.bounds.path_constraints = [[-1000, 1], [-1000, 1]]

        # Guess
        state.phase.guess.time = [self.t0, self.tF]
        state.phase.guess.state_variables = [[self.r_x_t0, self.r_x_tF],
                                             [self.r_y_t0, self.r_y_tF],
                                             [self.theta_t0, self.theta_tF],
                                             [self.v_x_t0, self.v_x_tF],
                                             [self.v_y_t0, self.v_y_tF],
                                             [self.omega_t0, self.omega_tF]]
        state.phase.guess.control_variables = [[0, 0], [0, 0], [0, 0], [0, 0]]
        state.phase.guess.integral_variables = [0]

        # Settings
        state.ocp.settings.mesh_tolerance = 1e-5
        state.ocp.settings.max_mesh_iterations = 15

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
        GPOPS_II_SOLUTION = 7.9101902
        SOS_SOLUTION = 7.910154646
        rtol = 1e-4
        atol = 0.0
        assert np.isclose(state.ocp.solution.objective,
                          GPOPS_II_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert np.isclose(state.ocp.solution.objective, SOS_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert state.ocp.mesh_tolerance_met is True
