"""Integration test based on the space shuttle reentry trajectory problem.

See the example `examples/space_shuttle_reentry_trajectory.py` for a
description of and reference for this optimal control problem.

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
        """Instantiate the required symbols and variables."""

        # Variable symbols
        self.h = sym.Symbol("h")
        self.phi = sym.Symbol("phi")
        self.theta = sym.Symbol("theta")
        self.nu = sym.Symbol("nu")
        self.gamma = sym.Symbol("gamma")
        self.psi = sym.Symbol("psi")
        self.alpha = sym.Symbol("alpha")
        self.beta = sym.Symbol("beta")

        # Auxiliary symbols
        self.D = sym.Symbol("D")
        self.L = sym.Symbol("L")
        self.g = sym.Symbol("g")
        self.r = sym.Symbol("r")
        self.rho = sym.Symbol("rho")
        self.rho_0 = sym.Symbol("rho_0")
        self.h_r = sym.Symbol("h_r")
        self.c_L = sym.Symbol("c_L")
        self.c_D = sym.Symbol("c_D")
        self.alpha_hat = sym.Symbol("alpha_hat")
        self.Re = sym.Symbol("Re")
        self.S = sym.Symbol("S")
        self.c_lift_0 = sym.Symbol("c_lift_0")
        self.c_lift_1 = sym.Symbol("c_lift_1")
        self.mu = sym.Symbol("mu")
        self.c_drag_0 = sym.Symbol("c_drag_0")
        self.c_drag_1 = sym.Symbol("c_drag_1")
        self.c_drag_2 = sym.Symbol("c_drag_2")
        self.q_r = sym.Symbol("q_r")
        self.q_a = sym.Symbol("q_a")
        self.c_0 = sym.Symbol("c_0")
        self.c_1 = sym.Symbol("c_1")
        self.c_2 = sym.Symbol("c_2")
        self.c_3 = sym.Symbol("c_3")
        self.w = sym.Symbol("w")
        self.m = sym.Symbol("m")
        self.g_0 = sym.Symbol("g_0")

        # Numerical bounds on time endpoints
        self.t_0 = 0.0
        self.t_f = None

        # Numerical bounds on times
        self.t_f_min = 0.0
        self.t_f_max = 3000.0

        # Numerical bounds on state endpoints
        self.h_0 = 79248
        self.h_f = 24384
        self.phi_0 = 0
        self.phi_f = None
        self.theta_0 = 0
        self.theta_f = None
        self.nu_0 = 7802.88
        self.nu_f = 762
        self.gamma_0 = -1 * np.pi / 180
        self.gamma_f = -5 * np.pi / 180
        self.psi_0 = 90 * np.pi / 180
        self.psi_f = None

        # Numerical bounds on states
        self.h_min = 0
        self.h_max = 300000
        self.phi_min = -np.pi
        self.phi_max = np.pi
        self.theta_min = -70 * np.pi / 180
        self.theta_max = 70 * np.pi / 180
        self.nu_min = 10
        self.nu_max = 45000
        self.gamma_min = -80 * np.pi / 180
        self.gamma_max = 80 * np.pi / 180
        self.psi_min = -np.pi
        self.psi_max = np.pi
        self.alpha_min = -np.pi / 2
        self.alpha_max = np.pi / 2
        self.beta_min = -np.pi / 2
        self.beta_max = np.pi / 180

        # Numerical guesses for state endpoints
        self.h_0_guess = self.h_0
        self.h_f_guess = self.h_f
        self.phi_0_guess = self.phi_0
        self.phi_f_guess = self.phi_0 + 10 * np.pi / 180
        self.theta_0_guess = self.theta_0
        self.theta_f_guess = self.theta_0 + 10 * np.pi / 180
        self.nu_0_guess = self.nu_0
        self.nu_f_guess = self.nu_f
        self.gamma_0_guess = self.gamma_0
        self.gamma_f_guess = self.gamma_f
        self.psi_0_guess = self.psi_0
        self.psi_f_guess = -self.psi_0
        self.alpha_0_guess = 0
        self.alpha_f_guess = 0
        self.beta_0_guess = 0
        self.beta_f_guess = 0

    def test_ocp_setup_and_solve(self, state):
        """Set up and solve the OCP."""

        # Set up the Pycollo OCP
        ocp_name = "Space shuttle reentry trajectory maximum crossrange"
        state.ocp = pycollo.OptimalControlProblem(name=ocp_name)
        state.phase = state.ocp.new_phase(name="A")

        # Phase information
        state.phase.state_variables = [self.h,
                                       self.phi,
                                       self.theta,
                                       self.nu,
                                       self.gamma,
                                       self.psi]
        state.phase.control_variables = [self.alpha, self.beta]

        dh = self.nu * sym.sin(self.gamma)

        dphi_1 = self.nu * sym.cos(self.gamma) * sym.sin(self.psi)
        dphi_2 = (self.r * sym.cos(self.theta))
        dphi = dphi_1 / dphi_2

        dtheta = self.nu * sym.cos(self.gamma) * sym.cos(self.psi) / self.r

        dnu = -(self.D / self.m) - self.g * sym.sin(self.gamma)

        dgamma_1 = self.L * sym.cos(self.beta) / (self.m * self.nu)
        dgamma_2 = sym.cos(self.gamma)
        dgamma_3 = ((self.nu / self.r) - (self.g / self.nu))
        dgamma = dgamma_1 + dgamma_2 * dgamma_3

        dpsi_1 = self.L * sym.sin(self.beta)
        dpsi_2 = (self.m * self.nu * sym.cos(self.gamma))
        dpsi_3 = self.nu * sym.cos(self.gamma)
        dpsi_4 = sym.sin(self.psi) * sym.sin(self.theta)
        dpsi_5 = self.r * sym.cos(self.theta)
        dpsi = (dpsi_1 / dpsi_2) + (dpsi_3 * dpsi_4) / dpsi_5

        state.phase.state_equations = {self.h: dh,
                                       self.phi: dphi,
                                       self.theta: dtheta,
                                       self.nu: dnu,
                                       self.gamma: dgamma,
                                       self.psi: dpsi}
        state.phase.auxiliary_data = {}

        # Problem information
        c_D_1 = self.c_drag_0 + (self.c_drag_1 * self.alpha)
        c_D_2 = (self.c_drag_2 * self.alpha**2)
        c_D = c_D_1 + c_D_2
        state.ocp.objective_function = -state.phase.final_state_variables[2]
        state.ocp.auxiliary_data = {
            self.rho_0: 1.225570827014494,
            self.h_r: 7254.24,
            self.Re: 6371203.92,
            self.S: 249.9091776,
            self.c_lift_0: -0.2070,
            self.c_lift_1: 1.6756,
            self.mu: 3.986031954093051e14,
            self.c_drag_0: 0.07854,
            self.c_drag_1: -0.3529,
            self.c_drag_2: 2.0400,
            self.D: 0.5 * self.c_D * self.S * self.rho * self.nu**2,
            self.L: 0.5 * self.c_L * self.S * self.rho * self.nu**2,
            self.g: self.mu / (self.r**2),
            self.r: self.Re + self.h,
            self.rho: self.rho_0 * sym.exp(-self.h / self.h_r),
            self.c_L: self.c_lift_0 + (self.c_lift_1 * self.alpha),
            self.c_D: c_D,
            self.m: 92079.2525560557,
        }

        # Bounds
        state.phase.bounds.initial_time = self.t_0
        state.phase.bounds.final_time = [self.t_f_min, self.t_f_max]
        y_bnds = {self.h: [self.h_min, self.h_max],
                  self.phi: [self.phi_min, self.phi_max],
                  self.theta: [self.theta_min, self.theta_max],
                  self.nu: [self.nu_min, self.nu_max],
                  self.gamma: [self.gamma_min, self.gamma_max],
                  self.psi: [self.psi_min, self.psi_max]}
        u_bnds = {self.alpha: [self.alpha_min, self.alpha_max],
                  self.beta: [self.beta_min, self.beta_max]}
        y_t0_bnds = {self.h: self.h_0,
                     self.phi: self.phi_0,
                     self.theta: self.theta_0,
                     self.nu: self.nu_0,
                     self.gamma: self.gamma_0,
                     self.psi: self.psi_0}
        y_tF_bnds = {self.h: [self.h_f, self.h_f],
                     self.nu: [self.nu_f, self.nu_f],
                     self.gamma: [self.gamma_f, self.gamma_f]}
        state.phase.bounds.state_variables = y_bnds
        state.phase.bounds.control_variables = u_bnds
        state.phase.bounds.initial_state_constraints = y_t0_bnds
        state.phase.bounds.final_state_constraints = y_tF_bnds

        # Guess
        y_guess = np.array([[self.h_0_guess, self.h_f_guess],
                            [self.phi_0_guess, self.phi_f_guess],
                            [self.theta_0_guess, self.theta_f_guess],
                            [self.nu_0_guess, self.nu_f_guess],
                            [self.gamma_0_guess, self.gamma_f_guess],
                            [self.psi_0_guess, self.psi_f_guess]])
        u_guess = np.array([[self.alpha_0_guess, self.alpha_f_guess],
                            [self.beta_0_guess, self.beta_f_guess]])
        state.phase.guess.time = np.array([self.t_0, 1000])
        state.phase.guess.state_variables = y_guess
        state.phase.guess.control_variables = u_guess

        # Settings
        state.ocp.settings.max_mesh_iterations = 10

        # Initialise and solve
        state.ocp.initialise()
        state.ocp.solve()

    def test_ocp_solution(self, state):
        """OCP solution is correct.

        The relative tolerance `rtol` is chosen because the GPOPS-II and SOS
        solutions differ at the third decimal place.

        """
        GPOPS_II_SOLUTION = -0.59628
        SOS_SOLUTION = -0.59588
        rtol = 1e-3
        atol = 0.0
        assert np.isclose(state.ocp.solution.objective,
                          GPOPS_II_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert np.isclose(state.ocp.solution.objective, SOS_SOLUTION,
                          rtol=rtol,
                          atol=atol)
        assert state.ocp.mesh_tolerance_met is True
