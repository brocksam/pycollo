"""Integration test based on the space station attitude control problem.

See the example `examples/space_station_attitude_control.py` for a
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

        # Symbol creation
        self.J_00 = sym.Symbol("J_00")
        self.J_01 = sym.Symbol("J_01")
        self.J_02 = sym.Symbol("J_02")
        self.J_10 = sym.Symbol("J_10")
        self.J_11 = sym.Symbol("J_11")
        self.J_12 = sym.Symbol("J_12")
        self.J_20 = sym.Symbol("J_20")
        self.J_21 = sym.Symbol("J_21")
        self.J_22 = sym.Symbol("J_22")
        self.J_inv_00 = sym.Symbol("J_inv_00")
        self.J_inv_01 = sym.Symbol("J_inv_01")
        self.J_inv_02 = sym.Symbol("J_inv_02")
        self.J_inv_10 = sym.Symbol("J_inv_10")
        self.J_inv_11 = sym.Symbol("J_inv_11")
        self.J_inv_12 = sym.Symbol("J_inv_12")
        self.J_inv_20 = sym.Symbol("J_inv_20")
        self.J_inv_21 = sym.Symbol("J_inv_21")
        self.J_inv_22 = sym.Symbol("J_inv_22")
        self.omega_x = sym.Symbol("omega_x")
        self.omega_y = sym.Symbol("omega_y")
        self.omega_z = sym.Symbol("omega_z")
        self.r_x = sym.Symbol("r_x")
        self.r_y = sym.Symbol("r_y")
        self.r_z = sym.Symbol("r_z")
        self.h_x = sym.Symbol("h_x")
        self.h_y = sym.Symbol("h_y")
        self.h_z = sym.Symbol("h_z")
        self.u_x = sym.Symbol("u_x")
        self.u_y = sym.Symbol("u_y")
        self.u_z = sym.Symbol("u_z")

        self.domega_x_dt = sym.Symbol("domega_x_dt")
        self.domega_y_dt = sym.Symbol("domega_y_dt")
        self.domega_z_dt = sym.Symbol("domega_z_dt")
        self.dr_x_dt = sym.Symbol("dr_x_dt")
        self.dr_y_dt = sym.Symbol("dr_y_dt")
        self.dr_z_dt = sym.Symbol("dr_z_dt")
        self.dh_x_dt = sym.Symbol("dh_x_dt")
        self.dh_y_dt = sym.Symbol("dh_y_dt")
        self.dh_z_dt = sym.Symbol("dh_z_dt")

        self.domega_x_dt_tF = sym.Symbol("domega_x_dt_tF")
        self.domega_y_dt_tF = sym.Symbol("domega_y_dt_tF")
        self.domega_z_dt_tF = sym.Symbol("domega_z_dt_tF")
        self.dr_x_dt_tF = sym.Symbol("dr_x_dt_tF")
        self.dr_y_dt_tF = sym.Symbol("dr_y_dt_tF")
        self.dr_z_dt_tF = sym.Symbol("dr_z_dt_tF")

        self.omega_orb = sym.Symbol("omega_orb")
        self.h_max = sym.Symbol("h_max")

        self.h_inner_prod_squared = sym.Symbol("h_inner_prod_squared")
        self.u_inner_prod_squared = sym.Symbol("u_inner_prod_squared")

        # Auxiliary information
        self.t0 = 0
        self.tF = 1800
        self.omega_x_t0 = -9.5380685844896e-6
        self.omega_y_t0 = -1.1363312657036e-3
        self.omega_z_t0 = 5.3472801108427e-6
        self.r_x_t0 = 2.9963689649816e-3
        self.r_y_t0 = 1.5334477761054e-1
        self.r_z_t0 = 3.8359805613992e-3
        self.h_x_t0 = 5000
        self.h_y_t0 = 5000
        self.h_z_t0 = 5000
        self.h_x_tF = 0
        self.h_y_tF = 0
        self.h_z_tF = 0

    def test_ocp_setup_and_solve(self, state):
        """Set up and solve the OCP."""

        # Set up the Pycollo OCP
        name = "Space Station Attitude Control"
        y_vars = [self.omega_x,
                  self.omega_y,
                  self.omega_z,
                  self.r_x,
                  self.r_y,
                  self.r_z,
                  self.h_x,
                  self.h_y,
                  self.h_z]
        u_vars = [self.u_x, self.u_y, self.u_z]
        state.ocp = pycollo.OptimalControlProblem(name=name)
        state.phase = state.ocp.new_phase(name="A",
                                          state_variables=y_vars,
                                          control_variables=u_vars)

        # Phase information
        state.phase.state_equations = {self.omega_x: self.domega_x_dt,
                                       self.omega_y: self.domega_y_dt,
                                       self.omega_z: self.domega_z_dt,
                                       self.r_x: self.dr_x_dt,
                                       self.r_y: self.dr_y_dt,
                                       self.r_z: self.dr_z_dt,
                                       self.h_x: self.dh_x_dt,
                                       self.h_y: self.dh_y_dt,
                                       self.h_z: self.dh_z_dt}
        state.phase.path_constraints = [self.h_inner_prod_squared]
        state.phase.integrand_functions = [1e-6 * self.u_inner_prod_squared]

        # Problem information
        state.ocp.objective_function = state.phase.integral_variables[0]
        state.ocp.endpoint_constraints = [self.domega_x_dt_tF,
                                          self.domega_y_dt_tF,
                                          self.domega_z_dt_tF,
                                          self.dr_x_dt_tF,
                                          self.dr_y_dt_tF,
                                          self.dr_z_dt_tF]

        # Problem bounds
        state.phase.bounds.initial_time = self.t0
        state.phase.bounds.final_time = self.tF
        state.phase.bounds.state_variables = {self.omega_x: [-2e-3, 2e-3],
                                              self.omega_y: [-2e-3, 2e-3],
                                              self.omega_z: [-2e-3, 2e-3],
                                              self.r_x: [-1, 1],
                                              self.r_y: [-1, 1],
                                              self.r_z: [-1, 1],
                                              self.h_x: [-15000, 15000],
                                              self.h_y: [-15000, 15000],
                                              self.h_z: [-15000, 15000]}
        state.phase.bounds.initial_state_constraints = {self.omega_x: self.omega_x_t0,
                                                        self.omega_y: self.omega_y_t0,
                                                        self.omega_z: self.omega_z_t0,
                                                        self.r_x: self.r_x_t0,
                                                        self.r_y: self.r_y_t0,
                                                        self.r_z: self.r_z_t0,
                                                        self.h_x: self.h_x_t0,
                                                        self.h_y: self.h_y_t0,
                                                        self.h_z: self.h_z_t0}
        state.phase.bounds.final_state_constraints = {self.h_x: self.h_x_tF,
                                                      self.h_y: self.h_y_tF,
                                                      self.h_z: self.h_z_tF}
        state.phase.bounds.control_variables = {self.u_x: [-150, 150],
                                                self.u_y: [-150, 150],
                                                self.u_z: [-150, 150]}
        state.phase.bounds.integral_variables = [[0, 10]]
        state.phase.bounds.path_constraints = [[0, self.h_max**2]]

        state.ocp.bounds.endpoint_constraints = [[0, 0],
                                                 [0, 0],
                                                 [0, 0],
                                                 [0, 0],
                                                 [0, 0],
                                                 [0, 0]]

        # Problem guesses
        state.phase.guess.time = np.array([self.t0, self.tF])
        state.phase.guess.state_variables = np.array([[self.omega_x_t0, self.omega_x_t0],
                                                      [self.omega_y_t0, self.omega_y_t0],
                                                      [self.omega_z_t0, self.omega_z_t0],
                                                      [self.r_x_t0, self.r_x_t0],
                                                      [self.r_y_t0, self.r_y_t0],
                                                      [self.r_z_t0, self.r_z_t0],
                                                      [self.h_x_t0, self.h_x_t0],
                                                      [self.h_y_t0, self.h_y_t0],
                                                      [self.h_z_t0, self.h_z_t0]])
        state.phase.guess.control_variables = np.array([[0, 0], [0, 0], [0, 0]])
        state.phase.guess.integral_variables = np.array([10])

        # J matrices
        J = sym.Matrix([
            [self.J_00, self.J_01, self.J_02],
            [self.J_10, self.J_11, self.J_12],
            [self.J_20, self.J_21, self.J_22]])
        J_inv = sym.Matrix([
            [self.J_inv_00, self.J_inv_01, self.J_inv_02],
            [self.J_inv_10, self.J_inv_11, self.J_inv_12],
            [self.J_inv_20, self.J_inv_21, self.J_inv_22]])

        # Continuous vectors
        omega = sym.Matrix([self.omega_x, self.omega_y, self.omega_z])
        r = sym.Matrix([self.r_x, self.r_y, self.r_z])
        h = sym.Matrix([self.h_x, self.h_y, self.h_z])
        u = sym.Matrix([self.u_x, self.u_y, self.u_z])

        # Calculating domega/dt
        r_skew_symmetric = skew_symmetric_cross_product_operator(r)
        I = sym.eye(3)
        D = 2 / (1 + col_vec_dot_row_vec(r.T, r))
        E = (r_skew_symmetric * r_skew_symmetric) - r_skew_symmetric
        C = I + (D * E)
        C_2_skew = skew_symmetric_cross_product_operator(C[:, 2])
        tau_gg = 3 * self.omega_orb**2 * C_2_skew * (J * C[:, 2])
        A = J * omega + h
        B = skew_symmetric_cross_product_operator(omega) * A
        K = tau_gg - B - u
        domega_dt = J_inv * K

        # Calculating dr/dt
        omega_0 = - self.omega_orb * C[:, 1]
        r_sqrd = row_vec_dot_col_vec(r, r.T)
        dr_dt = 0.5 * (r_sqrd + I + r_skew_symmetric) * (omega - omega_0)

        # Endpoint equations
        omega_tF = sym.Matrix(state.phase.final_state_variables[:3])
        r_tF = sym.Matrix(state.phase.final_state_variables[3:6])
        h_tF = sym.Matrix(state.phase.final_state_variables[6:])

        # Calculating domega(tF)/dt
        r_tF_skew_symmetric = skew_symmetric_cross_product_operator(r_tF)
        D_tF = 2 / (1 + col_vec_dot_row_vec(r_tF.T, r_tF))
        E_tF = (r_tF_skew_symmetric * r_tF_skew_symmetric) - r_tF_skew_symmetric
        C_tF = I + (D_tF * E_tF)
        C_tF_2_skew = skew_symmetric_cross_product_operator(C_tF[:, 2])
        tau_gg_tF = 3 * self.omega_orb**2 * C_tF_2_skew * J * C_tF[:, 2]
        A_tF = J * omega_tF + h_tF
        B_tF = skew_symmetric_cross_product_operator(omega_tF) * A_tF
        K_tF = tau_gg_tF - B_tF
        domega_dt_tF = J_inv * K_tF

        # Calculating dr(tF)/dt
        omega_0_tF = -self.omega_orb * C_tF[:, 1]
        r_tF_sqrd = row_vec_dot_col_vec(r_tF, r_tF.T)
        omega_tF_diff = (omega_tF - omega_0_tF)
        dr_dt_tF = 0.5 * (r_tF_sqrd + I + r_tF_skew_symmetric) * omega_tF_diff

        state.ocp.auxiliary_data = {
            self.J_00: 2.80701911616e7,
            self.J_01: 4.822509936e5,
            self.J_02: -1.71675094448e7,
            self.J_10: 4.822509936e5,
            self.J_11: 9.5144639344e7,
            self.J_12: 6.02604448e4,
            self.J_20: -1.71675094448e7,
            self.J_21: 6.02604448e4,
            self.J_22: 7.6594401336e7,
            self.J_inv_00: J.inv()[0, 0],
            self.J_inv_01: J.inv()[0, 1],
            self.J_inv_02: J.inv()[0, 2],
            self.J_inv_10: J.inv()[1, 0],
            self.J_inv_11: J.inv()[1, 1],
            self.J_inv_12: J.inv()[1, 2],
            self.J_inv_20: J.inv()[2, 0],
            self.J_inv_21: J.inv()[2, 1],
            self.J_inv_22: J.inv()[2, 2],
            self.omega_orb: 0.06511 * np.pi / 180,
            self.h_max: 10000,
            self.u_inner_prod_squared: self.u_x**2 + self.u_y**2 + self.u_z**2,
            self.h_inner_prod_squared: self.h_x**2 + self.h_y**2 + self.h_z**2,
            self.domega_x_dt: domega_dt[0, 0],
            self.domega_y_dt: domega_dt[1, 0],
            self.domega_z_dt: domega_dt[2, 0],
            self.dr_x_dt: dr_dt[0, 0],
            self.dr_y_dt: dr_dt[1, 0],
            self.dr_z_dt: dr_dt[2, 0],
            self.dh_x_dt: self.u_x,
            self.dh_y_dt: self.u_y,
            self.dh_z_dt: self.u_z,
            self.domega_x_dt_tF: domega_dt_tF[0, 0],
            self.domega_y_dt_tF: domega_dt_tF[1, 0],
            self.domega_z_dt_tF: domega_dt_tF[2, 0],
            self.dr_x_dt_tF: dr_dt_tF[0, 0],
            self.dr_y_dt_tF: dr_dt_tF[1, 0],
            self.dr_z_dt_tF: dr_dt_tF[2, 0]}

        state.ocp.initialise()
        state.ocp.solve()

    def test_ocp_solution(self, state):
        """OCP solution is correct.

        The relative tolerance `rtol` is chosen because the GPOPS-II and SOS
        solutions differ at the fourth decimal place.

        """
        GPOPS_II_SOLUTION = 3.58675
        SOS_SOLUTION = 3.58688
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


# Utility functions
def skew_symmetric_cross_product_operator(vec):
    if vec.shape != (3, 1):
        raise ValueError(f"Vector must be a column vector and have shape "
                         f"(3, 1) but is {vec.shape}")
    skew_symmetric_cross_product_operator = sym.Matrix([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]])
    return skew_symmetric_cross_product_operator


def row_vec_dot_col_vec(vec_1, vec_2):
    if vec_1.shape != (3, 1):
        raise ValueError(f"First vector must be a column vector and have "
                         f"shape (3, 1) but is {vec_1.shape}")
    if vec_2.shape != (1, 3):
        raise ValueError(f"Second vector must be a row vector and have shape "
                         f"(1, 3) but is {vec_2.shape}")
    matrix = sym.Matrix([[vec_1[0, 0] * vec_2[0, 0],
                          vec_1[0, 0] * vec_2[0, 1],
                          vec_1[0, 0] * vec_2[0, 2]],
                         [vec_1[1, 0] * vec_2[0, 0],
                          vec_1[1, 0] * vec_2[0, 1],
                          vec_1[1, 0] * vec_2[0, 2]],
                         [vec_1[2, 0] * vec_2[0, 0],
                          vec_1[2, 0] * vec_2[0, 1],
                          vec_1[2, 0] * vec_2[0, 2]]])
    return matrix


def col_vec_dot_row_vec(vec_1, vec_2):
    if vec_1.shape != (1, 3):
        raise ValueError(f"First vector must be a row vector and have shape "
                         f"(1, 3) but is {vec_1.shape}")
    if vec_2.shape != (3, 1):
        raise ValueError(f"Second vector must be a column vector and have "
                         f"shape (3, 1) but is {vec_2.shape}")
    return vec_1.dot(vec_2)
