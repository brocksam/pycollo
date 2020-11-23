"""Space Station Attitude Control problem.

Example 6.11 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p293 - 298.

"""

import numpy as np
import pycollo
import sympy as sym

# Symbol creation
J_00 = sym.Symbol("J_00")
J_01 = sym.Symbol("J_01")
J_02 = sym.Symbol("J_02")
J_10 = sym.Symbol("J_10")
J_11 = sym.Symbol("J_11")
J_12 = sym.Symbol("J_12")
J_20 = sym.Symbol("J_20")
J_21 = sym.Symbol("J_21")
J_22 = sym.Symbol("J_22")
J_inv_00 = sym.Symbol("J_inv_00")
J_inv_01 = sym.Symbol("J_inv_01")
J_inv_02 = sym.Symbol("J_inv_02")
J_inv_10 = sym.Symbol("J_inv_10")
J_inv_11 = sym.Symbol("J_inv_11")
J_inv_12 = sym.Symbol("J_inv_12")
J_inv_20 = sym.Symbol("J_inv_20")
J_inv_21 = sym.Symbol("J_inv_21")
J_inv_22 = sym.Symbol("J_inv_22")
omega_x, omega_y, omega_z = sym.symbols("omega_x omega_y omega_z")
r_x, r_y, r_z = sym.symbols("r_x r_y r_z")
h_x, h_y, h_z = sym.symbols("h_x h_y h_z")
u_x, u_y, u_z = sym.symbols("u_x u_y u_z")

domega_x_dt = sym.Symbol("domega_x_dt")
domega_y_dt = sym.Symbol("domega_y_dt")
domega_z_dt = sym.Symbol("domega_z_dt")
dr_x_dt, dr_y_dt, dr_z_dt = sym.symbols("dr_x_dt dr_y_dt dr_z_dt")
dh_x_dt, dh_y_dt, dh_z_dt = sym.symbols("dh_x_dt dh_y_dt dh_z_dt")

domega_x_dt_tF = sym.Symbol("domega_x_dt_tF")
domega_y_dt_tF = sym.Symbol("domega_y_dt_tF")
domega_z_dt_tF = sym.Symbol("domega_z_dt_tF")
dr_x_dt_tF = sym.Symbol("dr_x_dt_tF")
dr_y_dt_tF = sym.Symbol("dr_y_dt_tF")
dr_z_dt_tF = sym.Symbol("dr_z_dt_tF")

omega_orb, h_max = sym.symbols("omega_orb h_max")

h_inner_prod_squared = sym.Symbol("h_inner_prod_squared")
u_inner_prod_squared = sym.Symbol("u_inner_prod_squared")

# Auxiliary information
t0 = 0
tF = 1800
omega_x_t0 = -9.5380685844896e-6
omega_y_t0 = -1.1363312657036e-3
omega_z_t0 = 5.3472801108427e-6
r_x_t0 = 2.9963689649816e-3
r_y_t0 = 1.5334477761054e-1
r_z_t0 = 3.8359805613992e-3
h_x_t0 = 5000
h_y_t0 = 5000
h_z_t0 = 5000
h_x_tF = 0
h_y_tF = 0
h_z_tF = 0

# Optimal control problem definition
problem = pycollo.OptimalControlProblem(name="Space Station Attitude Control")
phase = problem.new_phase(name="A",
                          state_variables=[omega_x,
                                           omega_y,
                                           omega_z,
                                           r_x,
                                           r_y,
                                           r_z,
                                           h_x,
                                           h_y,
                                           h_z],
                          control_variables=[u_x, u_y, u_z])

phase.state_equations = {omega_x: domega_x_dt,
                         omega_y: domega_y_dt,
                         omega_z: domega_z_dt,
                         r_x: dr_x_dt,
                         r_y: dr_y_dt,
                         r_z: dr_z_dt,
                         h_x: dh_x_dt,
                         h_y: dh_y_dt,
                         h_z: dh_z_dt}
phase.path_constraints = [h_inner_prod_squared]
phase.integrand_functions = [1e-6 * u_inner_prod_squared]

problem.objective_function = phase.integral_variables[0]
problem.endpoint_constraints = [domega_x_dt_tF,
                                domega_y_dt_tF,
                                domega_z_dt_tF,
                                dr_x_dt_tF,
                                dr_y_dt_tF,
                                dr_z_dt_tF]

# Problem bounds
phase.bounds.initial_time = t0
phase.bounds.final_time = tF
phase.bounds.state_variables = {
    omega_x: [-2e-3, 2e-3],
    omega_y: [-2e-3, 2e-3],
    omega_z: [-2e-3, 2e-3],
    r_x: [-1, 1],
    r_y: [-1, 1],
    r_z: [-1, 1],
    h_x: [-15000, 15000],
    h_y: [-15000, 15000],
    h_z: [-15000, 15000],
}
phase.bounds.initial_state_constraints = {
    omega_x: omega_x_t0,
    omega_y: omega_y_t0,
    omega_z: omega_z_t0,
    r_x: r_x_t0,
    r_y: r_y_t0,
    r_z: r_z_t0,
    h_x: h_x_t0,
    h_y: h_y_t0,
    h_z: h_z_t0,
}
phase.bounds.final_state_constraints = {
    h_x: h_x_tF,
    h_y: h_y_tF,
    h_z: h_z_tF,
}
phase.bounds.control_variables = {
    u_x: [-150, 150],
    u_y: [-150, 150],
    u_z: [-150, 150],
}
phase.bounds.integral_variables = [[0, 10]]
phase.bounds.path_constraints = [[0, h_max**2]]

problem.bounds.endpoint_constraints = [[0, 0],
                                       [0, 0],
                                       [0, 0],
                                       [0, 0],
                                       [0, 0],
                                       [0, 0],
                                       ]

# Problem guesses
phase.guess.time = np.array([t0, tF])
phase.guess.state_variables = np.array([[omega_x_t0, omega_x_t0],
                                        [omega_y_t0, omega_y_t0],
                                        [omega_z_t0, omega_z_t0],
                                        [r_x_t0, r_x_t0],
                                        [r_y_t0, r_y_t0],
                                        [r_z_t0, r_z_t0],
                                        [h_x_t0, h_x_t0],
                                        [h_y_t0, h_y_t0],
                                        [h_z_t0, h_z_t0],
                                        ])
phase.guess.control_variables = np.array([[0, 0], [0, 0], [0, 0]])
phase.guess.integral_variables = np.array([10])


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


J = sym.Matrix([
    [J_00, J_01, J_02],
    [J_10, J_11, J_12],
    [J_20, J_21, J_22]])
J_inv = sym.Matrix([
    [J_inv_00, J_inv_01, J_inv_02],
    [J_inv_10, J_inv_11, J_inv_12],
    [J_inv_20, J_inv_21, J_inv_22]])

# Continuous vectors
omega = sym.Matrix([omega_x, omega_y, omega_z])
r = sym.Matrix([r_x, r_y, r_z])
h = sym.Matrix([h_x, h_y, h_z])
u = sym.Matrix([u_x, u_y, u_z])

# Calculating domega/dt
r_skew_symmetric = skew_symmetric_cross_product_operator(r)
I = sym.eye(3)
D = 2 / (1 + col_vec_dot_row_vec(r.T, r))
E = (r_skew_symmetric * r_skew_symmetric) - r_skew_symmetric
C = I + (D * E)
C_2_skew = skew_symmetric_cross_product_operator(C[:, 2])
tau_gg = 3 * omega_orb**2 * C_2_skew * (J * C[:, 2])
A = J * omega + h
B = skew_symmetric_cross_product_operator(omega) * A
K = tau_gg - B - u
domega_dt = J_inv * K

# Calculating dr/dt
omega_0 = - omega_orb * C[:, 1]
r_sqrd = row_vec_dot_col_vec(r, r.T)
dr_dt = 0.5 * (r_sqrd + I + r_skew_symmetric) * (omega - omega_0)

# Endpoint equations
omega_tF = sym.Matrix(phase.final_state_variables[:3])
r_tF = sym.Matrix(phase.final_state_variables[3:6])
h_tF = sym.Matrix(phase.final_state_variables[6:])

# Calculating domega(tF)/dt
r_tF_skew_symmetric = skew_symmetric_cross_product_operator(r_tF)
D_tF = 2 / (1 + col_vec_dot_row_vec(r_tF.T, r_tF))
E_tF = (r_tF_skew_symmetric * r_tF_skew_symmetric) - r_tF_skew_symmetric
C_tF = I + (D_tF * E_tF)
C_tF_2_skew = skew_symmetric_cross_product_operator(C_tF[:, 2])
tau_gg_tF = 3 * omega_orb**2 * C_tF_2_skew * J * C_tF[:, 2]
A_tF = J * omega_tF + h_tF
B_tF = skew_symmetric_cross_product_operator(omega_tF) * A_tF
K_tF = tau_gg_tF - B_tF
domega_dt_tF = J_inv * K_tF

# Calculating dr(tF)/dt
omega_0_tF = - omega_orb * C_tF[:, 1]
r_tF_sqrd = row_vec_dot_col_vec(r_tF, r_tF.T)
omega_tF_diff = (omega_tF - omega_0_tF)
dr_dt_tF = 0.5 * (r_tF_sqrd + I + r_tF_skew_symmetric) * omega_tF_diff

problem.auxiliary_data = {
    J_00: 2.80701911616e7,
    J_01: 4.822509936e5,
    J_02: -1.71675094448e7,
    J_10: 4.822509936e5,
    J_11: 9.5144639344e7,
    J_12: 6.02604448e4,
    J_20: -1.71675094448e7,
    J_21: 6.02604448e4,
    J_22: 7.6594401336e7,
    J_inv_00: J.inv()[0, 0],
    J_inv_01: J.inv()[0, 1],
    J_inv_02: J.inv()[0, 2],
    J_inv_10: J.inv()[1, 0],
    J_inv_11: J.inv()[1, 1],
    J_inv_12: J.inv()[1, 2],
    J_inv_20: J.inv()[2, 0],
    J_inv_21: J.inv()[2, 1],
    J_inv_22: J.inv()[2, 2],
    omega_orb: 0.06511 * np.pi / 180,
    h_max: 10000,
    u_inner_prod_squared: u_x**2 + u_y**2 + u_z**2,
    h_inner_prod_squared: h_x**2 + h_y**2 + h_z**2,
    domega_x_dt: domega_dt[0, 0],
    domega_y_dt: domega_dt[1, 0],
    domega_z_dt: domega_dt[2, 0],
    dr_x_dt: dr_dt[0, 0],
    dr_y_dt: dr_dt[1, 0],
    dr_z_dt: dr_dt[2, 0],
    dh_x_dt: u_x,
    dh_y_dt: u_y,
    dh_z_dt: u_z,
    domega_x_dt_tF: domega_dt_tF[0, 0],
    domega_y_dt_tF: domega_dt_tF[1, 0],
    domega_z_dt_tF: domega_dt_tF[2, 0],
    dr_x_dt_tF: dr_dt_tF[0, 0],
    dr_y_dt_tF: dr_dt_tF[1, 0],
    dr_z_dt_tF: dr_dt_tF[2, 0],
}

problem.settings.display_mesh_result_graph = True

problem.initialise()
problem.solve()
