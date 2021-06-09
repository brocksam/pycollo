"""Free-Flying Robot problem.

Example 6.13 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p326 - 330.

Attributes
----------
r_x : `Symbol <sympy>`
    First inertial coordinate of the centre of gravity (in m)
r_y : `Symbol <sympy>`
    Second inertial coordinate of the centre of gravity (in m)
theta : `Symbol <sympy>`
    Thrust direction (in rad)
v_x : `Symbol <sympy>`
    First inertial velocity of the centre of gravity (in m/s)
v_y : `Symbol <sympy>`
    Second inertial velocity of the centre of gravity  (in m/s)
omega : `Symbol <sympy>`
    Thurst angular velocity (in rad/s)
u_x_pos : `Symbol <sympy>`
    Positive component of the thrust in the x-direction (in N)
u_x_neg : `Symbol <sympy>`
    Negative component of the thrust in the x-direction (in N)
u_y_pos : `Symbol <sympy>`
    Positive component of the thrust in the y-direction (in N)
u_y_neg : `Symbol <sympy>`
    Negative component of the thrust in the y-direction (in N)
T_x : `Symbol <sympy>`
    Total thrust in the x-direction (in N)
T_y : `Symbol <sympy>`
    Total thrust in the y-direction (in N)
I_xx : `Symbol <sympy>`
    Moment of inertia about the x-axis (in kg/m^2)
I_yy : `Symbol <sympy>`
    Moment of inertia about the y-axis (in kg/m^2)

"""

import numpy as np
import pycollo
import sympy as sym

# Symbol creation
r_x = sym.Symbol("r_x")
r_y = sym.Symbol("r_y")
theta = sym.Symbol("theta")
v_x = sym.Symbol("v_x")
v_y = sym.Symbol("v_y")
omega = sym.Symbol("omega")

u_x_pos = sym.Symbol("u_x_pos")
u_x_neg = sym.Symbol("u_x_neg")
u_y_pos = sym.Symbol("u_y_pos")
u_y_neg = sym.Symbol("u_y_neg")

T_x = sym.Symbol("T_x")
T_y = sym.Symbol("T_y")

I_xx = sym.Symbol("I_xx")
I_yy = sym.Symbol("I_yy")

# Auxiliary information
u_x_pos_min = 0
u_x_pos_max = 1000
u_x_neg_min = 0
u_x_neg_max = 1000
u_y_pos_min = 0
u_y_pos_max = 1000
u_y_neg_min = 0
u_y_neg_max = 1000

t0 = 0.0
tF = 12.0

r_x_t0 = -10
r_x_tF = 0
r_y_t0 = -10
r_y_tF = 0
theta_t0 = np.pi / 2
theta_tF = 0
v_x_t0 = 0
v_x_tF = 0
v_y_t0 = 0
v_y_tF = 0
omega_t0 = 0
omega_tF = 0

r_x_min = -10
r_x_max = 10
r_y_min = -10
r_y_max = 10
theta_min = -np.pi
theta_max = np.pi
v_x_min = -2
v_x_max = 2
v_y_min = -2
v_y_max = 2
omega_min = -1
omega_max = 1

u_x_pos_min = 0
u_x_pos_max = 1000
u_x_neg_min = 0
u_x_neg_max = 1000
u_y_pos_min = 0
u_y_pos_max = 1000
u_y_neg_min = 0
u_y_neg_max = 1000

# Optimal control problem definition
problem = pycollo.OptimalControlProblem(name="Free-Flying Robot")
phase = problem.new_phase(name="A",
                          state_variables=[r_x, r_y, theta, v_x, v_y, omega],
                          control_variables=[u_x_pos,
                                             u_x_neg,
                                             u_y_pos,
                                             u_y_neg])

phase.state_equations = {r_x: v_x,
                         r_y: v_y,
                         theta: omega,
                         v_x: (T_x + T_y) * sym.cos(theta),
                         v_y: (T_x + T_y) * sym.sin(theta),
                         omega: (I_xx * T_x) - (I_yy * T_y)}
phase.integrand_functions = [u_x_pos + u_x_neg + u_y_pos + u_y_neg]
phase.path_constraints = [(u_x_pos + u_x_neg), (u_y_pos + u_y_neg)]

problem.objective_function = phase.integral_variables[0]
problem.auxiliary_data = {I_xx: 0.2,
                          I_yy: 0.2,
                          T_x: u_x_pos - u_x_neg,
                          T_y: u_y_pos - u_y_neg,
                          }

# Problem bounds
phase.bounds.initial_time = t0
phase.bounds.final_time = tF
phase.bounds.state_variables = {r_x: [r_x_min, r_x_max],
                                r_y: [r_y_min, r_y_max],
                                theta: [theta_min, theta_max],
                                v_x: [v_x_min, v_x_max],
                                v_y: [v_y_min, v_y_max],
                                omega: [omega_min, omega_max]}
phase.bounds.initial_state_constraints = {r_x: [r_x_t0, r_x_t0],
                                          r_y: [r_y_t0, r_y_t0],
                                          theta: [theta_t0, theta_t0],
                                          v_x: [v_x_t0, v_x_t0],
                                          v_y: [v_y_t0, v_y_t0],
                                          omega: [omega_t0, omega_t0]}
phase.bounds.final_state_constraints = {r_x: [r_x_tF, r_x_tF],
                                        r_y: [r_y_tF, r_y_tF],
                                        theta: [theta_tF, theta_tF],
                                        v_x: [v_x_tF, v_x_tF],
                                        v_y: [v_y_tF, v_y_tF],
                                        omega: [omega_tF, omega_tF]}
phase.bounds.control_variables = {u_x_pos: [u_x_pos_min, u_x_pos_max],
                                  u_x_neg: [u_x_neg_min, u_x_neg_max],
                                  u_y_pos: [u_y_pos_min, u_y_pos_max],
                                  u_y_neg: [u_y_neg_min, u_y_neg_max]}
phase.bounds.integral_variables = [[0, 100]]
phase.bounds.path_constraints = [[-1000, 1], [-1000, 1]]

# Problem guesses
phase.guess.time = [t0, tF]
phase.guess.state_variables = [[r_x_t0, r_x_tF],
                               [r_y_t0, r_y_tF],
                               [theta_t0, theta_tF],
                               [v_x_t0, v_x_tF],
                               [v_y_t0, v_y_tF],
                               [omega_t0, omega_tF]]
phase.guess.control_variables = [[0, 0], [0, 0], [0, 0], [0, 0]]
phase.guess.integral_variables = [0]

problem.settings.display_mesh_result_graph = True
problem.settings.mesh_tolerance = 1e-7
problem.settings.max_mesh_iterations = 25

problem.initialise()
problem.solve()
