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

u_a = sym.Symbol("u_a")
u_b = sym.Symbol("u_b")
u_c = sym.Symbol("u_c")
u_d = sym.Symbol("u_d")

F_a = sym.Symbol("F_a")
F_b = sym.Symbol("F_b")

alpha = sym.Symbol("alpha")
beta = sym.Symbol("beta")

# Auxiliary information
u_a_min = 0
u_a_max = 1000
u_b_min = 0
u_b_max = 1000
u_c_min = 0
u_c_max = 1000
u_d_min = 0
u_d_max = 1000

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

u_a_min = 0
u_a_max = 1000
u_b_min = 0
u_b_max = 1000
u_c_min = 0
u_c_max = 1000
u_d_min = 0
u_d_max = 1000

# Optimal control problem definition
problem = pycollo.OptimalControlProblem(name="Free-Flying Robot")
phase = problem.new_phase(name="A",
                          state_variables=[r_x, r_y, theta, v_x, v_y, omega],
                          control_variables=[u_a, u_b, u_c, u_d])

phase.state_equations = {r_x: v_x,
                         r_y: v_y,
                         theta: omega,
                         v_x: (F_a + F_b) * sym.cos(theta),
                         v_y: (F_a + F_b) * sym.sin(theta),
                         omega: (alpha * F_a) - (beta * F_b)}
phase.integrand_functions = [u_a + u_b + u_c + u_d]
phase.path_constraints = [(u_a + u_b), (u_c + u_d)]

problem.objective_function = phase.integral_variables[0]
problem.auxiliary_data = {alpha: 0.2,
                          beta: 0.2,
                          F_a: u_a - u_b,
                          F_b: u_c - u_d,
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
phase.bounds.control_variables = {u_a: [u_a_min, u_a_max],
                                  u_b: [u_b_min, u_b_max],
                                  u_c: [u_c_min, u_c_max],
                                  u_d: [u_d_min, u_d_max]}
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

problem.settings.display_mesh_result_graph = False
problem.settings.mesh_tolerance = 1e-6
problem.settings.max_mesh_iterations = 15

problem.initialise()
problem.solve()
