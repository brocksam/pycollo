"""Cart-pole swing-up optimal control problem.

This OCP is taken from from the description found in "Kelly, Matthew (2017). An
Introduction to Trajectory Optimization: How To Do Your Own Direct Collocation.
SIAM Review. Vol. 59, No. 4, pp. 849-904."

An animation of the optimal solution can be found on Matthew Kelly's YouTube at
https://www.youtube.com/watch?v=kAlhKJlu7O8. Credit to Matthew Kelly and his
Matlab software `OptimTraj`, which can be found on his GitHub at
https://github.com/MatthewPeterKelly/OptimTraj, for this solution and
animation.

"""
import matplotlib.pyplot as plt
import numpy as np
import pycollo
import sympy as sym


q1, q2 = sym.symbols("q1, q2")
q1d, q2d = sym.symbols("q1d, q2d")
q1dd, q2dd = sym.symbols("q1dd, q2dd")
u = sym.symbols("u")
m1, m2 = sym.symbols("m1, m2")
l, g = sym.symbols("l, g")

u_max = 20.0
d_max = 2.0
d = 1.0
T = 2.0

problem = pycollo.OptimalControlProblem(name="Cart-Pole Swing-Up")
phase = problem.new_phase(name="A")
phase.state_variables = [q1, q2, q1d, q2d]
phase.control_variables = u
phase.state_equations = [q1d, q2d, q1dd, q2dd]
phase.integrand_functions = [u**2]

phase.bounds.initial_time = 0
phase.bounds.final_time = T
phase.bounds.state_variables = {q1: [-d_max, d_max],
                                q2: [-10, 10],
                                q1d: [-10, 10],
                                q2d: [-10, 10]}
phase.bounds.control_variables = {u: [-u_max, u_max]}
phase.bounds.integral_variables = [[0, 100]]
phase.bounds.initial_state_constraints = {q1: 0,
                                          q2: 0,
                                          q1d: 0,
                                          q2d: 0}
phase.bounds.final_state_constraints = {q1: d,
                                        q2: np.pi,
                                        q1d: 0,
                                        q2d: 0}

phase.guess.time = [0, T]
phase.guess.state_variables = [[0, d], [0, np.pi], [0, 0], [0, 0]]
phase.guess.control_variables = [[0, 0]]
phase.guess.integral_variables = [0]

# These equations are taken from Kelly's paper
q1dd_eqn = (l * m2 * sym.sin(q2) * q2d**2 + u + m2 * g * sym.cos(q2) * sym.sin(q2)) / (m1 + m2 * (1 - sym.cos(q2)**2))
q2dd_eqn = - (l * m2 * sym.cos(q2) * sym.sin(q2) * q2d**2 + u * sym.cos(q2) + (m1 + m2) * g * sym.sin(q2)) / (l * m1 + l * m2 * (1 - sym.cos(q2)**2))

problem.objective_function = phase.integral_variables[0]
problem.auxiliary_data = {g: 9.81,
                          l: 0.5,
                          m1: 1.0,
                          m2: 0.3,
                          q1dd: q1dd_eqn,
                          q2dd: q2dd_eqn,
                          }

problem.initialise()
problem.solve()

time_solution = 0.5 * problem.solution.tau[0] + 0.5
position_solution = problem.solution.state[0][0]
angle_solution = problem.solution.state[0][1]
control_solution = problem.solution.control[0][0]

plt.subplot(3, 1, 1)
plt.plot(time_solution, position_solution, marker="x", color="tab:blue")

plt.subplot(3, 1, 2)
plt.plot(time_solution, angle_solution, marker="x", color="tab:blue")

plt.subplot(3, 1, 3)
plt.plot(time_solution, control_solution, marker="x", color="tab:blue")

plt.show()
