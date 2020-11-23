"""Brachistochrone problem.

Example 4.10 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p215 - 216.

"""

import numpy as np
import sympy as sym

import pycollo

x, y, v, u = sym.symbols('x y v u')

problem = pycollo.OptimalControlProblem(name="Brachistochrone")
phase = problem.new_phase(name="A")
phase.state_variables = [x, y, v]
phase.control_variables = u

g = 9.81
t0 = 0
tfmin = 0
tfmax = 10
x0 = 0
y0 = 0
v0 = 0
xf = 2
yf = 2
xmin = 0
xmax = 10
ymin = 0
ymax = 10
vmin = -50
vmax = 50
umin = -np.pi / 2
umax = np.pi / 2

phase.state_equations = [v * sym.sin(u), v * sym.cos(u), g * sym.cos(u)]

phase.auxiliary_data = {}

problem.objective_function = phase.final_time_variable

phase.bounds.initial_time = 0.0
phase.bounds.final_time = [tfmin, tfmax]
phase.bounds.state_variables = [[xmin, xmax], [ymin, ymax], [vmin, vmax]]
phase.bounds.control_variables = [[umin, umax]]
phase.bounds.initial_state_constraints = {x: x0, y: y0, v: v0}
phase.bounds.final_state_constraints = {x: xf, y: yf}

# Guess
phase.guess.time = np.array([t0, tfmax])
phase.guess.state_variables = np.array([[x0, xf], [y0, yf], [v0, v0]])
phase.guess.control_variables = np.array([[0, umax]])

problem.settings.display_mesh_result_graph = True

problem.initialise()
problem.solve()
