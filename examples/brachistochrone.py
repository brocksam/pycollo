import numpy as np
import sympy as sym

import pycollo as col

x, y, v, u = sym.symbols('x y v u')

problem = col.OptimalControlProblem(state_variables=[x, y, v], control_variables=u)

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
umin = -np.pi/2
umax = np.pi/2

problem.state_equations = [
	v*sym.sin(u), 
	v*sym.cos(u), 
	g*sym.cos(u),
	]

problem.objective_function = problem.final_time

problem.state_endpoint_constraints = [
	problem.initial_state[0],
	problem.initial_state[1],
	problem.initial_state[2],
	problem.final_state[0],
	problem.final_state[1]]

problem.bounds = col.Bounds(optimal_control_problem=problem, initial_time=0.0, final_time=[tfmin, tfmax], state=[[xmin, xmax], [ymin, ymax], [vmin, vmax]], control=[umin, umax], state_endpoint_lower=[x0, y0, v0, xf, yf], state_endpoint_upper=[x0, y0, v0, xf, yf])

# Guess
problem.initial_guess = col.Guess(optimal_control_problem=problem, time=[t0, tfmax], state=np.array([[x0, xf], [y0, yf], [v0, v0]]), control=np.array([0, umax]), state_endpoints_override=True)

# problem.settings.display_mesh_result_graph = True

problem.solve()