import numpy as np
import sympy as sym

import pycollo as pycollo

h, v, m, u = sym.symbols('h v m u')
g, alpha = sym.symbols('g alpha')

initial_mesh = pycollo.Mesh(mesh_sections=10, 
	mesh_section_fractions=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1000],
	mesh_collocation_points=8)

problem = pycollo.OptimalControlProblem(
	state_variables=[h, v, m], 
	control_variables=u,
	initial_mesh=initial_mesh)

problem.state_equations = [v, u/m - g, -alpha*u]

problem.objective_function = problem.initial_state[2] - problem.final_state[2]

problem.state_endpoint_constraints = [
	problem.initial_state[0],
	problem.initial_state[1],
	problem.initial_state[2],
	problem.final_state[0],
	]

h0 = 0
hf = 100000
v0 = 0
vmax = 2000
m0 = 500000
mmin = 0
umin = 0
umax = 1000000000

problem.auxiliary_data = {g: 9.81, alpha: 1/(300*g)}

problem.bounds = pycollo.Bounds(optimal_control_problem=problem,
	initial_time=[0, 0],
	final_time=[100, 100],
	state=[[h0, hf], [v0, vmax], [mmin, m0]],
	control=[umin, umax],
	state_endpoint=[[h0, h0], [v0, v0], [m0, m0], [hf, hf]])

problem.initial_guess = pycollo.Guess(optimal_control_problem=problem,
	time=[0, 100],
	state=[[h0, hf], [v0, vmax], [m0, mmin]],
	control=[umax, umin],
	state_endpoints_override=True)

problem.settings.display_mesh_result_graph = True
problem.settings.max_mesh_iterations = 10
problem.settings.scaling_method = 'bounds'

problem.solve()
