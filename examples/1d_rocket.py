import numpy as np
import sympy as sym

import pycollo as pycollo

h, v, m, u = sym.symbols('h v m u')
g, alpha = sym.symbols('g alpha')

h_nom = 1e5
v_nom = 2000
m_nom = 500e3
u_nom = 2e8

h_0 = 0
h_f = h_nom
v_0 = 0
v_f = None
m_0 = m_nom
m_f = None

h_min = 0
h_max = h_nom
v_min = 0
v_max = v_nom
m_min = 0
m_max = m_nom
u_min = 0
u_max = u_nom

h_stretch = h_max - h_min
h_shift = 0.5 - h_max / h_stretch
v_stretch = v_max - v_min
v_shift = 0.5 - v_max / v_stretch
m_stretch = m_max - m_min
m_shift = 0.5 - m_max / m_stretch
u_stretch = u_max - u_min
u_shift = 0.5 - u_max / u_stretch

initial_mesh = pycollo.Mesh(mesh_sections=50, 
mesh_collocation_points=3)

scaled = True

if scaled:

	h_tilde, v_tilde, m_tilde, u_tilde = sym.symbols('h_tilde v_tilde m_tilde u_tilde')	

	problem = pycollo.OptimalControlProblem(
		state_variables=[h_tilde, v_tilde, m_tilde], 
		control_variables=u_tilde,
		initial_mesh=initial_mesh)

	problem.state_equations = [
		(1/h_stretch)*v, 
		(1/v_stretch)*(u/m - g), 
		(1/m_stretch)*(-alpha*u)]

	problem.objective_function = (problem.initial_state[2] - problem.final_state[2])

	problem.state_endpoint_constraints = [
		problem.initial_state[0],
		problem.initial_state[1],
		problem.initial_state[2],
		problem.final_state[0],
		]

	problem.auxiliary_data = {g: 9.81, alpha: 1/(300*g), 
		h: h_stretch*(h_tilde - h_shift),
		v: v_stretch*(v_tilde - v_shift),
		m: m_stretch*(m_tilde - m_shift),
		u: u_stretch*(u_tilde - u_shift),
		}

	problem.bounds = pycollo.Bounds(optimal_control_problem=problem,
		initial_time=[0, 0],
		final_time=[100, 100],
		state=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
		control=[-0.5, 0.5],
		state_endpoint=[[-0.5, -0.5], [-0.5, -0.5], [0.5, 0.5], [0.5, 0.5]]
		)

	problem.initial_guess = pycollo.Guess(optimal_control_problem=problem,
		time=[0, 100],
		state=[[-0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]],
		control=[0.5, -0.5],
		state_endpoints_override=True)

else:

	problem = pycollo.OptimalControlProblem(
		state_variables=[h, v, m], 
		control_variables=u,
		initial_mesh=initial_mesh)

	problem.state_equations = [
		v, 
		u/m - g, 
		-alpha*u]

	problem.objective_function = (problem.initial_state[2] - problem.final_state[2])

	problem.state_endpoint_constraints = [
		problem.initial_state[0],
		problem.initial_state[1],
		problem.initial_state[2],
		problem.final_state[0],
		]

	problem.auxiliary_data = {g: 9.81, alpha: 1/(300*g)}

	problem.bounds = pycollo.Bounds(optimal_control_problem=problem,
		initial_time=[0, 0],
		final_time=[100, 100],
		state=[[h_min, h_max], [v_min, v_max], [m_min, m_max]],
		control=[u_min, u_max],
		state_endpoint=[[h_0, h_0], [v_0, v_0], [m_0, m_0], [h_f, h_f]])

	problem.initial_guess = pycollo.Guess(optimal_control_problem=problem,
		time=[0, 100],
		state=[[h_0, h_f], [v_0, v_f], [m_0, m_f]],
		control=[u_max, u_min],
		state_endpoints_override=True)

problem.settings.collocation_points_min = 2
problem.settings.display_mesh_result_graph = True
problem.settings.max_mesh_iterations = 3
problem.settings.scaling_method = None

problem.solve()


