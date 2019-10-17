import sys

sys.path.append('/Users/sambrockie/Documents/Cambridge/PhD/Code/Repositories/pycollo')

import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import pycollo

final_time = 3

# REMEMBER TO HANDLE NO SUBSTITUTIONS

if True:
	# Symbols
	y0, y1, u0 = sym.symbols('y0 y1 u0')
	g = sym.symbols('g')
	m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
	T0min, T0max = sym.symbols('T0min T0max')

	# Optimal Control Problem
	mesh = pycollo.Mesh(mesh_sections=2, mesh_section_fractions=None, mesh_collocation_points=[2,3])
	problem = pycollo.OptimalControlProblem(state_variables=[y0, y1], control_variables=[u0], parameter_variables=[m0, p0], initial_mesh=mesh)

	# State equations
	problem.state_equations = [y1, (g*m0*p0*sym.cos(y0) + (((T0min + T0max)/2) + u0*(T0max - T0min)))/(m0*(k0**2 + p0**2))]

	# Integrand functions
	problem.integrand_functions = [(u0**2)]

	# Objective function
	problem.objective_function = problem.integral_variables[0]

	# Point constraints
	problem.state_endpoint_constraints = [problem.initial_state[0],
		problem.initial_state[1],
		problem.final_state[0],
		problem.final_state[1]]

	# Bounds
	problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time=0, final_time=[1, final_time], state=[[-np.pi, np.pi], [-10, 10]], control=[[-0.5, 0.5]], integral=[0, 1000], parameter=[[0.5, 1.5], [0.5, 1.5]], state_endpoint=[[-np.pi/2, -np.pi/2], [0, 0], [np.pi/2, np.pi/2], [0, 0]])

	# Guess
	problem.initial_guess = pycollo.Guess(optimal_control_problem=problem, time=[0, final_time], state=[[-np.pi/2, np.pi/2], [0, 0]], control=[0, 0], integral=[100], state_endpoints_override=True)

	# Auxiliary data
	problem.auxiliary_data = {g: -9.81, d0: 0.5, k0: 1/12, T0min: -15, T0max: 15}

	# Solve
	problem.solve()































if False:
	# Symbols
	y0, y1, u0 = sym.symbols('y0 y1 u0')
	g = sym.symbols('g')
	m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
	phi0mu, phi0sigma = sym.symbols('phi0mu phi0sigma')
	dphi0mu, dphi0sigma = sym.symbols('dphi0mu dphi0sigma')
	T0mu, T0sigma = sym.symbols('T0mu T0sigma')
	phi0, dphi0, T0 = sym.symbols('phi0 dphi0 T0')
	c0, s0 = sym.symbols('c0 s0')

	# Optimal Control Problem
	mesh = pycollo.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
	problem = pycollo.OptimalControlProblem(state_variables=[y0, y1], control_variables=[u0], initial_mesh=mesh)

	# State equations
	problem.state_equations = [dphi0/phi0sigma, (m0*g*p0*c0 + T0)/(I0*dphi0sigma)]

	# Integrand functions
	problem.integrand_functions = [(T0**2)]

	# Objective function
	problem.objective_function = problem.integral_variables[0]

	# Point constraints
	problem.state_endpoint_constraints = [problem.initial_state[0],
		problem.initial_state[1],
		problem.final_state[0],
		problem.final_state[1]]

	# Bounds
	problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time=0, final_time=final_time, state=[[-0.5, 0.5], [-0.5, 0.5]], control=[-0.5, 0.5], integral=[0, 200], state_endpoint=[[-0.25, -0.25], [0, 0], [0.25, 0.25], [0, 0]])

	# Guess
	problem.initial_guess = pycollo.Guess(optimal_control_problem=problem, time=[0, final_time], state=[[-0.25, 0.25], [0, 0]], control=[0, 0], integral=[10], state_endpoints_override=True)

	phi0min, phi0max = -np.pi, np.pi
	dphi0min, dphi0max = -10, 10
	T0min, T0max = -15, 15
	q0min, q0max = 0, 200

	# Auxiliary data
	# problem.auxiliary_data = {g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, l0: p0 + d0, I0: m0*(k0**2 + p0**2), c0: sym.cos(y0)}
	problem.auxiliary_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, l0: p0 + d0, I0: m0*(k0**2 + p0**2), phi0mu: (phi0min + phi0max)/2, phi0sigma: phi0max - phi0min, dphi0mu: (dphi0min + dphi0max)/2, dphi0sigma: dphi0max - dphi0min, T0mu: (T0min + T0max)/2, T0sigma: T0max - T0min, phi0: phi0mu + y0*phi0sigma, dphi0: dphi0mu + y1*dphi0sigma, T0: T0mu + u0*T0sigma, c0: sym.cos(phi0)})

	# Solve
	problem.solve()




