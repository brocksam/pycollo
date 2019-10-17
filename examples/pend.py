import os
import sys

sys.path.insert(0, os.path.abspath('.'))

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