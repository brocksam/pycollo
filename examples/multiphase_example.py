import numpy as np
import sympy as sym

import pycollo

y0 = sym.Symbol('y0')
y1 = sym.Symbol('y1')
y2 = sym.Symbol('y2')
y3 = sym.Symbol('y3')
u0 = sym.Symbol('u0')
u1 = sym.Symbol('u1')
s0 = sym.Symbol('s0')

problem = pycollo.OptimalControlProblem(
	name="Multiphase example problem", 
	parameter_variables=s0)

phase_A = problem.new_phase(name="A")
phase_A.state_variables = [y0, y1, y2, y3]
phase_A.control_variables = [u0, u1]
phase_A.state_equations = {
	y0: y2,
	y1: y3,
	y2: u0 / s0,
	y3: u1 / s0,
	}
phase_A.path_constraints = [y0**2 + y1**2 - 1]
phase_A.integrand_functions = [u0**2, u1**2]
phase_A.auxiliary_data = {}

phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = 1
phase_A.bounds.state_variables = {
	y0: [-3, 3],
	y1: [-3, 3],
	y2: [-10, 10],
	y3: [-10, 10],
	}
phase_A.bounds.control_variables = {
	u0: [-10, 10],
	u1: [-10, 10],
	}
phase_A.bounds.integral_variables = [[0, 100], [0, 100]]
phase_A.bounds.path_constraints = [0, 10]
phase_A.bounds.initial_state_constraints = {
	y0: 1,
	y1: -2,
	y2: 0,
	y3: 0,
	}
phase_A.bounds.final_state_constraints = {
	y0: 0,
	y1: 2,
	y2: 0,
	y3: 0,
	}

phase_A.guess.time = np.array([0, 1])
phase_A.guess.state_variables = np.array([
	[1, 0],
	[-2, 2],
	[0, 0],
	[0, 0],
	])
phase_A.guess.control_variables = np.array([
	[0, 0],
	[0, 0],
	])
phase_A.guess_integral_variables = np.array([0, 0])

phase_B = problem.new_phase_like(
	phase_for_copying=phase_A,
	name="B",
	)
phase_B.auxiliary_data = {}

phase_B.bounds.initial_time = 1
phase_B.bounds.final_time = 2
phase_B.bounds.initial_state_constraints = {
	y0: 0,
	y1: 2,
	y2: 0,
	y3: 0,
	}
phase_B.bounds.final_state_constraints = {
	y0: -1,
	y1: -2,
	y2: 0,
	y3: 0,
	}

phase_B.guess.time = np.array([1, 2])
phase_B.guess.state_variables = np.array([
	[0, -1],
	[2, -2],
	[0, 0],
	[0, 0],
	])

problem.objective_function = (phase_A.integral_variables[0] 
	+ phase_A.integral_variables[1]
	+ phase_B.integral_variables[0]
	+ phase_B.integral_variables[1])

problem.bounds.parameter_variables = [[1, 2]]

problem.guess.parameter_variables = np.array([1.5])

problem.settings.nlp_tolerance = 10e-7
problem.settings.mesh_tolerance = 10e-6
problem.settings.maximise_objective = False
problem.settings.backend = "pycollo"
problem.settings.scaling_method = "bounds"
problem.settings.assume_inf_bounds = False
problem.settings.inf_value = 1e16

problem.initialise()

