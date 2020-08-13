import sys

import numpy as np
import sympy as sym

import pycollo

y, u = sym.symbols('y u')

problem = pycollo.OptimalControlProblem(name="Hypersensitive problem")
phase = problem.new_phase(name="A")
phase.state_variables = y
phase.control_variables = u
phase.state_equations = [-y**3 + u]
phase.integrand_functions = [0.5*(y**2 + u**2)]
phase.auxiliary_data = {}

phase.bounds.initial_time = 0.0
phase.bounds.final_time = 10000.0
phase.bounds.state_variables = [[0, 2]]
phase.bounds.control_variables = [[-1, 8]]
phase.bounds.integral_variables = [[0, 2000]]
phase.bounds.initial_state_constraints = [[1.0, 1.0]]
phase.bounds.final_state_constraints = [[1.5, 1.5]]

phase.guess.time = np.array([0.0, 10000.0])
phase.guess.state_variables = np.array([[1.0, 1.5]])
phase.guess.control_variables = np.array([[0.0, 0.0]])
phase.guess.integral_variables = np.array([4])

problem.objective_function = phase.integral_variables[0]

problem.settings.display_mesh_result_graph = True
problem.settings.derivative_level = 2
# problem.settings.scaling_method = None
problem.settings.quadrature_method = "lobatto"
problem.settings.max_mesh_iterations = 10

problem.initialise()
problem.solve()
