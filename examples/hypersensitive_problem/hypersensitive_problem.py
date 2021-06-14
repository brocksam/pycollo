"""Hypersensitive problem.

Example 4.4 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p170 - 171.

"""

import numpy as np
import sympy as sym

import pycollo

y, u = sym.symbols("y u")

problem = pycollo.OptimalControlProblem(name="Hypersensitive problem")
phase = problem.new_phase(name="A")
phase.state_variables = y
phase.control_variables = u
phase.state_equations = [-y**3 + u]
phase.integrand_functions = [0.5*(y**2 + u**2)]
phase.auxiliary_data = {}

phase.bounds.initial_time = 0.0
phase.bounds.final_time = 10000.0
phase.bounds.state_variables = [[-50, 50]]
phase.bounds.control_variables = [[-50, 50]]
phase.bounds.integral_variables = [[0, 100000]]
phase.bounds.initial_state_constraints = [[1.0, 1.0]]
phase.bounds.final_state_constraints = [[1.5, 1.5]]

phase.guess.time = np.array([0.0, 10000.0])
phase.guess.state_variables = np.array([[1.0, 1.5]])
phase.guess.control_variables = np.array([[0.0, 0.0]])
phase.guess.integral_variables = np.array([4])

problem.objective_function = phase.integral_variables[0]

problem.settings.display_mesh_result_graph = True

problem.initialise()
problem.solve()
