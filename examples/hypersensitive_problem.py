import sys

sys.path.append('/Users/sambrockie/Documents/Cambridge/PhD/Code/Repositories/pycollo')

import numpy as np
import sympy as sym

import pycollo

y, u = sym.symbols('y u')

mesh = pycollo.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
problem = pycollo.OptimalControlProblem(state_variables=y, control_variables=u, initial_mesh=mesh)

problem.state_equations = [-y**3 + u]

problem.integrand_functions = [0.5*(y**2 + u**2)]

problem.objective_function = problem.integral_variables[0]

problem.state_endpoint_constraints = [problem.initial_state[0], 
	problem.final_state[0]]

problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time_lower=0.0, initial_time_upper=0.0, final_time_lower=10000.0, final_time_upper=10000.0, state=[[0, 2]], control_lower=-1, control_upper=8, integral_lower=[0], integral_upper=[2000], state_endpoint_lower=[1.0, 1.5], state_endpoint_upper=[1.0, 1.5])

# Guess
problem.initial_guess = pycollo.Guess(optimal_control_problem=problem, time=[0.0, 10000.0], state=np.array([1.0, 1.5]), control=np.array([0.0, 0.0]), integral=[4], state_endpoints_override=True)

problem.settings.display_mesh_result_graph = True

problem.solve()
