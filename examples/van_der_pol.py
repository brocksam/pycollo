import numpy as np
import sympy as sym

import pycollo as col

y0, y1, u = sym.symbols('y0 y1 u')

mesh = col.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
problem = col.OptimalControlProblem(state_variables=[y0, y1], control_variables=u, initial_mesh=mesh)

problem.state_equations = [(1 - y1**2)*y0 - y1 + u, y0]

problem.integrand_functions = [y0**2 + y1**2 + u**2]

problem.objective_function = problem.integral_variables[0]

problem.state_endpoint_constraints = [problem.initial_state[0], 
	problem.initial_state[1]]

problem.bounds = col.Bounds(optimal_control_problem=problem, initial_time=0.0, final_time=10, state_lower=['-inf', -0.25], state_upper=['inf', 'inf'], control_lower=-1, control_upper=1, integral_lower=[0], integral_upper=[10], state_endpoint_lower=[0, 1], state_endpoint_upper=[0, 1])

# Guess
problem.initial_guess = col.Guess(optimal_control_problem=problem, time=[0.0, 10.0], state=np.array([[1.0, 0.0], [1.0, 0.0]]), control=np.array([0.0, 0.0]), integral=[3], state_endpoints_override=True)

problem.settings.display_mesh_result_graph = True

problem.solve()
