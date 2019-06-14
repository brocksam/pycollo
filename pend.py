import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import pycollo

# Symbols
q = me.dynamicsymbols('q')
u = me.dynamicsymbols('u')
T = sym.symbols('T')
m, l, I, g, k = sym.symbols('m l I g k')
ml, mgl, sinq, mglsinq = sym.symbols('ml mgl sinq mglsinq')

# Optimal Control Problem
# mesh = col.Mesh(mesh_sections=6, mesh_section_fractions=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1], mesh_collocation_points=[5,7,7,6,6,5])
# mesh = col.Mesh(mesh_sections=3, mesh_section_fractions=[0.2,0.5,0.3], mesh_collocation_points=[2,4,3])
mesh = pycollo.Mesh(mesh_sections=4, mesh_section_fractions=None, mesh_collocation_points=8)
problem = pycollo.OptimalControlProblem(state_variables=[q, u], control_variables=T, initial_mesh=mesh)

problem.parameter_variables = [m, l]

# State equations
problem.state_equations = [u, (m*g*l*sym.sin(q) - T)/I]
# problem.state_equations = [u, (m*g*l*sym.sin(q) - T)/I]

# Integrand functions
problem.integrand_functions = [T**2]

# Objective function
problem.objective_function = problem.integral_variables[0]

# Point constraints
problem.state_endpoint_constraints = [problem.initial_state[0], 
	problem.initial_state[1],
	problem.final_state[0],
	problem.final_state[1]]

problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time=0.0, final_time=(1.0, 2.0), state=[[-4, 4], [-15, 15]], control_lower='-inf', control_upper='inf', integral=[None, 1000], parameter=[[1.0, 2.0], [1.0, 2.0]], state_endpoint_lower=[0, 0, np.pi, 0], state_endpoint_upper=[0, 0, np.pi, 0], infer_bounds=True)

# Guess
# problem.initial_guess = pycollo.Guess(time=np.array([0.0, 1.5]), state=np.array([[0.0, np.pi], [0.0, 0.0]]), control=np.array([[0.0, 0.0]]), integral=[1000])#, parameter=[1.5, 1.5])

# Auxiliary data
problem.auxiliary_data = dict({I: m*(l**2+k**2), k: 1/12, g: -9.81, ml: m*l, mgl: ml*g, sinq: sym.sin(q), mglsinq: mgl*sinq})
# problem.auxiliary_data = dict({I: 1.0, g: -9.81})

# Solve
problem.solve()

