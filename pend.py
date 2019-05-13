import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import opytimise as opyt

# Symbols
q = me.dynamicsymbols('q')
u = me.dynamicsymbols('u')
T = sym.symbols('T')
m, l, I, g = sym.symbols('m l I g')

# Optimal Control Problem
# mesh = opyt.Mesh(mesh_sections=5, mesh_section_fractions=[0.1, 0.2, 0.4, 0.2, 0.1], mesh_collocation_points=[3,4,5,4,3])
mesh = opyt.Mesh(mesh_sections=3, mesh_section_fractions=None, mesh_collocation_points=5)
problem = opyt.OptimalControlProblem(state_variables=[q, u], control_variables=T, initial_mesh=mesh)

# State equations
problem.state_equations = [u, (m*g*l*sym.sin(q) - T)/I]

# Integrand functions
problem.integrand_functions = [T**2]

# Objective function
problem.objective_function = problem.integral_variables[0]

# Point constraints
problem.boundary_constraints = [problem.initial_state[0], 
	problem.initial_state[1],
	problem.final_state[0],
	problem.final_state[1]]

# Bounds
problem.bounds = opyt.Bounds(initial_time_lower=0.0, initial_time_upper=0.0, final_time_lower=1.0, final_time_upper=1.0, state=[['-inf', 'inf'], ['-inf', 'inf']], control_lower='-inf', control_upper='inf', integral_lower=[0], integral_upper=['inf'], boundary_lower=[0, 0, np.pi, 0], boundary_upper=[0, 0, np.pi, 0])

# Guess
problem.initial_guess = opyt.Guess(time=np.array([0.0, 1.0]), state=np.array([[0.0, np.pi], [0.0, 0.0]]), control=np.array([[0.0, 0.0]]), integral=1000)

# Auxiliary data
problem.auxiliary_data = dict({m:1.0, l: 1.0, I: 1.0, g: -9.81})

# Solve
problem.solve()

