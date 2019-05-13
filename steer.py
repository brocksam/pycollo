import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import opytimise as opyt

# Symbols
y1, y2, y3, y4, y5 = me.dynamicsymbols('y1 y2 y3 y4 y5')
u1, u2 = sym.symbols('u1 u2')
a = sym.symbols('a')
s = sym.symbols('s')

# Optimal Control Problem
problem = opyt.OptimalControlProblem(state_variables=[y1, y2, y3, y4, y5], control_variables=[u1, u2])

# Mesh
problem.mesh = opyt.Mesh(mesh_sections=5, mesh_section_fractions=[0.1, 0.2, 0.4, 0.2, 0.1], mesh_collocation_points=[3,4,5,4,3])

# State equations
problem.state_equations = [y3*y5, y4, a*sym.cos(u2), u1*a*sym.sin(u2), u1]

# Point constraints
# problem.boundary_constraints = [problem.initial_state[0], 
# 	problem.initial_state[1],
# 	problem.final_state[0],
# 	problem.final_state[1]]

# # Integral constraints
problem.integrand_functions = [y1*y2, u2**2]

# Path constraints
problem.path_constraints = [y1+y3, y2**2-u2]

# # Parameters
problem.parameter_variables = s

problem.objective_function = problem.final_time

# Bounds
problem.bounds.initial_time = [0.0, 0.0]
problem.bounds.final_time = [0.0, 10.0]
problem.bounds.state = [['-inf', 'inf'], ['-inf', 'inf'], ['-inf', 'inf'], ['-inf', 'inf'], [1, 1]]
problem.bounds.control = [[1, 1], ['-inf', 'inf']]
problem.bounds.integral = {problem.integral_variables[0]: [1000, 1000], problem.integral_variables[1]: [0, 1000]}
problem.bounds.parameter = [0, 10]
problem.bounds.path = [['-inf', 'inf'], ['-inf', 'inf']]

# Guess
problem.initial_guess = opyt.Guess(time=np.array([0.0, 0.5, 1.0]), state={y1: [0.0, 0.5, 1.0], y2: [0.0, 5.0, 10.0], y3: [0.0, 1.0, 0.0], y4: [0.0, 0.0, 0.0], y5: [0.0, 0.5, 0.0]}, control=[[1.0, 1.5, 2.0], [3.0, 3.5, 4.0]], integral=[10, 20], parameter={s: 10})

# Auxiliary data
problem.auxiliary_data = dict({a: 1.0})

# Solve
problem.solve()