import numpy as np
import sympy as sym

import pycollo as col

y, u = sym.symbols('y u')

mesh_secs = 20
mesh_sec_fracs = np.zeros(mesh_secs)
mesh_sec_fracs[round(0.0*mesh_secs):round(0.1*mesh_secs)] = 1
mesh_sec_fracs[round(0.1*mesh_secs):round(0.2*mesh_secs)] = 10
mesh_sec_fracs[round(0.2*mesh_secs):round(0.3*mesh_secs)] = 100
mesh_sec_fracs[round(0.3*mesh_secs):round(0.4*mesh_secs)] = 1000
mesh_sec_fracs[round(0.4*mesh_secs):round(0.5*mesh_secs)] = 10000
mesh_sec_fracs[round(0.5*mesh_secs):round(0.6*mesh_secs)] = 10000
mesh_sec_fracs[round(0.6*mesh_secs):round(0.7*mesh_secs)] = 1000
mesh_sec_fracs[round(0.7*mesh_secs):round(0.8*mesh_secs)] = 100
mesh_sec_fracs[round(0.8*mesh_secs):round(0.9*mesh_secs)] = 10
mesh_sec_fracs[round(0.9*mesh_secs):round(1.0*mesh_secs)] = 1

mesh = col.Mesh(mesh_sections=mesh_secs, mesh_section_fractions=mesh_sec_fracs, mesh_collocation_points=9)
# mesh = col.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
problem = col.OptimalControlProblem(state_variables=y, control_variables=u, initial_mesh=mesh)

problem.state_equations = [-y**3 + u]

problem.integrand_functions = [0.5*(y**2 + u**2)]

problem.objective_function = problem.integral_variables[0]

problem.boundary_constraints = [problem.initial_state[0], 
	problem.final_state[0]]

problem.bounds = col.Bounds(initial_time_lower=0.0, initial_time_upper=0.0, final_time_lower=10000.0, final_time_upper=10000.0, state=[[0, 2]], control_lower=-1, control_upper=8, integral_lower=[0], integral_upper=[2000], boundary_lower=[1, 1.5], boundary_upper=[1, 1.5])

# Guess
problem.initial_guess = col.Guess(time=np.array([0.0, 10000.0]), state=np.array([[1.0, 1.5]]), control=np.array([[0.0, 0.0]]), integral=[4])

problem.solve()