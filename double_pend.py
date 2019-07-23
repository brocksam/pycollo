import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import pycollo

# Symbols
y0, y1, y2, y3, u0, u1 = sym.symbols('y0 y1 y2 y3 u0 u1')
g = sym.symbols('g')
m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
m1, p1, d1, l1, k1, I1 = sym.symbols('m1 p1 d1 l1 k1 I1')
c0, s0, c1, s1 = sym.symbols('c0 s0 c1 s1')
M00, M01, M10, M11, K0, K1, detM = sym.symbols('M00 M01 M10 M11 K0 K1 detM')

# Optimal Control Problem
mesh = pycollo.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
problem = pycollo.OptimalControlProblem(state_variables=[y0, y1, y2, y3], control_variables=[u0, u1], initial_mesh=mesh)

# State equations
problem.state_equations = [y2, y3, (M11*K0 - M01*K1)/detM, (M00*K1 - M10*K0)/detM]

# Integrand functions
problem.integrand_functions = [(u0**2 + u1**2)]

# Objective function
problem.objective_function = problem.integral_variables[0]

# Point constraints
problem.state_endpoint_constraints = [problem.initial_state[0],
	problem.initial_state[1],
	problem.initial_state[2],
	problem.initial_state[3],
	problem.final_state[0],
	problem.final_state[1],
	problem.final_state[2],
	problem.final_state[3]]

# Bounds
problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time=0, final_time=3, state=[[-np.pi, np.pi], [-np.pi, np.pi], [-10, 10], [-10, 10]], control=[[-15, 15], [-15, 15]], integral=[0, 1000], state_endpoint=[[-np.pi/2, -np.pi/2], [-np.pi/2, -np.pi/2], [0, 0], [0, 0], [np.pi/2, np.pi/2], [np.pi/2, np.pi/2], [0, 0], [0, 0]])

# Guess
problem.initial_guess = pycollo.Guess(optimal_control_problem=problem, time=[0, 3], state=[[-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], [0, 0], [0, 0]], control=[[0, 0], [0, 0]], integral=[100], state_endpoints_override=True)

# Auxiliary data
problem.auxiliary_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, m1: 1.0, p1: 0.5, d1: 0.5, k1: 1/12, l0: p0 + d0, l1: p1 + d1, I0: m0*(k0**2 + p0**2), I1: m1*(k1**2 + p1**2), c0: sym.cos(y0), s0: sym.sin(y0), c1: sym.cos(y1), s1: sym.sin(y1), M00: I0 + m1*l0**2, M01: m1*p1*l0*(s0*s1 + c0*c1), M10: M01, M11: I1, K0: u0 + g*(m0*p0 + m1*l0)*c0 + m1*p1*l0*(s1*c0 - s0*c1)*y3**2, K1: u1 + g*m1*p1*c1 + m1*p1*l0*(s0*c1 - s1*c0)*y2**2, detM: M00*M11 - M01*M10})

# Solve
problem.solve()





















# import numpy as np
# import sympy as sym
# import sympy.physics.mechanics as me

# import pycollo

# # Symbols
# y0, y1, y2, y3, u0, u1 = sym.symbols('y0 y1 y2 y3 u0 u1')
# g = sym.symbols('g')
# m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
# m1, p1, d1, l1, k1, I1 = sym.symbols('m1 p1 d1 l1 k1 I1')
# phi0mu, phi0sigma = sym.symbols('phi0mu phi0sigma')
# phi1mu, phi1sigma = sym.symbols('phi1mu phi1sigma')
# dphi0mu, dphi0sigma = sym.symbols('dphi0mu dphi0sigma')
# dphi1mu, dphi1sigma = sym.symbols('dphi1mu dphi1sigma')
# T0mu, T0sigma = sym.symbols('T0mu T0sigma')
# T1mu, T1sigma = sym.symbols('T1mu T1sigma')
# phi0, phi1, dphi0, dphi1, T0, T1 = sym.symbols('phi0 phi1 dphi0 dphi1 T0 T1')
# c0, s0, c1, s1 = sym.symbols('c0 s0 c1 s1')
# M00, M01, M10, M11, K0, K1, detM = sym.symbols('M00 M01 M10 M11 K0 K1 detM')

# # Optimal Control Problem
# mesh = pycollo.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4)
# problem = pycollo.OptimalControlProblem(state_variables=[y0, y1, y2, y3], control_variables=[u0, u1], initial_mesh=mesh)

# # State equations
# problem.state_equations = [y2, y3, (M11*K0 - M01*K1)/detM, (M00*K1 - M10*K0)/detM]

# # Integrand functions
# problem.integrand_functions = [2*(T0**2 + T1**2)]

# # Objective function
# problem.objective_function = problem.integral_variables[0]

# # Point constraints
# problem.state_endpoint_constraints = [problem.initial_state[0],
# 	problem.initial_state[1],
# 	problem.initial_state[2],
# 	problem.initial_state[3],
# 	problem.final_state[0],
# 	problem.final_state[1],
# 	problem.final_state[2],
# 	problem.final_state[3]]

# # Bounds
# problem.bounds = pycollo.Bounds(optimal_control_problem=problem, initial_time=0, final_time=1, state=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]], control=[[-0.5, 0.5], [-0.5, 0.5]], integral=[0, 1], state_endpoint=[[-0.25, -0.25], [-0.25, -0.25], [0, 0], [0, 0], [0.25, 0.25], [0.25, 0.25], [0, 0], [0, 0]])

# # Guess
# problem.initial_guess = pycollo.Guess(optimal_control_problem=problem, time=[0, 1], state=[[-0.25, 0.25], [-0.25, 0.25], [0, 0], [0, 0]], control=[[0, 0], [0, 0]], integral=[0.5], state_endpoints_override=True)

# # Auxiliary data
# problem.auxiliary_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, m1: 1.0, p1: 0.5, d1: 0.5, k1: 1/12, l0: p0 + d0, l1: p1 + d1, I0: m0*(k0**2 + p0**2), I1: m1*(k1**2 + p1**2), phi0mu: 0, phi0sigma: 2*np.pi, phi1mu: 0, phi1sigma: 2*np.pi, dphi0mu: 0, dphi0sigma: 20, dphi1mu: 0, dphi1sigma: 20, T0mu: 0, T0sigma: 30, T1mu: 0, T1sigma: 30, phi0: phi0mu * y0*phi0sigma, phi1: phi1mu * y1*phi1sigma, dphi0: dphi0mu * y0*dphi0sigma, dphi1: dphi1mu * y1*dphi1sigma, T0: T0mu * y0*T0sigma, T1: T1mu * y1*T1sigma, c0: sym.cos(phi0), s0: sym.sin(phi0), c1: sym.cos(phi1), s1: sym.sin(phi1), M00: I0 + m1*l0**2, M01: m1*p1*l0*(s0*s1 + c0*c1), M10: M01, M11: I1, K0: T0 + g*(m0*p0 + m1*l0)*c0 + m1*p1*l0*(s1*c0 - s0*c1)*dphi1**2, K1: T1 + g*m1*p1*c1 + m1*p1*l0*(s0*c1 - s1*c0)*dphi0**2, detM: M00*M11 - M01*M10})

# # Solve
# problem.solve()








