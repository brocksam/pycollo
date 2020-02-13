import numpy as np
import sympy as sym

import pycollo as col

h, phi, theta, nu, gamma, psi, alpha, beta = sym.symbols('h phi theta nu gamma psi alpha beta')

D, L, g, r, rho, rho_0, h_r, c_L, c_D, alpha_hat, Re, S, a_0, a_1, mu, b_0, b_1, b_2, q_r, q_a, c_0, c_1, c_2, c_3 = sym.symbols('D L g r rho rho_0 h_r c_L c_D alpha_hat Re S a_0 a_1 mu b_0 b_1 b_2 q_r q_a c_0 c_1 c_2 c_3')

w, m, g_0 = sym.symbols('w m g_0')

mesh = col.Mesh(mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=9)
problem = col.OptimalControlProblem(state_variables=[h, phi, theta, nu, gamma, psi], control_variables=[alpha, beta], initial_mesh=mesh)

problem.state_equations = [nu*sym.sin(gamma),
	nu*sym.cos(gamma)*sym.sin(psi)/(r*sym.cos(theta)),
	nu*sym.cos(gamma)*sym.cos(psi)/r,
	- D/m - g*sym.sin(gamma),
	L*sym.cos(beta)/(m*nu) + sym.cos(gamma)*(nu/r - g/nu),
	L*sym.sin(beta)/(m*nu*sym.cos(gamma)) + nu*sym.cos(gamma)*sym.sin(psi)*sym.sin(theta)/(r*sym.cos(theta))]

problem.auxiliary_data = {rho_0: 0.002378, h_r: 23800, Re: 20902900, S: 2690, a_0: -0.20704, a_1: 0.029244, mu: 0.14076539e17, b_0: 0.07854, b_1: -0.61592e-2, b_2: 0.621408e-3, c_0: 1.0672181, c_1: -0.19213774e-1, c_2: 0.21286289e-3, c_3: -0.10117249e-5, w: 203000, g_0: 32.174, D: 0.5*c_D*S*rho*nu**2, L: 0.5*c_L*S*rho*nu**2, g: mu/r**2, r: Re + h, rho: rho_0*sym.exp(-h/h_r), c_L: a_0 + a_1*alpha_hat, c_D: b_0 + b_1*alpha_hat + b_2*alpha_hat**2, alpha_hat: 180*alpha/np.pi, q_r: 17700*sym.sqrt(rho)*(0.0001*nu)**3.07, q_a: c_0 + c_1*alpha_hat + c_2*alpha_hat**2 + c_3*alpha_hat**3, m: w/g_0}

problem.objective_function = -problem.final_state[2]

problem.state_endpoint_constraints = [problem.initial_state[0], 
	problem.initial_state[1],
	problem.initial_state[2],
	problem.initial_state[3],
	problem.initial_state[4],
	problem.initial_state[5],
	problem.final_state[0],
	problem.final_state[3],
	problem.final_state[4]]

problem.bounds = col.Bounds(optimal_control_problem=problem, initial_time=[0.0, 0.0], final_time=[0.0, 3000.0], state={h: [0, 80000], phi: np.deg2rad([-89, 89]), theta: np.deg2rad([-89, 89]), nu: [1, 10000], gamma: np.deg2rad([-89, 89]), psi: np.deg2rad([-179, 179])}, control={alpha: np.deg2rad([-89, 89]), beta: np.deg2rad([-89, 1])}, state_endpoint=[[260000, 260000], [0, 0], [0, 0], [25600, 25600], np.deg2rad([-1, -1]), np.deg2rad([90, 90]), [80000, 80000], [2500, 2500], np.deg2rad([-89, 89])])

# Guess
problem.initial_guess = col.Guess(optimal_control_problem=problem, time=np.array([0.0, 1000.0]), state={h: [0, 80000], phi: np.deg2rad([0, 10]), theta: np.deg2rad([0, 10]), nu: [2000, 70], gamma: np.deg2rad([-1, -5]), psi: np.deg2rad([90, -90])}, control={alpha: [0, 0], beta: [0, 0]})

problem.solve()