import numpy as np
import sympy as sym

import pycollo as col

h_tilde, phi_tilde, theta_tilde, nu_tilde, gamma_tilde, psi_tilde, alpha_tilde, beta_tilde = sym.symbols('h_tilde phi_tilde theta_tilde nu_tilde gamma_tilde psi_tilde alpha_tilde beta_tilde')
h, phi, theta, nu, gamma, psi, alpha, beta = sym.symbols('h phi theta nu gamma psi alpha beta')

D, L, g, r, rho, rho_0, h_r, c_L, c_D, alpha_hat, Re, S, a_0, a_1, mu, b_0, b_1, b_2, q_r, q_a, c_0, c_1, c_2, c_3 = sym.symbols('D L g r rho rho_0 h_r c_L c_D alpha_hat Re S a_0 a_1 mu b_0 b_1 b_2 q_r q_a c_0 c_1 c_2 c_3')

w, m, g_0 = sym.symbols('w m g_0')

# h_nom = 2.6e5
# phi_nom = np.deg2rad(90)
# theta_nom = np.deg2rad(90)
# nu_nom = 25600
# gamma_nom = np.deg2rad(90)
# psi_nom = np.deg2rad(90)
# alpha_nom = np.deg2rad(90)
# beta_nom = np.deg2rad(90)

t_0 = 0.0
t_f = None

t_f_min = 100.0
t_f_max = 4000.0

h_0 = 260000
h_f = 80000
phi_0 = np.deg2rad(0)
phi_f = None
theta_0 = np.deg2rad(0)
theta_f = None
nu_0 = 25600
nu_f = 2500
gamma_0 = np.deg2rad(-1)
gamma_f = np.deg2rad(-5)
psi_0 = np.deg2rad(90)
psi_f = None

h_min = 0
h_max = 300000
phi_min = np.deg2rad(-45)
phi_max = np.deg2rad(45)
theta_min = np.deg2rad(-89)
theta_max = np.deg2rad(89)
nu_min = 1000
nu_max = 40000
gamma_min = np.deg2rad(-89)
gamma_max = np.deg2rad(89)
psi_min = np.deg2rad(-180)
psi_max = np.deg2rad(180)
alpha_min = np.deg2rad(-89)
alpha_max = np.deg2rad(89)
beta_min = np.deg2rad(-90)
beta_max = np.deg2rad(1)

h_stretch = h_max - h_min
h_shift = 0.5 - h_max / h_stretch
phi_stretch = phi_max - phi_min
phi_shift = 0.5 - phi_max / phi_stretch
theta_stretch = theta_max - theta_min
theta_shift = 0.5 - theta_max / theta_stretch
nu_stretch = nu_max - nu_min
nu_shift = 0.5 - nu_max / nu_stretch
gamma_stretch = gamma_max - gamma_min
gamma_shift = 0.5 - gamma_max / gamma_stretch
psi_stretch = psi_max - psi_min
psi_shift = 0.5 - psi_max / psi_stretch
alpha_stretch = alpha_max - alpha_min
alpha_shift = 0.5 - alpha_max / alpha_stretch
beta_stretch = beta_max - beta_min
beta_shift = 0.5 - beta_max / beta_stretch

h_0_tilde = h_0/h_stretch + h_shift
phi_0_tilde = phi_0/phi_stretch + phi_shift
theta_0_tilde = theta_0/theta_stretch + theta_shift
nu_0_tilde = nu_0/nu_stretch + nu_shift
gamma_0_tilde = gamma_0/gamma_stretch + gamma_shift
psi_0_tilde = psi_0/psi_stretch + psi_shift
h_f_tilde = h_f/h_stretch + h_shift
nu_f_tilde = nu_f/nu_stretch + nu_shift
gamma_f_tilde = gamma_f/gamma_stretch + gamma_shift

mesh = col.Mesh(mesh_sections=10, 
	mesh_section_fractions=None, 
	mesh_collocation_points=4)

problem = col.OptimalControlProblem(
	state_variables=[h_tilde, phi_tilde, theta_tilde, nu_tilde, gamma_tilde, psi_tilde], 
	control_variables=[alpha_tilde, beta_tilde], 
	initial_mesh=mesh)

problem.state_equations = [
	(1/h_stretch)*(nu*sym.sin(gamma)),
	(1/phi_stretch)*(nu*sym.cos(gamma)*sym.sin(psi)/(r*sym.cos(theta))),
	(1/theta_stretch)*(nu*sym.cos(gamma)*sym.cos(psi)/r),
	(1/nu_stretch)*(- (D/m) - g*sym.sin(gamma)),
	(1/gamma_stretch)*(L*sym.cos(beta)/(m*nu) + sym.cos(gamma)*(nu/r - g/nu)),
	(1/psi_stretch)*(L*sym.sin(beta)/(m*nu*sym.cos(gamma)) + nu*sym.cos(gamma)*sym.sin(psi)*sym.sin(theta)/(r*sym.cos(theta)))]

problem.auxiliary_data = {
	rho_0: 0.002378, 
	h_r: 23800, 
	Re: 20902900, 
	S: 2690, 
	a_0: -0.20704, 
	a_1: 0.029244, 
	mu: 0.14076539e17, 
	b_0: 0.07854, 
	b_1: -0.61592e-2, 
	b_2: 0.621408e-3, 
	# c_0: 1.0672181, 
	# c_1: -0.19213774e-1, 
	# c_2: 0.21286289e-3, 
	# c_3: -0.10117249e-5, 
	w: 203000, 
	g_0: 32.174, 
	D: 0.5*c_D*S*rho*nu**2, 
	L: 0.5*c_L*S*rho*nu**2, 
	g: mu/(r**2), 
	r: Re + h, 
	rho: rho_0*sym.exp(-h/h_r), 
	c_L: a_0 + a_1*alpha_hat, 
	c_D: b_0 + b_1*alpha_hat + b_2*alpha_hat**2, 
	alpha_hat: 180*alpha/np.pi,
	# q_r: 17700*sym.sqrt(rho)*(0.0001*nu)**3.07, 
	# q_a: c_0 + c_1*alpha_hat + c_2*alpha_hat**2 + c_3*alpha_hat**3, 
	m: w/g_0,
	h: h_stretch*(h_tilde - h_shift),
	phi: phi_stretch*(phi_tilde - phi_shift),
	theta: theta_stretch*(theta_tilde - theta_shift),
	nu: nu_stretch*(nu_tilde - nu_shift),
	gamma: gamma_stretch*(gamma_tilde - gamma_shift),
	psi: psi_stretch*(psi_tilde - psi_shift),
	alpha: alpha_stretch*(alpha_tilde - alpha_shift),
	beta: beta_stretch*(beta_tilde - beta_shift),
	}

problem.objective_function = -(180/np.pi)*theta_stretch*(problem.final_state[2] - theta_shift)

problem.state_endpoint_constraints = [
	problem.initial_state[0], 
	problem.initial_state[1],
	problem.initial_state[2],
	problem.initial_state[3],
	problem.initial_state[4],
	problem.initial_state[5],
	problem.final_state[0],
	problem.final_state[3],
	problem.final_state[4]]

problem.bounds = col.Bounds(optimal_control_problem=problem, 
	initial_time=[t_0, t_0], 
	final_time=[100, 4000],
	state=[
		[-0.5, 0.5], 
		[-0.5, 0.5], 
		[-0.5, 0.5], 
		[-0.5, 0.5], 
		[-0.5, 0.5], 
		[-0.5, 0.5]
		], 
	control=[
		[-0.5, 0.5], 
		[-0.5, 0.5],
		], 
	state_endpoint=[
		[h_0_tilde, h_0_tilde], 
		[phi_0_tilde, phi_0_tilde], 
		[theta_0_tilde, theta_0_tilde], 
		[nu_0_tilde, nu_0_tilde], 
		[gamma_0_tilde, gamma_0_tilde], 
		[psi_0_tilde, psi_0_tilde], 
		[h_f_tilde, h_f_tilde], 
		[nu_f_tilde, nu_f_tilde], 
		[gamma_f_tilde, gamma_f_tilde],
		]
	)

# Guess
problem.initial_guess = col.Guess(optimal_control_problem=problem, 
	time=np.array([t_0, 2000]), 
	state=[
		[h_0_tilde, h_f_tilde], 
		[phi_0_tilde, phi_0_tilde], 
		[theta_0_tilde, theta_0_tilde], 
		[nu_0_tilde, nu_f_tilde], 
		[gamma_0_tilde, gamma_f_tilde], 
		[psi_0_tilde, psi_0_tilde],
		], 
	control=[
		[0, 0], 
		[0, 0],
		],
	state_endpoints_override=True,
	)

problem.settings.derivative_level = 2
problem.settings.collocation_points_min = 4
problem.settings.display_mesh_result_graph = True
problem.settings.max_mesh_iterations = 5
problem.settings.scaling_method = None
problem.settings.max_nlp_iterations = 500
problem.settings.nlp_tolerance=1e-7

problem.solve()