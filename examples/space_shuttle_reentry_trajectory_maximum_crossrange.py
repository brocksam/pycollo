import numpy as np
import sympy as sym

import pycollo

h_tilde, phi_tilde, theta_tilde, nu_tilde, gamma_tilde, psi_tilde, alpha_tilde, beta_tilde = sym.symbols('h_tilde phi_tilde theta_tilde nu_tilde gamma_tilde psi_tilde alpha_tilde beta_tilde')
h, phi, theta, nu, gamma, psi, alpha, beta = sym.symbols('h phi theta nu gamma psi alpha beta')

D, L, g, r, rho, rho_0, h_r, c_L, c_D, alpha_hat, Re, S, c_lift_0, c_lift_1, mu, c_drag_0, c_drag_1, c_drag_2, q_r, q_a, c_0, c_1, c_2, c_3 = sym.symbols('D L g r rho rho_0 h_r c_L c_D alpha_hat Re S c_lift_0 c_lift_1 mu c_drag_0 c_drag_1 c_drag_2 q_r q_a c_0 c_1 c_2 c_3')

w, m, g_0 = sym.symbols('w m g_0')

t_0 = 0.0
t_f = None

t_f_min = 0.0
t_f_max = 3000.0

h_0 = 79248
h_f = 24384
phi_0 = 0
phi_f = None
theta_0 = 0
theta_f = None
nu_0 = 7802.88
nu_f = 762
gamma_0 = -1*np.pi/180
gamma_f = -5*np.pi/180
psi_0 = 90*np.pi/180
psi_f = None

h_min = 0
h_max = 300000
phi_min = -np.pi
phi_max = np.pi
theta_min = -70*np.pi/180
theta_max = 70*np.pi/180
nu_min = 10
nu_max = 45000
gamma_min = -80*np.pi/180
gamma_max = 80*np.pi/180
psi_min = -np.pi
psi_max = np.pi
alpha_min = -np.pi/2
alpha_max = np.pi/2
beta_min = -np.pi/2
beta_max = np.pi/180

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

h_0_guess = h_0
h_f_guess = h_f
phi_0_guess = phi_0
phi_f_guess = phi_0 + 10*np.pi/180
theta_0_guess = theta_0
theta_f_guess = theta_0 + 10*np.pi/180
nu_0_guess = nu_0
nu_f_guess = nu_f
gamma_0_guess = gamma_0
gamma_f_guess = gamma_f
psi_0_guess = psi_0
psi_f_guess = -psi_0
alpha_0_guess = 0
alpha_f_guess = 0
beta_0_guess = 0
beta_f_guess = 0

h_0_guess_tilde = h_0_guess/h_stretch + h_shift
h_f_guess_tilde = h_f_guess/h_stretch + h_shift
phi_0_guess_tilde = phi_0_guess/phi_stretch + phi_shift
phi_f_guess_tilde = phi_f_guess/phi_stretch + phi_shift
theta_0_guess_tilde = theta_0_guess/theta_stretch + theta_shift
theta_f_guess_tilde = theta_f_guess/theta_stretch + theta_shift
nu_0_guess_tilde = nu_0_guess/nu_stretch + nu_shift
nu_f_guess_tilde = nu_f_guess/nu_stretch + nu_shift
gamma_0_guess_tilde = gamma_0_guess/gamma_stretch + gamma_shift
gamma_f_guess_tilde = gamma_f_guess/gamma_stretch + gamma_shift
psi_0_guess_tilde = psi_0_guess/psi_stretch + psi_shift
psi_f_guess_tilde = psi_f_guess/psi_stretch + psi_shift
alpha_0_guess_tilde = alpha_0_guess/alpha_stretch + alpha_shift
alpha_f_guess_tilde = alpha_f_guess/alpha_stretch + alpha_shift
beta_0_guess_tilde = beta_0_guess/beta_stretch + beta_shift
beta_f_guess_tilde = beta_f_guess/beta_stretch + beta_shift

problem = pycollo.OptimalControlProblem(name="Space shuttle reentry trajectory maximum crossrange")
phase = problem.new_phase(name="A")
phase.state_variables = [h_tilde, phi_tilde, theta_tilde, nu_tilde, gamma_tilde, psi_tilde]
phase.control_variables = [alpha_tilde, beta_tilde]
phase.state_equations = {
	h_tilde: (1/h_stretch)*(nu*sym.sin(gamma)),
	phi_tilde: (1/phi_stretch)*(nu*sym.cos(gamma)*sym.sin(psi)/(r*sym.cos(theta))),
	theta_tilde: (1/theta_stretch)*(nu*sym.cos(gamma)*sym.cos(psi)/r),
	nu_tilde: (1/nu_stretch)*(- (D/m) - g*sym.sin(gamma)),
	gamma_tilde: (1/gamma_stretch)*(L*sym.cos(beta)/(m*nu) + sym.cos(gamma)*(nu/r - g/nu)),
	psi_tilde: (1/psi_stretch)*(L*sym.sin(beta)/(m*nu*sym.cos(gamma)) + nu*sym.cos(gamma)*sym.sin(psi)*sym.sin(theta)/(r*sym.cos(theta)))
	}

phase.auxiliary_data = {}

phase.bounds.initial_time = [t_0, t_0] 
phase.bounds.final_time = [t_f_min, t_f_max]
phase.bounds.state_variables = {
		h_tilde: [-0.5, 0.5], 
		phi_tilde: [-0.5, 0.5], 
		theta_tilde: [-0.5, 0.5], 
		nu_tilde: [-0.5, 0.5], 
		gamma_tilde: [-0.5, 0.5], 
		psi_tilde: [-0.5, 0.5]
		} 
phase.bounds.control = {
		alpha_tilde: [-0.5, 0.5], 
		beta_tilde: [-0.5, 0.5],
		}
phase.bounds.initial_state_constraints = {
	h_tilde: [h_0_tilde, h_0_tilde], 
	phi_tilde: [phi_0_tilde, phi_0_tilde],
	theta_tilde: [theta_0_tilde, theta_0_tilde],
	nu_tilde: [nu_0_tilde, nu_0_tilde],
	gamma_tilde: [gamma_0_tilde, gamma_0_tilde],
	psi_tilde: [psi_0_tilde, psi_0_tilde],
	}
phase.bounds.final_state_constraints = {
	h_tilde: [h_f_tilde, h_f_tilde],
	nu_tilde: [nu_f_tilde, nu_f_tilde],
	gamma_tilde: [gamma_f_tilde, gamma_f_tilde],
	}

# Guess
phase.guess.time = np.array([t_0, 1000]), 
phase.guess.state_variables = np.array([
		[h_0_guess_tilde, h_f_guess_tilde], 
		[phi_0_guess_tilde, phi_f_guess_tilde], 
		[theta_0_guess_tilde, theta_f_guess_tilde], 
		[nu_0_guess_tilde, nu_f_guess_tilde], 
		[gamma_0_guess_tilde, gamma_f_guess_tilde], 
		[psi_0_guess_tilde, psi_f_guess_tilde],
		])
phase.guess.control_variables = np.array([
		[alpha_0_guess_tilde, alpha_f_guess_tilde], 
		[beta_0_guess_tilde, beta_f_guess_tilde],
		])

problem.objective_function = -theta_stretch*(phase.final_state_variables[2] - theta_shift)

problem.auxiliary_data = {
	rho_0: 1.225570827014494, 
	h_r: 7254.24, 
	Re: 6371203.92, 
	S: 249.9091776, 
	c_lift_0: -0.2070, 
	c_lift_1: 1.6756, 
	mu: 3.986031954093051e14, 
	c_drag_0: 0.07854, 
	c_drag_1: -0.3529, 
	c_drag_2: 2.0400, 
	# c_0: 1.0672181, 
	# c_1: -0.19213774e-1, 
	# c_2: 0.21286289e-3, 
	# c_3: -0.10117249e-5, 
	# w: 203000, 
	# g_0: 32.174, 
	D: 0.5*c_D*S*rho*nu**2, 
	L: 0.5*c_L*S*rho*nu**2, 
	g: mu/(r**2), 
	r: Re + h, 
	rho: rho_0*sym.exp(-h/h_r), 
	c_L: c_lift_0 + c_lift_1*alpha, 
	c_D: c_drag_0 + c_drag_1*alpha + c_drag_2*alpha**2, 
	# alpha_hat: 180*alpha/np.pi,
	# q_r: 17700*sym.sqrt(rho)*(0.0001*nu)**3.07, 
	# q_a: c_0 + c_1*alpha_hat + c_2*alpha_hat**2 + c_3*alpha_hat**3, 
	m: 92079.2525560557,
	h: h_stretch*(h_tilde - h_shift),
	phi: phi_stretch*(phi_tilde - phi_shift),
	theta: theta_stretch*(theta_tilde - theta_shift),
	nu: nu_stretch*(nu_tilde - nu_shift),
	gamma: gamma_stretch*(gamma_tilde - gamma_shift),
	psi: psi_stretch*(psi_tilde - psi_shift),
	alpha: alpha_stretch*(alpha_tilde - alpha_shift),
	beta: beta_stretch*(beta_tilde - beta_shift),
	}

problem.settings.derivative_level = 2
problem.settings.collocation_points_min = 4
problem.settings.display_mesh_result_graph = False
problem.settings.max_mesh_iterations = 5
problem.settings.scaling_method = 'bounds'
problem.settings.max_nlp_iterations = 500
problem.settings.nlp_tolerance = 1e-8
problem.settings.mesh_tolerance = 1e-7

problem.initialise()
problem.solve()
