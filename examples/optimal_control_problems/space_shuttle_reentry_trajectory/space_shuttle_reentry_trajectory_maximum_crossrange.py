"""Space Shuttle Reentry Trajectory problem - Maximise Crossrange.

Example 6.1 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p247 - 251.

"""

import numpy as np
import sympy as sym

import pycollo

# Variable symbols
h = sym.Symbol("h")
phi = sym.Symbol("phi")
theta = sym.Symbol("theta")
nu = sym.Symbol("nu")
gamma = sym.Symbol("gamma")
psi = sym.Symbol("psi")
alpha = sym.Symbol("alpha")
beta = sym.Symbol("beta")

# Auxiliary symbols
D = sym.Symbol("D")
L = sym.Symbol("L")
g = sym.Symbol("g")
r = sym.Symbol("r")
rho = sym.Symbol("rho")
rho_0 = sym.Symbol("rho_0")
h_r = sym.Symbol("h_r")
c_L = sym.Symbol("c_L")
c_D = sym.Symbol("c_D")
alpha_hat = sym.Symbol("alpha_hat")
Re = sym.Symbol("Re")
S = sym.Symbol("S")
c_lift_0 = sym.Symbol("c_lift_0")
c_lift_1 = sym.Symbol("c_lift_1")
mu = sym.Symbol("mu")
c_drag_0 = sym.Symbol("c_drag_0")
c_drag_1 = sym.Symbol("c_drag_1")
c_drag_2 = sym.Symbol("c_drag_2")
q_r = sym.Symbol("q_r")
q_a = sym.Symbol("q_a")
c_0 = sym.Symbol("c_0")
c_1 = sym.Symbol("c_1")
c_2 = sym.Symbol("c_2")
c_3 = sym.Symbol("c_3")
w = sym.Symbol("w")
m = sym.Symbol("m")
g_0 = sym.Symbol("g_0")

# Numerical bounds on time endpoints
t_0 = 0.0
t_f = None

# Numerical bounds on times
t_f_min = 0.0
t_f_max = 3000.0

# Numerical bounds on state endpoints
h_0 = 79248
h_f = 24384
phi_0 = 0
phi_f = None
theta_0 = 0
theta_f = None
nu_0 = 7802.88
nu_f = 762
gamma_0 = -1 * np.pi / 180
gamma_f = -5 * np.pi / 180
psi_0 = 90 * np.pi / 180
psi_f = None

# Numerical bounds on states
h_min = 0
h_max = 300000
phi_min = -np.pi
phi_max = np.pi
theta_min = -70 * np.pi / 180
theta_max = 70 * np.pi / 180
nu_min = 10
nu_max = 45000
gamma_min = -80 * np.pi / 180
gamma_max = 80 * np.pi / 180
psi_min = -np.pi
psi_max = np.pi
alpha_min = -np.pi / 2
alpha_max = np.pi / 2
beta_min = -np.pi / 2
beta_max = np.pi / 180

# Numerical guesses for state endpoints
h_0_guess = h_0
h_f_guess = h_f
phi_0_guess = phi_0
phi_f_guess = phi_0 + 10 * np.pi / 180
theta_0_guess = theta_0
theta_f_guess = theta_0 + 10 * np.pi / 180
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

# Set up the Pycollo OCP
ocp_name = "Space shuttle reentry trajectory maximum crossrange"
problem = pycollo.OptimalControlProblem(name=ocp_name)
phase = problem.new_phase(name="A")

# Phase information
phase.state_variables = [h, phi, theta, nu, gamma, psi]
phase.control_variables = [alpha, beta]
dgamma_1 = L * sym.cos(beta) / (m * nu)
dgamma_2 = sym.cos(gamma) * ((nu / r) - (g / nu))
dpsi_1 = L * sym.sin(beta) / (m * nu * sym.cos(gamma))
dpsi_2 = nu * sym.cos(gamma) * sym.sin(psi) * sym.sin(theta)
dpsi_3 = r * sym.cos(theta)
phase.state_equations = {
    h: nu * sym.sin(gamma),
    phi: nu * sym.cos(gamma) * sym.sin(psi) / (r * sym.cos(theta)),
    theta: nu * sym.cos(gamma) * sym.cos(psi) / r,
    nu: - (D / m) - g * sym.sin(gamma),
    gamma: dgamma_1 + dgamma_2,
    psi: dpsi_1 + dpsi_2 / dpsi_3,
}
phase.auxiliary_data = {}

# Problem information
problem.objective_function = -phase.final_state_variables[2]
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
    D: 0.5 * c_D * S * rho * nu**2,
    L: 0.5 * c_L * S * rho * nu**2,
    g: mu / (r**2),
    r: Re + h,
    rho: rho_0 * sym.exp(-h / h_r),
    c_L: c_lift_0 + (c_lift_1 * alpha),
    c_D: c_drag_0 + (c_drag_1 * alpha) + (c_drag_2 * alpha**2),
    # alpha_hat: 180*alpha/np.pi,
    # q_r: 17700*sym.sqrt(rho)*(0.0001*nu)**3.07,
    # q_a: c_0 + c_1*alpha_hat + c_2*alpha_hat**2 + c_3*alpha_hat**3,
    m: 92079.2525560557,
}

# Bounds
phase.bounds.initial_time = [t_0, t_0]
phase.bounds.final_time = [t_f_min, t_f_max]
phase.bounds.state_variables = {h: [h_min, h_max],
                                phi: [phi_min, phi_max],
                                theta: [theta_min, theta_max],
                                nu: [nu_min, nu_max],
                                gamma: [gamma_min, gamma_max],
                                psi: [psi_min, psi_max]
                                }
phase.bounds.control_variables = {alpha: [alpha_min, alpha_max],
                                  beta: [beta_min, beta_max],
                                  }
phase.bounds.initial_state_constraints = {h: h_0,
                                          phi: phi_0,
                                          theta: theta_0,
                                          nu: nu_0,
                                          gamma: gamma_0,
                                          psi: psi_0,
                                          }
phase.bounds.final_state_constraints = {h: [h_f, h_f],
                                        nu: [nu_f, nu_f],
                                        gamma: [gamma_f, gamma_f],
                                        }

# Guess
phase.guess.time = np.array([t_0, 1000])
phase.guess.state_variables = np.array([[h_0_guess, h_f_guess],
                                        [phi_0_guess, phi_f_guess],
                                        [theta_0_guess, theta_f_guess],
                                        [nu_0_guess, nu_f_guess],
                                        [gamma_0_guess, gamma_f_guess],
                                        [psi_0_guess, psi_f_guess],
                                        ])
phase.guess.control_variables = np.array([[alpha_0_guess, alpha_f_guess],
                                          [beta_0_guess, beta_f_guess],
                                          ])

# Settings
problem.settings.display_mesh_result_graph = True

# Initialise and solve
problem.initialise()
problem.solve()
