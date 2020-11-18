"""Delta III Launch Vehicle Ascent problem.

Example 6.15 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p336 - 345.

Attributes
----------

r_x : `Symbol <sympy>`
    Cartesian x-vector position relative to the centre of the earth
        using earth-centered inertial (ECI) coordinates (in m)
r_y : `Symbol <sympy>`
    Cartesian y-vector position using ECI coordinates (in m)
r_z : `Symbol <sympy>`
    Cartesian z-vector position using ECI coordinates (in m)
v_x : `Symbol <sympy>`
    Cartesian x-vector velocity using ECI coordinates (in m/s)
v_y : `Symbol <sympy>`
    Cartesian y-vector velocity using ECI coordinates (in m/s)
v_z : `Symbol <sympy>`
    Cartesian z-vector velocity using ECI coordinates (in m/s)
m : `Symbol <sympy>`
    Total mass (in kg)
u_x : `Symbol <sympy>`
    Cartesian x-vector inertial trust (N)
u_y : `Symbol <sympy>`
    Cartesian y-vector inertial trust (N)
u_z : `Symbol <sympy>`
    Cartesian z-vector inertial trust (N)
mu : `Symbol <sympy>`
    Gravitational parameter
R_E : `Symbol <sympy>`
    Radius of Earth (in m)
r_vec_norm : `Symbol <sympy>`
    Absolute position (in m)
v_vec_norm : `Symbol <sympy>`
    Absolute velocity (in m/s)
u_vec_norm : `Symbol <sympy>`
    Absolute control (in N)
D_x : `Symbol <sympy>`
    Drag for in x-direction (in N)
D_y : `Symbol <sympy>`
    Drag for in y-direction (in N)
D_z : `Symbol <sympy>`
    Drag for in z-direction (in N)
C_D : `Symbol <sympy>`
    Coefficient of drag (nondimensional)
S_rho : `Symbol <sympy>`
    Drag reference area (in m^3)
v_r_vec_norm : `Symbol <sympy>`
    Absolute Earth relative velocity (in m/s)
v_r_x : `Symbol <sympy>`
    Absolute Earth relative velocity in x-direction (in m/s)
v_r_y : `Symbol <sympy>`
    Absolute Earth relative velocity in y-direction (in m/s)
v_r_z : `Symbol <sympy>`
    Absolute Earth relative velocity in z-direction (in m/s)
omega_x_r_x : `Symbol <sympy>`
    x-component of cross product of `omega` and `r`
omega_x_r_y : `Symbol <sympy>`
    y-component of cross product of `omega` and `r`
omega_x_r_z : `Symbol <sympy>`
    z-component of cross product of `omega` and `r`
g_0 : `Symbol <sympy>`
    Accerlation due to gravity at sea level (in m/s^2)
h_0 : `Symbol <sympy>`
    Density scale height (in m)
h : `Symbol <sympy>`
    Absolute altitude above sea level (in m)
rho : `Symbol <sympy>`
    Atmospheric density (in kg/m^3)
rho_0 : `Symbol <sympy>`
    Atmospheric density at sea level (in kg/m^3)
omega_E : `Symbol <sympy>`
    Angular velocity of earth relative to inertial space (in rad/s)
m_tot_S : `Symbol <sympy>`
    Total mass of solid boosters (in kg)
m_tot_1 : `Symbol <sympy>`
    Total mass of stage 1 (in kg)
m_tot_2 : `Symbol <sympy>`
    Total mass of stage 2 (in kg)
m_prop_S : `Symbol <sympy>`
    Propellant mass of solid boosters (in kg)
m_prop_1 : `Symbol <sympy>`
    Propellant mass of stage 1 (in kg)
m_prop_2 : `Symbol <sympy>`
    Propellant mass of stage 2 (in kg)
m_struct_S : `Symbol <sympy>`
    Structure mass of solid boosters (in kg)
m_struct_1 : `Symbol <sympy>`
    Structure mass of stage 1 (in kg)
m_struct_2 : `Symbol <sympy>`
    Structure mass of stage 2 (in kg)
T_eng_S : `Symbol <sympy>`
    Engine thrust of solid boosters (in N)
T_eng_1 : `Symbol <sympy>`
    Engine thrust of stage 1 (in N)
T_eng_2 : `Symbol <sympy>`
    Engine thrust of stage 2 (in N)
I_S : `Symbol <sympy>`
    Specific impulse of solid boosters (in s)
I_1 : `Symbol <sympy>`
    Specific impulse of stage 1 (in s)
I_2 : `Symbol <sympy>`
    Specific impulse of stage 2 (in s)
tau_burn_S : `Symbol <sympy>`
    Burn time of solid boosters (in s)
tau_burn_1 : `Symbol <sympy>`
    Burn time of stage 1 (in s)
tau_burn_2 : `Symbol <sympy>`
    Burn time of stage 2 (in s)
m_payload : `Symbol <sympy>`
    Mass of payload (in kg)
T_over_m : `Symbol <sympy>`
    `T` divided by `m`
psi_L : `Symbol <sympy>`
    (Geocentric) latitude of Cape Canaveral launch site (in rad)

"""

import numpy as np
import pycollo
import sympy as sym

# Define all symbols for building problem equations
r_x = sym.Symbol("r_x")
r_y = sym.Symbol("r_y")
r_z = sym.Symbol("r_z")
v_x = sym.Symbol("v_x")
v_y = sym.Symbol("v_y")
v_z = sym.Symbol("v_z")
m = sym.Symbol("m")
u_x = sym.Symbol("u_x")
u_y = sym.Symbol("u_y")
u_z = sym.Symbol("u_z")
D_x = sym.Symbol("D_x")
D_y = sym.Symbol("D_y")
D_z = sym.Symbol("D_z")
T = sym.Symbol("T")
xi = sym.Symbol("xi")
C_D = sym.Symbol("C_D")
S_rho = sym.Symbol("S_rho")
omega_E = sym.Symbol("omega_E")
v_r_x = sym.Symbol("v_r_x")
v_r_y = sym.Symbol("v_r_y")
v_r_z = sym.Symbol("v_r_z")
omega_x_r_x = sym.Symbol("omega_x_r_x")
omega_x_r_y = sym.Symbol("omega_x_r_y")
omega_x_r_z = sym.Symbol("omega_x_r_z")
mu = sym.Symbol("mu")
R_E = sym.Symbol("R_E")
psi_L = sym.Symbol("psi_L")
g_0 = sym.Symbol("g_0")
h_0 = sym.Symbol("h_0")
h = sym.Symbol("h")
rho = sym.Symbol("rho")
rho_0 = sym.Symbol("rho_0")
r_vec_norm = sym.Symbol("r_vec_norm")
u_vec_norm = sym.Symbol("u_vec_norm")
v_vec_norm = sym.Symbol("v_vec_norm")
v_r_vec_norm = sym.Symbol("v_r_vec_norm")
I_S = sym.Symbol("I_S")
I_1 = sym.Symbol("I_1")
I_2 = sym.Symbol("I_2")
m_tot_S = sym.Symbol("m_tot_S")
m_tot_1 = sym.Symbol("m_tot_1")
m_tot_2 = sym.Symbol("m_tot_2")
m_payload = sym.Symbol("m_payload")
m_prop_S = sym.Symbol("m_prop_S")
m_prop_1 = sym.Symbol("m_prop_1")
m_prop_2 = sym.Symbol("m_prop_2")
m_struct_S = sym.Symbol("m_struct_S")
m_struct_1 = sym.Symbol("m_struct_1")
m_struct_2 = sym.Symbol("m_struct_2")
tau_burn_S = sym.Symbol("tau_burn_S")
tau_burn_1 = sym.Symbol("tau_burn_1")
tau_burn_2 = sym.Symbol("tau_burn_2")
T_eng_S = sym.Symbol("T_eng_S")
T_eng_1 = sym.Symbol("T_eng_1")
T_eng_2 = sym.Symbol("T_eng_2")
T_over_m = sym.Symbol("T_over_m")
scale_length = sym.Symbol("scale_length")
scale_speed = sym.Symbol("scale_speed")
scale_time = sym.Symbol("scale_time")
scale_acceleration = sym.Symbol("scale_acceleration")
scale_mass = sym.Symbol("scale_mass")
scale_force = sym.Symbol("scale_force")
scale_area = sym.Symbol("scale_area")
scale_volume = sym.Symbol("scale_volume")
scale_density = sym.Symbol("scale_density")
scale_gravity = sym.Symbol("scale_gravity")

problem_name = "Multi-stage launch vehicle ascent problem"
problem = pycollo.OptimalControlProblem(name=problem_name)

phase_A = problem.new_phase("A")
phase_A.state_variables = [r_x, r_y, r_z, v_x, v_y, v_z, m]
phase_A.control_variables = [u_x, u_y, u_z]

A = -(mu / (r_vec_norm**3))
phase_A.state_equations = {r_x: v_x,
                           r_y: v_y,
                           r_z: v_z,
                           v_x: (A * r_x) + (T_over_m * u_x) + ((1 / m) * D_x),
                           v_y: (A * r_y) + (T_over_m * u_y) + ((1 / m) * D_y),
                           v_z: (A * r_z) + (T_over_m * u_z) + ((1 / m) * D_z),
                           m: -xi}

phase_A.path_constraints = [u_vec_norm, r_vec_norm]

m_t0_A = (9 * m_tot_S) + m_tot_1 + m_tot_2 + m_payload
m_upper_bnd = m_t0_A
m_lower_bnd = m_t0_A - (6 * m_prop_S) + (tau_burn_S * m_prop_1 / tau_burn_1)
phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = [75.2, 75.2]
phase_A.bounds.state_variables = {r_x: [-2 * R_E, 2 * R_E],
                                  r_z: [-2 * R_E, 2 * R_E],
                                  r_y: [-2 * R_E, 2 * R_E],
                                  v_z: [-10000, 10000],
                                  v_x: [-10000, 10000],
                                  v_y: [-10000, 10000],
                                  m: [m_lower_bnd, m_upper_bnd]}
phase_A.bounds.control_variables = {u_x: [-10, 10],
                                    u_y: [-10, 10],
                                    u_z: [-10, 10]}
phase_A.bounds.path_constraints = [1, [R_E, "inf"]]
v_y_t0 = -omega_E * R_E * sym.cos(psi_L)
phase_A.bounds.initial_state_constraints = {r_x: R_E * sym.cos(psi_L),
                                            r_y: 0,
                                            r_z: R_E * sym.sin(psi_L),
                                            v_x: 0,
                                            v_y: v_y_t0,
                                            v_z: 0,
                                            m: m_t0_A,
                                            }

phase_A.guess.time = [0.0, 75.2]
phase_A.guess.state_variables = {r_x: []}
phase_A.guess.control = [[0, 0], [1, 1], [0, 0]]

phase_A.auxiliary_data = {T: 6 * T_eng_S + T_eng_1,
                          xi: (1 / g_0) * (6 * T_eng_S / I_S + T_eng_1 / I_1)}

new_phases = problem.new_phases_like(number=3,
                                     phase_for_copying=phase_A,
                                     names=["B", "C", "D"],
                                     copy_state_variables=True,
                                     copy_control_variables=True,
                                     copy_state_equations=True,
                                     copy_path_constraints=True,
                                     copy_integrand_functions=True,
                                     copy_state_endpoint_constraints=False,
                                     copy_bounds=True,
                                     copy_mesh=True)
phase_B, phase_C, phase_D = new_phases

# PHASE 2

phase_B.bounds.initial_time = phase_A.bounds.final_time
phase_B.bounds.final_time = 150.4

phase_B.auxiliary_data = {T: 3 * T_eng_S + T_eng_1,
                          xi: (1 / g_0) * (3 * T_eng_S / I_S + T_eng_1 / I_1)}

# PHASE 3

phase_C.bounds.initial_time = phase_B.bounds.final_time
phase_C.bounds.final_time = 261

phase_C.auxiliary_data = {T: T_eng_1,
                          xi: T_eng_1 / (g_0 * I_1)}

# PHASE 4

phase_D.bounds.initial_time = phase_C.bounds.final_time
phase_D.bounds.final_time = [phase_D.bounds.initial_time, 961]

phase_D.auxiliary_data = {
    T: T_eng_2,
    xi: T_eng_2 / (g_0 * I_2),
}

problem.objective_function = problem.phases[2].final_state_variables.m

problem.endpoint_constraints = [
    (phase_A.final_state_variables.m - phase_A.initial_state_variables.m 
        + 6*m_prop_S + (tau_burn_S/tau_burn_1)*m_prop_1),
    (phase_B.initial_state_variables.m - phase_A.final_state_variables.m 
        + 6*m_struct_S),
    (phase_B.final_state_variables.m - phase_B.initial_state_variables.m 
        + 3*m_prop_S + (tau_burn_S/tau_burn_1)*m_prop_1),
    (phase_C.initial_state_variables.m - phase_B.final_state_variables.m 
        + 3*m_struct_S),
    (phase_C.final_state_variables.m - phase_C.initial_state_variables.m
        + (1 - 2*(tau_burn_S/tau_burn_1))*m_prop_1),
    (phase_D.initial_state_variables.m - phase_C.final_state_variables.m
        + m_struct_1),
    phase_A.final_state_variables.r_x - phase_B.initial_state_variables.r_x,
    phase_A.final_state_variables.r_y - phase_B.initial_state_variables.r_y,
    phase_A.final_state_variables.r_z - phase_B.initial_state_variables.r_z,
    phase_A.final_state_variables.v_x - phase_B.initial_state_variables.v_x,
    phase_A.final_state_variables.v_y - phase_B.initial_state_variables.v_y,
    phase_A.final_state_variables.v_z - phase_B.initial_state_variables.v_z,
    phase_B.final_state_variables.r_x - phase_C.initial_state_variables.r_x,
    phase_B.final_state_variables.r_y - phase_C.initial_state_variables.r_y,
    phase_B.final_state_variables.r_z - phase_C.initial_state_variables.r_z,
    phase_B.final_state_variables.v_x - phase_C.initial_state_variables.v_x,
    phase_B.final_state_variables.v_y - phase_C.initial_state_variables.v_y,
    phase_B.final_state_variables.v_z - phase_C.initial_state_variables.v_z,
    phase_C.final_state_variables.r_x - phase_D.initial_state_variables.r_x,
    phase_C.final_state_variables.r_y - phase_D.initial_state_variables.r_y,
    phase_C.final_state_variables.r_z - phase_D.initial_state_variables.r_z,
    phase_C.final_state_variables.v_x - phase_D.initial_state_variables.v_x,
    phase_C.final_state_variables.v_y - phase_D.initial_state_variables.v_y,
    phase_C.final_state_variables.v_z - phase_D.initial_state_variables.v_z,
]

problem.bounds.endpoint_constraints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

problem.auxiliary_data = {
    mu: 3.986012e14,
    R_E: 6378145,
    r_vec_norm: sym.sqrt(r_x**2 + r_y**2 + r_z**2),
    v_vec_norm: sym.sqrt(v_x**2 + v_y**2 + v_z**2),
    u_vec_norm: sym.sqrt(u_x**2 + u_y**2 + u_z**2),
    D_x: - 0.5 * C_D * S_rho * v_r_vec_norm * v_r_x,
    D_y: - 0.5 * C_D * S_rho * v_r_vec_norm * v_r_y,
    D_z: - 0.5 * C_D * S_rho * v_r_vec_norm * v_r_z,
    C_D: 0.5,
    S_rho: 4 * np.pi,
    v_r_vec_norm: sym.sqrt(v_r_x**2 + v_r_y**2 + v_r_z**2),
    v_r_x: v_x - (omega_x_r_x),
    v_r_y: v_y - (omega_x_r_y),
    v_r_z: v_z - (omega_x_r_z),
    omega_x_r_x: - omega_E * r_y,
    omega_x_r_y: omega_E * r_x,
    omega_x_r_z: 0,
    g_0: 9.80665,
    h_0: 7200,
    h: r_vec_norm - R_E,
    rho: rho_0 * sym.exp(-h / h_0),
    rho_0: 1.225,
    omega_E: 7.29211585e-5,
    m_tot_S: 19290,
    m_tot_1: 104380,
    m_tot_2: 19300,
    m_prop_S: 17010,
    m_prop_1: 95550,
    m_prop_2: 16820,
    m_struct_S: 2280,
    m_struct_1: 8830,
    m_struct_2: 2480,
    T_eng_S: 628500,
    T_eng_1: 1083100,
    T_eng_2: 110094,
    I_S: 283.33364,
    I_1: 301.68776,
    I_2: 467.21311,
    tau_burn_S: 75.2,
    tau_burn_1: 261,
    tau_burn_2: 700,
    m_payload: 4146,
    T_over_m: T / m,
    psi_L: (28.5 / 180) * np.pi,
    scale_length: R_E,
    scale_speed: sym.sqrt(mu / scale_length),
    scale_time: scale_length / scale_speed,
    scale_acceleration: scale_speed / scale_time,
    scale_mass: m_t0_A,
    scale_force: scale_mass * scale_acceleration,
    scale_area: scale_length**2,
    scale_volume: scale_area * scale_length,
    scale_density: scale_mass / scale_volume,
    scale_gravity: scale_acceleration * scale_length**2,
}

# problem.settings.nlp_tolerance = 10e-7
# problem.settings.mesh_tolerance = 10e-6
# problem.settings.maximise_objective = True
# problem.settings.backend = "casadi"
# problem.settings.scaling_method = None
# problem.settings.assume_inf_bounds = True
# problem.settings.inf_value = 1e16
problem.settings.display_mesh_result_graph = True

problem.initialise()


