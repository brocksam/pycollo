"""An invented example to demonstrate multiphase problems with variable phase
lengths.

"""


import matplotlib.pyplot as plt
import numpy as np
import pycollo
import sympy as sym

pos_x, pos_y = sym.symbols("pos_x pos_y")
vel_x, vel_y = sym.symbols("vel_x vel_y")
F_x, F_y = sym.symbols("F_x F_y")
m = sym.Symbol("m")
D = sym.Symbol("D")

t_0 = 0.0
t_max = 10.0
pos_min = -5.0
pos_max = 5.0
vel_min = -20.0
vel_max = 20.0
F_min = -100.0
F_max = 100.0

problem = pycollo.OptimalControlProblem(name="Multiphase Obstacle Avoidance")
phase_A = problem.new_phase(name="A")
phase_A.state_variables = [pos_x, pos_y, vel_x, vel_y]
phase_A.control_variables = [F_x, F_y]

phase_A.state_equations = {
    pos_x: vel_x,
    pos_y: vel_y,
    vel_x: F_x / m,
    vel_y: F_y / m,
}
phase_A.path_constraints = [sym.sqrt(pos_x**2 + pos_y**2) - D]
phase_A.integrand_functions = [F_x**2 + F_y**2]

phase_A.bounds.initial_time = t_0
phase_A.bounds.final_time = [t_0, t_max / 2]
phase_A.bounds.state_variables = {
    pos_x: [pos_min, pos_max],
    pos_y: [pos_min, pos_max],
    vel_x: [vel_min, vel_max],
    vel_y: [vel_min, vel_max],
}
phase_A.bounds.control_variables = {
    F_x: [F_min, F_max],
    F_y: [F_min, F_max],
}
phase_A.bounds.path_constraints = [[0.0, pos_max**2]]
phase_A.bounds.integral_variables = [[0.0, 100.0]]
phase_A.bounds.initial_state_constraints = {
    pos_x: 0.0,
    pos_y: -2.0,
    vel_x: 0.0,
    vel_y: 0.0,
}
phase_A.bounds.final_state_constraints = {
    pos_x: 0.0,
    pos_y: 2.0,
    vel_x: 0.0,
    vel_y: 0.0,
}

phase_A.guess.time = [0.0, t_max / 2]
phase_A.guess.state_variables = [
    [0.0, 0.0],
    [-2.0, 2.0],
    [0.0, 0.0],
    [0.0, 0.0],
]
phase_A.guess.control_variables = [
    [0.0, 0.0],
    [0.0, 0.0],
]
phase_A.guess.integral_variables = [10.0]

problem.objective_function = phase_A.integral_variables[0]
problem.auxiliary_data = {
    m: 1.0,
    D: 1.0
}

problem.settings.max_mesh_iterations = 1

problem.solve()

plt.figure()
plt.plot(problem.solution.state[0][0], problem.solution.state[0][1])
plt.show()




