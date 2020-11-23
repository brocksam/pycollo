import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import pycollo

y0 = sym.Symbol('y0')
y1 = sym.Symbol('y1')
y2 = sym.Symbol('y2')
y3 = sym.Symbol('y3')
u0 = sym.Symbol('u0')
u1 = sym.Symbol('u1')
s0 = sym.Symbol('s0')
s1 = sym.Symbol('s1')

a0 = sym.Symbol('a0')
a1 = sym.Symbol('a1')

circle_radius = 1.1

problem = pycollo.OptimalControlProblem(
    name="Multiphase example problem",
    parameter_variables=s0
)

phase_A = problem.new_phase(name="A")
phase_A.state_variables = [y0, y1, y2, y3]
phase_A.control_variables = [u0, u1]
phase_A.state_equations = {
    y0: y2,
    y1: y3,
    y2: a0,
    y3: a1,
}
phase_A.path_constraints = [sym.sqrt(y0**2 + y1**2) - circle_radius]
phase_A.integrand_functions = [u0**2, u1**2]
phase_A.auxiliary_data = {}

phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = [0, 2]
phase_A.bounds.state_variables = {
    y0: [-3, 3],
    y1: [-3, 3],
    y2: [-50, 50],
    y3: [-50, 50],
}
phase_A.bounds.control_variables = {
    u0: [-50, 50],
    u1: [-50, 50],
}
phase_A.bounds.integral_variables = [[0, 1000], [0, 1000]]
phase_A.bounds.path_constraints = [[0, 10]]
phase_A.bounds.initial_state_constraints = {
    y0: 1,
    y1: -2,
    y2: 0,
    y3: 0,
}
phase_A.bounds.final_state_constraints = {
    y0: 0,
    y1: 2,
    y2: 0,
    y3: 0,
}

phase_A.guess.time = np.array([0, 1])
phase_A.guess.state_variables = np.array([
    [1, 0],
    [-2, 2],
    [0, 0],
    [0, 0],
])
phase_A.guess.control_variables = np.array([
    [0, 0],
    [0, 0],
])
phase_A.guess.integral_variables = np.array([0, 0])

phase_B = problem.new_phase_like(
    phase_for_copying=phase_A,
    name="B",
)
phase_B.auxiliary_data = {}

phase_B.bounds.initial_time = [0, 2]
phase_B.bounds.final_time = 2
# phase_B.bounds.initial_state_constraints = {
# 	y0: 0,
# 	y1: 2,
# 	y2: 0,
# 	y3: 0,
# 	}
phase_B.bounds.final_state_constraints = {
    y0: -1,
    y1: -2,
    y2: 0,
    y3: 0,
}

phase_B.guess.time = np.array([1, 2])
phase_B.guess.state_variables = np.array([
    [0, -1],
    [2, -2],
    [0, 0],
    [0, 0],
])
phase_B.guess.integral_variables = np.array([0, 0])

problem.objective_function = (phase_A.integral_variables[0]
                              + phase_A.integral_variables[1]
                              + phase_B.integral_variables[0]
                              + phase_B.integral_variables[1]
                              )

problem.auxiliary_data = {
    a0: u0 / s0,
    a1: u1 / s0,
}

problem.endpoint_constraints = [
    phase_A.final_time_variable - phase_B.initial_time_variable,
    phase_A.final_state_variables.y0 - phase_B.initial_state_variables.y0,
    phase_A.final_state_variables.y1 - phase_B.initial_state_variables.y1,
    phase_A.final_state_variables.y2 - phase_B.initial_state_variables.y2,
    phase_A.final_state_variables.y3 - phase_B.initial_state_variables.y3,
]

problem.bounds.parameter_variables = [[1, 2]]
problem.bounds.endpoint_constraints = [
    0,
    0,
    0,
    0,
    0,
]

problem.guess.parameter_variables = np.array([1.5])

# problem.settings.nlp_tolerance = 1e-8
# problem.settings.mesh_tolerance = 1e-7
# problem.settings.maximise_objective = False
# problem.settings.backend = "casadi"
problem.settings.scaling_method = "bounds"
# problem.settings.assume_inf_bounds = False
# problem.settings.inf_value = 1e16
problem.settings.check_nlp_functions = False
# problem.settings.dump_nlp_check_json = "pycollo"
# problem.settings.collocation_points_min = 2
# problem.settings.collocation_points_max = 8
# problem.settings.derivative_level = 2
# problem.settings.max_mesh_iterations = 10
problem.settings.display_mesh_result_graph = True

# phase_A.mesh.number_mesh_sections = 1
# phase_A.mesh.mesh_section_sizes = [1]
# phase_A.mesh.number_mesh_section_nodes = [2]
# phase_B.mesh.number_mesh_sections = 1
# phase_B.mesh.mesh_section_sizes = [1]
# phase_B.mesh.number_mesh_section_nodes = [2]

problem.initialise()
problem.solve()

r = np.linspace(0, 2 * np.pi, 1000)
x_circle = circle_radius * np.cos(r)
y_circle = circle_radius * np.sin(r)

x_P0 = problem.solution.state[0][0]
y_P0 = problem.solution.state[0][1]
x_P1 = problem.solution.state[1][0]
y_P1 = problem.solution.state[1][1]
plt.plot(x_P0, y_P0)
plt.plot(x_P1, y_P1)
plt.plot(x_circle, y_circle, color="#000000")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
