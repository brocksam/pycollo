import numpy as np
import sympy as sym

import pycollo

pos_x, pos_y, vel_x, vel_y = sym.symbols("pos_x pos_y vel_x vel_y")
F_x, F_y = sym.symbols("F_x F_y")

mesh = pycollo.Mesh(
    mesh_sections=2, mesh_section_fractions=None, mesh_collocation_points=4
)
problem = pycollo.OptimalControlProblem(
    state_variables=[pos_x, pos_y, vel_x, vel_y],
    control_variables=[F_x, F_y],
    initial_mesh=mesh,
)

problem.state_equations = [vel_x, vel_y, F_x, F_y]

problem.objective_function = problem.final_time

problem.path_constraints = [pos_x ** 2 + pos_y ** 2 - 1]

problem.state_endpoint_constraints = [
    problem.initial_state[0],
    problem.initial_state[1],
    problem.initial_state[2],
    problem.initial_state[3],
    problem.final_state[0],
    problem.final_state[1],
]

problem.bounds = pycollo.Bounds(
    optimal_control_problem=problem,
    initial_time=0.0,
    final_time=[0.1, 10],
    state=[[-1, 2], [-1, 2], [-10, 10], [-10, 10]],
    control=[[-1, 1], [-1, 1]],
    # parameter=[[1, 10]],
    path=[[0, 0]],
    state_endpoint=[[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
)

# Guess
problem.initial_guess = pycollo.Guess(
    optimal_control_problem=problem,
    time=[0, 1],
    state=np.array([[0, 0], [0, 0], [0, 0], [0, 0]]),
    control=np.array([[0, 0], [0, 0]]),
    state_endpoints_override=True,
)

problem.settings.scaling_method = None
problem.settings.display_mesh_result_graph = True
problem.settings.derivative_level = 2
problem.settings.mesh_tolerance = 1e-5

problem.solve()
