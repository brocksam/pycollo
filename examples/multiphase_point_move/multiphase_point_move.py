"""Multiphase point move problem.

Manufactured example to demonstrate a simple multiphase
problem. The problem involves finding the optimal trajectory
to move a point mass in the plane between three point while
avoiding a circular obstacle. The start coordinate is [1, -2],
the midpoint coordinate is [0, 2], and the end coordinate is
[-1, -2]. The obstacle is  circle of radius 1 centred at the
origin [0, 0].

"""


import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import pycollo


# State variables
x = sym.Symbol("x")  # Position (m) of the point horizontally from the origin (x-axis)
y = sym.Symbol("y")  # Position (m) of the point vertically from the origin (y-axis)
dx = sym.Symbol("dx")  # Velocity (m/s) of the point horizontally (x-axis)
dy = sym.Symbol("dy")  # Velocity (m/s) of the point vertically (y-axis)

# Control variables
Fx = sym.Symbol("Fx")  # Force (N) applied to the point horizontally (x-axis)
Fy = sym.Symbol("Fy")  # Force (N) applied to the point vertically (y-axis)

# Static parameter variable
m = sym.Symbol("m")  # Mass (kg) of the point

# Symbolic constants
ddx = sym.Symbol("ddx")  # Acceleration (m/s^2) of the point horizontally (x-axis)
ddy = sym.Symbol("ddy")  # Acceleration (m/s^2) of the point vertically (y-axis)

# Numerical constants
r = 1.0  # Radius (m) of the obstacle

# Problem instantiation
problem = pycollo.OptimalControlProblem(
    name="Multiphase point move",
    parameter_variables=m,
)

# Outbound phase definition
phase_A = problem.new_phase(name="A")
phase_A.state_variables = [x, y, dx, dy]
phase_A.control_variables = [Fx, Fy]
phase_A.state_equations = {
    x: dx,
    y: dy,
    dx: ddx,
    dy: ddy,
}
phase_A.path_constraints = [sym.sqrt(x ** 2 + y ** 2) - r]
phase_A.integrand_functions = [Fx ** 2, Fy ** 2]
phase_A.auxiliary_data = {}

# Outbound phase bounds
phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = [0.5, 1.5]
phase_A.bounds.state_variables = {
    x: [-3, 3],
    y: [-3, 3],
    dx: [-50, 50],
    dy: [-50, 50],
}
phase_A.bounds.control_variables = {
    Fx: [-50, 50],
    Fy: [-50, 50],
}
phase_A.bounds.integral_variables = [[0, 1000], [0, 1000]]
phase_A.bounds.path_constraints = [[0, 10]]
phase_A.bounds.initial_state_constraints = {
    x: 1,
    y: -2,
    dx: 0,
    dy: 0,
}
phase_A.bounds.final_state_constraints = {
    x: 0,
    y: 2,
    dx: 0,
    dy: 0,
}

# Outbound phase guess
phase_A.guess.time = np.array([0, 1])
phase_A.guess.state_variables = np.array(
    [
        [1, 0],
        [-2, 2],
        [0, 0],
        [0, 0],
    ]
)
phase_A.guess.control_variables = np.array(
    [
        [0, 0],
        [0, 0],
    ]
)
phase_A.guess.integral_variables = np.array([0, 0])

# Inbound phase definition
phase_B = problem.new_phase_like(
    phase_for_copying=phase_A,
    name="B",
)
phase_B.auxiliary_data = {}

# Inbound phase bounds
phase_B.bounds.initial_time = [0.5, 1.5]
phase_B.bounds.final_time = [1.5, 2.0]
phase_B.bounds.initial_state_constraints = {
    x: 0,
    y: 2,
    dx: 0,
    dy: 0,
}
phase_B.bounds.final_state_constraints = {
    x: -1,
    y: -2,
    dx: 0,
    dy: 0,
}

# Inbound phase guess
phase_B.guess.time = np.array([1, 2])
phase_B.guess.state_variables = np.array(
    [
        [0, -1],
        [2, -2],
        [0, 0],
        [0, 0],
    ]
)
phase_B.guess.integral_variables = np.array([0, 0])

# Problem definitions
problem.objective_function = (
    phase_A.integral_variables[0]
    + phase_A.integral_variables[1]
    + phase_B.integral_variables[0]
    + phase_B.integral_variables[1]
)

problem.auxiliary_data = {
    ddx: Fx / m,
    ddy: Fy / m,
}

problem.endpoint_constraints = [
    phase_A.final_time_variable - phase_B.initial_time_variable,
    phase_A.final_state_variables.x - phase_B.initial_state_variables.x,
    phase_A.final_state_variables.y - phase_B.initial_state_variables.y,
    phase_A.final_state_variables.dx - phase_B.initial_state_variables.dx,
    phase_A.final_state_variables.dy - phase_B.initial_state_variables.dy,
]

# Problem bounds
problem.bounds.parameter_variables = [[1, 2]]
problem.bounds.endpoint_constraints = [
    0,
    0,
    0,
    0,
    0,
]

# Problem guess
problem.guess.parameter_variables = np.array([1.5])


# Solve
problem.initialise()
problem.solve()

# Create obstacle coordinates
theta = np.linspace(0, 2 * np.pi, 1000)
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)

# Plot obstacle and solution in plan view
x_P0 = problem.solution.state[0][0]
y_P0 = problem.solution.state[0][1]
x_P1 = problem.solution.state[1][0]
y_P1 = problem.solution.state[1][1]
plt.plot(x_P0, y_P0)
plt.plot(x_P1, y_P1)
plt.plot(x_circle, y_circle, color="#000000")
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
