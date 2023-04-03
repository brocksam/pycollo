"""Trajectory tracking problem.

Manufactured example to demonstrate a simple path following
problem. The problem involves finding the control to manoeuvre
a point mass in the plane such that it follows a periodic
sinusoidal path.

The point mass is controllable by orthogonal point forces
parallel to both the horizontal and vertical directions.

"""


import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

import pycollo

# State variables
x = sm.Symbol("x")  # Position (m) of the point horizontally from the origin (x-axis)
y = sm.Symbol("y")  # Position (m) of the point vertically from the origin (y-axis)
dx = sm.Symbol("dx")  # Velocity (m/s) of the point horizontally (x-axis)
dy = sm.Symbol("dy")  # Velocity (m/s) of the point vertically (y-axis)

# Control variables
Fx = sm.Symbol("Fx")  # Force (N) applied to the point horizontally (x-axis)
Fy = sm.Symbol("Fy")  # Force (N) applied to the point vertically (y-axis)

# Static parameter variable
m = sm.Symbol("m")  # Mass (kg) of the point

# Symbolic constants
ddx = sm.Symbol("ddx")  # Acceleration (m/s^2) of the point horizontally (x-axis)
ddy = sm.Symbol("ddy")  # Acceleration (m/s^2) of the point vertically (y-axis)

# Problem instantiation
problem = pycollo.OptimalControlProblem(name="Path follow point mass")

# Phase definition
phase = problem.new_phase(name="A")
phase.state_variables = [x, y, dx, dy]
phase.control_variables = [Fx, Fy]
phase.state_equations = {
    x: dx,
    y: dy,
    dx: ddx,
    dy: ddy,
}
phase.integrand_functions = [(sm.sin(x) - y)**2, Fx**2 + Fy**2]

# Outbound phase bounds
phase.bounds.initial_time = 0.0
phase.bounds.final_time = 1.0
phase.bounds.state_variables = {
    x: [0, 2*sm.pi],
    y: [-1, 1],
    dx: [-50, 50],
    dy: [-50, 50],
}
phase.bounds.control_variables = {
    Fx: [-50, 50],
    Fy: [-50, 50],
}
phase.bounds.integral_variables = [[0, 1000], [0, 10_000]]
phase.bounds.initial_state_constraints = {x: 0.0}
phase.bounds.final_state_constraints = {x: 2*sm.pi}

# Outbound phase guess
phase.guess.time = np.array([0, 1])
phase.guess.state_variables = np.array(
    [
        [0, 2*np.pi],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
)
phase.guess.control_variables = np.array(
    [
        [0, 0],
        [0, 0],
    ]
)
phase.guess.integral_variables = np.array([0, 0])

# Problem definitions
TRACKING_WEIGHTING = 0.99
CONTROL_WEIGHTING = 0.01
problem.objective_function = (
    TRACKING_WEIGHTING * phase.integral_variables[0]
    + CONTROL_WEIGHTING * phase.integral_variables[1]
)
problem.auxiliary_data = {
    ddx: Fx / m,
    ddy: Fy / m,
    m: 1.0,
}

problem.endpoint_constraints = [
    phase.final_state_variables.y - phase.initial_state_variables.y,
    phase.final_state_variables.dx - phase.initial_state_variables.dx,
    phase.final_state_variables.dy - phase.initial_state_variables.dy,
]

# Problem bounds
problem.bounds.endpoint_constraints = [0, 0, 0]

# Solve
problem.initialise()
problem.solve()

# Create target path coordinates
x_path = np.linspace(0, 2 * np.pi, 1000)
y_path = np.sin(x_path)

# Plot target path and solution trajectory in plan view
x = problem.solution.state[0][0]
y = problem.solution.state[0][1]
plt.figure()
plt.plot(x_path, y_path, color="#000000")
plt.plot(x, y)
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# Plot control solution
t = problem.solution._time_[0]
Fx = problem.solution.control[0][0]
Fy = problem.solution.control[0][1]
plt.figure()
plt.plot(t, Fx, label="Fx")
plt.plot(t, Fy, label="Fx")
plt.xlabel("Time / s")
plt.ylabel("Control / N")
plt.legend()
plt.show()
