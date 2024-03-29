"""Trajectory tracking problem.

Manufactured example to demonstrate a simple path following
problem. The problem involves finding the control to manoeuvre
a point mass in the plane such that it follows a periodic
sinusoidal path.

The point mass is controllable by a single point force whose
direction can be controlled. However, the rate of change of the
magnitude of the applied force and the rate of change in the angle
of the applied force are the control variables in this problem
and are therefore limited.

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

F = sm.Symbol("F")  # Force (N) applied to the point
theta = sm.Symbol("theta")  # Force (rad) direction anticlockwise from the horizontal (x-axis)

# Control variables
dF = sm.Symbol("dF")  # Rate of force development (N/s)
dtheta = sm.Symbol("dtheta")  # Rate of change in force direction (rad/s)

# Static parameter variable
m = sm.Symbol("m")  # Mass (kg) of the point

# Symbolic constants
ddx = sm.Symbol("ddx")  # Acceleration (m/s^2) of the point horizontally (x-axis)
ddy = sm.Symbol("ddy")  # Acceleration (m/s^2) of the point vertically (y-axis)

# Problem instantiation
problem = pycollo.OptimalControlProblem(name="Path follow point mass")

# Phase definition
phase = problem.new_phase(name="A")
phase.state_variables = [x, y, dx, dy, F, theta]
phase.control_variables = [dF, dtheta]
phase.state_equations = {
    x: dx,
    y: dy,
    dx: ddx,
    dy: ddy,
    F: dF,
    theta: dtheta,
}
phase.integrand_functions = [(sm.sin(x) - y)**2]

# Outbound phase bounds
phase.bounds.initial_time = 0.0
phase.bounds.final_time = 1.0
phase.bounds.state_variables = {
    x: [0, 2*sm.pi],
    y: [-1, 1],
    dx: [0, 100],
    dy: [-100, 100],
    F: [0, 200],
    theta: [-sm.pi, sm.pi],
}
phase.bounds.control_variables = {
    dF: [-100, 100],
    dtheta: [-10, 10],
}
phase.bounds.integral_variables = [[0, 0.1]]
phase.bounds.initial_state_constraints = {x: 0.0}
phase.bounds.final_state_constraints = {x: 2*sm.pi}

# Outbound phase guess
phase.guess.time = np.array([0, 0.25, 0.5, 0.75, 1])
phase.guess.state_variables = np.array(
    [
        # [0, 1, np.pi, -1, 2*np.pi],
        # [0, 0, 0, 0, 0],
        [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
        [0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
phase.guess.control_variables = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
phase.guess.integral_variables = np.array([0])

# Problem definitions
problem.objective_function = phase.integral_variables[0]

problem.auxiliary_data = {
    ddx: (F * sm.cos(theta)) / m,
    ddy: (F * sm.sin(theta)) / m,
    m: 0.6,
}

problem.endpoint_constraints = [
    phase.final_state_variables.y - phase.initial_state_variables.y,
    phase.final_state_variables.dx - phase.initial_state_variables.dx,
    phase.final_state_variables.dy - phase.initial_state_variables.dy,
    phase.final_state_variables.F - phase.initial_state_variables.F,
    phase.final_state_variables.theta - phase.initial_state_variables.theta,
]

# Problem bounds
problem.bounds.endpoint_constraints = [0, 0, 0, 0, 0]

# Problem settings
problem.settings.mesh_tolerance = 1e-5

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

# Plot control solution
t = problem.solution._time_[0]
F = problem.solution.state[0][4]
theta = problem.solution.state[0][5]
dF = problem.solution.control[0][0]
dtheta = problem.solution.control[0][1]
plt.figure()
plt.plot(t, F, label="F")
plt.plot(t, theta, label="theta")
plt.plot(t, dF, label="dF")
plt.plot(t, dtheta, label="dtheta")
plt.xlabel("Time / s")
plt.ylabel("Control / N")
plt.legend()
plt.show()
