"""Trajectory tracking problem.

Manufactured example to demonstrate a simple path following
problem. The problem involves finding the control to manoeuvre
a point mass in the plane such that it follows a periodic
sinusoidal path.

The point mass is controllable by a single point force whose
direction can be directly controlled.

This problem is for illustration purposes only. In order to
keep it simple it ends up not being a well-defined optimal
control problem. This is because there is sufficient freedom
in the controls to "perfectly" match the trajectory in theory.
However, in practice it is not possible to do so due to slight
numerical errors between the "true" trajectory and the
simulated trajectory. The optimizer will always think that
it's possible to improve the solution accuracy and so it's
both difficult to meet the mesh tolerance and the optimal
control found ends up being very noisy.

To allow for the problem to be solved, a second term is
introduced to the objective function: the minimization of the
sum of squared forces. This introduces another challenge as
the two terms in the objective function now need to be
carefully weighted against one another. However, it does at
least allow for a non-noisy solution to be obtained.

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
F = sm.Symbol("F")  # Force (N) applied to the point
theta = sm.Symbol("theta")  # Force (N) applied to the point vertically (y-axis)

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
phase.control_variables = [F, theta]
phase.state_equations = {
    x: dx,
    y: dy,
    dx: ddx,
    dy: ddy,
}
phase.integrand_functions = [(sm.sin(x) - y)**2, F**2]

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
    F: [0, 100],
    theta: [-sm.pi, sm.pi],
}
phase.bounds.integral_variables = [[0, 0.01], [0, 1_000_000]]
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
TRACKING_WEIGHTING = 0.99999
CONTROL_WEIGHTING = 0.00001
problem.objective_function = (
    TRACKING_WEIGHTING * phase.integral_variables[0]
    + CONTROL_WEIGHTING * phase.integral_variables[1]
)

problem.auxiliary_data = {
    ddx: (F * sm.cos(theta)) / m,
    ddy: (F * sm.sin(theta)) / m,
    m: 1.0,
}

problem.endpoint_constraints = [
    phase.final_state_variables.y - phase.initial_state_variables.y,
    phase.final_state_variables.dx - phase.initial_state_variables.dx,
    phase.final_state_variables.dy - phase.initial_state_variables.dy,
]

# Problem bounds
problem.bounds.endpoint_constraints = [0, 0, 0]

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
F = problem.solution.control[0][0]
theta = problem.solution.control[0][1]
plt.figure()
plt.plot(t, F, label="F")
plt.plot(t, theta, label="theta")
plt.xlabel("Time / s")
plt.ylabel("Control / N")
plt.legend()
plt.show()
