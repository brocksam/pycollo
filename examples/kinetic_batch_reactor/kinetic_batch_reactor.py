"""Kinetic Batch Reactor problem.

NOTE: THIS EXAMPLE IS INCOMPLETE AND IS NOT CURRENTLY SOLVABLE USING PYCOLLO.

Example 6.15 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p331 - 336.

Attributes
----------

"""

import pycollo
import sympy as sym

# Define all symbols for building problem equations
y0 = pycollo.Symbol("y0")
y1 = pycollo.Symbol("y1")
y2 = pycollo.Symbol("y2")
y3 = pycollo.Symbol("y3")
y4 = pycollo.Symbol("y4")
y5 = pycollo.Symbol("y5")

u0 = pycollo.Symbol("u0")
u1 = pycollo.Symbol("u1")
u2 = pycollo.Symbol("u2")
u3 = pycollo.Symbol("u3")
u4 = pycollo.Symbol("u4")

p = pycollo.Symbol("p")

k0 = pycollo.Symbol("k0")
k1 = pycollo.Symbol("k1")
k2 = pycollo.Symbol("k2")
km1 = pycollo.Symbol("km1")
km2 = pycollo.Symbol("km2")
km3 = pycollo.Symbol("km3")

k0hat = pycollo.Symbol("k0hat")
k1hat = pycollo.Symbol("k1hat")
km1hat = pycollo.Symbol("km1hat")
beta0 = pycollo.Symbol("beta0")
beta1 = pycollo.Symbol("beta1")
betam1 = pycollo.Symbol("betam1")
K0 = pycollo.Symbol("K0")
K1 = pycollo.Symbol("K1")
K2 = pycollo.Symbol("K2")

# Creat the OCP
problem_name = "Kinetic Batch Reactor Problem"
problem = pycollo.OptimalControlProblem(name=problem_name)
problem.parameter_variables = [p]

# PHASE A

phase_A = problem.new_phase("A")
phase_A.state_variables = [y0, y1, y2, y3, y4, y5]
phase_A.control_variables = [u0, u1, u2, u3, u4]

k0y1y5 = k0 * y1 * y5
k1y1u1 = k1 * y1 * u1
k2y3y5 = k2 * y3 * y5
phase_A.state_equations = {y0: -k1y1u1,
                           y1: -(k0 * y1 * y5) + (km1 * u3) - k1y1u1,
                           y2: k1y1u1 + k2y3y5 - (km3 * u2),
                           y3: -k2y3y5 + (km3 * u2),
                           y4: k0y1y5 - (km1 * u3),
                           y5: -k0y1y5 + (km1 * u3) - k2y3y5 - (km3 * u2)}
phase_A.path_constraints = [p - y5 + (10**(-u0)) - u1 - u2 - u3]

problem.auxiliary_data = {k0hat: 1.3708e+12,
                          k1hat: 5.2282e+12,
                          km1hat: 1.6215e+20,
                          beta0: 9.2984e+3,
                          beta1: 9.5999e+3,
                          betam1: 1.3108e+4,
                          K0: 2.575e-16,
                          K1: 4.876e-14,
                          K2: 1.7884e-16,
}
