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
y0 = sym.Symbol("y0")
y1 = sym.Symbol("y1")
y2 = sym.Symbol("y2")
y3 = sym.Symbol("y3")
y4 = sym.Symbol("y4")
y5 = sym.Symbol("y5")

u0 = sym.Symbol("u0")
u1 = sym.Symbol("u1")
u2 = sym.Symbol("u2")
u3 = sym.Symbol("u3")
u4 = sym.Symbol("u4")

p = sym.Symbol("p")

k0 = sym.Symbol("k0")
k1 = sym.Symbol("k1")
k2 = sym.Symbol("k2")
km1 = sym.Symbol("km1")
km2 = sym.Symbol("km2")
km3 = sym.Symbol("km3")

k0hat = sym.Symbol("k0hat")
k1hat = sym.Symbol("k1hat")
km1hat = sym.Symbol("km1hat")
beta0 = sym.Symbol("beta0")
beta1 = sym.Symbol("beta1")
betam1 = sym.Symbol("betam1")
K0 = sym.Symbol("K0")
K1 = sym.Symbol("K1")
K2 = sym.Symbol("K2")

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
