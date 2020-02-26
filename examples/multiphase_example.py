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

problem = pycollo.OptimalControlProblem(parameter_variables=[s0, s1])
phase_1 = problem.new_phase(state_variables=[y0, y1, y2, y3])
phase_2 = problem.new_phase()
phase_2.state_variables = [y0, y1, y2]
# phase_3 = problem.new_phase(state_variables=[y0, y1])

print(problem.phases)