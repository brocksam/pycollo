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

problem = pycollo.OptimalControlProblem(name='Multiphase example problem', 
	parameter_variables=[s0, s1])

# phase_3 = pycollo.Phase(optimal_control_problem=problem)

# phase_4 = problem.new_phase()

phase_1 = pycollo.Phase()
# problem.add_phase(phase_1)

# phase_2 = pycollo.Phase()
# phase_2.optimal_control_problem = problem

# try:
# 	phase_1.optimal_control_problem = problem
# except AttributeError:
# 	pass

print(problem.phases)