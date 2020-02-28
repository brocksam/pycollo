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

phase_1 = pycollo.Phase(state_variables=y0, control_variables=u0)
phase_1.state_equations = [y0*y1, s0*s1]
phase_1.integrand_functions = [u0**2]
problem.add_phase(phase_1)

phase_2 = pycollo.Phase()
phase_2.optimal_control_problem = problem

phase_3 = pycollo.Phase(optimal_control_problem=problem)

phase_4 = problem.new_phase()

for phase in problem.phases:
	print(phase.optimal_control_problem)
	print(phase.phase_number)
	print(phase.state_variables)
	print(phase.control_variables)
	print(phase.integral_variables)
	print(phase.initial_time_variable)
	print(phase.final_time_variable)
	print(phase.initial_state_variables)
	print(phase.final_state_variables)
	print(phase.state_equations)
	print(phase.path_constraints)
	print(phase.integrand_functions)
	print(phase.state_endpoint_constraints)
	print('')