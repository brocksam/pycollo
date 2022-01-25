import pycollo
from sympy.abc import x,v,f

MAX_T = 1.0
MAX_V = 10.0
MAX_F = 10.0
def pairwise(a):
    return zip(a, a[1:])
def variable_phase_problem(phases = 1):
    # slide unit mass from x=0 to x=1 as fast as possible,
    # splitting into identical phases based on x-position with phases>1
    delta_T = MAX_T / (phases)
    state_equations = {x: v, v: f}
    problem = pycollo.OptimalControlProblem(f"{phases}-phase Sliding Mass")
    for idx in range(phases):
        start_x = idx/phases
        end_x = (idx+1)/phases
        phase = problem.new_phase(f"Stage{idx}", state_equations.keys(), f)
        phase.state_equations = state_equations
        # phase.bounds.initial_time = [0, MAX_T] if idx else 0
        # phase.bounds.final_time = [0, MAX_T]
        if idx == 0:
            phase.bounds.initial_time = 0
            phase.bounds.final_time = [(MAX_T / phases) - delta_T, (MAX_T / phases) + delta_T]
        else:
            phase.bounds.initial_time = [(MAX_T / phases) - delta_T, (MAX_T / phases) + delta_T]
            phase.bounds.final_time = [MAX_T - delta_T, MAX_T + delta_T]
        phase.bounds.initial_state_constraints = {
            x: start_x,
            v: [0,MAX_V] if idx else 0
        }
        phase.bounds.state_variables = {
            x: [start_x, end_x],
            v: [0,MAX_V],
        }
        phase.bounds.final_state_constraints = {
            x: end_x,
            v: [0,MAX_V] if idx-1!=phases else 0,
        }
        phase.bounds.control_variables = {
            f: [0, MAX_F]
        }
        phase.guess.time = [start_x * MAX_T, end_x * MAX_T]
        phase.guess.state_variables = [[start_x, end_x], [0,0]]
        phase.guess.control_variables = [[0, 0]]
    if phases > 1:
        endpoint_constraints = []
        for p1, p2 in pairwise(problem.phases):
            endpoint_constraints.append(p1.final_state_variables.v - p2.initial_state_variables.v)
            endpoint_constraints.append(p1.final_time_variable - p2.initial_time_variable)
        problem.endpoint_constraints = endpoint_constraints
        problem.bounds.endpoint_constraints = [[0, 0]] * len(endpoint_constraints)
    problem.objective_function = problem.phases[-1].final_time_variable
    problem.initialise()
    problem.solve(True)
    return problem

#problem works for one phase
variable_phase_problem(1)
#and then crashes for two phases
# variable_phase_problem(2)
#can see in data that objective functions are similar for first mesh iteration of both problems