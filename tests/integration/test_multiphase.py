"""Integration tests module for multiphase problems.

This module includes a trivial test problem in which a unit mass is slid from
position x = 0 to x = 1 as fast as possible, with the additional constraints
that velocity v = 0 at t0 and tF.

The problem is split into multiple phases of equal length (distance) by adding
phase endpoint constraints on the position variables. Continuity is imposed by
adding problem endpoint constraints between time and velocity of the unit mass
between adjacent phases.

"""


import numpy as np
import pytest
import sympy as sym

import pycollo


PHASE_NAMES = {0: "A", 1: "B", 2: "C", 3: "D"}
EXPECTED_SOLUTION = 0.4472136


def variable_phase_problem(num_phases):
    """Instantiate the trivial unit mass movement test OCP."""
    x = sym.Symbol("x")
    v = sym.Symbol("v")
    f = sym.Symbol("f")

    MAX_T = 1.0
    MAX_V = 10.0
    MAX_F = 20.0

    problem = pycollo.OptimalControlProblem(f"{num_phases}-phase Sliding Mass")

    for i in range(num_phases):
        start_x = i / num_phases
        end_x = (i + 1) / num_phases
        phase = problem.new_phase(PHASE_NAMES[i],
                                  state_variables=[x, v],
                                  control_variables=[f])
        phase.state_equations = state_equations = {x: v, v: f}
        phase.bounds.initial_time = [0, MAX_T] if i else 0
        phase.bounds.final_time = [0, MAX_T]
        phase.bounds.initial_state_constraints = {
            x: start_x,
            v: [0, MAX_V] if i else 0
        }
        phase.bounds.state_variables = {
            x: [start_x, end_x],
            v: [0, MAX_V],
        }
        phase.bounds.final_state_constraints = {
            x: end_x,
            v: [0, MAX_V] if ((i + 1) != num_phases) else 0,
        }
        phase.bounds.control_variables = {
            f: [-MAX_F, MAX_F]
        }
        phase.guess.time = [start_x * MAX_T, end_x * MAX_T]
        phase.guess.state_variables = [[start_x, end_x], [0, 0]]
        phase.guess.control_variables = [[0, 0]]

    if num_phases >= 2:
        endpoint_constraints = []
        for p1, p2 in zip(problem.phases[:-1], problem.phases[1:]):
            endpoint_constraints.append(p1.final_state_variables.v - p2.initial_state_variables.v)
            endpoint_constraints.append(p1.final_time_variable - p2.initial_time_variable)
        problem.endpoint_constraints = endpoint_constraints
        problem.bounds.endpoint_constraints = [[0, 0]] * len(endpoint_constraints)

    problem.objective_function = problem.phases[-1].final_time_variable

    return problem


@pytest.mark.parametrize("num_phases", [1, 2, 3, 4])
def test_multiphase(num_phases):
    """The multiphase OCP solves sucessfully with different numbers of phases."""
    problem = variable_phase_problem(num_phases)
    problem.solve()
    assert np.isclose(problem.solution.objective, EXPECTED_SOLUTION)
    assert problem.mesh_tolerance_met is True
