"""Fixture for unit tests."""

from collections import namedtuple

import numpy as np
import pytest
import sympy as sym

import pycollo


FixtureData = namedtuple("FixtureData", ["ocp", "syms"])


@pytest.fixture
def brachistochrone_fixture():
    """Fixture for the fully defined, but uninitialised, Brachistochrone OCP.

    See the example `examples/brachistochrone.py` for a description of and
    reference for this optimal control problem.

    """

    x, y, v, u = sym.symbols("x y v u")
    Symbols = namedtuple("Symbols", ["x", "y", "v", "u"])
    fixture_syms = Symbols(x, y, v, u)

    g = 9.81
    t0 = 0
    tfmin = 0
    tfmax = 10
    x0 = 0
    y0 = 0
    v0 = 0
    xf = 2
    yf = 2
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 10
    vmin = -50
    vmax = 50
    umin = -np.pi / 2
    umax = np.pi / 2

    problem = pycollo.OptimalControlProblem(name="Brachistochrone")
    phase = problem.new_phase(name="A")
    phase.state_variables = [x, y, v]
    phase.control_variables = u
    phase.state_equations = [v * sym.sin(u),
                             v * sym.cos(u),
                             g * sym.cos(u),
                             ]
    phase.auxiliary_data = {}
    problem.objective_function = phase.final_time_variable

    phase.bounds.initial_time = 0.0
    phase.bounds.final_time = [tfmin, tfmax]
    phase.bounds.state_variables = [[xmin, xmax], [ymin, ymax], [vmin, vmax]]
    phase.bounds.control_variables = [[umin, umax]]
    phase.bounds.initial_state_constraints = {x: x0, y: y0, v: v0}
    phase.bounds.final_state_constraints = {x: xf, y: yf}

    phase.guess.time = np.array([t0, tfmax])
    phase.guess.state_variables = np.array([[x0, xf], [y0, yf], [v0, v0]])
    phase.guess.control_variables = np.array([[0, umax]])

    problem.settings.display_mesh_result_graph = True
    problem.settings.derivative_level = 2
    problem.settings.scaling_method = "bounds"
    problem.settings.quadrature_method = "lobatto"
    problem.settings.max_mesh_iterations = 10

    fixture_data = FixtureData(problem, fixture_syms)

    return fixture_data


@pytest.fixture
def double_pendulum_fixture():
    """Fixture for the fully defined, but uninitialised, double pendulum
    swing-up problem.

    See the example `examples/double_pendulum.py` for a description of and
    reference for this optimal control problem.

    """

    a0, a1, v0, v1, T0, T1 = sym.symbols("a0 a1 v0 v1 T0 T1")
    g = sym.symbols("g")
    m0, p0, d0, l0, k0, I0 = sym.symbols("m0 p0 d0 l0 k0 I0")
    m1, p1, d1, l1, k1, I1 = sym.symbols("m1 p1 d1 l1 k1 I1")
    c0, s0, c1, s1 = sym.symbols("c0 s0 c1 s1")
    M00, M01, M10, M11, K0, K1 = sym.symbols("M00 M01 M10 M11 K0 K1")
    detM = sym.symbols("detM")

    syms = ["a0", "a1", "v0", "v1", "T0", "T1", "g", "m0", "p0", "d0", "l0",
            "k0", "I0", "m1", "p1", "d1", "l1", "k1", "I1", "c0", "s0", "c1",
            "s1", "M00", "M01", "M10", "M11", "K0", "K1", "detM"]
    Symbols = namedtuple("Symbols", syms)
    fixture_syms = Symbols(a0, a1, v0, v1, T0, T1, g, m0, p0, d0, l0, k0, I0,
                           m1, p1, d1, l1, k1, I1, c0, s0, c1, s1, M00, M01,
                           M10, M11, K0, K1, detM)

    K0_sub_expr_0 = T0 + g * (m0 * p0 + m1 * l0) * c0
    K0_sub_expr_1 = m1 * p1 * l0 * (s1 * c0 - s0 * c1) * v1 ** 2
    K0_eqn = K0_sub_expr_0 + K0_sub_expr_1
    K1_sub_expr_0 = T1 + g * m1 * p1 * c1
    K1_sub_expr_1 = m1 * p1 * l0 * (s0 * c1 - s1 * c0) * v0 ** 2
    K1_eqn = K1_sub_expr_0 + K1_sub_expr_1

    problem = pycollo.OptimalControlProblem(name="Double Pendulum Swing-Up")
    phase = problem.new_phase(name="A")
    phase.state_variables = [a0, a1, v0, v1]
    phase.control_variables = [T0, T1]
    phase.state_equations = [v0,
                             v1,
                             (M11 * K0 - M01 * K1) / detM,
                             (M00 * K1 - M10 * K0) / detM]
    phase.integrand_functions = [(T0**2 + T1**2)]
    phase.auxiliary_data = {g: -9.81,
                            k1: 1 / 12,
                            I0: m0 * (k0 ** 2 + p0 ** 2),
                            I1: m1 * (k1 ** 2 + p1 ** 2),
                            s0: sym.sin(a0),
                            c1: sym.cos(a1),
                            }
    problem.parameter_variables = [m0, p0]
    problem.objective_function = phase.integral_variables[0]
    problem.auxiliary_data = {g: 0,
                              d0: 0.5,
                              k0: 1 / 12,
                              m1: 1.0,
                              p1: 0.5,
                              d1: 0.5,
                              l0: p0 + d0,
                              l1: p1 + d1,
                              I0: m0 * (k0 ** 2 + p0 ** 2),
                              I1: m1 * (k1 ** 2 + p1 ** 2),
                              c0: sym.cos(a0),
                              s0: sym.sin(a0),
                              c1: sym.cos(a1),
                              s1: sym.sin(a1),
                              M00: I0 + m1 * l0 ** 2,
                              M01: m1 * p1 * l0 * (s0 * s1 + c0 * c1),
                              M10: M01,
                              M11: I1,
                              K0: K0_eqn,
                              K1: K1_eqn,
                              detM: M00 * M11 - M01 * M10,
                              }

    phase.bounds.initial_time = 0
    phase.bounds.final_time = [1, 3]
    phase.bounds.state_variables = [[-np.pi, np.pi],
                                    [-np.pi, np.pi],
                                    [-10, 10],
                                    [-10, 10]]
    phase.bounds.control_variables = [[-15, 15], [-15, 15]]
    phase.bounds.integral_variables = [0, 1000]
    phase.bounds.initial_state_constraints = [[-0.5 * np.pi, -0.5 * np.pi],
                                              [-0.5 * np.pi, -0.5 * np.pi],
                                              [0, 0],
                                              [0, 0]]
    phase.bounds.final_state_constraints = [[0.5 * np.pi, 0.5 * np.pi],
                                            [0.5 * np.pi, 0.5 * np.pi],
                                            [0, 0],
                                            [0, 0]]

    problem.bounds.parameter_variables = [[0.5, 1.5], [0.5, 1.5]]

    phase.guess.time = [0, 2]
    phase.guess.state_variables = [[-0.5 * np.pi, 0.5 * np.pi],
                                   [-0.5 * np.pi, 0.5 * np.pi],
                                   [0, 0],
                                   [0, 0]]
    phase.guess.control_variables = [[0, 0], [0, 0]]
    phase.guess.integral_variables = [[100]]

    problem.guess.parameter_variables = [1.0, 1.0]

    fixture_data = FixtureData(problem, fixture_syms)

    return fixture_data


@pytest.fixture
def hypersensitive_problem_fixture():
    """Fixture for the fully defined, but uninitialised, Hypersenstive Problem.

    See the example `examples/hypersenstive_problem.py` for a description of
    and reference for this optimal control problem.

    """

    y, u = sym.symbols('y u')
    Symbols = namedtuple("Symbols", ["y", "u"])
    fixture_syms = Symbols(y, u)

    problem = pycollo.OptimalControlProblem(name="Hypersensitive problem")
    phase = problem.new_phase(name="A")
    phase.state_variables = y
    phase.control_variables = u
    phase.state_equations = [-y**3 + u]
    phase.integrand_functions = [0.5 * (y**2 + u**2)]
    phase.auxiliary_data = {}

    problem.objective_function = phase.integral_variables[0]

    phase.bounds.initial_time = 0.0
    phase.bounds.final_time = 10000.0
    phase.bounds.state_variables = [[0, 2]]
    phase.bounds.control_variables = [[-1, 8]]
    phase.bounds.integral_variables = [[0, 2000]]
    phase.bounds.initial_state_constraints = [[1.0, 1.0]]
    phase.bounds.final_state_constraints = [[1.5, 1.5]]

    phase.guess.time = np.array([0.0, 10000.0])
    phase.guess.state_variables = np.array([[1.0, 1.5]])
    phase.guess.control_variables = np.array([[0.0, 0.0]])
    phase.guess.integral_variables = np.array([4])

    problem.settings.display_mesh_result_graph = True
    problem.settings.derivative_level = 2
    problem.settings.scaling_method = "bounds"
    problem.settings.quadrature_method = "lobatto"
    problem.settings.max_mesh_iterations = 10

    fixture_data = FixtureData(problem, fixture_syms)

    return fixture_data
