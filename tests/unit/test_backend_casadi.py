"""Testing of the CasADi-powered backend.

Attributes
----------
DP_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the double pendulum swing-up fixture.
    Note that DP is shorthand for 'double pendulum' here.
HS_FIXTURE_REF : :py:func:`fixture_ref <pytest_cases>`
    A pytest-cases fixture reference to the Hypersensitive problem fixture.
    Note that HS is shorthand for 'Hypersensitive problem' here.

"""

import casadi as ca
import pytest
import pytest_cases
import sympy as sym

import pycollo


BR_FIXTURE_REF = pytest_cases.fixture_ref("brachistochrone_fixture")
HS_FIXTURE_REF = pytest_cases.fixture_ref("hypersensitive_problem_fixture")
DP_FIXTURE_REF = pytest_cases.fixture_ref("double_pendulum_fixture")
ALL_FIXTURE_REF = [HS_FIXTURE_REF, DP_FIXTURE_REF, BR_FIXTURE_REF]


@pytest.fixture
def casadi_backend_fixture():
    """Unintialised CasADi backend object."""
    backend = object.__new__(pycollo.backend.Casadi)
    return backend


@pytest.fixture
def phase_backend_fixture():
    """Uninitialised phase backend object."""
    phase_backend = object.__new__(pycollo.backend.PycolloPhaseData)
    return phase_backend


@pytest.fixture
def double_pendulum_phase_backend_fixture(double_pendulum_fixture,
                                          casadi_backend_fixture,
                                          phase_backend_fixture):
    """Uninitialised phased backend for the double pendulum OCP."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    return phase_backend


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REF)
def test_sym_method(ocp_fixture, casadi_backend_fixture):
    """Check symbol creation correct handed off to chosen backend."""
    ocp, user_syms = ocp_fixture
    backend = casadi_backend_fixture
    for user_sym in user_syms:
        casadi_sym = backend.sym(user_sym.name)
        assert isinstance(casadi_sym, ca.SX)
        assert casadi_sym.name() == user_sym.name


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REF)
def test_create_point_variable_symbols(ocp_fixture, casadi_backend_fixture):
    """Endpoint symbol attributes exist and are of correct symbol type."""
    ocp, user_syms = ocp_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    assert hasattr(backend, "s_var_full")
    assert isinstance(backend.s_var_full, tuple)
    for s_sym in backend.s_var_full:
        assert isinstance(s_sym, ca.SX)
    assert hasattr(backend, "num_s_var_full")
    assert hasattr(backend, "V_s_var_full")
    assert hasattr(backend, "r_s_var_full")


def test_create_point_variable_symbols_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   utils):
    """Check correct insantiation of parameter variables.

    Use the double pendulum fixture to check that static parameter variables
    are correctly instantiated automatically by the CasADi backend. This
    includes creating the symbols, their associated scaling (stretch and shift)
    symbols, scaling mappings and symbol count variables.

    """
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()

    m0 = ca.SX.sym("m0")
    p0 = ca.SX.sym("p0")

    _s0 = ca.SX.sym("_s0")
    _V_s0 = ca.SX.sym("_V_s0")
    _r_s0 = ca.SX.sym("_r_s0")
    _s1 = ca.SX.sym("_s1")
    _V_s1 = ca.SX.sym("_V_s1")
    _r_s1 = ca.SX.sym("_r_s1")

    expect_sub_mapping = {user_syms.m0: m0,
                          user_syms.p0: p0}
    expect_aux_data = {m0: _V_s0 * _s0 + _r_s0,
                       p0: _V_s1 * _s1 + _r_s1}

    assert backend.s_var_user == ocp.parameter_variables
    assert backend.s_var_user == (user_syms.m0, user_syms.p0)
    assert backend.num_s_var_full == 2

    for backend_sym, expect_sym in zip(backend.s_var_full, (_s0, _s1)):
        utils.assert_ca_sym_identical(backend_sym, expect_sym)
    for backend_V, expect_V in zip(backend.V_s_var_full, (_V_s0, _V_s1)):
        utils.assert_ca_sym_identical(backend_V, expect_V)
    for backend_r, expect_r in zip(backend.r_s_var_full, (_r_s0, _r_s1)):
        utils.assert_ca_sym_identical(backend_r, expect_r)

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in backend.user_to_backend_mapping
        backend_sym = backend.user_to_backend_mapping[expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_create_state_variable_symbols_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   phase_backend_fixture,
                                                   utils):
    """Check correct instantiation of state variables by the phase backend.

    Use the double pendulum fixture to ensure the phase backend correctly
    handles state variables in the backend. The phase backend should
    instantiate the phase variables, the phase point variables and attributes
    containing the number of these different types in the full user-defined
    OCP.

    """
    ocp, user_syms = double_pendulum_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_state_variable_symbols()

    assert phase_backend.i == 0

    user_a0_t0 = sym.Symbol("a0_P0(t0)")
    user_a1_t0 = sym.Symbol("a1_P0(t0)")
    user_v0_t0 = sym.Symbol("v0_P0(t0)")
    user_v1_t0 = sym.Symbol("v1_P0(t0)")
    user_a0_tF = sym.Symbol("a0_P0(tF)")
    user_a1_tF = sym.Symbol("a1_P0(tF)")
    user_v0_tF = sym.Symbol("v0_P0(tF)")
    user_v1_tF = sym.Symbol("v1_P0(tF)")

    a0 = ca.SX.sym("a0_P0")
    a1 = ca.SX.sym("a1_P0")
    v0 = ca.SX.sym("v0_P0")
    v1 = ca.SX.sym("v1_P0")

    a0_t0 = ca.SX.sym("a0_P0(t0)")
    a1_t0 = ca.SX.sym("a1_P0(t0)")
    v0_t0 = ca.SX.sym("v0_P0(t0)")
    v1_t0 = ca.SX.sym("v1_P0(t0)")
    a0_tF = ca.SX.sym("a0_P0(tF)")
    a1_tF = ca.SX.sym("a1_P0(tF)")
    v0_tF = ca.SX.sym("v0_P0(tF)")
    v1_tF = ca.SX.sym("v1_P0(tF)")

    _y0 = ca.SX.sym("_y0_P0")
    _y0_t0 = ca.SX.sym("_y0_t0_P0")
    _y0_tF = ca.SX.sym("_y0_tF_P0")
    _y1 = ca.SX.sym("_y1_P0")
    _y1_t0 = ca.SX.sym("_y1_t0_P0")
    _y1_tF = ca.SX.sym("_y1_tF_P0")
    _y2 = ca.SX.sym("_y2_P0")
    _y2_t0 = ca.SX.sym("_y2_t0_P0")
    _y2_tF = ca.SX.sym("_y2_tF_P0")
    _y3 = ca.SX.sym("_y3_P0")
    _y3_t0 = ca.SX.sym("_y3_t0_P0")
    _y3_tF = ca.SX.sym("_y3_tF_P0")

    _V_y0 = ca.SX.sym("_V_y0_P0")
    _V_y1 = ca.SX.sym("_V_y1_P0")
    _V_y2 = ca.SX.sym("_V_y2_P0")
    _V_y3 = ca.SX.sym("_V_y3_P0")

    _r_y0 = ca.SX.sym("_r_y0_P0")
    _r_y1 = ca.SX.sym("_r_y1_P0")
    _r_y2 = ca.SX.sym("_r_y2_P0")
    _r_y3 = ca.SX.sym("_r_y3_P0")

    expect_sub_mapping = {user_syms.a0: a0,
                          user_syms.a1: a1,
                          user_syms.v0: v0,
                          user_syms.v1: v1,
                          user_a0_t0: a0_t0,
                          user_a1_t0: a1_t0,
                          user_v0_t0: v0_t0,
                          user_v1_t0: v1_t0,
                          user_a0_tF: a0_tF,
                          user_a1_tF: a1_tF,
                          user_v0_tF: v0_tF,
                          user_v1_tF: v1_tF,
                          }
    expect_aux_data = {a0: _V_y0 * _y0 + _r_y0,
                       a1: _V_y1 * _y1 + _r_y1,
                       v0: _V_y2 * _y2 + _r_y2,
                       v1: _V_y3 * _y3 + _r_y3,
                       a0_t0: _V_y0 * _y0_t0 + _r_y0,
                       a1_t0: _V_y1 * _y1_t0 + _r_y1,
                       v0_t0: _V_y2 * _y2_t0 + _r_y2,
                       v1_t0: _V_y3 * _y3_t0 + _r_y3,
                       a0_tF: _V_y0 * _y0_tF + _r_y0,
                       a1_tF: _V_y1 * _y1_tF + _r_y1,
                       v0_tF: _V_y2 * _y2_tF + _r_y2,
                       v1_tF: _V_y3 * _y3_tF + _r_y3,
                       }

    assert phase_backend.y_var_user == phase.state_variables
    assert phase_backend.y_var_user == (user_syms.a0, user_syms.a1,
                                        user_syms.v0, user_syms.v1)
    assert phase_backend.y_t0_var_user == (user_a0_t0, user_a1_t0,
                                           user_v0_t0, user_v1_t0)
    assert phase_backend.y_tF_var_user == (user_a0_tF, user_a1_tF,
                                           user_v0_tF, user_v1_tF)
    assert phase_backend.num_y_var_full == 4
    assert phase_backend.num_y_point_var_full == 8

    utils.assert_ca_syms_identical(phase_backend.y_var_full,
                                   (_y0, _y1, _y2, _y3))
    utils.assert_ca_syms_identical(phase_backend.V_y_var_full,
                                   (_V_y0, _V_y1, _V_y2, _V_y3))
    utils.assert_ca_syms_identical(phase_backend.r_y_var_full,
                                   (_r_y0, _r_y1, _r_y2, _r_y3))
    utils.assert_ca_syms_identical(phase_backend.y_t0_var_full,
                                   (_y0_t0, _y1_t0, _y2_t0, _y3_t0))
    utils.assert_ca_syms_identical(phase_backend.y_tF_var_full,
                                   (_y0_tF, _y1_tF, _y2_tF, _y3_tF))
    utils.assert_ca_syms_identical(phase_backend.y_point_var_full,
                                   (_y0_t0, _y0_tF, _y1_t0, _y1_tF,
                                    _y2_t0, _y2_tF, _y3_t0, _y3_tF))

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_create_state_variable_symbols_br_specific(brachistochrone_fixture,
                                                   casadi_backend_fixture,
                                                   phase_backend_fixture,
                                                   utils):
    """Check correct instantiation of state variables by the phase backend.

    Use the brachistochrone fixture to ensure the phase backend correctly
    handles state variables in the backend. The phase backend should
    instantiate the phase variables, the phase point variables and attributes
    containing the number of these different types in the full user-defined
    OCP.

    """
    ocp, user_syms = brachistochrone_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_state_variable_symbols()

    assert phase_backend.i == 0

    user_x_t0 = sym.Symbol("x_P0(t0)")
    user_y_t0 = sym.Symbol("y_P0(t0)")
    user_v_t0 = sym.Symbol("v_P0(t0)")
    user_x_tF = sym.Symbol("x_P0(tF)")
    user_y_tF = sym.Symbol("y_P0(tF)")
    user_v_tF = sym.Symbol("v_P0(tF)")

    x = ca.SX.sym("x_P0")
    y = ca.SX.sym("y_P0")
    v = ca.SX.sym("v_P0")

    x_t0 = ca.SX.sym("x_P0(t0)")
    y_t0 = ca.SX.sym("y_P0(t0)")
    v_t0 = ca.SX.sym("v_P0(t0)")
    x_tF = ca.SX.sym("x_P0(tF)")
    y_tF = ca.SX.sym("y_P0(tF)")
    v_tF = ca.SX.sym("v_P0(tF)")

    _y0 = ca.SX.sym("_y0_P0")
    _y0_t0 = ca.SX.sym("_y0_t0_P0")
    _y0_tF = ca.SX.sym("_y0_tF_P0")
    _y1 = ca.SX.sym("_y1_P0")
    _y1_t0 = ca.SX.sym("_y1_t0_P0")
    _y1_tF = ca.SX.sym("_y1_tF_P0")
    _y2 = ca.SX.sym("_y2_P0")
    _y2_t0 = ca.SX.sym("_y2_t0_P0")
    _y2_tF = ca.SX.sym("_y2_tF_P0")

    _V_y0 = ca.SX.sym("_V_y0_P0")
    _V_y1 = ca.SX.sym("_V_y1_P0")
    _V_y2 = ca.SX.sym("_V_y2_P0")

    _r_y0 = ca.SX.sym("_r_y0_P0")
    _r_y1 = ca.SX.sym("_r_y1_P0")
    _r_y2 = ca.SX.sym("_r_y2_P0")

    expect_sub_mapping = {user_syms.x: x,
                          user_syms.y: y,
                          user_syms.v: v,
                          user_x_t0: x_t0,
                          user_y_t0: y_t0,
                          user_v_t0: v_t0,
                          user_x_tF: x_tF,
                          user_y_tF: y_tF,
                          user_v_tF: v_tF,
                          }
    expect_aux_data = {x: _V_y0 * _y0 + _r_y0,
                       y: _V_y1 * _y1 + _r_y1,
                       v: _V_y2 * _y2 + _r_y2,
                       x_t0: _V_y0 * _y0_t0 + _r_y0,
                       y_t0: _V_y1 * _y1_t0 + _r_y1,
                       v_t0: _V_y2 * _y2_t0 + _r_y2,
                       x_tF: _V_y0 * _y0_tF + _r_y0,
                       y_tF: _V_y1 * _y1_tF + _r_y1,
                       v_tF: _V_y2 * _y2_tF + _r_y2,
                       }

    assert phase_backend.y_var_user == phase.state_variables
    assert phase_backend.y_var_user == (user_syms.x, user_syms.y, user_syms.v)
    assert phase_backend.y_t0_var_user == (user_x_t0, user_y_t0, user_v_t0)
    assert phase_backend.y_tF_var_user == (user_x_tF, user_y_tF, user_v_tF)
    assert phase_backend.num_y_var_full == 3
    assert phase_backend.num_y_point_var_full == 6

    utils.assert_ca_syms_identical(phase_backend.y_var_full,
                                   (_y0, _y1, _y2))
    utils.assert_ca_syms_identical(phase_backend.V_y_var_full,
                                   (_V_y0, _V_y1, _V_y2))
    utils.assert_ca_syms_identical(phase_backend.r_y_var_full,
                                   (_r_y0, _r_y1, _r_y2))
    utils.assert_ca_syms_identical(phase_backend.y_t0_var_full,
                                   (_y0_t0, _y1_t0, _y2_t0))
    utils.assert_ca_syms_identical(phase_backend.y_tF_var_full,
                                   (_y0_tF, _y1_tF, _y2_tF))
    utils.assert_ca_syms_identical(phase_backend.y_point_var_full,
                                   (_y0_t0, _y0_tF, _y1_t0, _y1_tF,
                                    _y2_t0, _y2_tF))

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_create_control_variable_symbols_dp_specific(double_pendulum_fixture,
                                                     casadi_backend_fixture,
                                                     phase_backend_fixture,
                                                     utils):
    """Check correct instantiation of control variables by the phase backend.

    Use the double pendulum fixture to ensure the phase backend correctly
    handles control variables in the backend. The phase backend should
    instantiate the phase variables and an attribute containing the number
    in the full user-defined OCP.

    """
    ocp, user_syms = double_pendulum_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_control_variable_symbols()

    assert phase_backend.i == 0

    T0 = ca.SX.sym("T0_P0")
    T1 = ca.SX.sym("T1_P0")

    _u0 = ca.SX.sym("_u0_P0")
    _u1 = ca.SX.sym("_u1_P0")

    _V_u0 = ca.SX.sym("_V_u0_P0")
    _V_u1 = ca.SX.sym("_V_u1_P0")

    _r_u0 = ca.SX.sym("_r_u0_P0")
    _r_u1 = ca.SX.sym("_r_u1_P0")

    expect_sub_mapping = {user_syms.T0: T0,
                          user_syms.T1: T1,
                          }
    expect_aux_data = {T0: _V_u0 * _u0 + _r_u0,
                       T1: _V_u1 * _u1 + _r_u1,
                       }

    assert phase_backend.u_var_user == phase.control_variables
    assert phase_backend.u_var_user == (user_syms.T0, user_syms.T1)
    assert phase_backend.num_u_var_full == 2

    utils.assert_ca_syms_identical(phase_backend.u_var_full,
                                   (_u0, _u1))
    utils.assert_ca_syms_identical(phase_backend.V_u_var_full,
                                   (_V_u0, _V_u1))
    utils.assert_ca_syms_identical(phase_backend.r_u_var_full,
                                   (_r_u0, _r_u1))

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_create_control_variable_symbols_br_specific(brachistochrone_fixture,
                                                     casadi_backend_fixture,
                                                     phase_backend_fixture,
                                                     utils):
    """Check correct instantiation of control variables by the phase backend.

    Use the brachistochrone fixture to ensure the phase backend correctly
    handles control variables in the backend. The phase backend should
    instantiate the phase variables and an attribute containing the number
    in the full user-defined OCP.

    """
    ocp, user_syms = brachistochrone_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_control_variable_symbols()

    assert phase_backend.i == 0

    u = ca.SX.sym("u_P0")
    _u0 = ca.SX.sym("_u0_P0")
    _V_u0 = ca.SX.sym("_V_u0_P0")
    _r_u0 = ca.SX.sym("_r_u0_P0")

    expect_sub_mapping = {user_syms.u: u}
    expect_aux_data = {u: _V_u0 * _u0 + _r_u0}

    assert phase_backend.u_var_user == phase.control_variables
    assert phase_backend.u_var_user == (user_syms.u, )
    assert phase_backend.num_u_var_full == 1

    utils.assert_ca_syms_identical(phase_backend.u_var_full, (_u0, ))
    utils.assert_ca_syms_identical(phase_backend.V_u_var_full, (_V_u0, ))
    utils.assert_ca_syms_identical(phase_backend.r_u_var_full, (_r_u0, ))

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_create_integral_variable_symbols_dp_specific(double_pendulum_fixture,
                                                      casadi_backend_fixture,
                                                      phase_backend_fixture,
                                                      utils):
    """Check correct instantiation of integral variables by the phase backend.

    Use the double pendulum fixture to ensure the phase backend correctly
    handles integral variables in the backend. The phase backend should
    instantiate the phase variables and an attribute containing the number
    in the full user-defined OCP.

    """
    ocp, user_syms = double_pendulum_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_integral_variable_symbols()

    assert phase_backend.i == 0

    user_q0 = sym.Symbol("q0_P0")
    q0 = ca.SX.sym("q0_P0")
    _q0 = ca.SX.sym("_q0_P0")
    _V_q0 = ca.SX.sym("_V_q0_P0")
    _r_q0 = ca.SX.sym("_r_q0_P0")

    expect_sub_mapping = {user_q0: q0,
                          }
    expect_aux_data = {q0: _V_q0 * _q0 + _r_q0,
                       }

    assert phase_backend.q_var_user == phase.integral_variables
    assert phase_backend.q_var_user == (user_q0, )
    assert phase_backend.num_q_var_full == 1

    utils.assert_ca_syms_identical(phase_backend.q_var_full, (_q0, ))
    utils.assert_ca_syms_identical(phase_backend.V_q_var_full, (_V_q0, ))
    utils.assert_ca_syms_identical(phase_backend.r_q_var_full, (_r_q0, ))

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REF)
def test_create_time_variable_symbols(ocp_fixture,
                                      casadi_backend_fixture,
                                      phase_backend_fixture,
                                      utils):
    """Check correct instantiation of time variables by the phase backend.

    Use the double pendulum fixture to ensure the phase backend correctly
    handles integral variables in the backend. The phase backend should
    instantiate the phase variables and an attribute containing the number
    in the full user-defined OCP.

    self.t_vars_full = (self.ocp_phase._t0, self.ocp_phase._tF)
    self.num_t_vars_full = len(self.t_vars_full)
    self.t_norm = self.ocp_phase._STRETCH

    """
    ocp, user_syms = ocp_fixture
    phase = ocp.phases.A
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_time_variable_symbols()

    assert phase_backend.i == 0

    user_t0 = sym.Symbol("t0_P0")
    user_tF = sym.Symbol("tF_P0")

    t0 = ca.SX.sym("t0_P0")
    tF = ca.SX.sym("tF_P0")

    _t0 = ca.SX.sym("_t0_P0")
    _tF = ca.SX.sym("_tF_P0")

    _V_t0 = ca.SX.sym("_V_t0_P0")
    _V_tF = ca.SX.sym("_V_tF_P0")

    _r_t0 = ca.SX.sym("_r_t0_P0")
    _r_tF = ca.SX.sym("_r_tF_P0")

    expect_sub_mapping = {user_t0: t0,
                          user_tF: tF,
                          }
    expect_aux_data = {t0: _V_t0 * _t0 + _r_t0,
                       tF: _V_tF * _tF + _r_tF,
                       }

    assert phase_backend.t_var_user == phase.time_variables
    assert phase_backend.t_var_user == (user_t0, user_tF)
    assert phase_backend.num_t_var_full == 2

    utils.assert_ca_syms_identical(phase_backend.t_var_full, (_t0, _tF))
    assert hasattr(phase_backend, "V_t_var_full")
    assert hasattr(phase_backend, "r_t_var_full")

    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping[
            expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        print(backend_expr)
        print(expect_backend_expr)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    iterable = phase_backend.phase_user_to_backend_mapping.items()
    for user_sym, backend_sym in iterable:
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        assert user_sym in expect_sub_mapping
        expect_backend_sym = expect_sub_mapping[user_sym]
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)

    expected_t_norm_expr = 0.5 * (_tF - _t0)
    utils.assert_ca_expr_identical(phase_backend.t_norm, expected_t_norm_expr)


def test_collect_pycollo_variables_full_dp_specific(double_pendulum_fixture,
                                                    casadi_backend_fixture,
                                                    phase_backend_fixture,
                                                    utils):
    """Test collection of Pycollo vars/data in to groups by phase backend."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_variable_symbols()
    phase_backend.collect_pycollo_variables_full()

    _y0 = ca.SX.sym("_y0_P0")
    _y1 = ca.SX.sym("_y1_P0")
    _y2 = ca.SX.sym("_y2_P0")
    _y3 = ca.SX.sym("_y3_P0")
    _u0 = ca.SX.sym("_u0_P0")
    _u1 = ca.SX.sym("_u1_P0")
    _q0 = ca.SX.sym("_q0_P0")
    _t0 = ca.SX.sym("_t0_P0")
    _tF = ca.SX.sym("_tF_P0")

    assert phase_backend.num_var_full == 9
    assert phase_backend.num_each_var_full == (4, 2, 1, 2)
    expected_x_var_full = (_y0, _y1, _y2, _y3, _u0, _u1, _q0, _t0, _tF)
    utils.assert_ca_syms_identical(phase_backend.x_var_full,
                                   expected_x_var_full)

    _y0_t0 = ca.SX.sym("_y0_t0_P0")
    _y0_tF = ca.SX.sym("_y0_tF_P0")
    _y1_t0 = ca.SX.sym("_y1_t0_P0")
    _y1_tF = ca.SX.sym("_y1_tF_P0")
    _y2_t0 = ca.SX.sym("_y2_t0_P0")
    _y2_tF = ca.SX.sym("_y2_tF_P0")
    _y3_t0 = ca.SX.sym("_y3_t0_P0")
    _y3_tF = ca.SX.sym("_y3_tF_P0")

    assert phase_backend.num_point_var_full == 11
    expected_x_point_var_full = (_y0_t0, _y0_tF, _y1_t0, _y1_tF, _y2_t0,
                                 _y2_tF, _y3_t0, _y3_tF, _q0, _t0, _tF)
    utils.assert_ca_syms_identical(phase_backend.x_point_var_full,
                                   expected_x_point_var_full)


def test_collect_pycollo_variables_full_br_specific(brachistochrone_fixture,
                                                    casadi_backend_fixture,
                                                    phase_backend_fixture,
                                                    utils):
    """Test collection of Pycollo vars/data in to groups by phase backend."""
    ocp, user_syms = brachistochrone_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_variable_symbols()
    phase_backend.collect_pycollo_variables_full()

    _y0 = ca.SX.sym("_y0_P0")
    _y1 = ca.SX.sym("_y1_P0")
    _y2 = ca.SX.sym("_y2_P0")
    _u0 = ca.SX.sym("_u0_P0")
    _t0 = ca.SX.sym("_t0_P0")
    _tF = ca.SX.sym("_tF_P0")

    assert phase_backend.num_var_full == 6
    assert phase_backend.num_each_var_full == (3, 1, 0, 2)
    expected_x_var_full = (_y0, _y1, _y2, _u0, _t0, _tF)
    utils.assert_ca_syms_identical(phase_backend.x_var_full,
                                   expected_x_var_full)

    _y0_t0 = ca.SX.sym("_y0_t0_P0")
    _y0_tF = ca.SX.sym("_y0_tF_P0")
    _y1_t0 = ca.SX.sym("_y1_t0_P0")
    _y1_tF = ca.SX.sym("_y1_tF_P0")
    _y2_t0 = ca.SX.sym("_y2_t0_P0")
    _y2_tF = ca.SX.sym("_y2_tF_P0")

    assert phase_backend.num_point_var_full == 8
    expected_x_point_var_full = (_y0_t0, _y0_tF, _y1_t0, _y1_tF, _y2_t0,
                                 _y2_tF, _t0, _tF)
    utils.assert_ca_syms_identical(phase_backend.x_point_var_full,
                                   expected_x_point_var_full)


def test_collect_user_variables_dp_specific(double_pendulum_fixture,
                                            casadi_backend_fixture,
                                            phase_backend_fixture,
                                            utils):
    """Test collection of user vars/data in to groups by phase backend."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_variable_symbols()
    phase_backend.collect_user_variables()

    q0 = sym.Symbol("q0_P0")
    t0 = sym.Symbol("t0_P0")
    tF = sym.Symbol("tF_P0")

    assert phase_backend.y_var_user == (user_syms.a0, user_syms.a1,
                                        user_syms.v0, user_syms.v1)
    assert phase_backend.u_var_user == (user_syms.T0, user_syms.T1)
    assert phase_backend.q_var_user == (q0, )
    assert phase_backend.t_var_user == (t0, tF)
    assert phase_backend.x_var_user == (user_syms.a0, user_syms.a1,
                                        user_syms.v0, user_syms.v1,
                                        user_syms.T0, user_syms.T1,
                                        q0, t0, tF)

    a0_t0 = sym.Symbol("a0_P0(t0)")
    a1_t0 = sym.Symbol("a1_P0(t0)")
    v0_t0 = sym.Symbol("v0_P0(t0)")
    v1_t0 = sym.Symbol("v1_P0(t0)")
    a0_tF = sym.Symbol("a0_P0(tF)")
    a1_tF = sym.Symbol("a1_P0(tF)")
    v0_tF = sym.Symbol("v0_P0(tF)")
    v1_tF = sym.Symbol("v1_P0(tF)")

    assert phase_backend.y_t0_var_user == (a0_t0, a1_t0, v0_t0, v1_t0)
    assert phase_backend.y_tF_var_user == (a0_tF, a1_tF, v0_tF, v1_tF)
    assert phase_backend.y_point_var_user == (a0_t0, a0_tF, a1_t0, a1_tF,
                                              v0_t0, v0_tF, v1_t0, v1_tF)
    assert phase_backend.x_point_var_user == (a0_t0, a0_tF, a1_t0, a1_tF,
                                              v0_t0, v0_tF, v1_t0, v1_tF,
                                              q0, t0, tF)
    assert phase_backend.all_user_var == set((user_syms.a0, user_syms.a1,
                                              user_syms.v0, user_syms.v1,
                                              a0_t0, a0_tF, a1_t0, a1_tF,
                                              v0_t0, v0_tF, v1_t0, v1_tF,
                                              user_syms.T0, user_syms.T1,
                                              q0, t0, tF))


def test_collect_user_variables_br_specific(brachistochrone_fixture,
                                            casadi_backend_fixture,
                                            phase_backend_fixture,
                                            utils):
    """Test collection of user vars/data in to groups by phase backend."""
    ocp, user_syms = brachistochrone_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    phase_backend = phase_backend_fixture
    phase_backend.ocp_backend = backend
    phase_backend.ocp_phase = ocp.phases.A
    phase_backend.i = ocp.phases.A.phase_number
    phase_backend.create_aux_data_containers()
    phase_backend.create_variable_symbols()
    phase_backend.collect_user_variables()

    t0 = sym.Symbol("t0_P0")
    tF = sym.Symbol("tF_P0")

    assert phase_backend.y_var_user == (user_syms.x, user_syms.y, user_syms.v)
    assert phase_backend.u_var_user == (user_syms.u, )
    assert phase_backend.q_var_user == ()
    assert phase_backend.t_var_user == (t0, tF)
    assert phase_backend.x_var_user == (user_syms.x, user_syms.y, user_syms.v,
                                        user_syms.u, t0, tF)

    x_t0 = sym.Symbol("x_P0(t0)")
    y_t0 = sym.Symbol("y_P0(t0)")
    v_t0 = sym.Symbol("v_P0(t0)")
    x_tF = sym.Symbol("x_P0(tF)")
    y_tF = sym.Symbol("y_P0(tF)")
    v_tF = sym.Symbol("v_P0(tF)")

    assert phase_backend.y_t0_var_user == (x_t0, y_t0, v_t0)
    assert phase_backend.y_tF_var_user == (x_tF, y_tF, v_tF)
    assert phase_backend.y_point_var_user == (x_t0, x_tF, y_t0, y_tF,
                                              v_t0, v_tF)
    assert phase_backend.x_point_var_user == (x_t0, x_tF, y_t0, y_tF,
                                              v_t0, v_tF, t0, tF)
    assert phase_backend.all_user_var == set((user_syms.x, user_syms.y,
                                              user_syms.v, user_syms.u,
                                              x_t0, x_tF, y_t0, y_tF,
                                              v_t0, v_tF, t0, tF))


def test_create_phase_backends_dp_specific(double_pendulum_fixture,
                                           casadi_backend_fixture,
                                           phase_backend_fixture,
                                           utils):
    """Test create of phase backend for double pendulum problem."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()

    q0 = sym.Symbol("q0_P0")
    t0 = sym.Symbol("t0_P0")
    tF = sym.Symbol("tF_P0")

    a0_t0 = sym.Symbol("a0_P0(t0)")
    a1_t0 = sym.Symbol("a1_P0(t0)")
    v0_t0 = sym.Symbol("v0_P0(t0)")
    v1_t0 = sym.Symbol("v1_P0(t0)")
    a0_tF = sym.Symbol("a0_P0(tF)")
    a1_tF = sym.Symbol("a1_P0(tF)")
    v0_tF = sym.Symbol("v0_P0(tF)")
    v1_tF = sym.Symbol("v1_P0(tF)")

    assert backend.all_phase_user_var == set((user_syms.a0, user_syms.a1,
                                              user_syms.v0, user_syms.v1,
                                              a0_t0, a0_tF, a1_t0, a1_tF,
                                              v0_t0, v0_tF, v1_t0, v1_tF,
                                              user_syms.T0, user_syms.T1,
                                              q0, t0, tF))
    assert backend.all_user_var == set((user_syms.a0, user_syms.a1,
                                        user_syms.v0, user_syms.v1,
                                        a0_t0, a0_tF, a1_t0, a1_tF,
                                        v0_t0, v0_tF, v1_t0, v1_tF,
                                        user_syms.T0, user_syms.T1,
                                        q0, t0, tF,
                                        user_syms.m0, user_syms.p0))


def test_create_phase_backends_br_specific(brachistochrone_fixture,
                                           casadi_backend_fixture,
                                           phase_backend_fixture,
                                           utils):
    """Test create of phase backend for brachistochrone problem."""
    ocp, user_syms = brachistochrone_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()

    t0 = sym.Symbol("t0_P0")
    tF = sym.Symbol("tF_P0")

    x_t0 = sym.Symbol("x_P0(t0)")
    y_t0 = sym.Symbol("y_P0(t0)")
    v_t0 = sym.Symbol("v_P0(t0)")
    x_tF = sym.Symbol("x_P0(tF)")
    y_tF = sym.Symbol("y_P0(tF)")
    v_tF = sym.Symbol("v_P0(tF)")

    assert backend.all_phase_user_var == set((user_syms.x, user_syms.y,
                                              user_syms.v, user_syms.u,
                                              x_t0, x_tF, y_t0, y_tF,
                                              v_t0, v_tF, t0, tF))
    assert backend.all_user_var == set((user_syms.x, user_syms.y,
                                        user_syms.v, user_syms.u,
                                        x_t0, x_tF, y_t0, y_tF,
                                        v_t0, v_tF, t0, tF))


def test_create_full_continuous_variable_indexes_slices(
        double_pendulum_phase_backend_fixture):
    """Slices for continuous Pycollo variables are correctly indexed."""
    phase_backend = double_pendulum_phase_backend_fixture
    phase_backend.create_variable_symbols()
    phase_backend.preprocess_variables()
    phase_backend.create_full_continuous_variable_indexes_slices()

    assert phase_backend.y_slice_full == slice(0, 4)
    assert phase_backend.u_slice_full == slice(4, 6)
    assert phase_backend.q_slice_full == slice(6, 7)
    assert phase_backend.t_slice_full == slice(7, 9)
    assert phase_backend.yu_slice_full == slice(0, 6)
    assert phase_backend.qt_slice_full == slice(6, 9)
    assert phase_backend.yu_qt_split_full == 6


def test_create_full_point_variable_indexes_slices(
        double_pendulum_phase_backend_fixture):
    """Slices for endpoint Pycollo variables are correctly indexed."""
    phase_backend = double_pendulum_phase_backend_fixture
    phase_backend.create_variable_symbols()
    phase_backend.preprocess_variables()
    phase_backend.create_full_point_variable_indexes_slices()

    assert phase_backend.y_point_full_slice == slice(0, 8)
    assert phase_backend.q_point_full_slice == slice(8, 9)
    assert phase_backend.t_point_full_slice == slice(9, 11)
    assert phase_backend.qt_point_full_slice == slice(8, 11)
    assert phase_backend.y_point_qt_point_full_split == 8


def test_preprocess_aux_data_partition_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   utils):
    """User-supplied aux data is categorised correctly during preprocessing."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    backend.partition_all_user_problem_phase_aux_data()

    expect_user_phase_aux_data_syms = {
        user_syms.g,
        user_syms.k1,
        user_syms.I0,
        user_syms.I1,
        user_syms.s0,
        user_syms.c1,
    }
    expect_user_aux_data_supplied_in_ocp_and_phase = {
        user_syms.g: sym.sympify(0),
        user_syms.I0: user_syms.m0 * (user_syms.k0 ** 2 + user_syms.p0 ** 2),
        user_syms.I1: user_syms.m1 * (user_syms.k1 ** 2 + user_syms.p1 ** 2),
        user_syms.s0: sym.sin(user_syms.a0),
        user_syms.c1: sym.cos(user_syms.a1)}

    M01_sub_expr_0 = user_syms.m1 * user_syms.p1 * user_syms.l0
    M01_sub_expr_1 = user_syms.s0 * user_syms.s1 + user_syms.c0 * user_syms.c1
    M01_eqn = M01_sub_expr_0 * M01_sub_expr_1
    K0_sub_expr_0 = user_syms.m0 * user_syms.p0 + user_syms.m1 * user_syms.l0
    K0_sub_expr_1 = user_syms.g * (K0_sub_expr_0) * user_syms.c0
    K0_sub_expr_2 = user_syms.T0 + K0_sub_expr_1
    K0_sub_expr_3 = user_syms.s1 * user_syms.c0 - user_syms.s0 * user_syms.c1
    K0_sub_expr_4 = user_syms.m1 * user_syms.p1 * user_syms.l0 * K0_sub_expr_3
    K0_sub_expr_5 = K0_sub_expr_4 * user_syms.v1 ** 2
    K0_eqn = K0_sub_expr_2 + K0_sub_expr_5
    K1_sub_expr_0 = user_syms.g * user_syms.m1
    K1_sub_expr_1 = user_syms.T1 + K1_sub_expr_0 * user_syms.p1 * user_syms.c1
    K1_sub_expr_2 = user_syms.s0 * user_syms.c1 - user_syms.s1 * user_syms.c0
    K1_sub_expr_3 = user_syms.m1 * user_syms.p1 * user_syms.l0 * K1_sub_expr_2
    K1_sub_expr_4 = K1_sub_expr_3 * user_syms.v0 ** 2
    K1_eqn = K1_sub_expr_1 + K1_sub_expr_4
    expect_user_aux_data_phase_dependent = {
        user_syms.M00: user_syms.I0 + user_syms.m1 * user_syms.l0 ** 2,
        user_syms.M01: M01_eqn,
        user_syms.M11: user_syms.I1,
        user_syms.K0: K0_eqn,
        user_syms.K1: K1_eqn,
    }

    expect_user_aux_data_phase_independent = {
        user_syms.d0: sym.sympify(0.5),
        user_syms.k0: sym.sympify(1 / 12),
        user_syms.m1: sym.sympify(1.0),
        user_syms.p1: sym.sympify(0.5),
        user_syms.d1: sym.sympify(0.5),
    }

    detM_eqn = user_syms.M00 * user_syms.M11 - user_syms.M01 * user_syms.M10
    expect_user_aux_data_for_preprocessing = {
        user_syms.l0: user_syms.p0 + user_syms.d0,
        user_syms.l1: user_syms.p1 + user_syms.d1,
        user_syms.c0: sym.cos(user_syms.a0),
        user_syms.s1: sym.sin(user_syms.a1),
        user_syms.M10: user_syms.M01,
        user_syms.detM: detM_eqn,
    }

    pycollo_length = len(backend.user_phase_aux_data_syms)
    expect_length = len(expect_user_phase_aux_data_syms)
    assert pycollo_length == expect_length
    for user_sym in expect_user_phase_aux_data_syms:
        assert user_sym in backend.user_phase_aux_data_syms

    pycollo_length = len(backend.user_aux_data_supplied_in_ocp_and_phase)
    expect_length = len(expect_user_aux_data_supplied_in_ocp_and_phase)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_supplied_in_ocp_and_phase.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_supplied_in_ocp_and_phase
        backend_eqn = backend.user_aux_data_supplied_in_ocp_and_phase[user_sym]
        assert user_eqn == backend_eqn

    pycollo_length = len(backend.user_aux_data_phase_dependent)
    expect_length = len(expect_user_aux_data_phase_dependent)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_phase_dependent.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_phase_dependent
        backend_eqn = backend.user_aux_data_phase_dependent[user_sym]
        assert user_eqn == backend_eqn

    pycollo_length = len(backend.user_aux_data_phase_independent)
    expect_length = len(expect_user_aux_data_phase_independent)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_phase_independent.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_phase_independent
        backend_eqn = backend.user_aux_data_phase_independent[user_sym]
        assert user_eqn == backend_eqn

    pycollo_length = len(backend.user_aux_data_for_preprocessing)
    expect_length = len(expect_user_aux_data_for_preprocessing)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_for_preprocessing.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_for_preprocessing
        backend_eqn = backend.user_aux_data_for_preprocessing[user_sym]
        assert user_eqn == backend_eqn


def test_partition_user_problem_aux_data_dp_specific(double_pendulum_fixture,
                                                     casadi_backend_fixture,
                                                     utils):
    """User-supplied aux data is categorised correctly during preprocessing."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    backend.preprocess_user_problem_aux_data()

    M01_sub_expr_0 = user_syms.m1 * user_syms.p1 * user_syms.l0
    M01_sub_expr_1 = user_syms.s0 * user_syms.s1 + user_syms.c0 * user_syms.c1
    M01_eqn = M01_sub_expr_0 * M01_sub_expr_1
    K0_sub_expr_0 = user_syms.m0 * user_syms.p0 + user_syms.m1 * user_syms.l0
    K0_sub_expr_1 = user_syms.g * (K0_sub_expr_0) * user_syms.c0
    K0_sub_expr_2 = user_syms.T0 + K0_sub_expr_1
    K0_sub_expr_3 = user_syms.s1 * user_syms.c0 - user_syms.s0 * user_syms.c1
    K0_sub_expr_4 = user_syms.m1 * user_syms.p1 * user_syms.l0 * K0_sub_expr_3
    K0_sub_expr_5 = K0_sub_expr_4 * user_syms.v1 ** 2
    K0_eqn = K0_sub_expr_2 + K0_sub_expr_5
    K1_sub_expr_0 = user_syms.g * user_syms.m1
    K1_sub_expr_1 = user_syms.T1 + K1_sub_expr_0 * user_syms.p1 * user_syms.c1
    K1_sub_expr_2 = user_syms.s0 * user_syms.c1 - user_syms.s1 * user_syms.c0
    K1_sub_expr_3 = user_syms.m1 * user_syms.p1 * user_syms.l0 * K1_sub_expr_2
    K1_sub_expr_4 = K1_sub_expr_3 * user_syms.v0 ** 2
    K1_eqn = K1_sub_expr_1 + K1_sub_expr_4
    detM_eqn = user_syms.M00 * user_syms.M11 - user_syms.M01 * user_syms.M10
    expect_user_aux_data_phase_dependent = {
        user_syms.c0: sym.cos(user_syms.a0),
        user_syms.s1: sym.sin(user_syms.a1),
        user_syms.M00: user_syms.I0 + user_syms.m1 * user_syms.l0 ** 2,
        user_syms.M01: M01_eqn,
        user_syms.M10: user_syms.M01,
        user_syms.M11: user_syms.I1,
        user_syms.K0: K0_eqn,
        user_syms.K1: K1_eqn,
        user_syms.detM: detM_eqn,
    }

    expect_user_aux_data_phase_independent = {
        user_syms.d0: sym.sympify(0.5),
        user_syms.k0: sym.sympify(1 / 12),
        user_syms.m1: sym.sympify(1.0),
        user_syms.p1: sym.sympify(0.5),
        user_syms.d1: sym.sympify(0.5),
        user_syms.l0: user_syms.p0 + user_syms.d0,
        user_syms.l1: user_syms.p1 + user_syms.d1,
    }

    pycollo_length = len(backend.user_aux_data_phase_dependent)
    expect_length = len(expect_user_aux_data_phase_dependent)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_phase_dependent.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_phase_dependent
        backend_eqn = backend.user_aux_data_phase_dependent[user_sym]
        assert user_eqn == backend_eqn

    pycollo_length = len(backend.user_aux_data_phase_independent)
    expect_length = len(expect_user_aux_data_phase_independent)
    assert pycollo_length == expect_length
    iterable = expect_user_aux_data_phase_independent.items()
    for user_sym, user_eqn in iterable:
        assert user_sym in backend.user_aux_data_phase_independent
        backend_eqn = backend.user_aux_data_phase_independent[user_sym]
        assert user_eqn == backend_eqn


def test_preprocess_phase_ind_aux_data_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   utils):
    """Phase-independent aux data correctly added to aux data."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    backend.preprocess_user_problem_aux_data()

    p0 = ca.SX.sym("p0")

    d0 = ca.SX.sym("d0")
    k0 = ca.SX.sym("k0")
    m1 = ca.SX.sym("m1")
    p1 = ca.SX.sym("p1")
    d1 = ca.SX.sym("d1")
    l0 = ca.SX.sym("l0")
    l1 = ca.SX.sym("l1")

    expect_sub_mapping = {
        user_syms.d0: d0,
        user_syms.k0: k0,
        user_syms.m1: m1,
        user_syms.p1: p1,
        user_syms.d1: d1,
        user_syms.l0: l0,
        user_syms.l1: l1,
    }
    expect_aux_data = {
        d0: ca.DM(0.5),
        k0: ca.DM(1 / 12),
        m1: ca.DM(1.0),
        p1: ca.DM(0.5),
        d1: ca.DM(0.5),
        l0: p0 + d0,
        l1: p1 + d1,
    }
    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in backend.user_to_backend_mapping
        backend_sym = backend.user_to_backend_mapping[expect_user_sym]
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_preprocess_phase_dep_aux_data_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   utils):
    """Phase-dependent aux data correctly added to aux data."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()

    a0 = ca.SX.sym("a0_P0")
    a1 = ca.SX.sym("a1_P0")
    v0 = ca.SX.sym("v0_P0")
    v1 = ca.SX.sym("v1_P0")
    T0 = ca.SX.sym("T0_P0")
    T1 = ca.SX.sym("T1_P0")
    m0 = ca.SX.sym("m0")
    p0 = ca.SX.sym("p0")

    g = ca.SX.sym("g_P0")
    k0 = ca.SX.sym("k0")
    m1 = ca.SX.sym("m1")
    p1 = ca.SX.sym("p1")
    k1 = ca.SX.sym("k1_P0")
    l0 = ca.SX.sym("l0")
    I0 = ca.SX.sym("I0_P0")
    I1 = ca.SX.sym("I1_P0")
    c0 = ca.SX.sym("c0_P0")
    s0 = ca.SX.sym("s0_P0")
    c1 = ca.SX.sym("c1_P0")
    s1 = ca.SX.sym("s1_P0")
    M00 = ca.SX.sym("M00_P0")
    M01 = ca.SX.sym("M01_P0")
    M10 = ca.SX.sym("M10_P0")
    M11 = ca.SX.sym("M11_P0")
    K0 = ca.SX.sym("K0_P0")
    K1 = ca.SX.sym("K1_P0")
    detM = ca.SX.sym("detM_P0")

    expect_sub_mapping = {
        user_syms.g: g,
        user_syms.k1: k1,
        user_syms.I0: I0,
        user_syms.I1: I1,
        user_syms.c0: c0,
        user_syms.s0: s0,
        user_syms.c1: c1,
        user_syms.s1: s1,
        user_syms.M00: M00,
        user_syms.M01: M01,
        user_syms.M10: M10,
        user_syms.M11: M11,
        user_syms.K0: K0,
        user_syms.K1: K1,
        user_syms.detM: detM,
    }

    K0_sub_expr_0 = T0 + g * (m0 * p0 + m1 * l0) * c0
    K0_sub_expr_1 = m1 * p1 * l0 * (s1 * c0 - s0 * c1) * v1 ** 2
    K0_eqn = K0_sub_expr_0 + K0_sub_expr_1
    K1_sub_expr_0 = T1 + g * m1 * p1 * c1
    K1_sub_expr_1 = m1 * p1 * l0 * (s0 * c1 - s1 * c0) * v0 ** 2
    K1_eqn = K1_sub_expr_0 + K1_sub_expr_1
    expect_aux_data = {
        g: ca.DM(-9.81),
        k1: ca.DM(1 / 12),
        I0: m0 * (k0 ** 2 + p0 ** 2),
        I1: m1 * (k1 ** 2 + p1 ** 2),
        c0: ca.cos(a0),
        s0: ca.sin(a0),
        c1: ca.cos(a1),
        s1: ca.sin(a1),
        M00: I0 + m1 * l0 ** 2,
        M01: m1 * p1 * l0 * (s0 * s1 + c0 * c1),
        M10: M01,
        M11: I1,
        K0: K0_eqn,
        K1: K1_eqn,
        detM: M00 * M11 - M01 * M10,
    }
    for expect_user_sym, expect_backend_sym in expect_sub_mapping.items():
        expect_backend_expr = expect_aux_data[expect_backend_sym]
        assert expect_user_sym in phase_backend.phase_user_to_backend_mapping
        backend_sym = phase_backend.phase_user_to_backend_mapping.get(
            expect_user_sym)
        assert backend_sym in backend.aux_data
        backend_expr = backend.aux_data[backend_sym]
        utils.assert_ca_sym_identical(backend_sym, expect_backend_sym)
        utils.assert_ca_expr_identical(backend_expr, expect_backend_expr)


def test_preprocess_phase_state_eqn_dp_specific(double_pendulum_fixture,
                                                casadi_backend_fixture,
                                                utils):
    """Phase state equations preprocessed correctly."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()
    backend.collect_variables_substitutions()
    phase_backend.preprocess_state_equations()

    _y0, _y1, _y2, _y3 = phase_backend.y_var_full
    _V_y0, _V_y1, _V_y2, _V_y3 = phase_backend.V_y_var_full
    _r_y0, _r_y1, _r_y2, _r_y3 = phase_backend.r_y_var_full

    _u0, _u1 = phase_backend.u_var_full
    _V_u0, _V_u1 = phase_backend.V_u_var_full
    _r_u0, _r_u1 = phase_backend.r_u_var_full

    _s0, _s1 = backend.s_var_full
    _V_s0, _V_s1 = backend.V_s_var_full
    _r_s0, _r_s1 = backend.r_s_var_full

    _a0 = _V_y0 * _y0 + _r_y0
    _a1 = _V_y1 * _y1 + _r_y1
    _v0 = _V_y2 * _y2 + _r_y2
    _v1 = _V_y3 * _y3 + _r_y3
    _T0 = _V_u0 * _u0 + _r_u0
    _T1 = _V_u1 * _u1 + _r_u1
    _m0 = _V_s0 * _s0 + _r_s0
    _p0 = _V_s1 * _s1 + _r_s1

    g = phase_backend.all_user_to_backend_mapping[user_syms.g]
    _g = backend.aux_data[g]
    d0 = phase_backend.all_user_to_backend_mapping[user_syms.d0]
    _d0 = backend.aux_data[d0]
    k0 = phase_backend.all_user_to_backend_mapping[user_syms.k0]
    _k0 = backend.aux_data[k0]
    m1 = phase_backend.all_user_to_backend_mapping[user_syms.m1]
    _m1 = backend.aux_data[m1]
    p1 = phase_backend.all_user_to_backend_mapping[user_syms.p1]
    _p1 = backend.aux_data[p1]
    k1 = phase_backend.all_user_to_backend_mapping[user_syms.k1]
    _k1 = backend.aux_data[k1]
    _l0 = _p0 + _d0
    _I0 = _m0 * (_k0 ** 2 + _p0 ** 2)
    _I1 = _m1 * (_k1 ** 2 + _p1 ** 2)
    _c0 = ca.cos(_a0)
    _s0 = ca.sin(_a0)
    _c1 = ca.cos(_a1)
    _s1 = ca.sin(_a1)
    _M00 = _I0 + _m1 * _l0 ** 2
    _M01 = _m1 * _p1 * _l0 * (_s0 * _s1 + _c0 * _c1)
    _M10 = _M01
    _M11 = _I1
    _K0_sub_expr_0 = _T0 + _g * (_m0 * _p0 + _m1 * _l0) * _c0
    _K0_sub_expr_1 = _m1 * _p1 * _l0 * (_s1 * _c0 - _s0 * _c1) * _v1 ** 2
    _K0 = _K0_sub_expr_0 + _K0_sub_expr_1
    _K1_sub_expr_0 = _T1 + _g * _m1 * _p1 * _c1
    _K1_sub_expr_1 = _m1 * _p1 * _l0 * (_s0 * _c1 - _s1 * _c0) * _v0 ** 2
    _K1 = _K1_sub_expr_0 + _K1_sub_expr_1
    _detM = _M00 * _M11 - _M01 * _M10

    expect_y_eqn = (
        _v0,
        _v1,
        (_M11 * _K0 - _M01 * _K1) / _detM,
        (_M00 * _K1 - _M10 * _K0) / _detM,
    )

    utils.assert_ca_exprs_identical(phase_backend.y_eqn, expect_y_eqn)
    assert phase_backend.num_y_eqn == 4


def test_preprocess_phase_state_eqn_br_specific(brachistochrone_fixture,
                                                casadi_backend_fixture,
                                                utils):
    """Phase state equations preprocessed correctly."""
    ocp, user_syms = brachistochrone_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()
    backend.collect_variables_substitutions()
    phase_backend.preprocess_state_equations()

    _y0, _y1, _y2 = phase_backend.y_var_full
    _V_y0, _V_y1, _V_y2 = phase_backend.V_y_var_full
    _r_y0, _r_y1, _r_y2 = phase_backend.r_y_var_full

    _u0, = phase_backend.u_var_full
    _V_u0, = phase_backend.V_u_var_full
    _r_u0, = phase_backend.r_u_var_full

    expect_y_eqn = (
        (_V_y2 * _y2 + _r_y2) * ca.sin(_V_u0 * _u0 + _r_u0),
        (_V_y2 * _y2 + _r_y2) * ca.cos(_V_u0 * _u0 + _r_u0),
        9.81 * ca.cos(_V_u0 * _u0 + _r_u0),
    )

    utils.assert_ca_exprs_identical(phase_backend.y_eqn, expect_y_eqn)
    assert phase_backend.num_y_eqn == 3


def test_preprocess_phase_path_con_dp_specific(double_pendulum_fixture,
                                               casadi_backend_fixture,
                                               utils):
    """Phase path constraint equations preprocessed correctly."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()
    backend.collect_variables_substitutions()
    phase_backend.preprocess_path_constraints()

    expect_p_con = ()

    assert phase_backend.p_con == expect_p_con
    assert phase_backend.num_p_con == 0


def test_preprocess_phase_integrand_fnc_dp_specific(double_pendulum_fixture,
                                                    casadi_backend_fixture,
                                                    utils):
    """Phase integrand functions preprocessed correctly."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()
    backend.collect_variables_substitutions()
    phase_backend.preprocess_integrand_functions()

    _u0, _u1 = phase_backend.u_var_full
    _V_u0, _V_u1 = phase_backend.V_u_var_full
    _r_u0, _r_u1 = phase_backend.r_u_var_full

    _T0 = _V_u0 * _u0 + _r_u0
    _T1 = _V_u1 * _u1 + _r_u1

    expect_q_fnc = (_T0 ** 2 + _T1 ** 2, )

    utils.assert_ca_exprs_identical(phase_backend.q_fnc, expect_q_fnc)
    assert phase_backend.num_q_fnc == 1


def test_preprocess_phase_collect_cons_dp_specific(double_pendulum_fixture,
                                                   casadi_backend_fixture,
                                                   utils):
    """Phase integrand functions preprocessed correctly."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    phase_backend = backend.p[0]
    backend.preprocess_user_problem_aux_data()
    phase_backend.preprocess_auxiliary_data()
    backend.collect_variables_substitutions()
    phase_backend.preprocess_state_equations()
    phase_backend.preprocess_path_constraints()
    phase_backend.preprocess_integrand_functions()
    phase_backend.collect_constraints()

    assert phase_backend.num_c == 5

    phase_backend.check_all_needed_user_phase_aux_data_supplied()
    phase_backend.create_constraint_indexes_slices()

    assert phase_backend.y_eqn_slice == slice(0, 4)
    assert phase_backend.p_con_slice == slice(4, 4)
    assert phase_backend.q_fnc_slice == slice(4, 5)


def test_preprocess_objective_fnc_dp_specific(double_pendulum_fixture,
                                              casadi_backend_fixture,
                                              utils):
    """Objective function preprocessed correctly."""
    ocp, user_syms = double_pendulum_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    backend.preprocess_user_problem_aux_data()
    backend.preprocess_phase_backends()
    backend.preprocess_problem_backend()
    phase_backend = backend.p[0]

    _q0, = phase_backend.q_var_full
    _V_q0, = phase_backend.V_q_var_full
    _r_q0, = phase_backend.r_q_var_full

    expect_J = _V_q0 * _q0 + _r_q0

    utils.assert_ca_expr_identical(backend.J, expect_J)


def test_preprocess_objective_fnc_br_specific(brachistochrone_fixture,
                                              casadi_backend_fixture,
                                              utils):
    """Objective function preprocessed correctly."""
    ocp, user_syms = brachistochrone_fixture
    backend = casadi_backend_fixture
    backend.ocp = ocp
    backend.create_aux_data_containers()
    backend.create_point_variable_symbols()
    backend.create_phase_backends()
    backend.preprocess_user_problem_aux_data()
    backend.preprocess_phase_backends()
    backend.preprocess_problem_backend()
    phase_backend = backend.p[0]

    _tF = phase_backend.t_var_full[1]
    _V_tF = phase_backend.V_t_var_full[1]
    _r_tF = phase_backend.r_t_var_full[1]

    expect_J = _V_tF * _tF + _r_tF

    utils.assert_ca_expr_identical(backend.J, expect_J)


@pytest_cases.parametrize("ocp_fixture", ALL_FIXTURE_REF)
def test_casadi_backend_init(ocp_fixture):
    """Backend initialises without error."""
    ocp, syms = ocp_fixture
    ocp._initialise_backend()
