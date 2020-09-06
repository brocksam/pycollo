import numpy as np
import pytest
import sympy as sym
import sympy.physics.mechanics as me

import pycollo


# @pytest.fixture
# def fixture_pendulum():
	
# 	# Symbols
# 	q = sym.Symbol('q')
# 	u = sym.Symbol('u')
# 	T = sym.Symbol('T')
# 	m, l, I, g, k = sym.symbols('m l I g k')
# 	ml, mgl, sinq, mglsinq = sym.symbols('ml mgl sinq mglsinq')

# 	# Optimal Control Problem
# 	problem = OptimalControlProblem(name="Pendulum swing-up problem")
# 	phase = problem.new_phase("phase")	
# 	phase.state_variables = [q, u]
# 	phase.control_variables = T
# 	phase.state_equations = [u, (m*g*l*sym.sin(q) - T)/I]
# 	phase.integrand_functions = [T**2]
# 	phase.auxiliary_data = {}
# 	problem.objective_function = phase.integral_variables[0]

# 	problem._check_variables_and_equations()
# 	problem._initialise_backend()

# 	return problem


# def test_bounds_mapping(fixture_pendulum):

# 	q = sym.Symbol('q')
# 	u = sym.Symbol('u')
# 	fixture_pendulum.phases[0].bounds.state_variables = {
# 		q: [0, np.pi],
# 		u: [-10, 10],
# 		}

# 	with pytest.raises(NotImplementedError):
# 		fixture_pendulum._check_problem_and_phase_bounds()

# def test_bounds_point_time_float(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, initial_time=0.0)
# 	assert fixture_pendulum.bounds.initial_time_lower == 0.0
# 	assert fixture_pendulum.bounds.initial_time_upper == 0.0
# 	assert fixture_pendulum.bounds.initial_time == (0.0, 0.0)


# def test_bounds_point_time_str(fixture_pendulum):

# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, initial_time='a')


# def test_bounds_point_time_iter_1(fixture_pendulum):
# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, initial_time=[0.0])


# def test_bounds_point_time_iter_2(fixture_pendulum):
# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, final_time=np.array([0.0, 1.0]))
# 	assert fixture_pendulum.bounds.final_time_lower == 0.0
# 	assert fixture_pendulum.bounds.final_time_upper == 1.0
# 	assert fixture_pendulum.bounds.final_time == (0.0, 1.0)


# def test_bounds_point_time_iter_3(fixture_pendulum):
# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, initial_time=(0.0, 1.0, 2.0))


# def test_bounds_point_time_iter_4(fixture_pendulum):
# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, final_time=[0.0, 1.0, 2.0, 3.0])


# def test_bounds_state_float(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=0.0)
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_lower, np.array([0.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_upper, np.array([0.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state, np.array([[0.0, 0.0]]))


# def test_bounds_state_iter_1(fixture_pendulum):

# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[0.0])


# def test_bounds_state_iter_2(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[0.0, 1.0])
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_lower, np.array([0.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_upper, np.array([1.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state, np.array([[0.0, 1.0]]))


# def test_bounds_state_iter_3(fixture_pendulum):

# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[0.0, 1.0, 2.0])


# def test_bounds_state_iter_4(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[0.0, 1.0, 2.0, 3.0])
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_lower, np.array([0.0, 2.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_upper, np.array([1.0, 3.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state, np.array([[0.0, 1.0], [2.0, 3.0]]))


# def test_bounds_state_iter_2_3(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_lower, np.array([0.0, 2.0, 4.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_upper, np.array([1.0, 3.0, 5.0]))
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state, np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))


# def test_bounds_state_iter_3_2(fixture_pendulum):

# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


# def test_bounds_state_dict_both(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state={fixture_pendulum.state_variables[0]: 0.0, fixture_pendulum.state_variables[1]: (1.0, 2.0)})
# 	assert fixture_pendulum.bounds.state_lower == {fixture_pendulum.state_variables[0]: 0.0, fixture_pendulum.state_variables[1]: 1.0}
# 	assert fixture_pendulum.bounds.state_upper == {fixture_pendulum.state_variables[0]: 0.0, fixture_pendulum.state_variables[1]: 2.0}
# 	assert fixture_pendulum.bounds.state == {fixture_pendulum.state_variables[0]: (0.0, 0.0), fixture_pendulum.state_variables[1]: (1.0, 2.0)}


# def test_bounds_state_dict_lower_upper(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state_lower={fixture_pendulum.state_variables[0]: 0.0}, state_upper={fixture_pendulum.state_variables[1]: 1.0})
# 	assert fixture_pendulum.bounds.state_lower == {fixture_pendulum.state_variables[0]: 0.0}
# 	assert fixture_pendulum.bounds.state_upper == {fixture_pendulum.state_variables[1]: 1.0}
# 	assert fixture_pendulum.bounds.state == {fixture_pendulum.state_variables[0]: (0.0, None), fixture_pendulum.state_variables[1]: (None, 1.0)}


# def test_bounds_state_dict_lower_upper_type_conflict(fixture_pendulum):

# 	fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, state_lower={fixture_pendulum.state_variables[0]: 0.0}, state_upper=[1.0, 2.0])
# 	assert fixture_pendulum.bounds.state_lower == {fixture_pendulum.state_variables[0]: 0.0}
# 	np.testing.assert_array_equal(fixture_pendulum.bounds.state_upper, np.array([1.0, 2.0]))
# 	with pytest.raises(TypeError):
# 		fixture_pendulum.bounds.state


# def test_bounds_state_dict_both_non_var_key(fixture_pendulum):

# 	with pytest.raises(ValueError):
# 		fixture_pendulum.bounds = Bounds(optimal_control_problem=fixture_pendulum, control={fixture_pendulum.state_variables[0]: 0.0, fixture_pendulum.control_variables[0]: 0.0})
