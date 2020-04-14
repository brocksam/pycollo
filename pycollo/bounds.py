from abc import (ABC, abstractmethod)
from collections import namedtuple
from numbers import Number
from typing import (Iterable, Optional, Union)

import numpy as np
import scipy.optimize as optimize
import sympy as sym

from .node import Node
from .typing import OptionalBoundsType
from .utils import (fast_sympify, format_multiple_items_for_output, 
	supported_iter_types)


__all__ = ["EndpointBounds", "PhaseBounds"]


class BoundsABC(ABC):

	_NUMERICAL_INF_DEFAULT = 1e20
	_ASSUME_INF_BOUNDS_DEFAULT = True
	_REMOVE_CONST_VARS_DEFAULT = True
	_OVERRIDE_ENDPOINTS_DEFAULT = True
	_BOUND_CLASH_TOLERANCE_DEFAULT = 1e-6

	@abstractmethod
	def optimal_control_problem(self): pass

	@abstractmethod
	def _process_and_check_user_values(self): pass


class EndpointBounds(BoundsABC):
	
	def __init__(self, optimal_control_problem):

		self._ocp = optimal_control_problem

	@property
	def optimal_control_problem(self):
		return self._ocp
	
	def _process_and_check_user_values(self):
		raise NotImplementedError


phase_info_fields = ("name", "index", "backend")
PhaseInfo = namedtuple("PhaseInfo", phase_info_fields)

bounds_info_fields = ("user_bnds", "user_syms", "bnds_type", "num")
BoundsInfo = namedtuple("BoundsInfo", bounds_info_fields)


class PhaseBounds(BoundsABC):
	"""Bounds on variables and constraints associated with a phase.

	This class currently behaves like a data class, however additional 
	functionality will be added in the future to support robust checking of the 
	user-supplied values for the bounds. 

	Intended behaviour will be::

		* None values will be treated as no bounds, i.e. ['-inf', 'inf'].
		* Single values will be treated as equal lower and upper bounds.
		* Mappings will be accepted for `state_variables`, `control_variables`, 
			`initial_state_constraints` and `final_state_constraints`.
		* Keys in the mappings should be the strings of the corresponding 
			`state_variables` or `control_variables` for the phase.
		* 'inf' values will be replaced by a large floating point value so that 
			scaling can be done automatically.
		* The 'inf' replacement value can be changed in 
			`OptimalControlProblem.settings.inf_value`, the default is 1e19.
		* If a :obj:`np.ndarray` with size = (2, 2) is passed as a value then 
			the first dimension will be treated as corresponding to the 
			variable or constraint to be bounded.
		* If iterables are passed then they may contain a combination of None,
			single numerical values, and pairs of numerical values
		* Symbolic expressions should also be allowed if they can be converted 
			into numerical values when processed alongside auxiliary data.

	Note:
		* 'inf' values should be avoided where possible in order to give better 
			automatic scaling.

	Attributes:
		phase: The phase with which these bounds will be associated. Default 
			value is None.
		initial_time: Bounds on when the phase starts. Default value is None.
		final_time: Bounds on when the phase ends.  Default value is None.
		state_variables: Bounds on the phase's state variables. Default value 
			is None.
		control_variables: Bounds on the phase's control variables. Default 
			value is None.
		integral_variables: Bounds on the phase's integral variables. Default 
			value is None.
		path_constraints: Bounds on the phase's path constraints. Default value 
			is None.
		initial_state_constraints: Bounds on the phase's state variables at the 
			initial time. Default value is None.
		final_state_constraints: Bounds on the phase's state variables at the 
			final time. Default value is None.
	"""
	
	def __init__(self, phase: "Phase", *, 
			initial_time: Optional[float] = None, 
			final_time: Optional[float] = None, 
			state_variables: OptionalBoundsType = None, 
			control_variables: OptionalBoundsType = None,
			integral_variables: OptionalBoundsType = None,
			path_constraints: OptionalBoundsType = None,
			initial_state_constraints: OptionalBoundsType = None,
			final_state_constraints: OptionalBoundsType = None,
			):
		"""Bounds on variables and constraints associated with a phase.

		Args:
			phase: The phase with which these bounds will be associated.
			initial_time: Bounds on when the phase starts. Default value is 
				None.
			final_time: Bounds on when the phase ends.  Default value is None.
			state_variables: Bounds on the phase's state variables. Default 
				value is None.
			control_variables: Bounds on the phase's control variables. Default 
				value is None.
			integral_variables: Bounds on the phase's integral variables. 
				Default value is None.
			path_constraints: Bounds on the phase's path constraints. Default 
				value is None.
			initial_state_constraints: Bounds on the phase's state variables at
				the initial time. Default value is None.
			final_state_constraints: Bounds on the phase's state variables at
				the final time. Default value is None.
		"""

		self.phase = phase
		self.initial_time = initial_time
		self.final_time = final_time
		self.state_variables = state_variables
		self.control_variables = control_variables
		self.integral_variables = integral_variables
		self.path_constraints = path_constraints
		self.initial_state_constraints = initial_state_constraints
		self.final_state_constraints = final_state_constraints

	@property
	def optimal_control_problem(self):
		return self.phase.optimal_control_problem

	def _process_and_check_user_values(self, phase_backend):
		self._backend = phase_backend
		self._expr_graph = phase_backend.ocp_backend.expression_graph
		self._INF = self.optimal_control_problem.settings.inf_value
		p_info = self._get_phase_info(phase_backend)
		y_bnds = self._process_state_vars(p_info)
		u_bnds = self._process_control_vars(p_info)
		q_bnds = self._process_integral_vars(p_info)
		p_cons = self._process_path_cons(p_info)
		t_bnds = self._process_time_vars(p_info)
		print('\n\n\n')
		raise NotImplementedError

	def _get_phase_info(self, phase_backend):
		phase_name = phase_backend.ocp_phase.name
		phase_index = phase_backend.ocp_phase.phase_number
		phase_info = PhaseInfo(phase_name, phase_index, phase_backend)
		return phase_info

	def _process_state_vars(self, p_info):
		user_bnds = self.state_variables
		user_syms = p_info.backend.y_vars_user
		bnds_type = "state variable"
		num_expected = p_info.backend.num_y_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		y_bnds = self._process_single_type_of_values(bnds_info, p_info)
		return y_bnds

	def _process_control_vars(self, p_info):
		user_bnds = self.control_variables
		user_syms = p_info.backend.u_vars_user
		bnds_type = "control variable"
		num_expected = p_info.backend.num_u_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		u_bnds = self._process_single_type_of_values(bnds_info, p_info)
		return u_bnds

	def _process_integral_vars(self, p_info):
		user_bnds = self.integral_variables
		user_syms = p_info.backend.q_vars_user
		bnds_type = "integral variable"
		num_expected = p_info.backend.num_q_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		q_bnds = self._process_single_type_of_values(bnds_info, p_info)
		return q_bnds

	def _process_path_cons(self, p_info):
		pass

	def _process_time_vars(self, p_info):
		user_bnds = self.time_variables
		user_syms = p_info.backend.t_vars_user
		bnds_type = "time variable"
		num_expected = p_info.backend.num_t_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		q_bnds = self._process_single_type_of_values(bnds_info, p_info)
		return q_bnds

	def _process_single_type_of_values(self, bnds_info, p_info):
		"""
		A single bound can either be a single value or a pair of values. If a 
		single value is supplied then the lower and upper bound are set equal. 
		For optimal control problem variables, if the lower and upper bound are 
		equal then that variable shouldn't be treated as an optimal control 
		problem variable. Pycollo should provide an option to automatically 
		override this and instead treat the variable as an auxiliary constant, 
		or raise an informative error to the user. If a pair of values are 
		supplied then the first value is the lower bound and the second value 
		is the upper bound. The lower bound must be less than or equal to the 
		upper bound (within a tolerance than can be set in the settings module).

		There are 3 options:
			1. The bounds are a mapping with the key being something sensible;
			2. The bounds are an iterable; and
			3. The bounds are a single value.
		"""
		if isinstance(bnds_info.user_bnds, dict):
			bnds = self._process_mapping_bounds_instance(bnds_info, p_info)
		elif bnds_info.user_bnds is None:
			bnds = self._process_none_bounds_instance(bnds_info, p_info)
		else:
			raise NotImplementedError
		bnds = self._check_lower_against_upper(bnds, bnds_info, p_info)
		return bnds

	def _process_mapping_bounds_instance(self, bnds_info, p_info):
		if bnds_info.user_syms is None:
			msg = f"Can't use mapping for {bnds_info.bnds_type} bounds."
			raise TypeError(msg)
		bnds = []
		for bnd_i, user_sym in enumerate(bnds_info.user_syms):
			bnd = bnds_info.user_bnds.get(user_sym)
			bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info, p_info)
			bnd = self._as_lower_upper_pair(bnd_info, p_info)
			bnds.append(bnd)
		return bnds

	def _check_user_bound_missing(self, bnd_info, p_info):
		ocp_settings = self.optimal_control_problem.settings
		is_bnd_none = bnd_info.user_bnds is None
		is_inf_assumed = ocp_settings.assume_inf_bounds
		if is_bnd_none and not is_inf_assumed:
			self._raise_user_bound_missing_error(bnd_info, p_info)

	def _raise_user_bound_missing_error(self, bnd_info, p_info):
		msg = (f"No bounds have been supplied for the {bnd_info.bnds_type} "
			f"'{bnd_info.user_syms}' (index #{bnd_info.num}) in phase "
			f"{p_info.name} (index #{p_info.index}).")
		raise ValueError(msg)		

	def _process_iterable_bounds_instance(self, bnds_info, p_info):
		pass

	def _process_single_value_bounds_instance(self, bnds_info, p_info):
		pass

	def _process_none_bounds_instance(self, bnds_info, p_info):
		bnds = []
		for bnd_i, user_sym in enumerate(bnds_info.user_syms):
			bnd = None
			bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info, p_info)
			bnd = self._as_lower_upper_pair(bnd_info, p_info)
			bnds.append(bnd)
		return bnds

	def _as_lower_upper_pair(self, bnd_info, p_info):
		bnds = np.array(bnd_info.user_bnds).flatten()
		if bnds.shape == (1, ):
			both = "lower and upper bounds"
			both_info = bnd_info._replace(user_bnds=bnds[0])
			lower_bnd = self._get_bound_as_number(both_info, p_info, both)
			upper_bnd = lower_bnd
		elif bnds.shape == (2, ):
			lower = "lower bound"
			upper = "upper bound"
			lower_info = bnd_info._replace(user_bnds=bnds[0])
			upper_info = bnd_info._replace(user_bnds=bnds[1])
			lower_bnd = self._get_bound_as_number(lower_info, p_info, lower)
			upper_bnd = self._get_bound_as_number(upper_info, p_info, upper)
		else:
			raise ValueError
		lower_bnd = - self._INF if lower_bnd is None else lower_bnd
		upper_bnd = self._INF if upper_bnd is None else upper_bnd
		return [lower_bnd, upper_bnd]

	def _get_bound_as_number(self, bnd_info, p_info, lower_upper):
		bnd = bnd_info.user_bnds
		if bnd is None:
			return bnd
		bnd = fast_sympify(bnd).xreplace(self._backend.all_subs_mappings)
		node = Node(bnd, self._expr_graph)
		if not node.is_precomputable:
			msg = (f"The user-supplied {lower_upper} for the {bnd_info.bnds_type} "
				f"'{bnd_info.user_sym}' (index #{bnd_info.num}) in phase "
				f"{p_info.name} (index #{p_info.index}) of '{bnd}' cannot be "
				f"precomputed.")
			raise ValueError(msg)
		return node.value

	def _check_lower_against_upper(self, bnds, bnds_info, p_info):
		if not bnds:
			return np.empty(shape=(0, 2), dtype=float)
		bnds = np.array(bnds, dtype=float)
		bnds = self._check_lower_same_as_upper_to_tol(bnds, bnds_info, p_info)
		bnds = self._check_lower_less_than_upper(bnds, bnds_info, p_info)
		return bnds

	def _check_lower_same_as_upper_to_tol(self, bnds, bnds_info, p_info):
		lower_bnds = bnds[:, 0]
		upper_bnds = bnds[:, 1]
		rtol = self.optimal_control_problem.settings.bound_clash_tolerance
		are_same = np.isclose(lower_bnds, upper_bnds, rtol=rtol, atol=0.0)
		if np.any(are_same):
			msg = (f"Can't currently handle casting values constrained by "
				f"bounds to constants.")
			raise NotImplementedError(msg)
		mean_bnds = (lower_bnds + upper_bnds)/2
		bnds[are_same, 0] = mean_bnds[are_same]
		bnds[are_same, 1] = mean_bnds[are_same]
		bnds = self._extract_variables_to_constants(bnds, bnds_info, p_info)
		return bnds

	def _check_lower_less_than_upper(self, bnds, bnds_info, p_info):
		lower_bnds = bnds[:, 0]
		upper_bnds = bnds[:, 1]
		lower_less_than_upper = lower_bnds <= upper_bnds
		all_less_than = np.all(lower_less_than_upper)
		if not all_less_than:
			error_indices = np.flatnonzero(~lower_less_than_upper)
			error_syms = np.array(bnds_info.user_syms)[error_indices]
			plural_needed = len(error_indices) > 1
			bound_plural = "bounds" if plural_needed else "bound"
			index_plural = "indices" if plural_needed else "index"
			bnds_type_plural = f"{bnds_info.bnds_type}{'s' if plural_needed else ''}"
			user_syms_formatted = format_multiple_items_for_output(error_syms)
			user_indices_formatted = format_multiple_items_for_output(
				error_indices, wrapping_char="", prefix_char="#")
			lower_bnds_formatted = format_multiple_items_for_output(
				lower_bnds[error_indices])
			upper_bnds_formatted = format_multiple_items_for_output(
				upper_bnds[error_indices])
			msg = (f"The user-supplied upper {bound_plural} for the "
				f"{bnds_type_plural} {user_syms_formatted} ({index_plural} "
				f"{user_indices_formatted}) in phase {p_info.name} (index "
				f"#{p_info.index}) of {upper_bnds_formatted} cannot be less "
				f"than the user-supplied lower {bound_plural} of "
				f"{lower_bnds_formatted}.")
			raise ValueError(msg)
		return bnds

	def _extract_variables_to_constants(self, bnds, bnds_info, p_info):
		# self.optimal_control_problem.settings.remove_constant_variables
		return bnds

		# self._y_l, self._y_u = parse_bounds_to_np_array(self.state, self._ocp._y_vars_user, 'state')
		# self._y_needed = compare_lower_and_upper(self._y_l, self._y_u, 'state')
		# self._y_b_needed = np.repeat(self._y_needed, 2)

		# self._y_l_needed = self._y_l[self._y_needed.astype(bool)]
		# self._y_u_needed = self._y_u[self._y_needed.astype(bool)]

		# # Control
		# self._u_l, self._u_u = parse_bounds_to_np_array(self.control, self._ocp._u_vars_user, 'control')
		# self._u_needed = compare_lower_and_upper(self._u_l, self._u_u, 'control')

		# self._u_l_needed = self._u_l[self._u_needed.astype(bool)]
		# self._u_u_needed = self._u_u[self._u_needed.astype(bool)]

		# # Integral
		# if self.infer_bounds:
		# 	raise NotImplementedError
		# 	self.integral = infer_bound(self.integral, self._ocp._q_vars_user, self._ocp._q_funcs_user, aux_data, aux_subs, 'integral')
		# self._q_l, self._q_u = parse_bounds_to_np_array(self.integral, self._ocp._q_vars_user, 'integral')
		# self._q_needed = compare_lower_and_upper(self._q_l, self._q_u, 'integral')

		# self._q_l_needed = self._q_l[self._q_needed.astype(bool)]
		# self._q_u_needed = self._q_u[self._q_needed.astype(bool)]

		# # Time
		# _ = parse_bounds_to_np_array(np.array([[self._t0_l, self._tF_l], [self._t0_u, self._tF_u]], dtype=np.float64), self._ocp._t_vars_user, 'time')
		# if self._t0_l > self._t0_u:
		# 	msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be greater than the lower bound ({self._t0_l}).")
		# 	raise ValueError(msg)
		# elif self._tF_l > self._tF_u:
		# 	msg = (f"The upper bound for `Bounds.final_time` ({self._tF_u}) must be greater than the lower bound ({self._tF_l}).")
		# 	raise ValueError(msg)
		# elif self._t0_l > self._tF_l:
		# 	msg = (f"The lower bound for `Bounds.initial_time` ({self._t0_l}) must be less than the lower bound for `Bounds.final_time` ({self._tF_l}).")
		# 	raise ValueError(msg)
		# elif self._t0_u > self._tF_u:
		# 	msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be less than the upper bound for `Bounds.final_time` ({self._tF_u}).")
		# 	raise ValueError(msg)
		# else:
		# 	self._t0_l = self.initial_time_lower
		# 	self._t0_u = self.initial_time_upper
		# 	self._tF_l = self.final_time_lower
		# 	self._tF_u = self.final_time_upper
		# 	self._t_l = np.array([self._t0_l, self._tF_l]).flatten()
		# 	self._t_u = np.array([self._t0_u, self._tF_u]).flatten()
		# self._t_needed = compare_lower_and_upper([self._t0_l, self._tF_l], [self._t0_u, self._tF_u], 'time')

		# self._t_l_needed = self._t_l[self._t_needed.astype(bool)]
		# self._t_u_needed = self._t_u[self._t_needed.astype(bool)]

		# # Parameter
		# self._s_l, self._s_u = parse_bounds_to_np_array(self.parameter, self._ocp._s_vars_user, 'parameter')
		# self._s_needed = compare_lower_and_upper(self._s_l, self._s_u, 'parameter')

		# self._s_l_needed = self._s_l[self._s_needed.astype(bool)]
		# self._s_u_needed = self._s_u[self._s_needed.astype(bool)]

		# self._x_needed = np.concatenate([self._y_needed, self._u_needed, self._q_needed, self._t_needed, self._s_needed])
		# self._x_b_needed = np.concatenate([self._y_b_needed, self._q_needed, self._t_needed, self._s_needed])

		# # Path
		# if self.infer_bounds:
		# 	if self._ocp._c_cons_user:
		# 		raise NotImplementedError
		# self._c_l, self._c_u = parse_bounds_to_np_array(self.path, self._ocp._c_cons_user, 'path constraint')
		# _ = compare_lower_and_upper(self._c_l, self._c_u, 'path constraint')

		# # Boundary
		# self._y_b_l, self._y_b_u = parse_bounds_to_np_array(self.state_endpoint, self._ocp._y_b_cons_user, 'state_endpoint_constraint')
		# _ = compare_lower_and_upper(self._b_l, self._b_u, 'state_endpoint_constraint')

		# # Boundary
		# self._b_l, self._b_u = parse_bounds_to_np_array(self.boundary, self._ocp._b_cons_user, 'boundary constraint')
		# _ = compare_lower_and_upper(self._b_l, self._b_u, 'boundary constraint')

		# self._bounds_checked = True











def replace_symbol_with_numeric_value(expression_graph, symbol):

	return num_val




class Bounds:

	def __init__(self, ocp_backend):
		self.ocp_backend = ocp_backend
		for p in self.ocp_backend.p:
			p.ocp_phase.bounds._process_and_check_user_values(p)








# class Bounds():

# 	INF = 1e19
	
# 	def __init__(self, optimal_control_problem, *, initial_time=None, initial_time_lower=None, initial_time_upper=None, final_time=None, final_time_lower=None, final_time_upper=None, state=None, state_lower=None, state_upper=None, control=None, control_lower=None, control_upper=None, integral=None, integral_lower=None, integral_upper=None, parameter=None, parameter_lower=None, parameter_upper=None, path=None, path_lower=None, path_upper=None, state_endpoint=None, state_endpoint_lower=None, state_endpoint_upper=None, boundary=None, boundary_lower=None, boundary_upper=None, default_inf=False, infer_bounds=False):

# 		# Optimal Control Problem
# 		self._ocp = optimal_control_problem
# 		self.default_inf=default_inf
# 		self.infer_bounds=infer_bounds
# 		self._bounds_checked = False

# 		if initial_time is not None:
# 			_ = self.kwarg_conflict_check(initial_time_lower, initial_time_upper, 'initial_time')
# 			self.initial_time = initial_time
# 		else:
# 			self.initial_time_lower = initial_time_lower
# 			self.initial_time_upper = initial_time_upper

# 		if final_time is not None:
# 			_ = self.kwarg_conflict_check(final_time_lower, final_time_upper, 'final_time')
# 			self.final_time = final_time
# 		else:
# 			self.final_time_lower = final_time_lower
# 			self.final_time_upper = final_time_upper

# 		if state is not None:
# 			_ = self.kwarg_conflict_check(state_lower, state_upper, 'state')
# 			self.state = state
# 		else:
# 			self.state_lower = state_lower
# 			self.state_upper = state_upper

# 		if control is not None:
# 			_ = self.kwarg_conflict_check(control_lower, control_upper, 'control')
# 			self.control = control
# 		else:
# 			self.control_lower = control_lower
# 			self.control_upper = control_upper

# 		if integral is not None:
# 			_ = self.kwarg_conflict_check(integral_lower, integral_upper, 'integral')
# 			self.integral = integral
# 		else:
# 			self.integral_lower = integral_lower
# 			self.integral_upper = integral_upper

# 		if parameter is not None:
# 			_ = self.kwarg_conflict_check(parameter_lower, parameter_upper, 'parameter')
# 			self.parameter = parameter
# 		else:
# 			self.parameter_lower = parameter_lower
# 			self.parameter_upper = parameter_upper

# 		if path is not None:
# 			_ = self.kwarg_conflict_check(path_lower, path_upper, 'path')
# 			self.path = path
# 		else:
# 			self.path_lower = path_lower
# 			self.path_upper = path_upper

# 		if state_endpoint is not None:
# 			_ = self.kwarg_conflict_check(state_endpoint_lower, state_endpoint_upper, 'state_endpoint')
# 			self.state_endpoint = state_endpoint
# 		else:
# 			self.state_endpoint_lower = state_endpoint_lower
# 			self.state_endpoint_upper = state_endpoint_upper

# 		if boundary is not None:
# 			_ = self.kwarg_conflict_check(boundary_lower, boundary_upper, 'boundary')
# 			self.boundary = boundary
# 		else:
# 			self.boundary_lower = boundary_lower
# 			self.boundary_upper = boundary_upper

# 	@property
# 	def infer_bounds(self):
# 		return self._infer_bounds
	
# 	@infer_bounds.setter
# 	def infer_bounds(self, val):
# 		self._infer_bounds = bool(val)

# 	@property
# 	def default_inf(self):
# 		return self._default_inf
	
# 	@default_inf.setter
# 	def default_inf(self, val):
# 		self._default_inf = bool(val)

# 	@property
# 	def initial_time(self):
# 		return self.initial_time_lower, self.initial_time_upper
	
# 	@initial_time.setter
# 	def initial_time(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'initial_time', vec=False)
# 		self.initial_time_lower = l
# 		self.initial_time_upper = u

# 	@property
# 	def initial_time_lower(self):
# 		return self._t0_l
	
# 	@initial_time_lower.setter
# 	def initial_time_lower(self, t0_l):
# 		self._t0_l = self.parse_single_bounds(t0_l, 'initial_time_lower', vec=False)

# 	@property
# 	def initial_time_upper(self):
# 		return self._t0_u
	
# 	@initial_time_upper.setter
# 	def initial_time_upper(self, t0_u):
# 		self._t0_u = self.parse_single_bounds(t0_u, 'initial_time_upper', vec=False)

# 	@property
# 	def final_time(self):
# 		return (self.final_time_lower, self.final_time_upper)
	
# 	@final_time.setter
# 	def final_time(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'final_time', vec=False)
# 		self.final_time_lower = l
# 		self.final_time_upper = u

# 	@property
# 	def final_time_lower(self):
# 		return self._tF_l
	
# 	@final_time_lower.setter
# 	def final_time_lower(self, tF_l):
# 		self._tF_l = self.parse_single_bounds(tF_l, 'final_time_lower', vec=False)

# 	@property
# 	def final_time_upper(self):
# 		return self._tF_u
	
# 	@final_time_upper.setter
# 	def final_time_upper(self, tF_u):
# 		self._tF_u = self.parse_single_bounds(tF_u, 'final_time_upper', vec=False)

# 	@property
# 	def state(self):
# 		return self.return_user_bounds(self._y_l, self._y_u, 'state')
	
# 	@state.setter
# 	def state(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'state', self._ocp._y_vars_user)
# 		self.state_lower = l
# 		self.state_upper = u

# 	@property
# 	def state_lower(self):
# 		return self._y_l
	
# 	@state_lower.setter
# 	def state_lower(self, y_l):
# 		self._y_l = self.parse_single_bounds(y_l, 'state_lower', self._ocp._y_vars_user)

# 	@property
# 	def state_upper(self):
# 		return self._y_u
	
# 	@state_upper.setter
# 	def state_upper(self, y_u):
# 		self._y_u = self.parse_single_bounds(y_u, 'state_upper', self._ocp._y_vars_user)

# 	@property
# 	def control(self):
# 		return self.return_user_bounds(self._u_l, self._u_u, 'control')
	
# 	@control.setter
# 	def control(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'control', self._ocp._u_vars_user)
# 		self.control_lower = l
# 		self.control_upper = u

# 	@property
# 	def control_lower(self):
# 		return self._u_l
	
# 	@control_lower.setter
# 	def control_lower(self, u_l):
# 		self._u_l = self.parse_single_bounds(u_l, 'control_lower', self._ocp._u_vars_user)

# 	@property
# 	def control_upper(self):
# 		return self._u_u
	
# 	@control_upper.setter
# 	def control_upper(self, u_u):
# 		self._u_u = self.parse_single_bounds(u_u, 'control_upper', self._ocp._u_vars_user)

# 	@property
# 	def integral(self):
# 		return self.return_user_bounds(self._q_l, self._q_u, 'integral')
	
# 	@integral.setter
# 	def integral(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'integral', self._ocp._q_vars_user)
# 		self.integral_lower = l
# 		self.integral_upper = u

# 	@property
# 	def integral_lower(self):
# 		return self._q_l
	
# 	@integral_lower.setter
# 	def integral_lower(self, q_l):
# 		self._q_l = self.parse_single_bounds(q_l, 'integral_lower', self._ocp._q_vars_user)

# 	@property
# 	def integral_upper(self):
# 		return self._q_u
	
# 	@integral_upper.setter
# 	def integral_upper(self, q_u):
# 		self._q_u = self.parse_single_bounds(q_u, 'integral_upper', self._ocp._q_vars_user)

# 	@property
# 	def time(self):
# 		return np.array([self.initial_time, self.final_time])

# 	# Include properties for time variables so that initial and final times can be set in the same manner as all other state, control, integral and parameter variables.

# 	@property
# 	def parameter(self):
# 		return self.return_user_bounds(self._s_l, self._s_u, 'parameter')
	
# 	@parameter.setter
# 	def parameter(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'parameter', self._ocp._s_vars_user)
# 		self.parameter_lower = l
# 		self.parameter_upper = u

# 	@property
# 	def parameter_lower(self):
# 		return self._s_l
	
# 	@parameter_lower.setter
# 	def parameter_lower(self, s_l):
# 		self._s_l = self.parse_single_bounds(s_l, 'parameter_lower', self._ocp._s_vars_user)

# 	@property
# 	def parameter_upper(self):
# 		return self._s_u
	
# 	@parameter_upper.setter
# 	def parameter_upper(self, s_u):
# 		self._s_u = self.parse_single_bounds(s_u, 'parameter_upper', self._ocp._s_vars_user)

# 	@property
# 	def path(self):
# 		return self.return_user_bounds(self._c_l, self._c_u, 'path')
	
# 	@path.setter
# 	def path(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'path', self._ocp._c_cons_user)
# 		self.path_lower = l
# 		self.path_upper = u

# 	@property
# 	def path_lower(self):
# 		return self._c_l
	
# 	@path_lower.setter
# 	def path_lower(self, c_l):
# 		self._c_l = self.parse_single_bounds(c_l, 'path_lower', self._ocp._c_cons_user)

# 	@property
# 	def path_upper(self):
# 		return self._c_u
	
# 	@path_upper.setter
# 	def path_upper(self, c_u):
# 		self._c_u = self.parse_single_bounds(c_u, 'path_upper', self._ocp._c_cons_user)

# 	@property
# 	def state_endpoint(self):
# 		return self.return_user_bounds(self._y_b_l, self._y_b_u, 'state_endpoint')
	
# 	@state_endpoint.setter
# 	def state_endpoint(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'state_endpoint', self._ocp._y_b_cons_user)
# 		self.state_endpoint_lower = l
# 		self.state_endpoint_upper = u

# 	@property
# 	def state_endpoint_lower(self):
# 		return self._y_b_l
	
# 	@state_endpoint_lower.setter
# 	def state_endpoint_lower(self, b_l):
# 		self._y_b_l = self.parse_single_bounds(b_l, 'state_endpoint_lower', self._ocp._y_b_cons_user)

# 	@property
# 	def state_endpoint_upper(self):
# 		return self._y_b_u
	
# 	@state_endpoint_upper.setter
# 	def state_endpoint_upper(self, b_u):
# 		self._y_b_u = self.parse_single_bounds(b_u, 'state_endpoint_upper', self._ocp._y_b_cons_user)

# 	@property
# 	def boundary(self):
# 		return self.return_user_bounds(self._b_l, self._b_u, 'boundary')
	
# 	@boundary.setter
# 	def boundary(self, bnd):
# 		l, u = self.parse_both_bounds(bnd, 'boundary', self._ocp._b_cons_user)
# 		self.boundary_lower = l
# 		self.boundary_upper = u

# 	@property
# 	def boundary_lower(self):
# 		return self._b_l
	
# 	@boundary_lower.setter
# 	def boundary_lower(self, b_l):
# 		self._b_l = self.parse_single_bounds(b_l, 'boundary_lower', self._ocp._b_cons_user)

# 	@property
# 	def boundary_upper(self):
# 		return self._b_u
	
# 	@boundary_upper.setter
# 	def boundary_upper(self, b_u):
# 		self._b_u = self.parse_single_bounds(b_u, 'boundary_upper', self._ocp._b_cons_user)

# 	def _bounds_check(self):

# 		def compare_lower_and_upper(lower, upper, name):
# 			if not np.logical_or(np.less(lower, upper), np.isclose(lower, upper)).all():
# 				msg = (f"Lower bound of {lower} for {name} must be less than or equal to upper bound of {upper}.")
# 				raise ValueError(msg)
# 			vars_needed = np.ones(len(lower), dtype=int)
# 			for i, (l, u) in enumerate(zip(lower, upper)):
# 				if np.isclose(l, u):
# 					vars_needed[i] = 0
# 			return vars_needed

# 		def parse_bounds_to_np_array(bnds, syms, bnd_str):
# 			if isinstance(bnds, np.ndarray):
# 				num_y = len(syms)
# 				if bnds.shape[0] < num_y:
# 					msg = (f"Insufficient number of bounds have been supplied for `Bounds.{bnd_str}`. Currently {bnds.shape[0]} pairs of bounds are supplied where {num_y} pairs are needed.")
# 					raise ValueError(msg)
# 				elif bnds.shape[0] > num_y:
# 					msg = (f"Excess bounds have been supplied for `Bounds.{bnd_str}`. Currently {bnds.shape[0]} pairs of bounds are supplied where only {num_y} pairs are needed.")
# 					raise ValueError(msg)
# 				return_bnds = bnds
# 			elif isinstance(bnds, dict):
# 				extra_keys = set(bnds.keys()).difference(set(syms))
# 				if extra_keys:
# 					raise ValueError
# 				return_bnds = np.array([bnds.get(sym, (None, None)) for sym in syms], dtype=np.float64)
# 			return replace_non_float_vals(return_bnds, syms)

# 		def replace_non_float_vals(bnds, syms):
# 			bnds[bnds < -self.INF] = -self.INF
# 			bnds[bnds > self.INF] = self.INF
# 			if self._default_inf:
# 				bnds[bnds == -np.nan] = -self.INF
# 				bnds[bnds == np.nan] = self.INF
# 			elif np.isnan(np.sum(bnds)):
# 				for bnd_pair, symb in zip(bnds, syms):
# 					l, u = bnd_pair
# 					if np.isnan(l) and np.isnan(u):
# 						msg = (f"Both lower and upper bounds are missing for {symb}.")
# 						raise ValueError(msg)
# 					elif np.isnan(l):
# 						msg = (f"A lower bound is missing for {symb}.")
# 						raise ValueError(msg)
# 					elif np.isnan(u):
# 						msg = (f"An upper bound is missing for {symb}.")
# 						raise ValueError(msg)
# 			return bnds[:, 0], bnds[:, 1]

# 		def infer_bound(bnds, syms, eqns, aux_data, aux_subs, bnd_str):

# 			def q_unpack(x):
# 				q = q_lambda(*x)
# 				return q

# 			def J_unpack(x):
# 				J = np.array(J_lambda(*x).tolist()).squeeze()
# 				return J

# 			if isinstance(bnds, np.ndarray):
# 				if bnds.size == 0:
# 					bnds = np.empty((len(syms), 2))
# 					bnds.fill(np.nan)
# 				elif bnds.shape == (len(syms), 2):
# 					pass
# 				else:
# 					msg = (f"Bounds for `Bounds.{bnd_str}` could not be automatically inferred as lower and upper bounds have been supplied as an iterable with length different to the number of {bnd_str} symbols in the optimal control problem. The supplied bounds could not be mapped to their corresponding symbols. Please resupply bounds as a {dict}.")
# 					raise ValueError(msg)
# 			elif isinstance(bnds, dict):
# 				bnds = np.array([bnds.get(s, (None, None)) for s in syms], dtype=np.float64)

# 			x_lb = np.concatenate((self._y_l, self._u_l, np.zeros(len(self._ocp._q_vars_user) + 2), self._s_l))
# 			x_ub = np.concatenate((self._y_u, self._u_u, np.zeros(len(self._ocp._q_vars_user) + 2), self._s_u))
# 			x_bnds = np.array([x_lb, x_ub]).transpose()
# 			q_bnds = []
# 			for symb, eqn, bnd_user in zip(syms, eqns, bnds):

# 				if np.isnan(bnd_user).any():

# 					# Prepare SQP problem
# 					eqn = eqn.subs(aux_subs)
# 					eqn = eqn.subs(aux_data)
# 					min_args = []
# 					min_bnds = []
# 					for x, bnd_pair in zip(self._ocp._x_vars_user, x_bnds):
# 						if x in eqn.free_symbols:
# 							min_args.append(x)
# 							min_bnds.append(bnd_pair.tolist())
# 					min_bnds = np.array(min_bnds)
# 					x0 = min_bnds[:, 0] + np.random.random(len(min_args))*(min_bnds[:, 1] - min_bnds[:, 0])

# 					# Minimize
# 					if np.isnan(bnd_user[0]):
# 						q_lambda = sym.lambdify(min_args, eqn, modules='numpy')
# 						jacobian = eqn.diff(sym.Matrix(min_args))
# 						J_lambda = sym.lambdify(min_args, jacobian, modules='numpy')
# 						rslts = optimize.minimize(q_unpack, x0, method='SLSQP', jac=J_unpack, bounds=min_bnds)
# 						q_lb = q_unpack(rslts.x)
# 					else:
# 						q_lb = bnd_user[0]

# 					# Maximize
# 					if np.isnan(bnd_user[1]):
# 						q_lambda = sym.lambdify(min_args, -eqn, modules='numpy')
# 						jacobian = eqn.diff(sym.Matrix(min_args))
# 						J_lambda = sym.lambdify(min_args, -jacobian, modules='numpy')
# 						rslts = optimize.minimize(q_unpack, x0, method='SLSQP', jac=J_unpack, bounds=min_bnds)
# 						q_ub = -q_unpack(rslts.x)
# 					else:
# 						q_ub = bnd_user[1]
# 				else:
# 					q_lb, q_ub = bnd_user

# 				q_bnds.append([q_lb, q_ub])

# 			return np.array(q_bnds)

# 		# State
# 		self._y_l, self._y_u = parse_bounds_to_np_array(self.state, self._ocp._y_vars_user, 'state')
# 		self._y_needed = compare_lower_and_upper(self._y_l, self._y_u, 'state')
# 		self._y_b_needed = np.repeat(self._y_needed, 2)

# 		self._y_l_needed = self._y_l[self._y_needed.astype(bool)]
# 		self._y_u_needed = self._y_u[self._y_needed.astype(bool)]

# 		# Control
# 		self._u_l, self._u_u = parse_bounds_to_np_array(self.control, self._ocp._u_vars_user, 'control')
# 		self._u_needed = compare_lower_and_upper(self._u_l, self._u_u, 'control')

# 		self._u_l_needed = self._u_l[self._u_needed.astype(bool)]
# 		self._u_u_needed = self._u_u[self._u_needed.astype(bool)]

# 		# Integral
# 		if self.infer_bounds:
# 			raise NotImplementedError
# 			self.integral = infer_bound(self.integral, self._ocp._q_vars_user, self._ocp._q_funcs_user, aux_data, aux_subs, 'integral')
# 		self._q_l, self._q_u = parse_bounds_to_np_array(self.integral, self._ocp._q_vars_user, 'integral')
# 		self._q_needed = compare_lower_and_upper(self._q_l, self._q_u, 'integral')

# 		self._q_l_needed = self._q_l[self._q_needed.astype(bool)]
# 		self._q_u_needed = self._q_u[self._q_needed.astype(bool)]

# 		# Time
# 		_ = parse_bounds_to_np_array(np.array([[self._t0_l, self._tF_l], [self._t0_u, self._tF_u]], dtype=np.float64), self._ocp._t_vars_user, 'time')
# 		if self._t0_l > self._t0_u:
# 			msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be greater than the lower bound ({self._t0_l}).")
# 			raise ValueError(msg)
# 		elif self._tF_l > self._tF_u:
# 			msg = (f"The upper bound for `Bounds.final_time` ({self._tF_u}) must be greater than the lower bound ({self._tF_l}).")
# 			raise ValueError(msg)
# 		elif self._t0_l > self._tF_l:
# 			msg = (f"The lower bound for `Bounds.initial_time` ({self._t0_l}) must be less than the lower bound for `Bounds.final_time` ({self._tF_l}).")
# 			raise ValueError(msg)
# 		elif self._t0_u > self._tF_u:
# 			msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be less than the upper bound for `Bounds.final_time` ({self._tF_u}).")
# 			raise ValueError(msg)
# 		else:
# 			self._t0_l = self.initial_time_lower
# 			self._t0_u = self.initial_time_upper
# 			self._tF_l = self.final_time_lower
# 			self._tF_u = self.final_time_upper
# 			self._t_l = np.array([self._t0_l, self._tF_l]).flatten()
# 			self._t_u = np.array([self._t0_u, self._tF_u]).flatten()
# 		self._t_needed = compare_lower_and_upper([self._t0_l, self._tF_l], [self._t0_u, self._tF_u], 'time')

# 		self._t_l_needed = self._t_l[self._t_needed.astype(bool)]
# 		self._t_u_needed = self._t_u[self._t_needed.astype(bool)]

# 		# Parameter
# 		self._s_l, self._s_u = parse_bounds_to_np_array(self.parameter, self._ocp._s_vars_user, 'parameter')
# 		self._s_needed = compare_lower_and_upper(self._s_l, self._s_u, 'parameter')

# 		self._s_l_needed = self._s_l[self._s_needed.astype(bool)]
# 		self._s_u_needed = self._s_u[self._s_needed.astype(bool)]

# 		self._x_needed = np.concatenate([self._y_needed, self._u_needed, self._q_needed, self._t_needed, self._s_needed])
# 		self._x_b_needed = np.concatenate([self._y_b_needed, self._q_needed, self._t_needed, self._s_needed])

# 		# Path
# 		if self.infer_bounds:
# 			if self._ocp._c_cons_user:
# 				raise NotImplementedError
# 		self._c_l, self._c_u = parse_bounds_to_np_array(self.path, self._ocp._c_cons_user, 'path constraint')
# 		_ = compare_lower_and_upper(self._c_l, self._c_u, 'path constraint')

# 		# Boundary
# 		self._y_b_l, self._y_b_u = parse_bounds_to_np_array(self.state_endpoint, self._ocp._y_b_cons_user, 'state_endpoint_constraint')
# 		_ = compare_lower_and_upper(self._b_l, self._b_u, 'state_endpoint_constraint')

# 		# Boundary
# 		self._b_l, self._b_u = parse_bounds_to_np_array(self.boundary, self._ocp._b_cons_user, 'boundary constraint')
# 		_ = compare_lower_and_upper(self._b_l, self._b_u, 'boundary constraint')

# 		self._bounds_checked = True


# def kwarg_conflict_check(lower, upper, kwarg_str):
# 	if lower is not None or upper is not None:
# 		msg = (f"If the key-word argument `{kwarg_str}` is used then the key-word arguments `{kwarg_str}_lower` and `{kwarg_str}_upper` cannot be used.")
# 		raise TypeError(msg)
# 	return None

# def split_both_bounds_dict(bnds, bnd_str, syms, vec):
# 	l = {}
# 	u = {}
# 	for bnd_sym, bnd_pair in bnds.items():
# 		# fromnbnd_paiimport supported_iter_types supported_iter_types):
# 			if len(bnd_pair) != 2:
# 				msg = (f"When values for lower and upper bounds are supplied together using a {dict} and the `Bounds.{bnd_str}` attribute, the  values must be a iterables of length 2: (`lower_bound`, `upper_bound`).")
# 				raise ValueError(msg)
# 		else:
# 			try:
# 				bnd_pair = np.array([bnd_pair, bnd_pair], dtype=np.float64)
# 			except ValueError:
# 				msg = (f"When values for lower and upper bounds are supplied together using a {dict} and the `Bounds.{bnd_str}` attribute, the  values must be a iterables of length 2: (`lower_bound`, `upper_bound`).")
# 				raise ValueError(msg)
# 		l.update({bnd_sym: bnd_pair[0]})
# 		u.update({bnd_sym: bnd_pair[1]})
# 	return l, u


# def split_both_bounds_iter(bnds, bnd_str, syms, vec):
		
# 	if vec is False:
# 		if len(bnds) != 2:
# 			msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be of length 2.")
# 			raise ValueError(msg)
# 		return float(bnds[0]), float(bnds[1])
# 	else:
# 		bnds = np.array(bnds)
# 		if bnds.ndim == 2:
# 			if bnds.shape[1] != 2:
# 				msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be a two-dimensional iterable with first dimension of length 2. Current dimensions are: {bnds.shape}.")
# 				raise ValueError(msg)
# 		elif bnds.ndim > 2:
# 			msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be a two-dimensional iterable. The currently supplied iterable's dimensions are: {bnds.shape}.")
# 			raise ValueError(msg)
# 		bnds = np.array(bnds, dtype=float).flatten().reshape((-1, 2), order='C')
# 		return bnds[:, 0], bnds[:, 1]
		

# def parse_single_bounds(bnds, bnd_str, syms=None, vec=True):

# 	if vec is True:
# 		if bnds is None:
# 			return np.array([])
# 		elif isinstance(bnds, dict):
# 			for bnd_sym in bnds.keys():
# 				if bnd_sym not in syms:
# 					msg = (f"{bnd_sym} cannot be supplied as a key value for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` as it is not a {bnd_str} variables in the optimal control problem.")
# 					raise ValueError(msg)
# 			try:
# 				return {bnd_sym: float(bnd) for bnd_sym, bnd in bnds.items()}
# 			except TypeError:
# 				msg = (f"Dictionary values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` must be convertable into {float} objects.")
# 				raise TypeError(msg)
# 		elif isinstance(bnds, pu.supported_iter_types):
# 			return np.array(bnds, dtype=np.float64)
# 		else:
# 			return np.array([bnds], dtype=np.float64)

# 	else:
# 		if isinstance(bnds, dict):
# 			msg = (f"Value for `Bounds.{bnd_str}` cannot be of type {dict}.")
# 			raise ValueError(msg)
# 		else:
# 			return None if bnds is None else float(bnds)

# def parse_both_bounds(bnds, bnd_str, syms=None, vec=True):

# 	if vec is True:
# 		if bnds is None:
# 			return None, None
# 		elif isinstance(bnds, dict):
# 			return split_both_bounds_dict(bnds, bnd_str, syms, vec)
# 		elif isinstance(bnds, pu.supported_iter_types):
# 			return split_both_bounds_iter(bnds, bnd_str, syms, vec)
# 		else:
# 			return np.array([bnds], dtype=np.float64), np.array([bnds], dtype=np.float64)

# 	else:
# 		if isinstance(bnds, dict):
# 			msg = (f"Value for `Bounds.{bnd_str}` cannot be of type {dict}.")
# 			raise ValueError(msg)
# 		elif isinstance(bnds, pu.supported_iter_types):
# 			return split_both_bounds_iter(bnds, bnd_str, syms, vec)
# 		else:
# 			return float(bnds), float(bnds)

# def return_user_bounds(l, u, bnd_str):
# 	if type(l) is type(u):
# 		if isinstance(l, np.ndarray):
# 			try:
# 				return np.array([l, u], dtype=np.float64).transpose()
# 			except ValueError:
# 				msg = (f"Values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` have been supplied as iterables of different length. Pycollo cannot infer which lower and upper bounds correspond to which {bnd_str} variables.")
# 				raise ValueError(msg)
# 		elif isinstance(l, dict):
# 			merged = {}
# 			sym_keys = set.union(set(l.keys()), set(u.keys()))
# 			for k in sym_keys:
# 				merged.update({k: (l.get(k), u.get(k))})
# 			return merged
# 	else:
# 		msg = (f"Values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` have been supplied as iterables of different types. If lower and upper bounds for {bnd_str} are being supplied separately, please ensure that these are both either of type {dict} or type {np.ndarray}.")
# 		raise TypeError(msg)
	
		
		