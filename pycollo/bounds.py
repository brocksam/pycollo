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

	def _extract_variables_to_constants(self, bnds, are_same):
		if not self.optimal_control_problem.settings.remove_constant_variables:
			needed = np.full(bnds.shape[0], True)
			return needed
		needed = ~are_same
		return needed


class EndpointBounds(BoundsABC):
	
	def __init__(self, optimal_control_problem, *,
			parameter_variables: OptionalBoundsType = None,
			endpoint_constraints: OptionalBoundsType = None,
			):

		self._ocp = optimal_control_problem
		self.parameter_variables = parameter_variables
		self.endpoint_constraints = endpoint_constraints

	@property
	def optimal_control_problem(self):
		return self._ocp
	
	def _process_and_check_user_values(self):
		self._backend = self.optimal_control_problem._backend
		self._expr_graph = self._backend.expression_graph
		self._INF = self.optimal_control_problem.settings.inf_value
		self._process_parameter_vars()
		self._process_endpoint_cons()

	def _process_parameter_vars(self):
		user_bnds = self.parameter_variables
		user_syms = self._backend.s_vars_user
		bnds_type = "parameter variable"
		num_expected = self._backend.num_s_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		self._s_bnds, self._s_needed = self._process_single_type_of_values(
			bnds_info)

	def _process_endpoint_cons(self):
		user_bnds = self.endpoint_constraints
		user_syms = None
		bnds_type = "endpoint constraints"
		num_expected = self._backend.num_c_endpoint
		bnds_info = BoundsInfo(
			user_bnds, user_syms, bnds_type, num_expected, False)
		self._c_endpoint_bnds, _ = self._process_single_type_of_values(
			bnds_info)

	def _process_single_type_of_values(self, bnds_info):
		if isinstance(bnds_info.user_bnds, dict):
			bnds = self._process_mapping_bounds_instance(bnds_info)
		elif bnds_info.user_bnds is None:
			bnds = self._process_none_bounds_instance(bnds_info)
		elif isinstance(bnds_info.user_bnds, supported_iter_types):
			bnds = self._process_iterable_bounds_instance(bnds_info)
		else:
			bnds = self._process_single_type_of_values(bnds_info)
		bnds, needed = self._check_lower_against_upper(bnds, bnds_info)
		return bnds, needed

	def _process_mapping_bounds_instance(self, bnds_info):
		if bnds_info.user_syms is None:
			msg = f"Can't use mapping for {bnds_info.bnds_type} bounds."
			raise TypeError(msg)
		bnds = []
		for bnd_i, user_sym in enumerate(bnds_info.user_syms):
			bnd = bnds_info.user_bnds.get(user_sym)
			bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info)
			bnd = self._as_lower_upper_pair(bnd_info)
			bnds.append(bnd)
		return bnds

	def _check_user_bound_missing(self, bnd_info):
		ocp_settings = self.optimal_control_problem.settings
		is_bnd_none = bnd_info.user_bnds is None
		is_inf_assumed = ocp_settings.assume_inf_bounds
		if is_bnd_none and not is_inf_assumed:
			self._raise_user_bound_missing_error(bnd_info)

	def _raise_user_bound_missing_error(self, bnd_info):
		msg = (f"No bounds have been supplied for the {bnd_info.bnds_type} "
			f"'{bnd_info.user_syms}' (index #{bnd_info.num}).")
		raise ValueError(msg)		

	def _process_iterable_bounds_instance(self, bnds_info):
		bnds = []
		for bnd_i, bnd in enumerate(bnds_info.user_bnds):
			bnd_info = BoundsInfo(bnd, None, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info)
			bnd = self._as_lower_upper_pair(bnd_info)
			bnds.append(bnd)
		return bnds

	def _process_single_value_bounds_instance(self, bnds_info):
		if not isinstance(bnds_info.user_bnds, (Number, sym.Expr)):
			msg = ()
			raise TypeError(msg)
		bnds_info = bnds_info._replace(user_bnds=[bnds_info.user_bnds]*2)
		return self._process_iterable_bounds_instance(bnds_info)

	def _process_none_bounds_instance(self, bnds_info):
		bnds = []
		for bnd_i, user_sym in enumerate(bnds_info.user_syms):
			bnd = None
			bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info)
			bnd = self._as_lower_upper_pair(bnd_info)
			bnds.append(bnd)
		return bnds

	def _as_lower_upper_pair(self, bnd_info):
		bnds = np.array(bnd_info.user_bnds).flatten()
		if bnds.shape == (1, ):
			both = "lower and upper bounds"
			both_info = bnd_info._replace(user_bnds=bnds[0])
			lower_bnd = self._get_bound_as_number(both_info, both)
			upper_bnd = lower_bnd
		elif bnds.shape == (2, ):
			lower = "lower bound"
			upper = "upper bound"
			lower_info = bnd_info._replace(user_bnds=bnds[0])
			upper_info = bnd_info._replace(user_bnds=bnds[1])
			lower_bnd = self._get_bound_as_number(lower_info, lower)
			upper_bnd = self._get_bound_as_number(upper_info, upper)
		else:
			raise ValueError
		lower_bnd = - self._INF if lower_bnd is None else lower_bnd
		upper_bnd = self._INF if upper_bnd is None else upper_bnd
		return [lower_bnd, upper_bnd]

	def _get_bound_as_number(self, bnd_info, lower_upper):
		bnd = bnd_info.user_bnds
		if bnd is None:
			return bnd
		elif isinstance(bnd, str):
			if bnd == "inf":
				return self._INF
			elif bnd == "-inf":
				return - self._INF
			else:
				msg = (f"A bound value of {bnd} is not supported.")
				raise NotImplementedError(msg)
		bnd = fast_sympify(bnd).xreplace(self._backend.all_subs_mappings)
		node = Node(bnd, self._expr_graph)
		if not node.is_precomputable:
			msg = (f"The user-supplied {lower_upper} for the {bnd_info.bnds_type} "
				f"'{bnd_info.user_sym}' (index #{bnd_info.num}) of '{bnd}' "
				f"cannot be precomputed.")
			raise ValueError(msg)
		return node.value

	def _check_lower_against_upper(self, bnds, bnds_info):
		if not bnds:
			bnds = np.empty(shape=(0, 2), dtype=float)
			needed = np.empty(shape=0, dtype=bool)
			return bnds, needed
		bnds = np.array(bnds, dtype=float)
		bnds, needed = self._check_lower_same_as_upper_to_tol(
			bnds, bnds_info)
		bnds = self._check_lower_less_than_upper(bnds, bnds_info)
		return bnds, needed

	def _check_lower_same_as_upper_to_tol(self, bnds, bnds_info):
		lower_bnds = bnds[:, 0]
		upper_bnds = bnds[:, 1]
		rtol = self.optimal_control_problem.settings.bound_clash_tolerance
		are_same = np.isclose(lower_bnds, upper_bnds, rtol=rtol, atol=0.0)
		if bnds_info.is_variable:
			needed = self._extract_variables_to_constants(bnds, are_same)
		else:
			needed = None
		mean_bnds = (lower_bnds + upper_bnds)/2
		bnds[are_same, 0] = mean_bnds[are_same]
		bnds[are_same, 1] = mean_bnds[are_same]
		return bnds, needed

	def _check_lower_less_than_upper(self, bnds, bnds_info):
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
				f"{user_indices_formatted}) of {upper_bnds_formatted} cannot "
				f"be less than the user-supplied lower {bound_plural} of "
				f"{lower_bnds_formatted}.")
			raise ValueError(msg)
		return bnds


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
		self._process_state_vars(p_info)
		self._process_control_vars(p_info)
		self._process_integral_vars(p_info)
		self._process_time_vars(p_info)
		self._process_path_cons(p_info)
		self._process_initial_state_cons(p_info)
		self._process_final_state_cons(p_info)

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
		self._y_bnds, self._y_needed = self._process_single_type_of_values(
			bnds_info, p_info)

	def _process_control_vars(self, p_info):
		user_bnds = self.control_variables
		user_syms = p_info.backend.u_vars_user
		bnds_type = "control variable"
		num_expected = p_info.backend.num_u_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		self._u_bnds, self._u_needed = self._process_single_type_of_values(
			bnds_info, p_info)

	def _process_integral_vars(self, p_info):
		user_bnds = self.integral_variables
		user_syms = p_info.backend.q_vars_user
		bnds_type = "integral variable"
		num_expected = p_info.backend.num_q_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		self._q_bnds, self._q_needed = self._process_single_type_of_values(
			bnds_info, p_info)

	def _process_path_cons(self, p_info):
		user_bnds = self.path_constraints
		user_syms = None
		bnds_type = "path constraints"
		num_expected = p_info.backend.num_c_path
		bnds_info = BoundsInfo(
			user_bnds, user_syms, bnds_type, num_expected, False)
		self._c_path_bnds, _ = self._process_single_type_of_values(
			bnds_info, p_info)

	def _process_time_vars(self, p_info):
		user_bnds = [self.initial_time, self.final_time]
		user_syms = p_info.backend.t_vars_user
		bnds_type = "time variable"
		num_expected = p_info.backend.num_t_vars
		bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
		self._t_bnds, self._t_needed = self._process_single_type_of_values(
			bnds_info, p_info)
		self._check_time_bounds_error((0, 0), (1, 0), p_info)
		self._check_time_bounds_error((0, 1), (1, 1), p_info)

	def _check_time_bounds_error(self, i_1, i_2, p_info):
		arg_1 = self._t_bnds[i_1]
		arg_2 = self._t_bnds[i_2]
		if arg_1 > arg_2:
			self._raise_time_bounds_error(i_1, i_2, arg_1, arg_2, p_info)

	def _raise_time_bounds_error(self, i_1, i_2, bnd_1, bnd_2, p_info):
		bnd_1_t0_or_tF = "initial" if i_1[0] == 0 else "final"
		bnd_1_lower_or_upper = "lower" if i_1[1] == 0 else "upper"
		bnd_2_t0_or_tF = "initial" if i_2[0] == 0 else "final"
		bnd_2_lower_or_upper = "lower" if i_2[1] == 0 else "upper"
		msg = (f"The {bnd_2_lower_or_upper} bound for the {bnd_2_t0_or_tF} "
			f"time ('{bnd_2}') must be greater than the {bnd_1_lower_or_upper} "
			f"bound for the {bnd_1_t0_or_tF} time ('{bnd_1}') in phase "
			f"{p_info.name} (index #{p_info.index}).")
		raise ValueError(msg)

	def _process_initial_state_cons(self, p_info):
		user_bnds = self.initial_state_constraints
		user_syms = p_info.backend.y_vars_user
		bnds_type = "initial state constraint"
		num_expected = p_info.backend.num_y_vars
		bnds_info = BoundsInfo(
			user_bnds, user_syms, bnds_type, num_expected, False)
		y_t0_bnds, self._y_t0_needed = self._process_single_type_of_values(
			bnds_info, p_info)
		if self.optimal_control_problem.settings.override_endpoint_bounds:
			y_t0_bnds = self._override_endpoint_bounds(y_t0_bnds)
		self._y_t0_bnds = y_t0_bnds

	def _process_final_state_cons(self, p_info):
		user_bnds = self.final_state_constraints
		user_syms = p_info.backend.y_vars_user
		bnds_type = "final state constraint"
		num_expected = p_info.backend.num_y_vars
		bnds_info = BoundsInfo(
			user_bnds, user_syms, bnds_type, num_expected, False)
		y_tF_bnds, self._y_tF_needed = self._process_single_type_of_values(
			bnds_info, p_info)
		if self.optimal_control_problem.settings.override_endpoint_bounds:
			y_tF_bnds = self._override_endpoint_bounds(y_tF_bnds)
		self._y_tF_bnds = y_tF_bnds

	def _override_endpoint_bounds(self, y_b_bnds):
		lower_is_less = y_b_bnds[:, 0] < self._y_bnds[:, 0]
		y_b_bnds[lower_is_less, 0] = self._y_bnds[lower_is_less, 0]
		upper_is_more = y_b_bnds[:, 1] > self._y_bnds[:, 1]
		y_b_bnds[upper_is_more, 1] = self._y_bnds[upper_is_more, 1]
		return y_b_bnds

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
		elif isinstance(bnds_info.user_bnds, supported_iter_types):
			bnds = self._process_iterable_bounds_instance(bnds_info, p_info)
		else:
			bnds = self._process_single_type_of_values(bnds_info, p_info)
		bnds, needed = self._check_lower_against_upper(bnds, bnds_info, p_info)
		return bnds, needed

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
		bnds = []
		for bnd_i, bnd in enumerate(bnds_info.user_bnds):
			bnd_info = BoundsInfo(bnd, None, bnds_info.bnds_type, bnd_i)
			self._check_user_bound_missing(bnd_info, p_info)
			bnd = self._as_lower_upper_pair(bnd_info, p_info)
			bnds.append(bnd)
		return bnds

	def _process_potential_dual_value_to_single_value(self, bnd_info, p_info):
		bnd = bnd_info.user_bnds
		msg = (f"Single bounds in this form ('{bnd}') are not supported.")
		is_list = isinstance(bnd, supported_iter_types)
		if not is_list:
			raise TypeError(msg)
		is_len_2 = len(bnd) == 2
		if not is_len_2:
			raise ValueError(msg)
		is_pair_same = bnd[0] == bnd[1]
		if not is_pair_same:
			raise ValueError(msg)
		bnd = bnd[0]
		bnd_info = bnd_info._replace(user_bnds=bnd)
		return bnd_info

	def _process_single_value_bounds_instance(self, bnds_info, p_info):
		if not isinstance(bnds_info.user_bnds, (Number, sym.Expr)):
			msg = ()
			raise TypeError(msg)
		bnds_info = bnds_info._replace(user_bnds=[bnds_info.user_bnds]*2)
		return self._process_iterable_bounds_instance(bnds_info, p_info)

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
		elif isinstance(bnd, str):
			if bnd == "inf":
				return self._INF
			elif bnd == "-inf":
				return - self._INF
			else:
				msg = (f"A bound value of {bnd} is not supported.")
				raise NotImplementedError(msg)
		elif not isinstance(bnd, (Number, sym.Expr)):
			bnd_info = self._process_potential_dual_value_to_single_value(
				bnd_info, p_info)
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
			return np.empty(shape=(0, 2), dtype=float), np.empty(shape=0, dtype=bool)
		bnds = np.array(bnds, dtype=float)
		bnds, needed = self._check_lower_same_as_upper_to_tol(
			bnds, bnds_info, p_info)
		bnds = self._check_lower_less_than_upper(bnds, bnds_info, p_info)
		return bnds, needed

	def _check_lower_same_as_upper_to_tol(self, bnds, bnds_info, p_info):
		lower_bnds = bnds[:, 0]
		upper_bnds = bnds[:, 1]
		rtol = self.optimal_control_problem.settings.bound_clash_tolerance
		are_same = np.isclose(lower_bnds, upper_bnds, rtol=rtol, atol=0.0)
		if bnds_info.is_variable:
			needed = self._extract_variables_to_constants(bnds, are_same)
		else:
			needed = None
		mean_bnds = (lower_bnds + upper_bnds)/2
		bnds[are_same, 0] = mean_bnds[are_same]
		bnds[are_same, 1] = mean_bnds[are_same]
		return bnds, needed

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


phase_info_fields = ("name", "index", "backend")
PhaseInfo = namedtuple("PhaseInfo", phase_info_fields)

bounds_info_fields = ("user_bnds", "user_syms", "bnds_type", "num", 
	"is_variable", "none_default_allowed")
BoundsInfo = namedtuple("BoundsInfo", bounds_info_fields, 
	defaults=[True, True])


class Bounds:

	def __init__(self, ocp_backend):
		self.ocp_backend = ocp_backend
		for p in self.ocp_backend.p:
			p.ocp_phase.bounds._process_and_check_user_values(p)
		self.ocp_backend.ocp.bounds._process_and_check_user_values()

		
		