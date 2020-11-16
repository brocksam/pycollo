"""Objects, functions and constants relating to OCP bounds.

Attributes
----------
DEFAULT_ASSUME_INF_BOUNDS : bool
    Default as to whether Pycollo should treat unspecified bounds as being
    numerically infinite.
DEFAULT_INF_VALUE : float
    Default numerical value for when Pycollo needs to use a finite numerical
    approximation for infinity.

"""


__all__ = ["EndpointBounds", "PhaseBounds"]


from abc import (ABC, abstractmethod)
from collections import namedtuple
from numbers import Number
from typing import (Iterable, Optional, Union)

import numpy as np
import scipy.optimize as optimize
import sympy as sym

from .node import Node
from .typing import OptionalBoundsType
from .utils import (fast_sympify,
                    format_multiple_items_for_output,
                    SUPPORTED_ITER_TYPES,
                    symbol_primitives,
                    )


DEFAULT_ASSUME_INF_BOUNDS = True
DEFAULT_BOUND_CLASH_TOLERANCE = 1e-6
DEFAULT_BOUND_CLASH_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_BOUND_CLASH_RELATIVE_TOLERANCE = 1e-6
DEFAULT_NUMERICAL_INF = 10e19
DEFAULT_OVERRIDE_ENDPOINTS = True
DEFAULT_REMOVE_CONSTANT_VARIABLES = True


class BoundsABC(ABC):

    @abstractmethod
    def optimal_control_problem(self):
        pass

    @abstractmethod
    def _process_and_check_user_values(self):
        pass

    def _extract_variables_to_constants(self, bnds, are_same):
        if not self.optimal_control_problem.settings.remove_constant_variables:
            needed = np.full(bnds.shape[0], True)
            return needed
        needed = ~are_same
        return needed

    @abstractmethod
    def _required_variable_bounds(self):
        pass


class EndpointBounds(BoundsABC):

    def __init__(self,
                 optimal_control_problem,
                 *,
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
        self._INF = self.optimal_control_problem.settings.numerical_inf
        self._process_parameter_vars()
        self._process_endpoint_cons()

    def _process_parameter_vars(self):
        user_bnds = self.parameter_variables
        user_syms = self._backend.s_vars_user
        bnds_type = "parameter variable"
        num_expected = self._backend.num_s_vars_full
        bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
        self._s_bnd, self._s_needed = self._process_single_type_of_values(
            bnds_info)

    def _process_endpoint_cons(self):
        num_b_con = self.optimal_control_problem.number_endpoint_constraints
        user_bnds = self.endpoint_constraints
        user_syms = [None] * num_b_con
        bnds_type = "endpoint constraints"
        num_expect = num_b_con
        bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expect,
                              False)
        self._b_con_bnd, needed = self._process_single_type_value(bnds_info)
        if any(needed):
            msg = (f"Pycollo cannot currently handle automatic conversion of "
                   f"endpoint constraints to state endpoint constraints. "
                   f"Please reformulate using "
                   f"`bounds.initial_state_constraints` or "
                   f"`bounds.final_state_constraints`.")
            raise NotImplementedError(msg)

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
        if any(user_sym is None for user_sym in bnds_info.user_syms):
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
        is_bnd_none = bnd_info.user_bnd is None
        is_inf_assumed = ocp_settings.assume_inf_bounds
        if is_bnd_none and not is_inf_assumed:
            self._raise_user_bound_missing_error(bnd_info)

    def _raise_user_bound_missing_error(self, bnd_info):
        msg = (f"No bounds have been supplied for the {bnd_info.bnds_type} "
               f"'{bnd_info.user_syms}' (index #{bnd_info.num}).")
        raise ValueError(msg)

    def _process_iterable_bounds_instance(self, bnd_info):
        if bnd_info.num == 1 and not isinstance(bnd_info.user_bnd[0],
                                                SUPPORTED_ITER_TYPES):
            bnd_info = bnd_info._replace(user_bnd=[bnd_info.user_bnd])
        bnds = []
        for bnd_i, bnd in enumerate(bnd_info.user_bnd):
            bnd_info = BoundsInfo(bnd, None, bnds_info.bnds_type, bnd_i)
            self._check_user_bound_missing(bnd_info_i)
            bnd = self._as_lower_upper_pair(bnd_info_i)
            bnds.append(bnd)
        return bnds

    def _process_single_value_bounds_instance(self, bnds_info):
        if not isinstance(bnds_info.user_bnds, (Number, sym.Expr)):
            msg = ""
            raise TypeError(msg)
        bnds_info = bnds_info._replace(user_bnds=[bnds_info.user_bnds] * 2)
        return self._process_iterable_bounds_instance(bnds_info)

    def _process_none_bounds_instance(self, bnds_info):
        bnds = []
        for bnd_i, user_sym in enumerate(bnds_info.user_sym):
            bnd = None
            bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
            self._check_user_bound_missing(bnd_info)
            bnd = self._as_lower_upper_pair(bnd_info)
            bnds.append(bnd)
        return bnds

    def _as_lower_upper_pair(self, bnd_info):
        bnd = np.array(bnd_info.user_bnd).flatten()
        if bnd.shape == (1, ):
            both = "lower and upper bounds"
            both_info = bnd_info._replace(user_bnd=bnd[0])
            lower_bnd = self._get_bound_as_number(both_info, both)
            upper_bnd = lower_bnd
        elif bnd.shape == (2, ):
            lower = "lower bound"
            upper = "upper bound"
            lower_info = bnd_info._replace(user_bnd=bnd[0])
            upper_info = bnd_info._replace(user_bnd=bnd[1])
            lower_bnd = self._get_bound_as_number(lower_info, lower)
            upper_bnd = self._get_bound_as_number(upper_info, upper)
        else:
            raise ValueError
        lower_bnd = - self._INF if lower_bnd is None else lower_bnd
        upper_bnd = self._INF if upper_bnd is None else upper_bnd
        return [lower_bnd, upper_bnd]

    def _get_bound_as_number(self, bnd_info, lower_upper):
        bnd = bnd_info.user_bnd
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
        if isinstance(bnd, (np.float64, np.int64)):
            return bnd
        bnd = self._backend.substitute_pycollo_sym(bnd)
        if symbol_primitives(bnd):
            msg = (f"The user-supplied {lower_upper} for the "
                   f"{bnd_info.bnds_type} '{bnd_info.user_sym}' "
                   f"(index #{bnd_info.num}) of '{bnd}' "
                   f"cannot be precomputed.")
            raise ValueError(msg)
        return float(bnd)

    def _check_lower_against_upper(self, bnd, bnd_info):
        if not bnd:
            bnd = np.empty(shape=(0, 2), dtype=float)
            needed = np.empty(shape=0, dtype=bool)
            return bnd, needed
        bnd = np.array(bnd, dtype=float)
        bnd, needed = self._check_lower_same_as_upper_to_tol(
            bnd, bnd_info)
        bnd = self._check_lower_less_than_upper(bnd, bnd_info)
        return bnd, needed

    def _check_lower_same_as_upper_to_tol(self, bnd, bnd_info):
        lower_bnd = bnd[:, 0]
        upper_bnd = bnd[:, 1]
        rtol = self.optimal_control_problem.settings.bound_clash_tolerance
        are_same = np.isclose(lower_bnd, upper_bnd, rtol=rtol, atol=0.0)
        if bnd_info.is_variable:
            needed = self._extract_variables_to_constants(bnd, are_same)
        else:
            needed = None
        mean_bnd = (lower_bnd + upper_bnd) / 2
        bnd[are_same, 0] = mean_bnd[are_same]
        bnd[are_same, 1] = mean_bnd[are_same]
        return bnd, needed

    def _check_lower_less_than_upper(self, bnd, bnd_info):
        lower_bnd = bnd[:, 0]
        upper_bnd = bnd[:, 1]
        lower_less_than_upper = lower_bnd <= upper_bnd
        all_less_than = np.all(lower_less_than_upper)
        if not all_less_than:
            error_indices = np.flatnonzero(~lower_less_than_upper)
            error_syms = np.array(bnd_info.user_sym)[error_indices]
            plural_needed = len(error_indices) > 1
            bound_plural = "bounds" if plural_needed else "bound"
            index_plural = "indices" if plural_needed else "index"
            bnd_type_plural = (f"{bnd_info.bnd_type}"
                                f"{'s' if plural_needed else ''}")
            user_syms_formatted = format_multiple_items_for_output(error_syms)
            user_indices_formatted = format_multiple_items_for_output(
                error_indices, wrapping_char="", prefix_char="#")
            lower_bnd_formatted = format_multiple_items_for_output(
                lower_bnd[error_indices])
            upper_bnd_formatted = format_multiple_items_for_output(
                upper_bnd[error_indices])
            msg = (f"The user-supplied upper {bound_plural} for the "
                   f"{bnd_type_plural} {user_syms_formatted} ({index_plural} "
                   f"{user_indices_formatted}) of {upper_bnd_formatted} "
                   f"cannot be less than the user-supplied lower "
                   f"{bound_plural} of {lower_bnd_formatted}.")
            raise ValueError(msg)
        return bnd

    def _required_variable_bounds(self):
        x_bnd = self._s_bnd[self._s_needed]
        return x_bnd


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
            `OptimalControlProblem.settings.numerical_inf`, the default is 1e19.
        * If a :obj:`np.ndarray` with size = (2, 2) is passed as a value then
            the first dimension will be treated as corresponding to the
            variable or constraint to be bounded.
        * If iterables are passed then they may contain a combination of None,
            single numerical values, and pairs of numerical values
        * Symbolic expressions should also be allowed if they can be converted
            into numerical values when processed alongside auxiliary data.

    Notes
    -----
        * 'inf' values should be avoided where possible in order to give better
          automatic scaling.

    Attributes
    ----------
    phase
        The phase with which these bounds will be associated. Default value is
        `None`.
    initial_time
        Bounds on when the phase starts. Default value is `None`.
    final_time
        Bounds on when the phase ends.  Default value is `None`.
    state_variables:
        Bounds on the phase's state variables. Default value is `None`.
    control_variables
        Bounds on the phase's control variables. Default value is `None`.
    integral_variables
        Bounds on the phase's integral variables. Default value is `None`.
    path_constraints
        Bounds on the phase's path constraints. Default value is `None`.
    initial_state_constraints
        Bounds on the phase's state variables at the initial time. Default
        value is `None`.
    final_state_constraints
        Bounds on the phase's state variables at the final time. Default value
        is `None`.
    """

    def __init__(self,
                 phase: "Phase",
                 *,
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

        Args
        ----
        phase
            The phase with which these bounds will be associated.
        initial_time
            Bounds on when the phase starts. Default value is `None`.
        final_time
            Bounds on when the phase ends.  Default value is `None`.
        state_variables
            Bounds on the phase's state variables. Default value is `None`.
        control_variables
            Bounds on the phase's control variables. Default value is `None`.
        integral_variables
            Bounds on the phase's integral variables. Default value is `None`.
        path_constraints
            Bounds on the phase's path constraints. Default value is `None`.
        initial_state_constraints
            Bounds on the phase's state variables at the initial time. Default
            value is `None`.
        final_state_constraints
            Bounds on the phase's state variables at the final time. Default
            value is `None`.
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
        self._INF = self.optimal_control_problem.settings.numerical_inf
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
        user_bnd = self.state_variables
        user_sym = p_info.backend.y_var_user
        bnd_type = "state variable"
        num_expect = p_info.backend.num_y_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._y_bnd, self._y_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)

    def _process_control_vars(self, p_info):
        user_bnd = self.control_variables
        user_sym = p_info.backend.u_var_user
        bnd_type = "control variable"
        num_expect = p_info.backend.num_u_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._u_bnd, self._u_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)

    def _process_integral_vars(self, p_info):
        user_bnd = self.integral_variables
        user_sym = p_info.backend.q_var_user
        bnd_type = "integral variable"
        num_expect = p_info.backend.num_q_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._q_bnd, self._q_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)

    def _process_path_cons(self, p_info):
        user_bnd = self.path_constraints
        user_sym = [None] * p_info.backend.num_p_con
        bnd_type = "path constraints"
        num_expect = p_info.backend.num_p_con
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect, False)
        self._p_con_bnd, _ = self._process_single_type_value(bnd_info, p_info)

    def _process_time_vars(self, p_info):
        user_bnd = [self.initial_time, self.final_time]
        user_sym = p_info.backend.t_var_user
        bnd_type = "time variable"
        num_expect = p_info.backend.num_t_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._t_bnd, self._t_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)
        self._check_time_bounds_error((0, 0), (1, 0), p_info)
        self._check_time_bounds_error((0, 1), (1, 1), p_info)

    def _check_time_bounds_error(self, i_1, i_2, p_info):
        arg_1 = self._t_bnd[i_1]
        arg_2 = self._t_bnd[i_2]
        if arg_1 > arg_2:
            self._raise_time_bounds_error(i_1, i_2, arg_1, arg_2, p_info)

    def _raise_time_bounds_error(self, i_1, i_2, bnd_1, bnd_2, p_info):
        bnd_1_t0_or_tF = "initial" if i_1[0] == 0 else "final"
        bnd_1_lower_or_upper = "lower" if i_1[1] == 0 else "upper"
        bnd_2_t0_or_tF = "initial" if i_2[0] == 0 else "final"
        bnd_2_lower_or_upper = "lower" if i_2[1] == 0 else "upper"
        msg = (f"The {bnd_2_lower_or_upper} bound for the {bnd_2_t0_or_tF} "
               f"time ('{bnd_2}') must be greater than the "
               f"{bnd_1_lower_or_upper} bound for the {bnd_1_t0_or_tF} time "
               f"('{bnd_1}') in phase {p_info.name} (index #{p_info.index}).")
        raise ValueError(msg)

    def _process_initial_state_cons(self, p_info):
        user_bnd = self.initial_state_constraints
        user_sym = p_info.backend.y_var_user
        bnd_type = "initial state constraint"
        num_expect = p_info.backend.num_y_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect, False)
        y_t0_bnd, self._y_t0_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)
        if self.optimal_control_problem.settings.override_endpoint_bounds:
            y_t0_bnd = self._override_endpoint_bounds(y_t0_bnd)
        self._y_t0_bnd = y_t0_bnd

    def _process_final_state_cons(self, p_info):
        user_bnd = self.final_state_constraints
        user_sym = p_info.backend.y_var_user
        bnd_type = "final state constraint"
        num_expect = p_info.backend.num_y_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect, False)
        y_tF_bnd, self._y_tF_needed = self._process_single_type_value(bnd_info,
                                                                      p_info)
        if self.optimal_control_problem.settings.override_endpoint_bounds:
            y_tF_bnd = self._override_endpoint_bounds(y_tF_bnd)
        self._y_tF_bnd = y_tF_bnd

    def _override_endpoint_bounds(self, y_con_bnd):
        settings = self.optimal_control_problem.settings
        override = settings.override_endpoint_bounds
        lower_is_less = y_con_bnd[:, 0] < self._y_bnd[:, 0]
        if not override and np.any(lower_is_less):
            msg = (f"")
            raise ValueError(msg)
        y_con_bnd[lower_is_less, 0] = self._y_bnd[lower_is_less, 0]
        upper_is_more = y_con_bnd[:, 1] > self._y_bnd[:, 1]
        if not override and np.any(upper_is_more):
            msg = (f"")
            raise ValueError(msg)
        y_con_bnd[upper_is_more, 1] = self._y_bnd[upper_is_more, 1]
        return y_con_bnd

    def _process_single_type_value(self, bnds_info, p_info):
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
        elif isinstance(bnd_info.user_bnd, SUPPORTED_ITER_TYPES):
            bnds = self._process_iterable_bounds_instance(bnds_info, p_info)
        else:
            bnds = self._process_single_type_bounds_instance(bnds_info, p_info)
        bnds, needed = self._check_lower_against_upper(bnds, bnds_info, p_info)
        return bnds, needed

    def _process_mapping_bounds_instance(self, bnds_info, p_info):
        if any(user_sym is None for user_sym in bnds_info.user_sym):
            msg = f"Can't use mapping for {bnds_info.bnd_type} bounds."
            raise TypeError(msg)
        bnds = []
        for bnd_i, user_sym in enumerate(bnds_info.user_sym):
            bnd = bnds_info.user_bnd.get(user_sym)
            bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnd_type, bnd_i)
            self._check_user_bound_missing(bnd_info, p_info)
            bnd = self._as_lower_upper_pair(bnd_info, p_info)
            bnds.append(bnd)
        return bnds

    def _check_user_bound_missing(self, bnd_info, p_info):
        ocp_settings = self.optimal_control_problem.settings
        is_bnd_none = bnd_info.user_bnd is None
        is_inf_assumed = ocp_settings.assume_inf_bounds
        if is_bnd_none and not is_inf_assumed:
            self._raise_user_bound_missing_error(bnd_info, p_info)

    def _raise_user_bound_missing_error(self, bnd_info, p_info):
        msg = (f"No bounds have been supplied for the {bnd_info.bnds_type} "
            f"'{bnd_info.user_syms}' (index #{bnd_info.num}) in phase "
            f"{p_info.name} (index #{p_info.index}).")
        raise ValueError(msg)       

    def _process_iterable_bounds_instance(self, bnd_info, p_info):
        if bnd_info.num == 1 and not isinstance(bnd_info.user_bnd[0],
                                                SUPPORTED_ITER_TYPES):
            bnd_info = bnd_info._replace(user_bnd=[bnd_info.user_bnd])
        bnds = []
        for bnd_i, bnd in enumerate(bnd_info.user_bnd):
            bnd_info = BoundsInfo(bnd, None, bnd_info.bnd_type, bnd_i)
            self._check_user_bound_missing(bnd_info, p_info)
            bnd = self._as_lower_upper_pair(bnd_info, p_info)
            bnds.append(bnd)
        return bnds

    def _process_potential_dual_value_to_single_value(self, bnd_info, p_info):
        bnd = bnd_info.user_bnd
        msg = (f"Single bounds in this form ('{bnd}') are not supported.")
        is_list = isinstance(bnd, SUPPORTED_ITER_TYPES)
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
        for bnd_i, user_sym in enumerate(bnds_info.user_sym):
            bnd = None
            bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
            self._check_user_bound_missing(bnd_info, p_info)
            bnd = self._as_lower_upper_pair(bnd_info, p_info)
            bnds.append(bnd)
        return bnds

    def _as_lower_upper_pair(self, bnd_info, p_info):
        bnd = np.array(bnd_info.user_bnd).flatten()
        if bnd.shape == (1, ):
            both = "lower and upper bounds"
            both_info = bnd_info._replace(user_bnd=bnd[0])
            lower_bnd = self._get_bound_as_number(both_info, p_info, both)
            upper_bnd = lower_bnd
        elif bnd.shape == (2, ):
            lower = "lower bound"
            upper = "upper bound"
            lower_info = bnd_info._replace(user_bnd=bnd[0])
            upper_info = bnd_info._replace(user_bnd=bnd[1])
            lower_bnd = self._get_bound_as_number(lower_info, p_info, lower)
            upper_bnd = self._get_bound_as_number(upper_info, p_info, upper)
        else:
            raise ValueError
        lower_bnd = - self._INF if lower_bnd is None else lower_bnd
        upper_bnd = self._INF if upper_bnd is None else upper_bnd
        return [lower_bnd, upper_bnd]

    def _get_bound_as_number(self, bnd_info, p_info, lower_upper):
        bnd = bnd_info.user_bnd
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
        if isinstance(bnd, (np.float64, np.int64)):
            return bnd
        ocp_backend = self._backend.ocp_backend
        bnd = ocp_backend.substitute_pycollo_sym(bnd, self._backend)
        if symbol_primitives(bnd):
            msg = (f"The user-supplied {lower_upper} for the "
                   f"{bnd_info.bnd_type} '{bnd_info.user_sym}' "
                   f"(index #{bnd_info.num}) in phase {p_info.name} "
                   f"(index #{p_info.index}) of '{bnd}' cannot be "
                   f"precomputed.")
            raise ValueError(msg)
        return float(bnd)

    def _check_lower_against_upper(self, bnd, bnd_info, p_info):
        if not bnd:
            empty_bnd = np.empty(shape=(0, 2), dtype=float)
            empty_ind = np.empty(shape=0, dtype=bool)
            return empty_bnd, empty_ind
        bnd = np.array(bnd, dtype=float)
        bnd, needed = self._check_lower_same_as_upper_to_tol(
            bnd, bnd_info, p_info)
        bnd = self._check_lower_less_than_upper(bnd, bnd_info, p_info)
        return bnd, needed

    def _check_lower_same_as_upper_to_tol(self, bnd, bnd_info, p_info):
        lower_bnd = bnd[:, 0]
        upper_bnd = bnd[:, 1]
        rtol = self.optimal_control_problem.settings.bound_clash_tolerance
        are_same = np.isclose(lower_bnd, upper_bnd, rtol=rtol, atol=0.0)
        if bnd_info.is_variable:
            needed = self._extract_variables_to_constants(bnd, are_same)
        else:
            needed = None
        mean_bnd = (lower_bnd + upper_bnd) / 2
        bnd[are_same, 0] = mean_bnd[are_same]
        bnd[are_same, 1] = mean_bnd[are_same]
        return bnd, needed

    def _check_lower_less_than_upper(self, bnd, bnd_info, p_info):
        lower_bnd = bnd[:, 0]
        upper_bnd = bnd[:, 1]
        lower_less_than_upper = lower_bnd <= upper_bnd
        all_less_than = np.all(lower_less_than_upper)
        if not all_less_than:
            error_indices = np.flatnonzero(~lower_less_than_upper)
            error_syms = np.array(bnd_info.user_sym)[error_indices]
            plural_needed = len(error_indices) > 1
            bound_plural = "bounds" if plural_needed else "bound"
            index_plural = "indices" if plural_needed else "index"
            plural = "s" if plural_needed else ""
            bnd_type_plural = f"{bnd_info.bnd_type}{plural}"
            user_syms_formatted = format_multiple_items_for_output(error_syms)
            user_indices_formatted = format_multiple_items_for_output(
                error_indices, wrapping_char="", prefix_char="#")
            lower_bnd_formatted = format_multiple_items_for_output(
                lower_bnd[error_indices])
            upper_bnd_formatted = format_multiple_items_for_output(
                upper_bnd[error_indices])
            msg = (f"The user-supplied upper {bound_plural} for the "
                   f"{bnd_type_plural} {user_syms_formatted} ({index_plural} "
                   f"{user_indices_formatted}) in phase {p_info.name} (index "
                   f"#{p_info.index}) of {upper_bnd_formatted} cannot be "
                   f"less than the user-supplied lower {bound_plural} of "
                   f"{lower_bnd_formatted}.")
            raise ValueError(msg)
        return bnd

    def _required_variable_bounds(self):
        y_bnd = self._y_bnd[self._y_needed]
        u_bnd = self._u_bnd[self._u_needed]
        q_bnd = self._q_bnd[self._q_needed]
        t_bnd = self._t_bnd[self._t_needed]
        x_bnd = np.vstack([y_bnd, u_bnd, q_bnd, t_bnd])
        return x_bnd


phase_info_fields = ("name", "index", "backend")
PhaseInfo = namedtuple("PhaseInfo", phase_info_fields)

bounds_info_fields = ("user_bnds", "user_syms", "bnds_type", "num",
                      "is_variable", "none_default_allowed")
BoundsInfo = namedtuple("BoundsInfo",
                       bounds_info_fields,
                       defaults=[True, True])


class Bounds:

    def __init__(self, ocp_backend):
        self.ocp_backend = ocp_backend
        self.process_and_check_user_values()
        self.collect_required_variable_bounds()
        self.collect_required_state_variable_endpoint_bounds()
        self.collect_constraint_bounds()
        self.add_unrequired_variables_to_auxiliary_data()

    def process_and_check_user_values(self):
        for p in self.ocp_backend.p:
            p.ocp_phase.bounds._process_and_check_user_values(p)
        self.ocp_backend.ocp.bounds._process_and_check_user_values()

    def collect_required_variable_bounds(self):
        x_bnd = []
        for p in self.ocp_backend.p:
            p_bnds = p.ocp_phase.bounds
            x_bnd.append(p_bnds._required_variable_bounds())
        x_bnd.append(self.ocp_backend.ocp.bounds._required_variable_bounds())
        self.x_bnd = np.vstack(x_bnd)

    def collect_required_state_variable_endpoint_bounds(self):
        y_t0_bnd = []
        y_tF_bnd = []
        for p in self.ocp_backend.p:
            p_bnd = p.ocp_phase.bounds
            y_t0_bnd.append(p_bnd._y_t0_bnd[p_bnd._y_needed])
            y_tF_bnd.append(p_bnd._y_tF_bnd[p_bnd._y_needed])
        self.y_t0_bnd = np.vstack(y_t0_bnd)
        self.y_tF_bnd = np.vstack(y_tF_bnd)

    @property
    def x_bnd_lower(self):
        return self.x_bnd[:, 0]

    @property
    def x_bnd_upper(self):
        return self.x_bnd[:, 1]

    def collect_constraint_bounds(self):
        pass

    def add_unrequired_variables_to_auxiliary_data(self):
        self.aux_data = {}
        for p in self.ocp_backend.p:
            p_bnd = p.ocp_phase.bounds
            self.aux_data.update({y: np.mean(value) 
                for y, y_needed, value in zip(
                    p.y_var_full, p_bnd._y_needed, p_bnd._y_bnd) 
                if not y_needed})
            self.aux_data.update({u: np.mean(value) 
                for u, u_needed, value in zip(
                    p.u_var_full, p_bnd._u_needed, p_bnd._u_bnd) 
                if not u_needed})
            self.aux_data.update({q: np.mean(value) 
                for q, q_needed, value in zip(
                    p.q_var_full, p_bnd._q_needed, p_bnd._q_bnd) 
                if not q_needed})
            self.aux_data.update({t: np.mean(value) 
                for t, t_needed, value in zip(
                    p.t_var_full, p_bnd._t_needed, p_bnd._t_bnd) 
                if not t_needed})
        prob_bnd = self.ocp_backend.ocp.bounds
        self.aux_data.update({s: np.mean(value) 
            for s, s_needed, value in zip(
                self.ocp_backend.s_var_full, prob_bnd._s_needed, prob_bnd._s_bnd) 
            if not s_needed})
