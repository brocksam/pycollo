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
                    format_for_output,
                    SUPPORTED_ITER_TYPES,
                    symbol_primitives,
                    )


# Default values for settings
DEFAULT_ASSUME_INF_BOUNDS = True
DEFAULT_BOUND_CLASH_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_BOUND_CLASH_RELATIVE_TOLERANCE = 1e-6
DEFAULT_NUMERICAL_INF = 10e19
DEFAULT_OVERRIDE_ENDPOINTS = True
DEFAULT_REMOVE_CONSTANT_VARIABLES = True


# Data structures
phase_info_fields = ("name", "index", "backend")
PhaseInfo = namedtuple("PhaseInfo", phase_info_fields)
"""Data structure for information about OCP phases.

These are mostly used to format descriptive error messages for the user.

Fields
------
name : str
    The name associated with the phase
index : int
    The index of the phase.
backend : :py:class:`PycolloPhaseData`
    The phase backend associated with the specified OCP phase.

"""

bounds_info_fields = ("user_bnds", "user_syms", "bnds_type", "num",
                      "is_variable", "none_default_allowed")
BoundsInfo = namedtuple("BoundsInfo",
                        bounds_info_fields,
                        defaults=[True, True])
"""Data structure for storing information about user-supplied bounds.

Fields
------
user_bnds : obj
    The bounds that the user has supplied.
user_syms : Iterable[sym.Symbols]
    An iterable of symbols relating to the user-supplied bounds (if available).
bnds_type : str
    String indentifying the aspect of the OCP that the bounds relate to. Mostly
    used for formatting descriptive error messages for the user.
num : int
    The number of variables/constraints that should be expected for the type of
    bounds in question.
is_variable : bool
    `True` if the bound type in question is a variable, `False` if it is a
    constraint.
none_default_allowed : bool
    `True` if Pycollo should automatically handle the situation where no bounds
    have been supplied. `False` if an error should be raised.

"""


class BoundsABC(ABC):

    @abstractmethod
    def optimal_control_problem(self):
        pass

    @abstractmethod
    def _process_and_check_user_values(self):
        pass

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

        self.ocp = optimal_control_problem
        self.parameter_variables = parameter_variables
        self.endpoint_constraints = endpoint_constraints

    @property
    def optimal_control_problem(self):
        return self.ocp

    def _process_and_check_user_values(self):
        self._backend = self.optimal_control_problem._backend
        self._INF = self.optimal_control_problem.settings.numerical_inf
        self._process_parameter_vars()
        self._process_endpoint_cons()

    def _process_parameter_vars(self):
        user_bnds = self.parameter_variables
        user_syms = self._backend.s_var_user
        bnds_type = "parameter variable"
        num_expected = self._backend.num_s_var_full
        bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expected)
        self._s_bnd, self._s_needed = process_single_type_of_values(self,
                                                                    bnds_info)

    def _process_endpoint_cons(self):
        num_b_con = self.optimal_control_problem.number_endpoint_constraints
        user_bnds = self.endpoint_constraints
        user_syms = [None] * num_b_con
        bnds_type = "endpoint constraints"
        num_expect = num_b_con
        bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expect,
                               False)
        self._b_con_bnd, needed = process_single_type_of_values(self,
                                                                bnds_info)

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
        self.ocp = phase.optimal_control_problem
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
        user_bnds = self.state_variables
        user_syms = p_info.backend.y_var_user
        bnds_type = "state variable"
        num_expect = p_info.backend.num_y_var_full
        bnds_info = BoundsInfo(user_bnds, user_syms, bnds_type, num_expect)
        self._y_bnd, self._y_needed = process_single_type_of_values(self,
                                                                    bnds_info,
                                                                    p_info)

    def _process_control_vars(self, p_info):
        user_bnd = self.control_variables
        user_sym = p_info.backend.u_var_user
        bnd_type = "control variable"
        num_expect = p_info.backend.num_u_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._u_bnd, self._u_needed = process_single_type_of_values(self,
                                                                    bnd_info,
                                                                    p_info)

    def _process_integral_vars(self, p_info):
        user_bnd = self.integral_variables
        user_sym = p_info.backend.q_var_user
        bnd_type = "integral variable"
        num_expect = p_info.backend.num_q_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._q_bnd, self._q_needed = process_single_type_of_values(self,
                                                                    bnd_info,
                                                                    p_info)

    def _process_path_cons(self, p_info):
        user_bnd = self.path_constraints
        user_sym = [None] * p_info.backend.num_p_con
        bnd_type = "path constraints"
        num_expect = p_info.backend.num_p_con
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect, False)
        self._p_con_bnd, needed = process_single_type_of_values(self,
                                                                bnd_info,
                                                                p_info)

    def _process_time_vars(self, p_info):
        user_bnd = [self.initial_time, self.final_time]
        user_sym = p_info.backend.t_var_user
        bnd_type = "time variable"
        num_expect = p_info.backend.num_t_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect)
        self._t_bnd, self._t_needed = process_single_type_of_values(self,
                                                                    bnd_info,
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
        y_t0_bnd, self._y_t0_needed = process_single_type_of_values(self,
                                                                    bnd_info,
                                                                    p_info)
        if self.ocp.settings.override_endpoint_bounds:
            y_t0_bnd = self._override_endpoint_bounds(y_t0_bnd)
        self._y_t0_bnd = y_t0_bnd

    def _process_final_state_cons(self, p_info):
        user_bnd = self.final_state_constraints
        user_sym = p_info.backend.y_var_user
        bnd_type = "final state constraint"
        num_expect = p_info.backend.num_y_var_full
        bnd_info = BoundsInfo(user_bnd, user_sym, bnd_type, num_expect, False)
        y_tF_bnd, self._y_tF_needed = process_single_type_of_values(self,
                                                                    bnd_info,
                                                                    p_info)
        if self.ocp.settings.override_endpoint_bounds:
            y_tF_bnd = self._override_endpoint_bounds(y_tF_bnd)
        self._y_tF_bnd = y_tF_bnd

    def _override_endpoint_bounds(self, y_con_bnd):
        settings = self.ocp.settings
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

    # def _process_potential_dual_value_to_single_value(self, bnd_info, p_info):
    #     bnd = bnd_info.user_bnd
    #     msg = (f"Single bounds in this form ('{bnd}') are not supported.")
    #     is_list = isinstance(bnd, SUPPORTED_ITER_TYPES)
    #     if not is_list:
    #         raise TypeError(msg)
    #     is_len_2 = len(bnd) == 2
    #     if not is_len_2:
    #         raise ValueError(msg)
    #     is_pair_same = bnd[0] == bnd[1]
    #     if not is_pair_same:
    #         raise ValueError(msg)
    #     bnd = bnd[0]
    #     bnd_info = bnd_info._replace(user_bnds=bnd)
    #     return bnd_info

    def _required_variable_bounds(self):
        y_bnd = self._y_bnd[self._y_needed]
        u_bnd = self._u_bnd[self._u_needed]
        q_bnd = self._q_bnd[self._q_needed]
        t_bnd = self._t_bnd[self._t_needed]
        x_bnd = np.vstack([y_bnd, u_bnd, q_bnd, t_bnd])
        return x_bnd


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


"""









"""


def process_single_type_of_values(bnds_obj, bnds_info, p_info=None):
    """Given a `BoundsInfo` object, process and determine if needed.

    Bounds can either be passed by the user as:
        * a dictionary with the keys as the OCP symbols and the values as the
            bounds;
        * no bounds via the use of `None`; or
        * an iterable of supported type (e.g. tuple, list, np.ndarray) provided
            that the first dimension is the number of variables/constraints of
            that type and the second dimension is either 1 or 2 (depending on
            the circumstance).

    Note that some forms of bounds are not supported for specific types of
    bounds.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns:
    --------
    `tuple`
    Of length 2 with the first item being a :py:class:`ndarray <numpy>` with
        the correctly formatted bounds and the second item being another
        :py:class:`ndarray <numpy>` of type `bool` stating whether the bounds
        are needed (i.e. have they been determined to be equal in upper and
        lower bound so that Pycollo can remove them from the OCP and instead
        treat them as variables).

    Raises
    ------
    TypeError
        If the bounds supplied by the user are of a type that cannot be handled
        by Pycollo.

    """
    if isinstance(bnds_info.user_bnds, dict):
        bnds = process_mapping_bounds_instance(bnds_obj, bnds_info, p_info)
    elif bnds_info.user_bnds is None:
        bnds = process_none_bounds_instance(bnds_obj, bnds_info, p_info)
    elif isinstance(bnds_info.user_bnds, SUPPORTED_ITER_TYPES):
        bnds = process_iterable_bounds_instance(bnds_obj, bnds_info, p_info)
    else:
        formatted_valid_types = format_for_output(SUPPORTED_ITER_TYPES)
        msg = (f"Bounds for {bnds_info.bnds_type} cannot be supplied as a "
               f"{type(bnds_info.user_bnds)}, use one of: "
               f"{formatted_valid_types}")
        raise TypeError(msg)
    bnds, needed = check_lower_against_upper(bnds_obj, bnds, bnds_info, p_info)
    return bnds, needed


def process_mapping_bounds_instance(bnds_obj, bnds_info, p_info):
    """Used to process bounds supplied by the user as a `dict`.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    list
        A list of lists with the outer length equal to the number of expected
        bounds and the inner lengths all equal to 2.

    Raises
    ------
    TypeError
        If the bounds type is not supported for use of dictionary because there
        aren't symbols associated with every variable/constraint of that type.

    """
    if any(user_sym is None for user_sym in bnds_info.user_syms):
        msg = f"Can't use mapping for {bnds_info.bnds_type} bounds."
        raise TypeError(msg)
    bnds = []
    for bnd_i, user_sym in enumerate(bnds_info.user_syms):
        bnd = bnds_info.user_bnds.get(user_sym)
        bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
        check_user_bound_missing(bnds_obj, bnd_info, p_info)
        bnd = as_lower_upper_pair(bnds_obj, bnd_info, p_info)
        bnds.append(bnd)
    return bnds


def check_user_bound_missing(bnds_obj, bnds_info, p_info):
    """Check if any user-supplied bounds for a specific type are missing.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Raises
    ------
    ValueError
        If there are bounds that need to be supplied by aren't.

    """
    is_bnd_none = bnds_info.user_bnds is None
    is_inf_assumed = bnds_obj.ocp.settings.assume_inf_bounds
    if is_bnd_none and not is_inf_assumed:
        msg = (f"No bounds have been supplied for the {bnds_info.bnds_type} "
               f"'{bnds_info.user_syms}' (index #{bnds_info.num}).")
        raise ValueError(msg)


def process_iterable_bounds_instance(bnds_obj, bnds_info, p_info):
    """Used to process bounds supplied by the user as a `dict`.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    list
        A list of lists with the outer length equal to the number of expected
        bounds and the inner lengths all equal to 2.

    Raises
    ------
    TypeError
        If the bounds type is not supported for use of dictionary because there
        aren't symbols associated with every variable/constraint of that type.

    """
    supported_iter = isinstance(bnds_info.user_bnds[0], SUPPORTED_ITER_TYPES)
    if bnds_info.num == 1 and not supported_iter:
        bnds_info = bnds_info._replace(user_bnds=[bnds_info.user_bnds])
    bnds = []
    for bnd_i, bnd in enumerate(bnds_info.user_bnds):
        bnd_info = BoundsInfo(bnd, None, bnds_info.bnds_type, bnd_i)
        check_user_bound_missing(bnds_obj, bnd_info, p_info)
        bnd = as_lower_upper_pair(bnds_obj, bnd_info, p_info)
        bnds.append(bnd)
    return bnds


def process_none_bounds_instance(bnds_obj, bnds_info, p_info):
    """Used to process bounds supplied by the user as a `dict`.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    list
        A list of lists with the outer length equal to the number of expected
        bounds and the inner lengths all equal to 2.

    Raises
    ------
    TypeError
        If the bounds type is not supported for use of dictionary because there
        aren't symbols associated with every variable/constraint of that type.

    """
    bnds = []
    for bnd_i, user_sym in enumerate(bnds_info.user_syms):
        bnd = None
        bnd_info = BoundsInfo(bnd, user_sym, bnds_info.bnds_type, bnd_i)
        check_user_bound_missing(bnds_obj, bnd_info, p_info)
        bnd = as_lower_upper_pair(bnds_obj, bnd_info, p_info)
        bnds.append(bnd)
    return bnds


def as_lower_upper_pair(bnds_obj, bnds_info, p_info):
    """Get the user-supplied bounds as a lower-upper pair of numeric values.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    `list`
        Pair of bounds as a lower bound (first) and an upper bound (second) in
        a `list`.

    Raises
    ------
    ValueError
        If the flattened user-supplied bounds are not either shape (1, ) or
        (2, ).

    """
    bnds = np.array(bnds_info.user_bnds).flatten()
    if bnds.shape == (1, ):
        both = "lower and upper bounds"
        both_info = bnds_info._replace(user_bnds=bnds[0])
        lower_bnd = get_bound_as_number(bnds_obj, both_info, both, p_info)
        upper_bnd = lower_bnd
    elif bnds.shape == (2, ):
        lower = "lower bound"
        upper = "upper bound"
        lower_info = bnds_info._replace(user_bnds=bnds[0])
        upper_info = bnds_info._replace(user_bnds=bnds[1])
        lower_bnd = get_bound_as_number(bnds_obj, lower_info, lower, p_info)
        upper_bnd = get_bound_as_number(bnds_obj, upper_info, upper, p_info)
    else:
        raise ValueError
    lower_bnd = -bnds_obj._INF if lower_bnd is None else lower_bnd
    upper_bnd = bnds_obj._INF if upper_bnd is None else upper_bnd
    return [lower_bnd, upper_bnd]


def get_bound_as_number(bnds_obj, bnds_info, lower_upper, p_info):
    """Format user-supplied bounds to be a number.

    Users can potentially supply bounds as strings (such as "inf" etc.),
    numerical values from non-core Python (e.g. :py:type`float64 <numpy>`,
    :py:type:`DM <casadi>`), or as symbols (e.g. :py:type:`Symbol <sympy>`,
    :py:type:`SX <casadi>`) provided that they can be resolved as constants due
    to auxiliary data supplied by the user.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    float
        The bound as a numerical value.

    Raises
    ------
    ValueError
        If the user-supplied bound is symbolic and contains a symbol primitive
        that cannot be resolved down to a numerical value.
    NotImplementedError
        If the user supplies a string bound that is unsupported, e.g. 'nan'.

    """
    bnds = bnds_info.user_bnds
    if bnds is None:
        return bnds
    elif isinstance(bnds, str):
        if bnds == "inf":
            return bnds_obj._INF
        elif bnds == "-inf":
            return -bnds_obj._INF
        try:
            bnds = float(bnds)
        except TypeError:
            msg = (f"A bound value of {bnds} is not supported.")
            raise NotImplementedError(msg)
    if isinstance(bnds, (np.float64, np.int64, float, int)):
        return float(bnds)
    bnds = bnds_obj.ocp._backend.substitute_pycollo_sym(bnds)
    if symbol_primitives(bnds):
        msg = (f"The user-supplied {lower_upper} for the "
               f"{bnds_info.bnds_type} '{bnd_info.user_syms}' "
               f"(index #{bnds_info.num}) of '{bnds}' "
               f"cannot be precomputed.")
        raise ValueError(msg)
    return float(bnds)


def check_lower_against_upper(bnds_obj, bnds, bnds_info, p_info):
    """Abstraction layer for checking lower bound against upper bound in pair.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds : `list`
        The pre-processed bounds.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    `tuple`
        The first index is an :py:type:`ndarray <numpy>` of shape (2, ) with
        the numerical lower and upper bounds for the bound in question and the
        second index is a `bool` of whether that bound pair is needed in the
        OCP (`True`) or if it can be treated as a constant (`False`).

    """
    if not bnds:
        bnds = np.empty(shape=(0, 2), dtype=float)
        needed = np.empty(shape=0, dtype=bool)
        return bnds, needed
    bnds = np.array(bnds, dtype=float)
    bnds, needed = check_lower_same_as_upper_to_tol(bnds_obj, bnds, bnds_info,
                                                    p_info)
    bnds = check_lower_less_than_upper(bnds_obj, bnds, bnds_info, p_info)
    return bnds, needed


def check_lower_same_as_upper_to_tol(bnds_obj, bnds, bnd_info, p_info):
    """Handle case where bounds are equal to floating precision.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds : `list`
        The pre-processed bounds.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    `tuple`
        The first index is an :py:type:`ndarray <numpy>` of shape (2, ) with
        the numerical lower and upper bounds for the bound in question and the
        second index is a `bool` of whether that bound pair is needed in the
        OCP (`True`) or if it can be treated as a constant (`False`).

    """
    lower_bnds = bnds[:, 0]
    upper_bnds = bnds[:, 1]
    atol = bnds_obj.ocp.settings.bound_clash_relative_tolerance
    rtol = bnds_obj.ocp.settings.bound_clash_absolute_tolerance
    are_same = np.isclose(lower_bnds, upper_bnds, rtol=rtol, atol=atol)
    needed = extract_variables_to_constants(bnds_obj, bnds, are_same)
    mean_bnds = (lower_bnds + upper_bnds) / 2
    bnds[are_same, 0] = mean_bnds[are_same]
    bnds[are_same, 1] = mean_bnds[are_same]
    return bnds, needed


def check_lower_less_than_upper(bnds_obj, bnds, bnds_info, p_info):
    """Ensure the lower bound is less than the upper bound.

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds : `list`
        The pre-processed bounds.
    bnds_info : `BoundsInfo`
        The bounds info that is being processed.

    Returns
    -------
    :py:type:`ndarray <numpy>`
        The lower-upper bound pair with shape (2, ).

    Raises
    ------
    ValueError
        If any lower bounds are greater than their upper bound.

    """
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
        bnds_type_plural = (f"{bnds_info.bnds_type}"
                            f"{'s' if plural_needed else ''}")
        user_syms_formatted = format_for_output(error_syms)
        user_indices_formatted = format_for_output(
            error_indices, wrapping_char="", prefix_char="#")
        lower_bnds_formatted = format_for_output(lower_bnds[error_indices])
        upper_bnds_formatted = format_for_output(upper_bnds[error_indices])
        msg = (f"The user-supplied upper {bound_plural} for the "
               f"{bnds_type_plural} {user_syms_formatted} ({index_plural} "
               f"{user_indices_formatted}) of {upper_bnds_formatted} "
               f"cannot be less than the user-supplied lower "
               f"{bound_plural} of {lower_bnds_formatted}.")
        raise ValueError(msg)
    return bnds


def extract_variables_to_constants(bnds_obj, bnds, are_same):
    """

    Parameters
    ----------
    bnds_obj : Union[`EndpointBounds`, `PhaseBounds`]
        The parent bounds-related object for which this function is processing
        bounds for.
    bnds : `list`
        The pre-processed bounds.
    are_same : `bool`
        If bounds are equal.

    Returns
    -------
    bool
        `True` if the bounds pair are needed, `False` if not.

    """
    if not bnds_obj.ocp.settings.remove_constant_variables:
        needed = np.full(bnds.shape[0], True)
        return needed
    needed = ~are_same
    return needed



























