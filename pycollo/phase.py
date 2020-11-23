"""Everything needed for defining phases within an optimal control problem.

Classes:
	Phase
"""


import copy
import itertools
from typing import (Optional, Tuple)

import sympy as sym

from .bounds import PhaseBounds
from .guess import PhaseGuess
from .mesh import PhaseMesh
from .scaling import PhaseScaling
from .typing import (OptionalExprsType, OptionalSymsType, TupleSymsType)
from .utils import (check_sym_name_clash, format_as_named_tuple)


__all__ = ["Phase"]


class Phase:
    """A single continuous time phase as part of an optimal control problem.

    Attributes:
            name: The name associated with a problem. Should be something short 
                    like 'A'.
            optimal_control_problem: The :obj:`OptimalControlProblem` with which 
                    this phase is to be associated.
            state_variables: The continuous time state variables in this phase.
            control_variables: The continuous time control variables in this phase.
            state_equations: The dynamical state equations associated with this 
                    state variables in this phase.
            integrand_functions: The integrand functions corresponding to the 
                    integral variables in this phase.
            path_constraints: The continuous time path constraints associated with 
                    this phase.
            bounds: The phase bounds on this phase. See :obj:PhaseBounds for more 
                    details.
            scaling: The phase scaling on this phase. See :obj:PhaseScaling for 
                    more details.
            guess: The initial guess at which this phase is to be solved.
            mesh: This initial mesh on which this phase is to be solved.
            _name: Protected version of :attr:`name`.
            _ocp: Protected version of :attr:`optimal_control_problem`.
            _phase_number: Protected integer number associated with this phase. If 
                    not associated with any optimal control problem then defaults to 
                    None until one is associated. These are ordered sequentially 
                    starting at '0' in the order with which phases are added to an 
                    optimal control problem.
            _phase_suffix: Protected str which is used in the naming of auto-
                    generated Pycollo variables such as the endpoint state variables.
            _y_var_user: Protected version of :attr:`state_variables`. 
            _u_var_user: Protected version of :attr:`control_variables`.
            _q_var_user: Protected version of :attr:`integral_variables`.
            _t_var_user: Protected version of :attr:`time_variables`.
            _y_eqn_user: Protected version of :attr:`state_equations`. 
            _c_con_user: Protected version of :attr:`path_constraints`. 
            _q_fnc_user: Protected version of :attr:`integrand_functions`.
            _t0_USER: Protected version of :attr:`initial_time_variable`.
            _tF_USER: Protected version of :attr:`final_time_variable`.
            _t0: Internal Pycollo symbol for phase initial time.
            _tF: Internal Pycollo symbol for phase final time.
            _STRETCH: Convenience expression for phase time scaling stretch.
            _SHIFT: Convenience expression for phase time scaling shift.
    """

    def __init__(self,
                 name: str,
                 *,
                 optimal_control_problem: Optional["OptimalControlProblem"] = None,
                 state_variables: OptionalSymsType = None,
                 control_variables: OptionalSymsType = None,
                 state_equations: OptionalExprsType = None,
                 integrand_functions: OptionalExprsType = None,
                 path_constraints: OptionalExprsType = None,
                 bounds: Optional[PhaseBounds] = None,
                 scaling: Optional[PhaseScaling] = None,
                 guess: Optional[PhaseGuess] = None,
                 mesh: Optional[PhaseMesh] = None,
                 ):
        """Initialise the Phase object with minimum a name.

        Args: 
                name: The name associated with a problem. Should be something short 
                        like 'A'.
                optimal_control_problem: The :obj:`OptimalControlProblem` with 
                        which this phase is to be associated. Default value is None in 
                        which case the phase remain uninitialised to an optimal control 
                        problem.
                state_variables: The continuous time state variables in this phase. 
                        Default value is None in which case the phase has no associated 
                        state variables and no phase-specific endpoint time or state 
                        variables are created.
                control_variables: The continuous time control variables in this 
                        phase. Default value is None in which case the phase has no 
                        associated control variables.
                state_equations: The dynamical state equations associated with this 
                        state variables in this phase. Default value is None in which 
                        case no dynamical equations have been added to the phase yet.
                integrand_functions: The integrand functions corresponding to the 
                        integral variables in this phase. Default value is None in 
                        which case the phase has no integrand functions associated with 
                        it and no phase-specific integral variables are created.
                path_constraints: The continuous time path constraints associated 
                        with this phase. Default value is None in which case the phase 
                        has no path constraints associated with it.
                bounds: The phase bounds on this phase. See :obj:PhaseBounds for 
                        more details. Default value is None in which case an empty 
                        :obj:`PhaseBounds` object is instantiated and associated with 
                        the phase.
                scaling: The phase scaling on this phase. See :obj:PhaseScaling for 
                        more details. Default value is None in which case an empty 
                        :obj:`PhaseScaling` object is instantiated and associated with 
                        the phase.
                guess: The initial guess at which this phase is to be solved. 
                        Default value is None in which case an empty :obj:`PhaseGuess` 
                        object is instantiated and associated with the phase.
                mesh: This initial mesh on which this phase is to be solved. 
                        Default value is None in which case an empty :obj:`PhaseMesh` 
                        object is instantiated and associated with the phase.
        """

        self._name = str(name)
        self._ocp = None
        self._phase_number = None
        self._phase_suffix = "X"

        self._y_var_user = ()
        self._u_var_user = ()
        self._q_var_user = ()
        self._t_var_user = ()

        self._y_eqn_user = ()
        self._c_con_user = ()
        self._q_fnc_user = ()

        if optimal_control_problem is not None:
            self.optimal_control_problem = optimal_control_problem

        self.state_variables = state_variables
        self.control_variables = control_variables

        self.state_equations = state_equations
        self.integrand_functions = integrand_functions
        self.path_constraints = path_constraints

        self.bounds = bounds
        self.scaling = scaling
        self.guess = guess
        self.mesh = mesh

        self.auxiliary_data = {}

    def create_new_copy(self,
                        name: str,
                        *,
                        copy_state_variables: bool = True,
                        copy_control_variables: bool = True,
                        copy_state_equations: bool = True,
                        copy_path_constraints: bool = True,
                        copy_integrand_functions: bool = True,
                        copy_state_endpoint_constraints: bool = False,
                        copy_bounds: bool = True,
                        copy_mesh: bool = True,
                        copy_scaling: bool = True,
                        copy_guess: bool = True,
                        ):

        self._check_variables_and_equations()
        new_phase = Phase(name,
                          optimal_control_problem=self.optimal_control_problem)

        if copy_state_variables:
            new_phase.state_variables = copy.deepcopy(self.state_variables)
            if copy_bounds:
                new_phase.bounds.state_variables = copy.deepcopy(self.bounds.state_variables)
            if copy_guess:
                new_phase.guess.state_variables = copy.deepcopy(self.guess.state_variables)
            if copy_scaling:
                new_phase.scaling.state_variables = copy.deepcopy(self.scaling.state_variables)

        if copy_control_variables:
            new_phase.control_variables = copy.deepcopy(self.control_variables)
            if copy_bounds:
                new_phase.bounds.control_variables = copy.deepcopy(self.bounds.control_variables)
            if copy_guess:
                new_phase.guess.control_variables = copy.deepcopy(self.guess.control_variables)
            if copy_scaling:
                new_phase.scaling.control_variables = copy.deepcopy(self.scaling.control_variables)

        if copy_state_equations:
            new_phase.state_equations = copy.deepcopy(self.state_equations)

        if copy_path_constraints:
            new_phase.path_constraints = copy.deepcopy(self.path_constraints)
            if copy_bounds:
                new_phase.bounds.path_constraints = copy.deepcopy(self.bounds.path_constraints)

        if copy_integrand_functions:
            new_phase.integrand_functions = copy.deepcopy(self.integrand_functions)
            if copy_bounds:
                new_phase.bounds.integral_variables = copy.deepcopy(self.bounds.integral_variables)

        if copy_state_endpoint_constraints and copy_bounds:
            new_phase.bounds.state_endpoint_constraints = copy.deepcopy(self.bounds.state_endpoint_constraints)

        if copy_mesh:
            new_phase.mesh = copy.deepcopy(self.mesh)

        return new_phase

    @staticmethod
    def create_new_copy_like(phase_for_copying: "Phase", name: str, **kwargs):
        """Constructor class to copy a phase."""
        return phase_for_copying.create_new_copy(name, **kwargs)

    @property
    def name(self):
        """Name of the phase."""
        return self._name

    @property
    def optimal_control_problem(self) -> Optional["OptimalControlProblem"]:
        """The optimal control problem with which this phase is associated.

        There are two allowable scenarios. In the first scenario a phase may be 
        instantiated without being associated with an optimal control problem. 
        If this is the case then the default values of `None` for the phase 
        number and 'X' for the phase suffix remain. 

        In the second scenario a phase is instantiated with an associated 
        optimal control problem or is associated with an optimal control 
        problem after the first type of instantiation. In this case the phase 
        is appended to the protected `_phases` attribute of the 
        :obj:`OptimalControlProblem`, the phase number is set according to its 
        position in the order of addition to the optimal controls problem's 
        phases, and its phase suffix is set as a string version of the phase 
        number. Finally a replacement of any symbols that may have been used in 
        supplementary information about the phase that contained the placeholder 
        'X' phase suffix are renamed and substituted.

        No checking is done to see whether the phase is already associated with 
        the optimal control problem in question or any other optimal control 
        problem. The reason being that if the setter method for this property is 
        accessed after having already been set then an `AttributeError` is 
        raised (see below). The reason this class works like that is to avoid 
        having to allow phases to be disassociated from an 
        :obj:`OptimalControlProblem` and thus having to handled the complexities 
        that would come with the phase renumbering and substitution of any 
        phase-related information that has already been given to the optimal 
        control problem.

        Raises:
                AttributeError: If an :obj:`OptimalControlProblem` has already been 
                        associated with `self`. If a argument of any type other than 
                        :obj:`OptimalControlProblem` is passed to the 
                        `optimal_control_problem` property setter.
        """
        return self._ocp

    @optimal_control_problem.setter
    def optimal_control_problem(self, ocp):
        if self._ocp is not None:
            msg = ('Optimal control problem is already set for this phase and '
                   'cannot be reset.')
            raise AttributeError(msg)

        try:
            previous_phase_names = ocp._phases._fields
        except AttributeError:
            previous_phase_names = ()
        phase_names = (*previous_phase_names, self.name)
        ocp._phases = format_as_named_tuple([*ocp._phases, self],
                                            named_keys=phase_names, sympify=False)

        self._ocp = ocp
        self._phase_number = self._ocp.number_phases - 1
        self._phase_suffix = str(self.phase_number)

        self.state_variables = self.state_variables
        self.integrand_functions = self.integrand_functions

    @property
    def phase_number(self) -> Optional[int]:
        """The integer numerical identifier for the phase.

        If this phase has not yet been associated with an optimal control 
        problem then None is returned.

        Corresponds to the chronological order in which it was associated with 
        the optimal control problem in question.
        """
        return self._phase_number

    @property
    def initial_time_variable(self) -> sym.Symbol:
        """Symbol for the time at which this phase begins."""
        try:
            return self._t0_USER
        except AttributeError:
            msg = ("Can't access initial time until associated with an optimal "
                   "control problem.")
            raise AttributeError(msg)

    @property
    def final_time_variable(self) -> sym.Symbol:
        """Symbol for the time at which this phase begins."""
        try:
            return self._tF_USER
        except AttributeError:
            msg = ("Can't access final time until associated with an optimal "
                   "control problem.")
            raise AttributeError(msg)

    @property
    def initial_state_variables(self) -> TupleSymsType:
        """Symbols for this phase's state variables at the initial time.

        Raises:
                AttributeError: If `optimal_control_problem` property has not yet 
                        been set to a not None value. See docstring for 
                        `state_variables` for details about why.
        """
        try:
            return self._y_t0_user
        except AttributeError:
            msg = ("Can't access initial state until associated with an optimal "
                   "control problem.")
            raise AttributeError(msg)

    @property
    def final_state_variables(self) -> TupleSymsType:
        """Symbols for this phase's state variables at the final time.

        Raises:
                AttributeError: If `optimal_control_problem` property has not yet 
                        been set to a not None value. See docstring for 
                        `state_variables` for details about why.
        """
        try:
            return self._y_tF_user
        except AttributeError:
            msg = ("Can't access initial state until associated with an optimal "
                   "control problem.")
            raise AttributeError(msg)

    @property
    def state_variables(self) -> TupleSymsType:
        """Symbols for this phase's state variables in order added by user.

        The user may supply either a single symbol or an iterable of symbols. 
        The supplied argument is handled by the `format_as_tuple` method from 
        the `utils` module. Additional protected attributes `_y_t0_user` and 
        `_y_tF_user` are set by post-appending either '_PX(t0)' or '_PX(tF)' to 
        the user supplied symbols where the X is replaced by the phase suffix. 
        As such if this phase has not yet been associated with an optimal 
        control problem yet then `self` will not have attributes `_y_t0_user` 
        and `_y_tF_user` and accessing either the `initial_state` or 
        `final_state` property will raise an AttributeError.
        """
        return self._y_var_user

    @state_variables.setter
    def state_variables(self, y_vars: OptionalSymsType):

        self._y_var_user = format_as_named_tuple(y_vars)
        check_sym_name_clash(self._y_var_user)

        # Generate the state endpoint variable symbols only if phase has number
        if self.optimal_control_problem is not None:
            self._t0_USER = sym.Symbol(f't0_P{self._phase_suffix}')
            self._tF_USER = sym.Symbol(f'tF_P{self._phase_suffix}')
            self._t0 = sym.Symbol(f'_t0_P{self._phase_suffix}')
            self._tF = sym.Symbol(f'_tF_P{self._phase_suffix}')
            self._STRETCH = 0.5 * (self._tF - self._t0)
            self._SHIFT = 0.5 * (self._t0 + self._tF)
            self._t_var_user = (self._t0_USER, self._tF_USER)

            try:
                named_keys = self._y_var_user._fields
            except AttributeError:
                named_keys = ()

            self._y_t0_user = format_as_named_tuple(
                (sym.Symbol(f'{y}_P{self._phase_suffix}(t0)')
                 for y in self._y_var_user),
                named_keys=named_keys)
            self._y_tF_user = format_as_named_tuple(
                (sym.Symbol(f'{y}_P{self._phase_suffix}(tF)')
                 for y in self._y_var_user),
                named_keys=named_keys)

    @property
    def number_state_variables(self) -> int:
        """Integer number of state variables in the phase."""
        return len(self._y_var_user)

    @property
    def control_variables(self) -> TupleSymsType:
        """Symbols for this phase's control variables in order added by user.

        The user may supply either a single symbol or an iterable of symbols.
        The supplied argument is handled by the `format_as_tuple` method from
        the `utils` module.
        """
        return self._u_var_user

    @control_variables.setter
    def control_variables(self, u_vars: OptionalSymsType):
        self._u_var_user = format_as_named_tuple(u_vars)
        check_sym_name_clash(self._u_var_user)

    @property
    def number_control_variables(self) -> int:
        """Integer number of control variables in the phase."""
        return len(self._u_var_user)

    @property
    def integral_variables(self) -> TupleSymsType:
        """Symbols for this phase's integral variables.

        These symbols are auto generated as required by the user-supplied
        integrand functions.
        """
        return self._q_var_user

    @property
    def time_variables(self) -> TupleSymsType:
        """The initial and final time symbols as a pair."""
        return (self.initial_time_variable, self.final_time_variable)

    @property
    def number_integral_variables(self) -> int:
        """Integer number of integral variables in the phase."""
        return len(self._q_var_user)

    @property
    def state_equations(self) -> Tuple[sym.Expr, ...]:
        """User-supplied dynamical equations in the phase.

        These equations are the dynamical equations associated with each of the 
        state variables in the phase. There should therefore be exactly one 
        state equation for each dynamics symbol.

        State equations can be supplied in a compact form by the user defining additional auxiliary symbols and 
        """
        return self._y_eqn_user

    @state_equations.setter
    def state_equations(self, y_eqns: OptionalExprsType):
        try:
            named_keys = self._y_var_user._fields
        except AttributeError:
            named_keys = ()
        self._y_eqn_user = format_as_named_tuple(y_eqns, use_named=True,
                                                  named_keys=named_keys)

    @property
    def number_state_equations(self) -> int:
        """Integer number of state equations in the phase.

        Should be the same as the number of state variables, i.e. there should 
        be a direct mapping between the two.
        """
        return len(self._y_eqn_user)

    @property
    def path_constraints(self):
        return self._c_con_user

    @path_constraints.setter
    def path_constraints(self, c_cons):
        self._c_con_user = format_as_named_tuple(c_cons, use_named=False)

    @property
    def number_path_constraints(self):
        return len(self._c_con_user)

    @property
    def integrand_functions(self):
        return self._q_fnc_user

    @integrand_functions.setter
    def integrand_functions(self, integrands):
        self._q_fnc_user = format_as_named_tuple(integrands, use_named=False)
        self._q_var_user = tuple(sym.Symbol(f'q{i_q}_P{self._phase_suffix}')
                                  for i_q, _ in enumerate(self._q_fnc_user))

    @property
    def number_integrand_functions(self):
        return len(self._q_fnc_user)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = PhaseBounds(phase=self)
        else:
            self._bounds = bounds

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if scaling is None:
            self._scaling = PhaseScaling(phase=self)
        else:
            self._scaling = scaling

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if mesh is None:
            self._mesh = PhaseMesh(phase=self)
        else:
            self._mesh = mesh

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, guess):
        if guess is None:
            self._guess = PhaseGuess(phase=self)
        else:
            self._guess = guess

    def _check_variables_and_equations(self):
        """Check the user-supplied variables and equations for this OCP phase.

        Steps involved are:
            * Ensure that the same number of state variables and state
              equations are supplied.
            * Ensure that the symbols related to the state equations are the
              same set as the set of state variables.
            * If the two sets are not the same then issue an informative
              warning to the user describing what is wrong about the supplied
              state variables and state equations.

        Raises
        ------
        ValueError
            If the state variables and symbols associated with the state
            equations are not the same.

        """
        try:
            set_state_variables_keys = set(self.state_variables._fields)
        except AttributeError:
            set_state_variables_keys = set()

        try:
            set_state_equations_keys = set(self.state_equations._fields)
        except AttributeError:
            set_state_equations_keys = set()

        if set_state_variables_keys != set_state_equations_keys:

            intersection = set_state_variables_keys.intersection(
                set_state_equations_keys)
            spacer = "', '"
            msg_other = []

            if len(set_state_variables_keys) != len(set_state_equations_keys):
                if len(set_state_variables_keys) == 1:
                    msg_vars_len = (f"{len(set_state_variables_keys)} state "
                                    f"variable is")
                else:
                    msg_vars_len = (f"{len(set_state_variables_keys)} state "
                                    f"variables are")
                if len(set_state_equations_keys) == 1:
                    msg_eqns_len = (f"{len(set_state_equations_keys)} state "
                                    f"equation is")
                else:
                    msg_eqns_len = (f"{len(set_state_equations_keys)} state "
                                    f"equations are")
                msg_len = (f"{msg_vars_len} defined while {msg_eqns_len} "
                           f"supplied")
                msg_other.append(msg_len)

            only_in_variables = set_state_variables_keys.difference(
                intersection)
            if only_in_variables:
                immutable_only_in_variables = list(only_in_variables)
                if len(only_in_variables) == 1:
                    msg_vars = (f"the state variable "
                                f"'{immutable_only_in_variables[0]}' is defined "
                                f"without a state equation")
                else:
                    msg_vars = (f"the state variables "
                                f"'{spacer.join(immutable_only_in_variables[:-1])}' "
                                f"and '{immutable_only_in_variables[-1]}' are defined "
                                f"without state equations")
                msg_other.append(msg_vars)

            only_in_equations = set_state_equations_keys.difference(
                intersection)
            if only_in_equations:
                if len(only_in_equations) == 1:
                    immutable_only_in_equations = list(only_in_equations)
                    msg_eqns = (f"a state derivative is supplied for "
                                f"'{immutable_only_in_equations[0]}' which is not a "
                                f"state variable")
                else:
                    msg_eqns = (f"state derivatives are supplied for "
                                f"'{spacer.join(immutable_only_in_equations[:-1])}' "
                                f"and '{immutable_only_in_equations[-1]}' which are "
                                f"not defined as state variables")
                msg_other.append(msg_eqns)

            msg = ("A state equation must be supplied for each state variable "
                   f"in each phase. Currently in phase '{self.name}'")
            if len(msg_other) == 1:
                full_msg = (f"{msg}, {msg_other[0]}.")
            else:
                full_msg = (f"{msg}: {'; '.join(msg_other[:-1])}; and "
                            f"{msg_other[-1]}.")
            raise ValueError(full_msg)

    def __str__(self):
        string = (f"Phase {self.phase_number} of {self.optimal_control_problem}")
        return string

    def __repr__(self):
        string = (f"Phase({repr(self.optimal_control_problem)}, "
                  f"phase_number={self.phase_number})")
        return string
