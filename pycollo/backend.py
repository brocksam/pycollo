"""

Attributes
----------
CASADI : str
    Constant keyword string identifier for the Pycollo-CasADi backend.
PYCOLLO : str
    Constant keyword string identifier for the Pycollo-only (hSAD) backend.
SYMPY : str
    Constant keyword string identifier for the Pycollo-Sympy backend.
BACKENDS : :py:class:`Options <pyproprop>`
    The default backend to be used (via its constant keyword string
    identifier).

"""


import itertools
from abc import ABC, abstractmethod
from collections import namedtuple
from timeit import default_timer as timer

import casadi as ca
import numpy as np
import scipy.sparse as sparse
import sympy as sym
from pyproprop import Options, processed_property

from .bounds import Bounds
from .compiled import CompiledFunctions
from .expression_graph import ExpressionGraph
from .guess import Guess
from .iteration import Iteration
from .mesh import Mesh
from .quadrature import Quadrature
from .scaling import (Scaling,
                      CasadiIterationScaling,
                      HsadIterationScaling,
                      PycolloIterationScaling,
                      SympyIterationScaling,
                      )
from .solution import CasadiSolution, NlpResult
from .utils import (casadi_substitute,
                    console_out,
                    dict_merge,
                    fast_sympify,
                    format_multiple_items_for_output,
                    needed_to_tuple,
                    SUPPORTED_ITER_TYPES,
                    symbol_name,
                    symbol_primitives,
                    sympy_to_casadi,
                    )


__all__ = []


CASADI = "casadi"
HSAD = "hsad"
PYCOLLO = "pycollo"
SYMPY = "sympy"


class BackendABC(ABC):
    """Abstract base class for backends"""

    _MAX_AUX_DATA_SUBSTITUTION_DEPTH = 100

    ocp = processed_property("ocp", read_only=True)

    def __init__(self, ocp):
        self.ocp = ocp
        self.create_aux_data_containers()
        self.create_point_variable_symbols()
        self.create_phase_backends()
        self.preprocess_user_problem_aux_data()
        self.preprocess_phase_backends()
        self.preprocess_problem_backend()
        self.console_out_variables_constraints_preprocessed()

    @staticmethod
    @abstractmethod
    def sym(name):
        """Return an instantiated symbol with a name/identifier.

        This method must be overriden and should return a symbol primitive of
        type associated with the specific backend and with its name/identifier
        set to be the :arg:`name` argument.

        Args
        ----
        name : str
            The str which should be attached to the new symbol as its name to
            identify it. How exactly this is attached to the new symbol will
            vary depending on which dependent package actually implements the
            symbol.

        """
        pass

    @staticmethod
    @abstractmethod
    def const(val):
        """Convert a value to a constant of correct type for specific backend.

        Args
        ----
        val : float
            The numerical value to be converted in type.

        """
        pass

    @abstractmethod
    def substitute_pycollo_sym(self, expr):
        """Substitute an expression for backend Pycollo symbols.

        Returns
        -------
        Union[ca.SX, sym.Expr, pycollo.Expression]
            The substituted expression.

        """
        pass

    @staticmethod
    @abstractmethod
    def iteration_scaling(*args, **kwargs):
        """Instantiate an IterationScaling object specific to the backend type.

        Returns
        -------
        IterationScaling
            Of the specific backend type.

        """
        pass

    @abstractmethod
    def generate_nlp_function_callables(self, iteration):
        """Create iteration-specific OCP callables.

        Needs to be overriden as different backends will require different NLP
        functions to be compiled to callables.

        """
        pass

    def create_aux_data_containers(self):
        """Instantiate containers for mapping symbols to expressions.

        Two types of container are made: :attr:`user_to_backend_mapping` holds
        a mapping from user-defined symbols to the corresponding symbols
        automatically-created by the backend; and :attr:`aux_data` holds a
        mapping from the backend symbols to the backend equations defining the
        symbol in question.

        """
        self.user_to_backend_mapping = {}
        self.user_aux_data_for_preprocessing = {}
        self.user_aux_data_phase_dependent = {}
        self.user_aux_data_phase_independent = {}
        self.user_aux_data_supplied_in_ocp_and_phase = {}
        self.aux_data = {}

    def create_point_variable_symbols(self):
        """Abstraction layer for creating problem-level variable symbols."""
        self.create_parameter_variable_symbols()

    def create_parameter_variable_symbols(self):
        """Create static parameter variables symbols and associated data.

        The user supplies symbols to the OCP that are unscaled (as these
        symbols crop up in the OCP equations etc.). Pycollo automatically
        handles problem scaling by introducing a scaling for every variable
        when introduced. This is done by shifting and stretching that
        variable's domain on to the window [-0.5, 0.5] (which is handled by the
        :mod:`scaling` module within Pycollo). The Pycollo backend deals with
        the scaled version of the symbols, or the "tilde" symbols. The scaled
        symbols are moved to the original domain first by stretching them by
        the "stretch" factor `V`, then by shifting them by the "shift" factor
        `r`. To move from the scaled (tilde) variable to the unscaled
        (user-defined) variable the following transformation is used:

        ..math::
            x = V * x_tilde + r

        """
        self.s_var_user = tuple(self.ocp._s_var_user)
        self.num_s_var_full = len(self.s_var_user)
        s = tuple(self.sym(symbol_name(var)) for var in self.s_var_user)
        self.add_user_to_backend_mapping(self.s_var_user, s)
        s_tilde = (self.sym(f"_s{i_s}") for i_s in range(self.num_s_var_full))
        self.s_var_full = tuple(s_tilde)
        V_s, r_s = self.create_variable_scaling_symbols("s",
                                                        self.num_s_var_full)
        self.V_s_var_full = V_s
        self.r_s_var_full = r_s
        s_exprs = self.create_variable_scaling_expression(self.s_var_full,
                                                          self.V_s_var_full,
                                                          self.r_s_var_full)
        self.add_aux_data_mapping(s, s_exprs)

    def add_user_to_backend_mapping(self, user_sym, backend_sym):
        """Add mapping of (iterable) of user symbols to backend symbols.

        Args
        ----
        user_sym : Tuple[Union[sym.Symbol, ca.SX, pycollo.Symbol]]
            Iterable of user symbols.
        backend_sym : Tuple[Union[sym.Symbol, ca.SX, pycollo.Symbol]]
            Iterable of Pycollo backend symbols.

        """
        self.user_to_backend_mapping.update(dict(zip(user_sym, backend_sym)))

    def add_aux_data_mapping(self, backend_sym, backend_expr):
        """Add mapping of (iterable) of backend symbols to backend expressions.

        Args
        ----
        backend_sym : Tuple[Union[sym.Symbol, ca.SX, pycollo.Symbol]]
            Iterable of Pycollo backend symbols.
        backend_expr : Tuple[Union[sym.Expr, ca.SX, pycollo.Expression]]
            Iterable of Pycollo backend expressions.

        """
        self.aux_data.update(dict(zip(backend_sym, backend_expr)))

    def create_variable_scaling_symbols(self,
                                        identifier,
                                        var_ids,
                                        phase_num=None):
        """Instantiate stretch and shift variables as backend symbols.

        Args
        ----
        identifier : str
            Character shorthand for the type of variable, e.g. "y" for state.
        num_var_full : int
            The number of variables in that category. Corresponds to the number
            of stretching/shifting symbols that need to be instantiated.
        phase_num : int, optional
            Default `None` corresponds to endpoint (i.e. parameter variables).
            Otherwise the integer ("x") is appended as "_Px" to the symbols.

        Returns
        -------
        Tuple[Tuple[Union[sym.Symbol, ca.SX]], Tuple[Union[sym.Symbol, ca.SX]]]
            The stretching (V_x) and shifting (r_x) variables.

        """
        if not isinstance(var_ids, tuple):
            var_ids = tuple(range(var_ids))
        phase_suffix = "" if phase_num is None else f"_P{phase_num}"
        V_x = (self.sym(f"_V_{identifier}{i}{phase_suffix}")
               for i in var_ids)
        r_x = (self.sym(f"_r_{identifier}{i}{phase_suffix}")
               for i in var_ids)
        return tuple(V_x), tuple(r_x)

    @staticmethod
    def create_variable_scaling_expression(x_tildes, Vs, rs):
        """Create scaling expressions for tilde variable to unscaled variable.

        The scaling expression is the stretch/shift expression given in Betts,
        2010.

        Args
        ----
        x_tildes : Tuple[Union[ca.SX]]
            The backend scaled variable symbols.
        Vs : Tuple[Union[ca.SX]]
            The backend stretching parameters.
        rs : Tuple[Union[ca.SX]]
            The backend shifting parameters.

        """
        return tuple(V * x_tilde + r
                     for x_tilde, V, r in zip(x_tildes, Vs, rs))

    def create_phase_backends(self):
        """Create and initialise a phase backend for each OCP phase.

        Phase backend initialisation involes setting the number of phases in
        the OCP for the backend and collating all user variables, both phase
        and non-phase, in to sets.

        """
        p = (PycolloPhaseData(self, phase) for phase in self.ocp.phases)
        self.p = tuple(p)
        self.num_phases = len(self.p)
        self.all_phase_user_var = {var
                                   for p in self.p
                                   for var in p.all_user_var}
        self.all_user_var = self.all_phase_user_var.union(set(self.s_var_user))
        self.all_phase_var = {var for p in self.p for var in p.all_var}
        all_endpoint_var = itertools.chain(self.s_var_full,
                                           self.V_s_var_full,
                                           self.r_s_var_full)
        self.all_var = self.all_phase_var.union(all_endpoint_var)

    def preprocess_user_problem_aux_data(self):
        """Abstraction layer for preprocessing of all problem aux data."""
        self.partition_all_user_problem_phase_aux_data()
        self.preprocess_user_phase_and_problem_aux_data()
        self.partition_user_problem_aux_data()
        self.preprocess_user_phase_independent_aux_data()

    def partition_all_user_problem_phase_aux_data(self):
        """Determine category of all user-supplied aux data"""
        self.user_phase_aux_data_syms = {symbol
                                         for p in self.ocp.phases
                                         for symbol in p.auxiliary_data}
        for symbol, equation in self.ocp.auxiliary_data.items():
            self.partition_user_problem_phase_aux_data(symbol, equation)

    def partition_user_problem_phase_aux_data(self, user_sym, user_eqn):
        """Separate aux data into phase-independent/phase-dependent/constant.

        User-supplied auxiliary data be one of three types: (A) constants,
        (B) phase-independent, and (C) phase-dependent. Constants and phase-
        independent auxiliary data can be handled trivially as the associated
        symbols can be used in all phases. Phase-dependent auxiliary data is
        more complicated as every phase-dependent symbol must have a unique
        symbol for each phase that it appears in. The method processes the
        user-supplied auxiliary data to determine what further processing it
        requires.

        Firstly, the user equation is transformed to a Sympy expression so that
        types are uniform across all processing. Secondly, if the user symbol
        is in a phase auxiliary data then it is added to the
        `user_aux_data_supplied_in_ocp_and_phase` mapping for processing within
        that category. Thirdly, if any of the primitives in the user equation
        are in the phase auxiliary data then the user symbol and user equation
        pair are added to the `user_aux_data_phase_dependent` mapping for
        processing as phase-dependent auxiliary data. Finally if none of these
        categories are met then the user symbol and user equation auxiliary
        data are added to the `user_aux_data_for_processing` mapping.

        Args
        ----
        user_sym : Union[sympy.Symbol, ca.SX, pycollo.Symbol]
            The key in the user-supplied auxiliary data mappping.
        user_eqn : Union[sympy.Expr, ca.SX, pycollo.Expression]
            The value in the user-supplied auxiliary data mapping.

        """
        user_eqn = fast_sympify(user_eqn)
        user_eqn_prims = symbol_primitives(user_eqn)
        if user_sym in self.user_phase_aux_data_syms:
            self.user_aux_data_supplied_in_ocp_and_phase[user_sym] = user_eqn
        elif self.user_phase_aux_data_syms.intersection(user_eqn_prims):
            self.user_aux_data_phase_dependent[user_sym] = user_eqn
        elif not user_eqn_prims:
            self.user_aux_data_phase_independent[user_sym] = user_eqn
        else:
            self.user_aux_data_for_preprocessing[user_sym] = user_eqn

    def preprocess_user_phase_and_problem_aux_data(self):
        """Alert user of any problem- and phase-level aux data definitions.

        Aux data can only be supplied at either the problem-level or the
        phase-level. If a user has supplied an auxiliary equation mapped to an
        auxiliary symbol in both the :class:`OptimalControlProblem` object and
        a :class:`Phase` object then a `ValueError` should be raised.

        Raises
        ------
        ValueError
            If it has been explicitly specified not to allow auxiliary
            equations for the same symbol at both the phase and problem level
            but the user has done so.

        """
        self.allow_aux_data_in_phase_and_problem = True
        condition_1 = not self.allow_aux_data_in_phase_and_problem
        condition_2 = self.user_aux_data_supplied_in_ocp_and_phase
        if condition_1 and condition_2:
            formatted_syms = format_multiple_items_for_output(
                self.aux_data_supplied_in_ocp_and_phase)
            msg = (f"Auxiliary data for {formatted_syms} has been supplied at "
                    f"a per-phase level and therefore cannot be supplied at a "
                    f"problem level.")
            raise ValueError(msg)

    def partition_user_problem_aux_data(self):
        """Iterate over all user-supplied aux-data for processing."""
        for symbol, equation in self.user_aux_data_for_preprocessing.items():
            self.process_aux_data_pair_is_phase_dependent(symbol, equation)

    def process_aux_data_pair_is_phase_dependent(self, symbol, equation):
        """Check if an aux data pair is phase dependent.

        An aux data pair is phase dependent if any of the primitives of its
        equation are themselves phase dependent. Conversely, an aux data symbol
        is trivially phase independent if: (1) its equation directly maps to an
        endpoint symbol, or (2) it is a constant. Otherwise, the equation
        primitives must be found and checked recursively.

        Returns
        -------
        bool
            `True` if the aux data pair supplied in the args is phase-dependent
            or `False` if it is phase-independent.

        """
        if symbol is equation:
            if symbol in self.all_phase_user_var:
                return True
            elif symbol in self.s_var_user:
                return False
            else:
                msg = (f"The non-root symbol '{symbol}' has been mapped to "
                       f"itself and auxiliary data cannot be rectified.")
                raise ValueError(msg)
        elif symbol in self.user_aux_data_phase_dependent:
            return True
        elif equation in self.all_phase_user_var:
            self.new_aux_data_pair_phase_dependent(symbol, equation)
            return True
        elif equation in self.s_var_user:
            return False
        elif equation.is_Number:
            self.new_aux_data_pair_phase_independent(symbol, equation)
            return False
        else:
            return self.check_aux_data_pair_children(symbol, equation)

    def new_aux_data_pair_phase_dependent(self, symbol, equation):
        """Mark an aux data pair as phase-dependent.

        Args
        ----
        symbol : Union[sym.Symbol, ca.SX, pycollo.Symbol]
            The symbol in the aux data pair.
        equation : Union[sym.Expr, ca.SX, pycollo.Expression]
            The equation in the aux data pair.

        """
        self.user_aux_data_phase_dependent[symbol] = equation

    def new_aux_data_pair_phase_independent(self, symbol, equation):
        """Mark an aux data pair as phase-independent.

        Args
        ----
        symbol : Union[sym.Symbol, ca.SX, pycollo.Symbol]
            The symbol in the aux data pair.
        equation : Union[sym.Expr, ca.SX, pycollo.Expression]
            The equation in the aux data pair.

        """
        self.user_aux_data_phase_independent[symbol] = equation

    def check_aux_data_pair_children(self, symbol, equation):
        """Discern phase dependency status by analysing equation primitives.

        Look at the primitives (child symbols) and their associated auxiliary
        equations. If any of them are phase-dependent then the aux data pair
        in question is itself phase dependent. Otherwise the aux data pair is
        phase independent.

        This method will recursively traverse the tree of mappings that
        constitute an auxiliary equation until either a root that has already
        be determined to be either phase-dependent or phase-independent is
        encountered or until an error is raised (because Pycollo) has not been
        able to traverse through the tree of the auxiliary equation to a point
        where the status can be determined.

        Args
        ----
        symbol : Union[sym.Symbol, ca.SX, pycollo.Symbol]
            The symbol in the aux data pair.
        equation : Union[sym.Expr, ca.SX, pycollo.Expression]
            The equation in the aux data pair.

        Returns
        -------
        bool
            `True` if the aux data pair supplied in the args is phase-dependent
            or `False` if it is phase-independent.

        """
        child_syms = list(symbol_primitives(equation))
        child_eqns = [self.get_child_equation(child_sym)
                      for child_sym in child_syms]
        child_is_phase_dependent = [
            self.process_aux_data_pair_is_phase_dependent(child_sym, child_eqn)
            for child_sym, child_eqn in zip(child_syms, child_eqns)]
        if any(child_is_phase_dependent):
            self.new_aux_data_pair_phase_dependent(symbol, equation)
            return True
        else:
            self.new_aux_data_pair_phase_independent(symbol, equation)
            return False

    def get_child_equation(self, symbol):
        """Get an auxiliary equation from an auxiliary symbol.

        At this point in the auxiliary data processing, auxiliary data can
        either be held in :attr:`user_aux_data_for_processing`,
        :attr:`user_aux_data_phase_dependent` or
        :attr:`user_aux_data_phase_independent`.

        Args
        ----
        symbol : Union[sym.Symbol, ca.SX, pycollo.Symbol]
            The symbol in the aux data pair.

        Returns
        -------
        Union[sym.Symbol, sym.Expr, ca.SX]
            A symbol or expression corresponding to the auxiliary mapping.

        Raises
        ------
        ValueError
            If the symbol is not defined. Probably because the user has not
            supplied sufficient auxiliary information for the problem.

        """
        all_sym_eqn_mapping = dict_merge(self.user_aux_data_for_preprocessing,
                                         self.user_aux_data_phase_dependent,
                                         self.user_aux_data_phase_independent)
        equation = all_sym_eqn_mapping.get(symbol)
        if symbol in self.all_user_var:
            return symbol
        elif equation is None:
            msg = (f"'{symbol}' is not defined.")
            raise ValueError(msg)
        return equation

    def preprocess_user_phase_independent_aux_data(self):
        """Add phase-independent user aux data to Pycollo aux data."""
        for user_sym, user_eqn in self.user_aux_data_phase_independent.items():
            if user_sym not in self.user_to_backend_mapping:
                backend_sym = self.sym(symbol_name(user_sym))
                self.add_user_to_backend_mapping((user_sym, ), (backend_sym, ))
            if isinstance(user_eqn, sym.Expr):
                backend_eqn, self.user_to_backend_mapping = sympy_to_casadi(
                    user_eqn, self.user_to_backend_mapping)
            else:
                msg = (f"Cannot process independent auxiliary data equation "
                       f"of type '{type(user_eqn)}'")
                raise TypeError(msg)
            self.add_aux_data_mapping((backend_sym, ), (backend_eqn, ))

    def preprocess_phase_backends(self):
        """Abstraction layer for preprocessing phase backends."""
        for p in self.p:
            p.preprocess_auxiliary_data()
        self.collect_variables_substitutions()
        for p in self.p:
            p.preprocess_constraints()

    def collect_variables_substitutions(self):
        """Substitute aux data equations to only contain backend symbols.

        Backend symbols include: (1) backend variable symbols, and (2) scaling
        symbols for backend variables in the for `V_x` ("stretch" symbols) and
        `r_s` ("shift" symbols).

        Raises
        ------
        ValueError
            If an auxiliary equation cannot be converted to a form to only
            contain symbols of types (1) and (2). This is indicative to the
            user that they have supplied insufficient auxiliary data/there is
            an error in their auxiliary data.

        """
        point_sym = set(self.s_var_full)
        V_point_sym = set(self.V_s_var_full)
        r_point_sym = set(self.r_s_var_full)
        phase_sym = set.union(*[set(p.x_var_full) for p in self.p])
        phase_V_sym = set.union(*[set(itertools.chain(p.V_y_var_full,
                                                      p.V_u_var_full,
                                                      p.V_q_var_full))
                                  for p in self.p])
        phase_r_sym = set.union(*[set(itertools.chain(p.r_y_var_full,
                                                      p.r_u_var_full,
                                                      p.r_q_var_full))
                                  for p in self.p])
        phase_point_sym = set.union(*[set(p.x_point_var_full) for p in self.p])
        all_backend_sym = set.union(point_sym,
                                    V_point_sym,
                                    r_point_sym,
                                    phase_sym,
                                    phase_V_sym,
                                    phase_r_sym,
                                    phase_point_sym)
        for backend_sym in self.aux_data.copy().keys():
            backend_eqn = self.aux_data[backend_sym]
            before_prims = set(symbol_primitives(backend_eqn))
            for _ in range(self._MAX_AUX_DATA_SUBSTITUTION_DEPTH):
                backend_eqn = casadi_substitute(backend_eqn, self.aux_data)
                after_prims = set(symbol_primitives(backend_eqn))
                if before_prims == after_prims:
                    diff = after_prims.difference(self.all_var)
                    if diff:
                        msg = (f"Cannot rectify aux data for symbol "
                               f"{backend_sym} with equation "
                               f"{self.aux_data[backend_sym]} as it contains "
                               f"the non-root symbols: {diff}.")
                        raise ValueError(msg)
                    break
                self.aux_data[backend_sym] = backend_eqn
                before_prims = after_prims

    def preprocess_problem_backend(self):
        """Abstraction layer for backend-specific postprocessing."""
        self.process_objective_function()

    def process_objective_function(self):
        """Substitute objetive function with backend symbols."""
        self.J = self.substitute_pycollo_sym(self.ocp.objective_function)

    def console_out_variables_constraints_preprocessed(self):
        """Console out success message variable/constraint preprocessing."""
        if self.ocp.settings.console_out_progress:
            msg = "Pycollo variables and constraints preprocessed."
            console_out(msg)

    def create_bounds(self):
        """Create bounds object for backend."""
        self.bounds = Bounds(self)
        self.recollect_variables()
        self.process_point_constraints()
        self.collect_constraints()

    def recollect_variables(self):
        """Take in to account variables that Pycollo's determined constant."""
        chain_from_iterable = itertools.chain.from_iterable

        for p in self.p:
            p.collect_pycollo_variables()
            p.create_variable_indexes_slices()

        self.s_var = needed_to_tuple(self.s_var_full,
                                     self.ocp.bounds._s_needed)
        self.num_s_var = len(self.s_var)

        self.V_s_var = needed_to_tuple(self.V_s_var_full,
                                       self.ocp.bounds._s_needed)
        self.r_s_var = needed_to_tuple(self.r_s_var_full,
                                       self.ocp.bounds._s_needed)

        self.V_x_var = tuple(itertools.chain(*list(p.V_x_var for p in self.p),
                                             self.V_s_var))
        self.r_x_var = tuple(itertools.chain(*list(p.r_x_var for p in self.p),
                                             self.r_s_var))

        all_phase_var = chain_from_iterable(p.x_var for p in self.p)
        continuous_var = tuple(all_phase_var) + self.s_var
        self.x_var = continuous_var
        self.num_var = len(continuous_var)
        all_point_var = chain_from_iterable(p.x_point_var for p in self.p)
        endpoint_var = tuple(all_point_var) + self.s_var
        self.x_point_var = endpoint_var
        self.num_point_var = len(endpoint_var)
        self.variables = (continuous_var, endpoint_var)

        self.phase_y_var_slices = []
        self.phase_u_var_slices = []
        self.phase_q_var_slices = []
        self.phase_t_var_slices = []
        self.phase_variable_slices = []
        phase_start = 0
        for p in self.p:
            start = phase_start
            stop = start + p.num_y_var
            p_slice = slice(start, stop)
            self.phase_y_var_slices.append(p_slice)
            start = stop
            stop = start + p.num_u_var
            p_slice = slice(start, stop)
            self.phase_u_var_slices.append(p_slice)
            start = stop
            stop = start + p.num_q_var
            p_slice = slice(start, stop)
            self.phase_q_var_slices.append(p_slice)
            start = stop
            stop = start + p.num_t_var
            p_slice = slice(start, stop)
            self.phase_t_var_slices.append(p_slice)
            start = stop
            phase_stop = phase_start + p.num_var
            p_slice = slice(phase_start, phase_stop)
            self.phase_variable_slices.append(p_slice)
            phase_start = phase_stop
        self.s_var_slice = slice(
            self.num_var - self.num_s_var, self.num_var)
        self.variable_slice = self.s_var_slice

        self.phase_endpoint_variable_slices = []
        start = 0
        for p in self.p:
            stop = start + p.num_point_var
            p_slice = slice(start, stop)
            start = stop
            self.phase_endpoint_variable_slices.append(p_slice)
        self.endpoint_variable_slice = slice(
            self.num_point_var - self.num_s_var, self.num_point_var)

    def process_point_constraints(self):
        """Process user-supplied point constraints."""
        all_y_var = []
        all_y_t0_var = []
        all_y_tF_var = []
        for p in self.p:
            all_y_var.extend(list(p.y_var))
            all_y_t0_var.extend(list(p.y_t0_var))
            all_y_tF_var.extend(list(p.y_tF_var))
        all_y_var_set = set(all_y_var)
        all_y_bnd = []
        for var, x_bnd in zip(self.x_var, self.bounds.x_bnd):
            if var in all_y_var_set:
                all_y_bnd.append(x_bnd)
        endpoint_state_constraints = []
        endpoint_state_constraints_bounds = []
        zipped = zip(all_y_t0_var,
                     all_y_tF_var,
                     all_y_bnd,
                     self.bounds.y_t0_bnd,
                     self.bounds.y_tF_bnd)
        for y_t0_var, y_tF_var, y_bnd, y_t0_bnd, y_tF_bnd in zipped:
            if np.any(~np.isclose(np.array(y_t0_bnd), np.array(y_bnd))):
                endpoint_state_constraints.append(y_t0_var)
                endpoint_state_constraints_bounds.append(y_t0_bnd)
            if np.any(~np.isclose(np.array(y_tF_bnd), np.array(y_bnd))):
                endpoint_state_constraints.append(y_tF_var)
                endpoint_state_constraints_bounds.append(y_tF_bnd)
        self.y_con = tuple(endpoint_state_constraints)
        self.num_y_con = len(self.y_con)
        self.bounds.c_y_bnd = endpoint_state_constraints_bounds
        b_cons = []
        for b_con in self.ocp.endpoint_constraints:
            b_con = self.replace_phase_dependent_point_constraint(b_con)
            b_con = self.substitute_pycollo_sym(b_con)
            b_con = self.check_point_constraint_primitives(b_con)
            b_cons.append(b_con)
        self.b_con = tuple(b_cons)
        self.num_b_con = len(self.b_con)

    def replace_phase_dependent_point_constraint(self, b_con):
        """Replace ambiguous point symbols in point constraints."""
        prims = symbol_primitives(b_con)
        ambiguous_syms = prims.intersection(self.user_aux_data_phase_dependent)
        if ambiguous_syms:
            if len(self.p) != 1:
                msg = (f"Ambiguous point constraint '{b_con}' supplied "
                       f"which contains phase-dependent symbols and "
                       f"cannot be rectified by Pycollo. Please rewrite "
                       f"using phase-specific symbols.")
                raise ValueError(msg)
            all_user_to_backend_mapping = dict_merge(
                self.user_to_backend_mapping,
                self.p[0].phase_user_to_backend_mapping)
            b_con, _ = sympy_to_casadi(b_con, all_user_to_backend_mapping,
                                       phase=self.p[0].i)
        return b_con

    def check_point_constraint_primitives(self, b_con):
        """Ensure point constraints aren't being used for state endpoints."""
        if b_con in set(self.x_point_var):
            msg = (f"Pycollo cannot automatically transform point constraints "
                   f"to state endpoint constraints. Use state endpoint "
                   f"constraints for '{b_con}'.")
            raise ValueError(msg)
        prims = set(symbol_primitives(b_con))
        allowed_syms = itertools.chain(self.x_point_var,
                                       self.V_x_var,
                                       self.r_x_var)
        if prims.difference(set(allowed_syms)):
            msg = (f"Endpoint constraint {b_con} is invalid as it contains "
                   f"symbols that aren't OCP point variables or constants.")
            raise ValueError(msg)
        return b_con

    def collect_constraints(self):
        """Collect phase and problem constraints and related data."""
        phase_c = itertools.chain.from_iterable(p.c for p in self.p)
        self.c = (tuple(phase_c) + self.b_con)
        self.num_c = len(self.c)

        self.phase_y_eqn_slices = []
        self.phase_p_con_slices = []
        self.phase_q_fnc_slices = []
        self.phase_c_slices = []
        phase_start = 0
        for p in self.p:
            start = phase_start
            stop = start + p.num_y_eqn
            p_slice = slice(start, stop)
            self.phase_y_eqn_slices.append(p_slice)
            start = stop
            stop = start + p.num_p_con
            p_slice = slice(start, stop)
            self.phase_p_con_slices.append(p_slice)
            start = stop
            stop = start + p.num_q_fnc
            p_slice = slice(start, stop)
            self.phase_q_fnc_slices.append(p_slice)
            start = stop
            phase_stop = phase_start + p.num_c
            p_slice = slice(phase_start, phase_stop)
            self.phase_c_slices.append(p_slice)
            phase_start = phase_stop

        start = 0
        stop = self.phase_c_slices[-1].stop
        self.c_continuous_slice = slice(start, stop)
        start = stop
        stop = start + self.num_b_con
        self.c_endpoint_slice = slice(start, stop)

    def create_guess(self):
        phase_guesses = [p.ocp_phase.guess for p in self.p]
        endpoint_guess = self.ocp.guess
        self.initial_guess = Guess(self, phase_guesses, endpoint_guess)

    def create_initial_mesh(self):
        phase_meshes = [p.ocp_phase.mesh for p in self.p]
        self.initial_mesh = Mesh(self, phase_meshes)

    def create_mesh_iterations(self):
        self.mesh_iterations = []
        _ = self.new_mesh_iteration(self.initial_mesh, self.initial_guess)

    def create_quadrature(self):
        self.quadrature = Quadrature(self)

    def create_scaling(self):
        self.scaling = Scaling(self)

    @abstractmethod
    def postprocess_problem_backend(self):
        """Abstraction layer for backend-specific postprocessing."""
        pass

    def new_mesh_iteration(self, mesh, guess):
        index = len(self.mesh_iterations)
        new_iteration = Iteration(
            backend=self,
            index=index,
            mesh=mesh,
            guess=guess,
        )
        self.mesh_iterations.append(new_iteration)
        return new_iteration


class PycolloPhaseData:

    def __init__(self, ocp_backend, ocp_phase):

        self.ocp_backend = ocp_backend
        self.ocp_phase = ocp_phase
        self.i = ocp_phase.phase_number
        self.create_aux_data_containers()
        self.create_variable_symbols()
        self.preprocess_variables()
        self.create_full_variable_indexes_slices()

    @property
    def ocp(self):
        """Utility property to hand-off collecting OCP to OCP backend."""
        return self.ocp_backend.ocp

    def sym(self, *args, **kwargs):
        """Handoff method for symbol creation.

        Symbol creation is handled by the OCP backend so phase backends pass
        these requests off to their :py:attr:`ocp_backend`.

        """
        return self.ocp_backend.sym(*args, **kwargs)

    def create_aux_data_containers(self):
        """Instantiate containers for mapping symbols to expressions."""
        self.phase_user_to_backend_mapping = {}

    def create_variable_symbols(self):
        """Abstraction layer converning creation of all phase OCP variables."""
        self.create_state_variable_symbols()
        self.create_control_variable_symbols()
        self.create_integral_variable_symbols()
        self.create_time_variable_symbols()

    def add_phase_user_to_backend_mapping(self, user_sym, phase_backend_sym):
        """Add mapping of (iterable) of user symbols to backend symbols.

        Args
        ----
        user_sym : Tuple[Union[sym.Symbol, ca.SX, pycollo.Symbol]]
            Iterable of user symbols.
        phase_backend_sym : Tuple[Union[sym.Symbol, ca.SX, pycollo.Symbol]]
            Iterable of Pycollo backend symbols.

        """
        self.phase_user_to_backend_mapping.update(dict(zip(user_sym,
                                                           phase_backend_sym)))

    def create_state_variable_symbols(self):
        """Instantiate phase state variables (including enpoint symbols)."""
        self.y_var_user = tuple(self.ocp_phase.state_variables)
        self.num_y_var_full = len(self.y_var_user)
        y = tuple(self.sym(f"{symbol_name(var)}_P{self.i}")
                  for var in self.y_var_user)
        self.add_phase_user_to_backend_mapping(self.y_var_user, y)
        y_tilde = (self.sym(f"_y{i_y}_P{self.i}")
                   for i_y in range(self.num_y_var_full))
        self.y_var_full = tuple(y_tilde)
        V_y, r_y = self.ocp_backend.create_variable_scaling_symbols(
            "y", self.num_y_var_full, self.i)
        self.V_y_var_full = V_y
        self.r_y_var_full = r_y
        y_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.y_var_full, self.V_y_var_full, self.r_y_var_full)
        self.ocp_backend.add_aux_data_mapping(y, y_exprs)

        self.y_t0_var_user = tuple(self.ocp_phase.initial_state_variables)
        self.y_tF_var_user = tuple(self.ocp_phase.final_state_variables)
        self.num_y_point_var_full = 2 * self.num_y_var_full
        y_t0 = tuple(self.sym(symbol_name(var)) for var in self.y_t0_var_user)
        y_tF = tuple(self.sym(symbol_name(var)) for var in self.y_tF_var_user)
        self.add_phase_user_to_backend_mapping(self.y_t0_var_user, y_t0)
        self.ocp_backend.add_user_to_backend_mapping(self.y_t0_var_user, y_t0)
        self.add_phase_user_to_backend_mapping(self.y_tF_var_user, y_tF)
        self.ocp_backend.add_user_to_backend_mapping(self.y_tF_var_user, y_tF)
        y_t0_tilde = (self.sym(f"_y{i_y}_t0_P{self.i}")
                      for i_y in range(self.num_y_var_full))
        y_tF_tilde = (self.sym(f"_y{i_y}_tF_P{self.i}")
                      for i_y in range(self.num_y_var_full))
        self.y_t0_var_full = tuple(y_t0_tilde)
        self.y_tF_var_full = tuple(y_tF_tilde)
        self.y_point_var_full = tuple(itertools.chain.from_iterable(
            y for y in zip(self.y_t0_var_full, self.y_tF_var_full)))
        y_t0_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.y_t0_var_full, self.V_y_var_full, self.r_y_var_full)
        y_tF_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.y_tF_var_full, self.V_y_var_full, self.r_y_var_full)
        self.ocp_backend.add_aux_data_mapping(y_t0, y_t0_exprs)
        self.ocp_backend.add_aux_data_mapping(y_tF, y_tF_exprs)

    def create_control_variable_symbols(self):
        """Instantiate phase control variable symbols."""
        self.u_var_user = tuple(self.ocp_phase.control_variables)
        self.num_u_var_full = len(self.u_var_user)
        u = tuple(self.sym(f"{symbol_name(var)}_P{self.i}")
                  for var in self.u_var_user)
        self.add_phase_user_to_backend_mapping(self.u_var_user, u)
        u_tilde = (self.sym(f"_u{i_u}_P{self.i}")
                   for i_u in range(self.num_u_var_full))
        self.u_var_full = tuple(u_tilde)
        V_u, r_u = self.ocp_backend.create_variable_scaling_symbols(
            "u", self.num_u_var_full, self.i)
        self.V_u_var_full = V_u
        self.r_u_var_full = r_u
        u_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.u_var_full, self.V_u_var_full, self.r_u_var_full)
        self.ocp_backend.add_aux_data_mapping(u, u_exprs)

    def create_integral_variable_symbols(self):
        """Instantiate phase integral variable symbols."""
        self.q_var_user = tuple(self.ocp_phase.integral_variables)
        self.num_q_var_full = len(self.q_var_user)
        q = tuple(self.sym(symbol_name(var)) for var in self.q_var_user)
        self.add_phase_user_to_backend_mapping(self.q_var_user, q)
        self.ocp_backend.add_user_to_backend_mapping(self.q_var_user, q)
        q_tilde = (self.sym(f"_q{i_q}_P{self.i}")
                   for i_q in range(self.num_q_var_full))
        self.q_var_full = tuple(q_tilde)
        V_q, r_q = self.ocp_backend.create_variable_scaling_symbols(
            "q", self.num_q_var_full, self.i)
        self.V_q_var_full = V_q
        self.r_q_var_full = r_q
        q_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.q_var_full, self.V_q_var_full, self.r_q_var_full)
        self.ocp_backend.add_aux_data_mapping(q, q_exprs)

    def create_time_variable_symbols(self):
        """Instantiate phase time variable symbols."""
        self.t_var_user = tuple(self.ocp_phase.time_variables)
        self.num_t_var_full = 2
        t = tuple(self.sym(symbol_name(var)) for var in self.t_var_user)
        self.add_phase_user_to_backend_mapping(self.t_var_user, t)
        self.ocp_backend.add_user_to_backend_mapping(self.t_var_user, t)
        t_tilde = (self.sym(f"_t0_P{self.i}"), self.sym(f"_tF_P{self.i}"))
        self.t_var_full = tuple(t_tilde)
        V_t, r_t = self.ocp_backend.create_variable_scaling_symbols(
            "t", ("0", "F"), self.i)
        self.V_t_var_full = V_t
        self.r_t_var_full = r_t
        t_exprs = self.ocp_backend.create_variable_scaling_expression(
            self.t_var_full, self.V_t_var_full, self.r_t_var_full)
        self.ocp_backend.add_aux_data_mapping(t, t_exprs)
        self.t_norm = 0.5 * (self.t_var_full[1] - self.t_var_full[0])

    def preprocess_variables(self):
        """Abstraction layer for preprocessing of Pycollo/user variables."""
        self.collect_pycollo_variables_full()
        self.collect_user_variables()

    def collect_pycollo_variables_full(self):
        """Collect Pycollo variables and associated data."""
        x_var_full = itertools.chain(self.y_var_full,
                                     self.u_var_full,
                                     self.q_var_full,
                                     self.t_var_full)
        self.x_var_full = tuple(x_var_full)
        V_x_var_full = itertools.chain(self.V_y_var_full,
                                       self.V_u_var_full,
                                       self.V_q_var_full,
                                       self.V_t_var_full)
        self.V_x_var_full = tuple(V_x_var_full)
        r_x_var_full = itertools.chain(self.r_y_var_full,
                                       self.r_u_var_full,
                                       self.r_q_var_full,
                                       self.r_t_var_full)
        self.r_x_var_full = tuple(r_x_var_full)
        self.num_var_full = len(self.x_var_full)
        self.num_each_var_full = (self.num_y_var_full,
                                  self.num_u_var_full,
                                  self.num_q_var_full,
                                  self.num_t_var_full)
        x_point_var_full = itertools.chain(self.y_point_var_full,
                                           self.q_var_full,
                                           self.t_var_full)
        self.x_point_var_full = tuple(x_point_var_full)
        self.num_point_var_full = len(self.x_point_var_full)
        self.all_var = set(itertools.chain(self.x_var_full,
                                           self.V_x_var_full,
                                           self.r_x_var_full,
                                           self.x_point_var_full))

    def collect_user_variables(self):
        """Collect user variables and associated data."""
        self.y_var_user = self.ocp_phase._y_var_user
        self.y_t0_var_user = self.ocp_phase._y_t0_user
        self.y_tF_var_user = self.ocp_phase._y_tF_user
        y_zip = zip(self.y_t0_var_user, self.y_tF_var_user)
        y_point_var_user = itertools.chain.from_iterable(y for y in y_zip)
        self.y_point_var_user = tuple(y_point_var_user)
        self.u_var_user = self.ocp_phase._u_var_user
        self.q_var_user = self.ocp_phase._q_var_user
        self.t_var_user = self.ocp_phase._t_var_user
        x_var_user = itertools.chain(self.y_var_user,
                                     self.u_var_user,
                                     self.q_var_user,
                                     self.t_var_user)
        self.x_var_user = tuple(x_var_user)
        x_point_var_user = itertools.chain(self.y_point_var_user,
                                           self.q_var_user,
                                           self.t_var_user)
        self.x_point_var_user = tuple(x_point_var_user)
        self.all_user_var = set(self.x_var_user + self.x_point_var_user)

    def create_full_variable_indexes_slices(self):
        """Abstraction layer for preprocessing of Pycollo/user slices."""
        self.create_full_continuous_variable_indexes_slices()
        self.create_full_point_variable_indexes_slices()

    def create_full_continuous_variable_indexes_slices(self):
        """Create slices and indices for the continuous Pycollo variables."""
        y_start = 0
        u_start = y_start + self.num_y_var_full
        q_start = u_start + self.num_u_var_full
        t_start = q_start + self.num_q_var_full
        x_end = self.num_var_full
        self.y_slice_full = slice(y_start, u_start)
        self.u_slice_full = slice(u_start, q_start)
        self.q_slice_full = slice(q_start, t_start)
        self.t_slice_full = slice(t_start, x_end)
        self.yu_slice_full = slice(y_start, q_start)
        self.qt_slice_full = slice(q_start, x_end)
        self.yu_qt_split_full = q_start

    def create_full_point_variable_indexes_slices(self):
        """Create slices and indices for the endpoint Pycollo variables."""
        y_point_start = 0
        q_point_start = y_point_start + self.num_y_point_var_full
        t_point_start = q_point_start + self.num_q_var_full
        x_point_end = self.num_point_var_full
        self.y_point_full_slice = slice(y_point_start, q_point_start)
        self.q_point_full_slice = slice(q_point_start, t_point_start)
        self.t_point_full_slice = slice(t_point_start, x_point_end)
        self.qt_point_full_slice = slice(q_point_start, x_point_end)
        self.y_point_qt_point_full_split = q_point_start

    def preprocess_auxiliary_data(self):
        """Abstraction layer for preprocessing of user-supplied aux data."""
        self.preprocess_user_phase_dependent_aux_data()
        self.check_all_user_phase_aux_data_supplied()
        self.collect_variables_substitutions()

    def preprocess_user_phase_dependent_aux_data(self):
        """Add phase-dependent user aux data to Pycollo aux data."""
        phase_aux_data = dict_merge(
            self.ocp_backend.user_aux_data_phase_dependent,
            self.ocp_phase.auxiliary_data)
        for user_sym, user_eqn in phase_aux_data.items():
            user_eqn = fast_sympify(user_eqn)
            if user_sym not in self.phase_user_to_backend_mapping:
                backend_sym = self.sym(f"{user_sym}_P{self.i}")
                self.phase_user_to_backend_mapping[user_sym] = backend_sym
            else:
                backend_sym = self.phase_user_to_backend_mapping[user_sym]
            if not isinstance(user_eqn, sym.Expr):
                msg = (f"Cannot process independent auxiliary data equation "
                       f"of type '{type(user_eqn)}'")
                raise TypeError(msg)
            all_user_to_backend_mapping = dict_merge(
                self.ocp_backend.user_to_backend_mapping,
                self.phase_user_to_backend_mapping)
            backend_eqn, all_user_to_backend_mapping = sympy_to_casadi(
                user_eqn, all_user_to_backend_mapping, phase=self.i)
            for k, v in all_user_to_backend_mapping.items():
                if k not in self.ocp_backend.user_to_backend_mapping:
                    self.phase_user_to_backend_mapping[k] = v
            self.ocp_backend.add_aux_data_mapping((backend_sym, ),
                                                  (backend_eqn, ))

    def check_all_user_phase_aux_data_supplied(self):
        """Determine any phase-dependent aux data not built for this phase."""
        missing_syms = self.ocp_backend.user_phase_aux_data_syms.difference(
            set(self.phase_user_to_backend_mapping.keys()))
        self.missing_phase_user_aux_data_syms = missing_syms

    def collect_variables_substitutions(self):
        """All user symbols mapped to their phase-specific backend symbol."""
        self.all_user_to_backend_mapping = dict_merge(
            self.ocp_backend.user_to_backend_mapping,
            self.phase_user_to_backend_mapping)

    def preprocess_constraints(self):
        """Abstraction layer for preprocessing of phase constraints."""
        self.preprocess_state_equations()
        self.preprocess_path_constraints()
        self.preprocess_integrand_functions()
        self.collect_constraints()
        self.check_all_needed_user_phase_aux_data_supplied()
        self.create_constraint_indexes_slices()

    def preprocess_state_equations(self):
        """Substitute state equations with backend symbols."""
        y_eqns = []
        for y_eqn in self.ocp_phase.state_equations:
            y_eqn = self.ocp_backend.substitute_pycollo_sym(y_eqn, self)
            y_eqns.append(y_eqn)
        self.y_eqn = tuple(y_eqns)
        self.num_y_eqn = self.ocp_phase.number_state_equations

    def preprocess_path_constraints(self):
        """Substitute path constraint equations with backend symbols."""
        p_cons = []
        for p_con in self.ocp_phase.path_constraints:
            p_con = self.ocp_backend.substitute_pycollo_sym(p_con, self)
            p_cons.append(p_con)
        self.p_con = tuple(p_cons)
        self.num_p_con = self.ocp_phase.number_path_constraints

    def preprocess_integrand_functions(self):
        """Substitute integrand functions with backend symbols."""
        q_fncs = []
        for q_fnc in self.ocp_phase.integrand_functions:
            q_fnc = self.ocp_backend.substitute_pycollo_sym(q_fnc, self)
            q_fncs.append(q_fnc)
        self.q_fnc = tuple(q_fncs)
        self.num_q_fnc = self.ocp_phase.number_integrand_functions

    def collect_constraints(self):
        """Collect phase constraint information together."""
        self.c = (self.y_eqn + self.p_con + self.q_fnc)
        self.num_c = sum([self.num_y_eqn, self.num_p_con, self.num_q_fnc])

    def check_all_needed_user_phase_aux_data_supplied(self):
        """Check if any phase-specific symbols are missing."""
        needed_syms = {s for c in self.c for s in symbol_primitives(c)}
        missing_syms = needed_syms.intersection(
            self.missing_phase_user_aux_data_syms)
        if missing_syms:
            self.raise_needed_user_phases_aux_data_missing_error(missing_syms)

    def raise_needed_user_phases_aux_data_missing_error(self, missing_syms):
        """Raise ValueError detailing missing phase-specific symbols.

        Raises
        ------
        ValueError
            Detailing missing phase-dependent symbols for the phase.

        """
        formatted_syms = format_multiple_items_for_output(missing_syms)
        msg = (f"Phase-dependent auxiliary data must be supplied for "
               f"all phase-dependent symbols in each phase. Please supply "
               f"auxiliary data for {formatted_syms} in {self.ocp_phase.name} "
               f"(phase index: {self.i}).")
        raise ValueError(msg)

    def create_constraint_indexes_slices(self):
        """Create slices for backend constraint components."""
        y_eqn_start = 0
        p_con_start = y_eqn_start + self.num_y_eqn
        q_fnc_start = p_con_start + self.num_p_con
        c_end = self.num_c
        self.y_eqn_slice = slice(y_eqn_start, p_con_start)
        self.p_con_slice = slice(p_con_start, q_fnc_start)
        self.q_fnc_slice = slice(q_fnc_start, c_end)

    def collect_pycollo_variables(self):
        """Recollect Pycollo variables accounting for 'constant' variables."""
        self.y_var = needed_to_tuple(self.y_var_full,
                                     self.ocp_phase.bounds._y_needed)
        self.u_var = needed_to_tuple(self.u_var_full,
                                     self.ocp_phase.bounds._u_needed)
        self.q_var = needed_to_tuple(self.q_var_full,
                                     self.ocp_phase.bounds._q_needed)
        self.t_var = needed_to_tuple(self.t_var_full,
                                     self.ocp_phase.bounds._t_needed)
        self.x_var = tuple(itertools.chain(self.y_var,
                                           self.u_var,
                                           self.q_var,
                                           self.t_var))

        self.num_y_var = len(self.y_var)
        self.num_u_var = len(self.u_var)
        self.num_q_var = len(self.q_var)
        self.num_t_var = len(self.t_var)
        self.num_var = len(self.x_var)

        self.V_y_var = needed_to_tuple(self.V_y_var_full,
                                       self.ocp_phase.bounds._y_needed)
        self.V_u_var = needed_to_tuple(self.V_u_var_full,
                                       self.ocp_phase.bounds._u_needed)
        self.V_q_var = needed_to_tuple(self.V_q_var_full,
                                       self.ocp_phase.bounds._q_needed)
        self.V_t_var = needed_to_tuple(self.V_t_var_full,
                                       self.ocp_phase.bounds._t_needed)
        self.V_x_var = tuple(itertools.chain(self.V_y_var,
                                             self.V_u_var,
                                             self.V_q_var,
                                             self.V_t_var))
        self.r_y_var = needed_to_tuple(self.r_y_var_full,
                                       self.ocp_phase.bounds._y_needed)
        self.r_u_var = needed_to_tuple(self.r_u_var_full,
                                       self.ocp_phase.bounds._u_needed)
        self.r_q_var = needed_to_tuple(self.r_q_var_full,
                                       self.ocp_phase.bounds._q_needed)
        self.r_t_var = needed_to_tuple(self.r_t_var_full,
                                       self.ocp_phase.bounds._t_needed)
        self.r_x_var = tuple(itertools.chain(self.r_y_var,
                                             self.r_u_var,
                                             self.r_q_var,
                                             self.r_t_var))

        self.y_t0_var = needed_to_tuple(self.y_t0_var_full,
                                        self.ocp_phase.bounds._y_needed)
        self.y_tF_var = needed_to_tuple(self.y_tF_var_full,
                                        self.ocp_phase.bounds._y_needed)
        y_point_var = (y for y in zip(self.y_t0_var, self.y_tF_var))
        self.y_point_var = tuple(itertools.chain.from_iterable(y_point_var))
        self.num_y_point_var = len(self.y_point_var)

        self.num_each_var = (self.num_y_var,
                             self.num_u_var,
                             self.num_q_var,
                             self.num_t_var)
        self.x_point_var = tuple(itertools.chain(self.y_point_var,
                                                 self.q_var,
                                                 self.t_var))
        self.num_point_var = len(self.x_point_var)

    def create_variable_indexes_slices(self):
        """Abstraction layer for creating continuous/point var index slices."""
        self.create_continuous_variable_indexes_slices()
        self.create_point_variable_indexes_slices()

    def create_continuous_variable_indexes_slices(self):
        """Create variable index slices for phase continuous variables."""
        y_start = 0
        u_start = y_start + self.num_y_var
        q_start = u_start + self.num_u_var
        t_start = q_start + self.num_q_var
        x_end = self.num_var
        self.y_slice = slice(y_start, u_start)
        self.u_slice = slice(u_start, q_start)
        self.q_slice = slice(q_start, t_start)
        self.t_slice = slice(t_start, x_end)
        self.yu_slice = slice(y_start, q_start)
        self.qt_slice = slice(q_start, x_end)
        self.yu_qt_split = q_start

    def create_point_variable_indexes_slices(self):
        """Create variable index slices for phase point variables."""
        y_point_start = 0
        q_point_start = y_point_start + self.num_y_point_var
        t_point_start = q_point_start + self.num_q_var
        x_point_end = self.num_point_var
        self.y_point_slice = slice(y_point_start, q_point_start)
        self.q_point_slice = slice(q_point_start, t_point_start)
        self.t_point_slice = slice(t_point_start, x_point_end)
        self.qt_point_slice = slice(q_point_start, x_point_end)
        self.y_point_qt_point_split = q_point_start


class Pycollo(BackendABC):

    def __init__(self, ocp):
        super().__init__(ocp)
        self.create_expression_graph()

    @staticmethod
    def syms(name, rows=1, cols=1):
        if (rows == 1) and (cols == 1):
            return sym.Symbol(name)
        raise NotImplementedError

    @staticmethod
    def iteration_scaling(*args, **kwargs):
        """Instantiate a PycolloIterationScaling object for iteration.

        Returns
        -------
        PycolloIterationScaling
            Initialised iteration scaling for a mesh iteration.

        """
        return PycolloIterationScaling(*args, **kwargs)

    def postprocess_problem_backend(self):
        """Abstraction layer for backend-specific postprocessing."""
        self.build_expression_graph()
        self.create_compiled_functions()

    def create_compiled_functions(self):
        self.compiled_functions = CompiledFunctions(self)


class Casadi(BackendABC):

    @staticmethod
    def sym(name, rows=1, cols=1):
        return ca.SX.sym(name, rows, cols)

    @staticmethod
    def const(val):
        return ca.DM(val)

    def substitute_pycollo_sym(self, expr, phase=None):
        """Convert to ca.SX and replace user syms with Pycollo backend syms."""
        expr_sym = expr
        if isinstance(expr, sym.Expr):
            if isinstance(phase, PycolloPhaseData):
                user_to_backend_mapping = phase.all_user_to_backend_mapping
            elif isinstance(phase, int):
                phase = self.p[phase]
                user_to_backend_mapping = phase.all_user_to_backend_mapping
            elif phase is None:
                user_to_backend_mapping = self.user_to_backend_mapping
            expr, _ = sympy_to_casadi(expr, user_to_backend_mapping)
        elif isinstance(expr, SUPPORTED_ITER_TYPES + (sym.Matrix, )):
            return self.substitute_matrix_pycollo_sym(expr)
        if isinstance(expr, (int, float, np.int32, np.int64, np.float64)):
            expr = ca.SX(expr)
        if not isinstance(expr, ca.SX):
            msg = (f"Unsupported type of {type(expr)} for substitution.")
            raise NotImplementedError(msg)
        expr = casadi_substitute(expr, self.aux_data)
        return expr

    def substitute_matrix_pycollo_sym(self, expr, phase=None):
        """Convert and replace a non-scalar with Pycollo backend syms."""
        expr_array = np.array(expr)
        expr_array_shape = expr_array.shape
        expr_array_flat = expr_array.flatten()
        for i, single_expr in enumerate(expr_array_flat):
            expr_array_flat[i] = self.substitute_pycollo_sym(single_expr)
        expr_array = expr_array_flat.reshape(*expr_array_shape)
        return ca.SX(expr_array)

    def expr_as_numeric(self, expr):
        """Convert an expression of backend syms to numerical values."""
        return np.array(ca.DM(expr)).astype(np.float64)

    @staticmethod
    def iteration_scaling(*args, **kwargs):
        """Instantiate a CasadiIterationScaling object for iteration.

        Returns
        -------
        CasadiIterationScaling
            Initialised iteration scaling for a mesh iteration.

        """
        scaling = CasadiIterationScaling(*args, **kwargs)
        return scaling

    def postprocess_problem_backend(self):
        """CasADi backend doesn't need to do any postprocessing."""
        pass

    def generate_nlp_function_callables(self, iteration):
        """Create iteration-specific OCP callables required by CasADi backend.
        """
        self.current_iteration = iteration
        self.create_iteration_specific_symbols()
        self.generate_objective_function_callable()
        self.generate_objective_function_gradient_callable()
        self.generate_constraint_function_callable()
        self.generate_jacobian_constraint_function_callable()

    def create_iteration_specific_symbols(self):
        """Abstraction layer for creating iteration-specific symbols.

        There is a 1:1 mapping from temporal node on the mesh discretisation to
        the state and control variables (which are continuous functions of
        time). Integral, time and static parameters variables can just use the
        variables previously created as these do not change when the mesh
        discretisation changes.

        This includes the stretch and shift symbols for the OCP variables, as
        well as the scaling symbols for the objective function and constraint
        function. Like for the variable symbols, the state and control stretch
        and shift symbols are 1:1 related to the mesh discretisation, as are
        the scaling symbols for the defect and path constraints.

        """
        self.create_iteration_specific_variable_symbols()
        self.create_iteration_specific_variable_scaling_mappings()
        self.create_iteration_specific_constraint_scaling_symbols()

    def create_iteration_specific_variable_symbols(self):
        """Create iteration specific variabiel symbols."""
        mapping = {}
        mapping_point = {}
        for p, N in zip(self.p, self.current_iteration.mesh.N):
            mapping.update({y: self.sym(symbol_name(y), N) for y in p.y_var})
            mapping_point.update({y_t0: mapping[y][0]
                                  for y, y_t0 in zip(p.y_var, p.y_t0_var)})
            mapping_point.update({y_tF: mapping[y][-1]
                                  for y, y_tF in zip(p.y_var, p.y_tF_var)})
            mapping.update({u: self.sym(symbol_name(u), N) for u in p.u_var})
            mapping_point.update({q: q for q in p.q_var})
            mapping_point.update({t: t for t in p.t_var})
        mapping_point.update({s: s for s in self.s_var})
        self.ocp_iter_sym_mapping = mapping
        self.ocp_iter_sym_point_mapping = mapping_point
        var_iter = []
        for x in self.x_var:
            var = self.ocp_iter_sym_mapping.get(x)
            if var is not None:
                var_iter.append(var)
            var = self.ocp_iter_sym_point_mapping.get(x)
            if var is not None:
                var_iter.append(var)
        self.x_var_iter = ca.vertcat(*var_iter)

    def create_iteration_specific_variable_scaling_mappings(self):
        """Create iteration-specific stretch/shift scaling mappings."""
        scaling = self.current_iteration.scaling
        self.V_sym_val_mapping_iter = dict(zip(self.V_x_var, scaling.V_ocp))
        self.r_sym_val_mapping_iter = dict(zip(self.r_x_var, scaling.r_ocp))

    def create_iteration_specific_constraint_scaling_symbols(self):
        """Create iteration-specific constraint scaling symbols.

        (1) w_J - objective function
        (2) W_d - defect constraints (mesh-specific)
        (3) W_p - path constraints (mesh-specific)
        (4) W_i - integral constraints
        (5) W_b - endpoint constraints

        """
        self.w_J_iter = self.sym("w_J")
        W = []
        self.W_iter_mapping = {}
        for p in self.p:
            phase_W_iter_mapping = {}
            W_d = list(self.sym(f"W_d{i}_P{p.i}") for i in range(p.num_y_eqn))
            phase_W_iter_mapping["d"] = W_d
            W.extend(W_d)
            W_p = list(self.sym(f"W_p{i}_P{p.i}") for i in range(p.num_p_con))
            phase_W_iter_mapping["p"] = W_p
            W.extend(W_p)
            W_i = list(self.sym(f"W_i{i}_P{p.i}") for i in range(p.num_q_fnc))
            phase_W_iter_mapping["i"] = W_i
            W.extend(W_i)
            self.W_iter_mapping[p] = phase_W_iter_mapping
        W_e = list(self.sym(f"W_b{i}") for i in range(self.num_b_con))
        self.W_iter_mapping["e"] = W_e
        W.extend(W_e)
        self.W_iter = ca.vertcat(*W)

    def generate_objective_function_callable(self):
        """Compile a callable function to evaluate J."""
        J = self.w_J_iter * self.J
        subs = dict_merge(self.ocp_iter_sym_point_mapping,
                          self.V_sym_val_mapping_iter,
                          self.r_sym_val_mapping_iter)
        self.J_iter = casadi_substitute(J, subs)
        args = ca.vertcat(self.x_var_iter,
                          self.w_J_iter)
        self.J_iter_scale_callable = ca.Function("J", [args], [self.J_iter])

    def generate_objective_function_gradient_callable(self):
        """Compile a callable function to evaluate g."""
        self.g_iter = ca.gradient(self.J_iter, self.x_var_iter)
        args = ca.vertcat(self.x_var_iter,
                          self.w_J_iter)
        self.g_iter_scale_callable = ca.Function("g", [args], [self.g_iter])

    def generate_constraint_function_callable(self):
        """Compile a function to evaluate c.

        Generate the iteration-specific constraint vector. The involves
        constructing the iteration-specific: (1) defect constraints, (2) path
        constraints, (3) integral constraints, and (4) endpoint constraints.
        Note that endpoint constraints are iteration specfific and if they
        contain state variables at a phase final time then these are iteration-
        specific due to the fact that mesh sizes (and therefore the number of
        discretised states) vary between mesh iterations.

        """

        def make_all_phase_mapping(mesh):
            """Create mapping from OCP symbol to iteration symbols."""
            all_phase_mapping = {}
            for p, N in zip(self.p, mesh.N):
                phase_mapping = {}
                for i in range(N):
                    mapping = {}
                    for y in p.y_var:
                        mapping[y] = self.ocp_iter_sym_mapping[y][i]
                    for u in p.u_var:
                        mapping[u] = self.ocp_iter_sym_mapping[u][i]
                    phase_mapping[i] = mapping
                all_phase_mapping[p] = phase_mapping
            return all_phase_mapping

        def make_state_derivatives(all_phase_mapping, mesh):
            """Construct all state derivatives for the mesh iteration."""
            dy = []
            for p in self.p:
                phase_mapping = all_phase_mapping[p]
                for y_eqn in p.y_eqn:
                    y_eqn = expand_eqn_to_vec(y_eqn, phase_mapping)
                    dy.append(y_eqn)
            return ca.vertcat(*dy)

        def make_constraints(all_phase_mapping, mesh):
            """Construct all constraints for the mesh iteration."""
            c_d = make_defect_constraints(all_phase_mapping, mesh)
            c_p = make_path_constraints(all_phase_mapping)
            c_i = make_integral_constraints(all_phase_mapping, mesh)
            c_e = make_endpoint_constraints()
            c = []
            for c_d_phase, c_p_phase, c_i_phase in zip(c_d, c_p, c_i):
                c.extend(c_d_phase)
                c.extend(c_p_phase)
                c.extend(c_i_phase)
            c.extend(c_e)
            return ca.vertcat(*c)

        def expand_eqn_to_vec(eqn, phase_mapping):
            """Convert an equation in OCP base to vector iteration base."""
            vec = []
            for mapping in phase_mapping.values():
                vec.append(casadi_substitute(eqn, mapping))
            return ca.vertcat(*vec)

        def make_defect_constraints(all_phase_mapping, mesh):
            """Constraint all defect constraints for the mesh iteration."""
            c_d = []
            for p, A_mat, I_mat in zip(self.p, mesh.sA_matrix, mesh.sI_matrix):
                c = []
                phase_mapping = all_phase_mapping[p]
                W_d_phase = self.W_iter_mapping[p]["d"]
                if p.ocp_phase.bounds._t_needed[0]:
                    t0 = p.V_t_var[0] * p.t_var[0] + p.r_t_var[0]
                else:
                    t0 = p.t_var_full[0]
                if p.ocp_phase.bounds._t_needed[1]:
                    tF = p.V_t_var[-1] * p.t_var[-1] + p.r_t_var[-1]
                else:
                    tF = p.t_var_full[1]
                zipped = zip(p.y_var, p.V_y_var, p.r_y_var, p.y_eqn, W_d_phase)
                for y_var, V_y_var, r_y_var, y_eqn, W_d in zipped:
                    y_var = self.ocp_iter_sym_mapping[y_var]
                    y_var_unscaled = V_y_var * y_var + r_y_var
                    y_eqn = expand_eqn_to_vec(y_eqn, phase_mapping)
                    c.append(W_d * make_defect_constraint(y_var_unscaled,
                                                          y_eqn,
                                                          t0,
                                                          tF,
                                                          A_mat,
                                                          I_mat))
                c_d.append(c)
            return c_d

        def make_defect_constraint(y, y_eqn, t0, tF, A, I):
            """Construct a defect constraint from components."""
            return ca.mtimes(A, y) + 0.5 * (tF - t0) * ca.mtimes(I, y_eqn)

        def make_path_constraints(all_phase_mapping):
            """Constraint all path constraints for the mesh iteration."""
            c_p = []
            for p in self.p:
                c = []
                phase_mapping = all_phase_mapping[p]
                W_p_phase = self.W_iter_mapping[p]["p"]
                for p_con, W_p in zip(p.p_con, W_p_phase):
                    p_con = expand_eqn_to_vec(p_con, phase_mapping)
                    c.append(W_p * p_con)
                c_p.append(c)
            return c_p

        def make_integral_constraints(all_phase_mapping, mesh):
            """Constraint all integral constraints for the mesh iteration."""
            c_i = []
            for p, W_mat in zip(self.p, mesh.W_matrix):
                c = []
                phase_mapping = all_phase_mapping[p]
                W_i_phase = self.W_iter_mapping[p]["i"]
                if p.ocp_phase.bounds._t_needed[0]:
                    t0 = p.V_t_var[0] * p.t_var[0] + p.r_t_var[0]
                else:
                    t0 = p.t_var_full[0]
                if p.ocp_phase.bounds._t_needed[1]:
                    tF = p.V_t_var[-1] * p.t_var[-1] + p.r_t_var[-1]
                else:
                    tF = p.t_var_full[1]
                zipped = zip(p.q_var, p.V_q_var, p.r_q_var, p.q_fnc, W_i_phase)
                for q_var, V_q_var, r_q_var, q_fnc, W_i in zipped:
                    q_var_unscaled = V_q_var * q_var + r_q_var
                    q_fnc = expand_eqn_to_vec(q_fnc, phase_mapping)
                    c.append(W_i * make_integral_constraint(q_var_unscaled,
                                                            q_fnc,
                                                            t0,
                                                            tF,
                                                            W_mat))
                c_i.append(c)
            return c_i

        def make_integral_constraint(q, q_fnc, t0, tF, W):
            """Construct an integral constraint from components."""
            return q - 0.5 * (tF - t0) * ca.dot(W, q_fnc)

        def make_endpoint_constraints():
            """Constraint all endpoint constraints for the mesh iteration."""
            c_e = []
            W_e_problem = self.W_iter_mapping["e"]
            for b_con, W_e in zip(self.b_con, W_e_problem):
                c_e.append(W_e * b_con)
            return c_e

        all_phase_mapping = make_all_phase_mapping(self.current_iteration.mesh)
        dy = make_state_derivatives(all_phase_mapping,
                                    self.current_iteration.mesh)
        c = make_constraints(all_phase_mapping, self.current_iteration.mesh)
        subs = dict_merge(self.ocp_iter_sym_point_mapping,
                          self.V_sym_val_mapping_iter,
                          self.r_sym_val_mapping_iter,
                          self.bounds.aux_data)
        self.dy_iter = casadi_substitute(dy, subs)
        self.dy_iter_callable = ca.Function("dy",
                                            [self.x_var_iter],
                                            [self.dy_iter])
        self.c_iter = casadi_substitute(c, subs)
        args = ca.vertcat(self.x_var_iter,
                          self.W_iter)
        self.c_iter_scale_callable = ca.Function("c", [args], [self.c_iter])

    def generate_jacobian_constraint_function_callable(self):
        """Compile a callable function to evaluate G."""
        self.G_iter = ca.jacobian(self.c_iter, self.x_var_iter)
        args = ca.vertcat(self.x_var_iter,
                          self.W_iter)
        self.G_iter_scale_callable = ca.Function("G", [args], [self.G_iter])

    def create_nlp_solver(self):
        """Create CasADi NLP solver interface to IPOPT."""
        x_iter = self.x_var_iter
        J_subs = {self.w_J_iter: self.current_iteration.scaling.w}
        J_iter = casadi_substitute(self.J_iter, J_subs)
        c_subs = {}
        for i, W_val in enumerate(self.current_iteration.scaling.W_ocp):
            c_subs.update({self.W_iter[i]: W_val})
        c_iter = casadi_substitute(self.c_iter, c_subs)
        nlp = {"x": x_iter, "f": J_iter, "g": c_iter}
        ipopt_settings = self.create_nlp_solver_settings()
        settings = {"ipopt": ipopt_settings}
        self.nlp_solver = ca.nlpsol("solver", "ipopt", nlp, settings)

    def create_nlp_solver_settings(self):
        """Create settings for CasADi IPOPT NLP solver.

        `"mu_strategy"` and `"mu_min"` are overridden from the IPOPT defaults
        to match the overridden settings used by Cyipopt which have been
        amended for good performance in solving NLPs for OCPs.

        """
        warm_start = "yes" if self.ocp.settings.warm_start else "no"
        ipopt_settings = {"tol": self.ocp.settings.nlp_tolerance,
                          "max_iter": self.ocp.settings.max_nlp_iterations,
                          "linear_solver": self.ocp.settings.linear_solver,
                          "mu_strategy": "adaptive",
                          "mu_min": 1e-11,
                          "warm_start_init_point": warm_start,
                          }
        return ipopt_settings

    def evaluate_J(self, x):
        """Evaluate `J` at a point `x` using CasADi compiled function."""
        return float(self.nlp_solver.get_function("nlp_f")(x, False))

    def evaluate_g(self, x):
        """Evaluate `g` at a point `x` using CasADi compiled function."""
        g = np.array(self.nlp_solver.get_function("nlp_grad_f")(x, False)[1]).squeeze()
        return g

    def evaluate_c(self, x):
        """Evaluate `c` at a point `x` using CasADi compiled function."""
        c = np.array(self.nlp_solver.get_function("nlp_g")(x, False)).squeeze()
        return c

    def evaluate_G(self, x):
        """Evaluate `G` at a point `x` using CasADi compiled function.

        This returns `G` as a sparse matrix, the form expected by Pycollo, but
        which is a different form to what CasADi will naturally produce.

        """
        G = self.nlp_solver.get_function("nlp_jac_g")(x, False)[1]
        sG = sparse.coo_matrix(np.array(G))
        return sG

    def evaluate_G_nonzeros(self, x):
        """Evaluate `G` at a point `x` using CasADi compiled function.

        This returns just the nonzero values of `G`.

        """
        G = self.nlp_solver.get_function("nlp_jac_g")(x, False)[1].nonzeros()
        return G

    def evaluate_G_structure(self):
        """Evaluate `G` at a point `x` using CasADi compiled function.

        This returns just the row and column indices of `G` in the form
        expected by Pycollo.

        """
        G = self.nlp_solver.get_function("nlp_jac_g").sx_out()[1]
        arg_1 = range(G.size2())
        arg_2 = np.diff(np.array(G.colind(), dtype=int))
        zipped = zip(arg_1, arg_2)
        iterable = (itertools.repeat(*args) for args in zipped)
        col_indices = np.array(list(itertools.chain.from_iterable(iterable)))
        row_indices = np.array(G.row(), dtype=int)
        return (row_indices, col_indices)

    def evaluate_G_num_nonzero(self):
        """Evaluate `G` at a point `x` using CasADi compiled function.

        This returns just the number of nonzero elements in `G`.

        """
        G = self.nlp_solver.get_function("nlp_jac_g").sx_out()[1]
        nnz = G.nnz()
        return nnz

    def evaluate_H(self, x, obj, l):
        """Evaluate `H` at a point `x` using CasADi compiled function.

        This returns `H` as a sparse matrix, the form expected by Pycollo, but
        which is a different form to what CasADi will naturally produce.

        """
        raise NotImplementedError

    def evaluate_H_nonzeros(self, x):
        """Evaluate `H` at a point `x` using CasADi compiled function.

        This returns just the nonzero values of `H`.

        """
        raise NotImplementedError

    def evaluate_H_structure(self):
        """Evaluate the structure of `H` using CasADi compiled function.

        This returns just the row and column indices of `H` in the form
        expected by Pycollo.

        """
        raise NotImplementedError

    def evaluate_H_num_nonzero(self):
        """Evaluate number of nonzeros in `H` using CasADi compiled function.

        This returns just the number of nonzero elements in `H`.

        """
        raise NotImplementedError

    def solve_nlp(self):
        """Solve the NLP.

        Returns
        -------
        NlpResult
            Named tuple including the solution, solution info, and solve time.

        """
        nlp_start_time = timer()
        nlp_solver_output = self.nlp_solver(x0=self.current_iteration.guess_x,
                                            lbx=self.current_iteration.x_bnd_l,
                                            ubx=self.current_iteration.x_bnd_u,
                                            lbg=self.current_iteration.c_bnd_l,
                                            ubg=self.current_iteration.c_bnd_u)
        nlp_stop_time = timer()
        nlp_solve_time = nlp_stop_time - nlp_start_time
        nlp_result = NlpResult(solution=nlp_solver_output,
                               info=None,
                               solve_time=nlp_solve_time)
        return nlp_result

    @staticmethod
    def process_solution(*args, **kwargs):
        """Instantiate a CasadiSolution object for iteration.

        Returns
        -------
        CasadiSolution
            Solution class with processed NLP solution.

        """
        solution = CasadiSolution(*args, **kwargs)
        return solution


class Hsad(BackendABC):

    not_implemented_error_msg = ("The hSAD backend for Pycollo is not "
                                 "currently supported or implemented.")

    def __init__(self, ocp):
        raise NotImplementedError(self.not_implemented_error_msg)

    @classmethod
    def sym(cls, name):
        raise NotImplementedError(cls.not_implemented_error_msg)

    @staticmethod
    def const(val):
        raise NotImplementedError(not_implemented_error_msg)

    def substitute_pycollo_sym(self, expr):
        raise NotImplementedError(not_implemented_error_msg)

    @staticmethod
    def iteration_scaling(*args, **kwargs):
        """Instantiate a `HsadIterationScaling` object for iteration.

        Returns
        -------
        HsadIterationScaling
            Initialised iteration scaling for a mesh iteration.

        """
        return HsadIterationScaling(*args, **kwargs)

    def postprocess_problem_backend(self):
        """Abstraction layer for backend-specific postprocessing."""
        pass

    def generate_nlp_function_callables(self, iteration):
        """Create iteration-specific OCP callables required by hSAD backend.
        """
        pass


class Sympy(BackendABC):

    not_implemented_error_msg = ("The Sympy backend for Pycollo is not "
                                 "currently supported or implemented.")

    def __init__(self, ocp):
        raise NotImplementedError(self.not_implemented_error_msg)

    @staticmethod
    def sym(name):
        raise NotImplementedError(not_implemented_error_msg)

    @staticmethod
    def const(val):
        raise NotImplementedError(not_implemented_error_msg)

    def substitute_pycollo_sym(self, expr):
        raise NotImplementedError(not_implemented_error_msg)

    @staticmethod
    def iteration_scaling(*args, **kwargs):
        """Instantiate a `SympyIterationScaling` object for iteration.

        Returns
        -------
        SympyIterationScaling
            Initialised iteration scaling for a mesh iteration.

        """
        return SympyIterationScaling(*args, **kwargs)

    def postprocess_problem_backend(self):
        """Abstraction layer for backend-specific postprocessing."""
        pass

    def generate_nlp_function_callables(self, iteration):
        """Create iteration-specific OCP callables required by Sympy backend.
        """
        pass


BACKENDS = Options((PYCOLLO, HSAD, CASADI, SYMPY), default=CASADI,
                   unsupported=(PYCOLLO, HSAD, SYMPY),
                   handles=(Pycollo, Hsad, Casadi, Sympy))
