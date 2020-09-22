from abc import (ABC, abstractmethod)
from collections import namedtuple
import itertools

import numpy as np
import sympy as sym

from .bounds import Bounds
from .compiled import CompiledFunctions
from .expression_graph import ExpressionGraph
from .guess import Guess
from .iteration import Iteration
from .mesh import Mesh
from .quadrature import Quadrature
from .scaling import Scaling
from .utils import (console_out, dict_merge, fast_sympify,
                    format_multiple_items_for_output)


__all__ = []


CASADI = "casadi"
PYCOLLO = "pycollo"
SYMPY = "sympy"


class BackendABC(ABC):

    # _DEFAULT_BACKEND = PYCOLLO
    # _BACKENDS = {PYCOLLO}

    def create_bounds(self):
        self.bounds = Bounds(self)
        self.recollect_variables_and_slices()

    def create_compiled_functions(self):
        self.compiled_functions = CompiledFunctions(self)

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
        self.create_variable_symbols()
        self.preprocess_variables()
        self.create_full_variable_indexes_slices()

    def create_variable_symbols(self):
        self.create_state_variable_symbols()
        self.create_control_variable_symbols()
        self.create_integral_variable_symbols()
        self.create_time_variable_symbols()

    def create_state_variable_symbols(self):
        self.y_vars_full = tuple(sym.Symbol(f'_y{i_y}_P{self.i}')
                                 for i_y, _ in enumerate(self.ocp_phase._y_vars_user))
        self.num_y_vars_full = len(self.y_vars_full)
        self.y_t0_vars_full = tuple(sym.Symbol(f'{y}_t0')
                                    for y in self.y_vars_full)
        self.y_tF_vars_full = tuple(sym.Symbol(f'{y}_tF')
                                    for y in self.y_vars_full)
        self.y_point_vars_full = tuple(itertools.chain.from_iterable(y
                                                                     for y in zip(self.y_t0_vars_full, self.y_tF_vars_full)))
        self.num_y_point_vars_full = len(self.y_point_vars_full)

    def create_control_variable_symbols(self):
        self.u_vars_full = tuple(sym.Symbol(f'_u{i_u}_P{self.i}')
                                 for i_u, _ in enumerate(self.ocp_phase._u_vars_user))
        self.num_u_vars_full = len(self.u_vars_full)

    def create_integral_variable_symbols(self):
        self.q_vars_full = tuple(sym.Symbol(f'_{q}')
                                 for q in self.ocp_phase._q_vars_user)
        self.num_q_vars_full = len(self.q_vars_full)

    def create_time_variable_symbols(self):
        self.t_vars_full = (self.ocp_phase._t0, self.ocp_phase._tF)
        self.num_t_vars_full = len(self.t_vars_full)
        self.t_norm = self.ocp_phase._STRETCH

    def preprocess_variables(self):
        self.collect_pycollo_variables_full()
        self.collect_user_variables()

    def collect_pycollo_variables_full(self):
        self.x_vars_full = (self.y_vars_full + self.u_vars_full
                            + self.q_vars_full + self.t_vars_full)
        self.num_vars_full = len(self.x_vars_full)
        self.num_each_vars_full = (self.num_y_vars_full + self.num_u_vars_full
                                   + self.num_q_vars_full + self.num_t_vars_full)
        self.x_point_vars_full = (self.y_point_vars_full + self.q_vars_full
                                  + self.t_vars_full)
        self.num_point_vars_full = len(self.x_point_vars_full)

    def collect_user_variables(self):
        self.y_vars_user = self.ocp_phase._y_vars_user
        self.y_t0_vars_user = self.ocp_phase._y_t0_user
        self.y_tF_vars_user = self.ocp_phase._y_tF_user
        self.y_point_vars_user = tuple(itertools.chain.from_iterable(y
                                                                     for y in zip(self.y_t0_vars_user, self.y_tF_vars_user)))
        self.u_vars_user = self.ocp_phase._u_vars_user
        self.q_vars_user = self.ocp_phase._q_vars_user
        self.t_vars_user = self.ocp_phase._t_vars_user
        self.x_vars_user = (self.y_vars_user + self.u_vars_user
                            + self.q_vars_user + self.t_vars_user)
        self.x_point_vars_user = (self.y_point_vars_user + self.q_vars_user
                                  + self.t_vars_user)
        self.all_user_vars = set(self.x_vars_user + self.x_point_vars_user)

    def create_full_variable_indexes_slices(self):
        self.create_full_continuous_variable_indexes_slices()
        self.create_full_point_variable_indexes_slices()

    def create_full_continuous_variable_indexes_slices(self):
        self.y_slice_full = slice(0, self.num_y_vars_full)
        self.u_slice_full = slice(self.y_slice_full.stop,
                                  self.y_slice_full.stop + self.num_u_vars_full)
        self.q_slice_full = slice(self.u_slice_full.stop,
                                  self.u_slice_full.stop + self.num_q_vars_full)
        self.t_slice_full = slice(self.q_slice_full.stop, self.num_vars_full)
        self.yu_slice_full = slice(self.y_slice_full.start,
                                   self.u_slice_full.stop)
        self.qt_slice_full = slice(self.q_slice_full.start, self.num_vars_full)
        self.yu_qt_split_full = self.yu_slice_full.stop

    def create_full_point_variable_indexes_slices(self):
        self.y_point_full_slice = slice(0, self.num_y_point_vars_full)
        self.q_point_full_slice = slice(self.y_point_full_slice.stop,
                                        self.y_point_full_slice.stop + self.num_q_vars_full)
        self.t_point_full_slice = slice(self.q_point_full_slice.stop,
                                        self.num_point_vars_full)
        self.qt_point_full_slice = slice(self.q_point_full_slice.start,
                                         self.num_point_vars_full)
        self.y_point_qt_point_full_split = self.y_point_full_slice.stop

    def preprocess_auxiliary_data(self):
        self.create_auxiliary_variables_for_user_phase_aux_data()
        self.check_all_user_phase_aux_data_supplied()
        self.collect_variables_substitutions()

    def create_auxiliary_variables_for_user_phase_aux_data(self):
        self.aux_data = {}
        self.user_phase_aux_data_mapping = {}
        per_phase_aux_data = dict_merge(self.ocp_phase.auxiliary_data,
                                        self.ocp_backend.aux_data_phase_dependent)
        for user_sym, user_eqn in per_phase_aux_data.items():
            phase_sym = sym.Symbol(f"_p_{user_sym}_P{self.i}")
            self.aux_data[phase_sym] = fast_sympify(user_eqn)
            self.user_phase_aux_data_mapping[user_sym] = phase_sym

    def check_all_user_phase_aux_data_supplied(self):
        self_aux_user_keys = set(self.user_phase_aux_data_mapping.keys())
        missing_syms = self.ocp_backend.user_phase_aux_data_syms.difference(
            self_aux_user_keys)
        self.missing_user_phase_aux_data_syms = missing_syms

    def collect_variables_substitutions(self):
        vars_subs = dict(zip(self.x_vars_user, self.x_vars_full))
        point_vars_subs = dict(
            zip(self.x_point_vars_user, self.x_point_vars_full))
        self.all_subs_mappings = dict_merge(vars_subs, point_vars_subs,
                                            self.user_phase_aux_data_mapping,
                                            self.ocp_backend.s_vars_subs_mappings)
        self.aux_data = {phase_sym: user_eqn.xreplace(self.all_subs_mappings)
                         for phase_sym, user_eqn in self.aux_data.items()}

    def preprocess_constraints(self):
        self.preprocess_defect_constraints()
        self.preprocess_path_constraints()
        self.preprocess_integral_constraints()
        self.collect_constraints()
        self.check_all_needed_user_phase_aux_data_supplied()

    def preprocess_defect_constraints(self):
        self.zeta = tuple(y_eqn.xreplace(self.all_subs_mappings)
                          for y_eqn in self.ocp_phase.state_equations)
        self.num_c_defect = self.ocp_phase.number_state_equations

    def preprocess_path_constraints(self):
        self.gamma = tuple(p_con.xreplace(self.all_subs_mappings)
                           for p_con in self.ocp_phase.path_constraints)
        self.num_c_path = self.ocp_phase.number_path_constraints

    def preprocess_integral_constraints(self):
        self.rho = tuple(q_fnc.xreplace(self.all_subs_mappings)
                         for q_fnc in self.ocp_phase.integrand_functions)
        self.num_c_integral = self.ocp_phase.number_integrand_functions

    def collect_constraints(self):
        self.c = (self.zeta + self.gamma + self.rho)
        self.num_c = (self.num_c_defect +
                      self.num_c_path + self.num_c_integral)

    def create_constraint_indexes_slices(self):
        self.c_defect_slice = slice(0, self.num_c_defect)
        self.c_path_slice = slice(self.c_defect_slice.stop,
                                  self.c_defect_slice.stop + self.num_c_path)
        self.c_integral_slice = slice(self.c_path_slice.stop, self.num_c)

    def check_all_needed_user_phase_aux_data_supplied(self):
        needed_syms = {s for c in self.c for s in c.free_symbols}
        missing_syms = needed_syms.intersection(
            self.missing_user_phase_aux_data_syms)
        if missing_syms:
            self.raise_needed_user_phases_aux_data_missing_error(missing_syms)

    def raise_needed_user_phases_aux_data_missing_error(self, missing_syms):
        formatted_syms = format_multiple_items_for_output(missing_syms)
        msg = (f"Phase-dependent auxiliary data must be supplied for "
                f"all phase-dependent symbols in each phase. Please supply "
                f"auxiliary data for {formatted_syms} in {self.ocp_phase.name} "
                f"(phase index: {self.i}).")
        raise ValueError(msg)

    def collect_pycollo_variables(self):
        self.y_vars = tuple(np.array(self.y_vars_full)[
                            self.ocp_phase.bounds._y_needed].tolist())
        self.u_vars = tuple(np.array(self.u_vars_full)[
                            self.ocp_phase.bounds._u_needed].tolist())
        self.q_vars = tuple(np.array(self.q_vars_full)[
                            self.ocp_phase.bounds._q_needed].tolist())
        self.t_vars = tuple(np.array(self.t_vars_full)[
                            self.ocp_phase.bounds._t_needed].tolist())
        self.x_vars = self.y_vars + self.u_vars + self.q_vars + self.t_vars

        self.num_y_vars = len(self.y_vars)
        self.num_u_vars = len(self.u_vars)
        self.num_q_vars = len(self.q_vars)
        self.num_t_vars = len(self.t_vars)
        self.num_vars = len(self.x_vars)

        self.y_t0_vars = tuple(np.array(self.y_t0_vars_full)[
                               self.ocp_phase.bounds._y_needed].tolist())
        self.y_tF_vars = tuple(np.array(self.y_tF_vars_full)[
                               self.ocp_phase.bounds._y_needed].tolist())
        self.y_point_vars = tuple(itertools.chain.from_iterable(y
                                                                for y in zip(self.y_t0_vars, self.y_tF_vars)))
        self.num_y_point_vars = len(self.y_point_vars)

        self.num_each_vars = (self.num_y_vars, self.num_u_vars,
                              self.num_q_vars, self.num_t_vars)
        self.x_point_vars = (self.y_point_vars + self.q_vars
                             + self.t_vars)
        self.num_point_vars = len(self.x_point_vars)

        self.create_variable_indexes_slices()

    def create_variable_indexes_slices(self):
        self.create_continuous_variable_indexes_slices()
        self.create_point_variable_indexes_slices()

    def create_continuous_variable_indexes_slices(self):
        self.y_slice = slice(0, self.num_y_vars)
        self.u_slice = slice(self.y_slice.stop,
                             self.y_slice.stop + self.num_u_vars)
        self.q_slice = slice(self.u_slice.stop,
                             self.u_slice.stop + self.num_q_vars)
        self.t_slice = slice(self.q_slice.stop, self.num_vars)
        self.yu_slice = slice(self.y_slice.start,
                              self.u_slice.stop)
        self.qt_slice = slice(self.q_slice.start, self.num_vars)
        self.yu_qt_split = self.yu_slice.stop

    def create_point_variable_indexes_slices(self):
        self.y_point_slice = slice(0, self.num_y_point_vars)
        self.q_point_slice = slice(self.y_point_slice.stop,
                                   self.y_point_slice.stop + self.num_q_vars)
        self.t_point_slice = slice(self.q_point_slice.stop,
                                   self.num_point_vars)
        self.qt_point_slice = slice(self.q_point_slice.start,
                                    self.num_point_vars)
        self.y_point_qt_point_split = self.y_point_slice.stop


class Pycollo(BackendABC):

    def __init__(self, ocp):
        self.ocp = ocp
        self.create_point_variable_symbols()
        self.create_phase_backends()
        self.preprocess_user_problem_aux_data()
        self.preprocess_phase_backends()
        self.collect_variables_substitutions()
        self.console_out_variables_constraints_preprocessed()
        self.create_expression_graph()

    def create_point_variable_symbols(self):
        self.create_parameter_variable_symbols()

    def create_parameter_variable_symbols(self):
        self.s_vars_full = tuple(sym.Symbol(f'_s{i_s}')
                                 for i_s, _ in enumerate(self.ocp._s_vars_user))
        self.num_s_vars_full = len(self.s_vars_full)
        self.s_vars_subs_mappings = dict(
            zip(self.ocp._s_vars_user, self.s_vars_full))
        self.s_vars_user = self.ocp._s_vars_user

    def create_phase_backends(self):
        self.p = tuple(PycolloPhaseData(self, phase)
                       for phase in self.ocp.phases)
        self.num_phases = len(self.p)
        self.all_phase_vars = {var for p in self.p for var in p.all_user_vars}
        self.all_vars = self.all_phase_vars.union(set(self.s_vars_user))

    def preprocess_user_problem_aux_data(self):
        self.user_phase_aux_data_syms = {symbol
                                         for phase in self.ocp.phases
                                         for symbol in phase.auxiliary_data}
        self.aux_data = {}
        self.aux_data_phase_dependent = {}
        self.aux_data_phase_independent = {}
        self.aux_data_supplied_in_ocp_and_phase = {}
        for symbol, equation in self.ocp.auxiliary_data.items():
            self.partition_user_problem_phase_aux_data(symbol, equation)
        self.check_user_phase_aux_data_not_user_problem_aux_data()
        self.partition_user_problem_aux_data()

    def partition_user_problem_phase_aux_data(self, symbol, equation):
        equation = fast_sympify(equation)
        if symbol in self.user_phase_aux_data_syms:
            self.aux_data_supplied_in_ocp_and_phase[symbol] = equation
        elif self.user_phase_aux_data_syms.intersection(equation.free_symbols):
            self.aux_data_phase_dependent[symbol] = equation
        else:
            self.aux_data[symbol] = equation

    def check_user_phase_aux_data_not_user_problem_aux_data(self):
        if self.aux_data_supplied_in_ocp_and_phase:
            formatted_syms = format_multiple_items_for_output(
                self.aux_data_supplied_in_ocp_and_phase)
            msg = (f"Auxiliary data for {formatted_syms} has been supplied at "
                    f"a per-phase level and therefore cannot be supplied at a "
                    f"problem level.")
            raise ValueError(msg)

    def partition_user_problem_aux_data(self):
        for symbol, equation in self.aux_data.items():
            self.process_aux_data_pair_is_phase_dependent(symbol, equation)

    def process_aux_data_pair_is_phase_dependent(self, symbol, equation):
        if symbol in self.aux_data_phase_dependent:
            return True
        elif equation in self.all_phase_vars:
            self.new_aux_data_pair_phase_independent(symbol, equation)
            return True
        elif equation in self.s_vars_user:
            return False
        elif equation.is_Number:
            self.new_aux_data_pair_phase_independent(symbol, equation)
            return False
        else:
            return self.check_aux_data_pair_children(symbol, equation)

    def new_aux_data_pair_phase_dependent(self, symbol, equation):
        self.aux_data_phase_dependent[symbol] = equation

    def new_aux_data_pair_phase_independent(self, symbol, equation):
        self.aux_data_phase_independent[symbol] = equation

    def check_aux_data_pair_children(self, symbol, equation):
        child_syms = list(equation.free_symbols)
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
        all_symbol_equation_mappings = dict_merge(self.aux_data,
                                                  self.aux_data_phase_dependent, self.aux_data_phase_independent)
        equation = all_symbol_equation_mappings.get(symbol)
        if symbol in self.all_vars:
            return symbol
        elif equation is None:
            msg = (f"'{symbol}' is not defined.")
            raise ValueError(msg)
        return equation

    def preprocess_phase_backends(self):
        for p in self.p:
            p.preprocess_auxiliary_data()
            p.preprocess_constraints()
            p.create_constraint_indexes_slices()

    def collect_variables_substitutions(self):
        self.all_subs_mappings = dict_merge(self.s_vars_subs_mappings,
                                            *[p.all_subs_mappings for p in self.p])
        self.aux_data = {phase_sym: user_eqn.xreplace(self.all_subs_mappings)
                         for phase_sym, user_eqn in self.aux_data_phase_independent.items()}

    def console_out_variables_constraints_preprocessed(self):
        if self.ocp.settings.console_out_progress:
            msg = "Pycollo variables and constraints preprocessed."
            console_out(msg)

    def create_expression_graph(self):
        variables = self.collect_variables()
        objective = None
        constraints = None
        aux_data = self.collect_aux_data()
        self.expression_graph_full = ExpressionGraph(self, variables, objective,
                                                     constraints, aux_data)

    def collect_variables(self):
        continuous_vars_full = (tuple(itertools.chain.from_iterable(p.x_vars_full
                                                                    for p in self.p)) + self.s_vars_full)
        self.num_vars_full = len(continuous_vars_full)
        endpoint_vars_full = (tuple(itertools.chain.from_iterable(p.x_point_vars_full
                                                                  for p in self.p)) + self.s_vars_full)
        variables = (continuous_vars_full, endpoint_vars_full)
        self.phase_variable_full_slices = []
        start = 0
        for p in self.p:
            stop = start + p.num_vars_full
            p_slice = slice(start, stop)
            start = stop
            self.phase_variable_full_slices.append(p_slice)
        return variables

    def collect_aux_data(self):
        aux_data = dict_merge(self.aux_data, *(p.aux_data for p in self.p))
        return aux_data

    def recollect_variables_and_slices(self):
        self.recollect_variables()
        self.process_objective_function()
        self.process_point_constraints()
        self.build_expression_graph()

    def recollect_variables(self):
        for p in self.p:
            p.collect_pycollo_variables()

        self.s_vars = tuple(np.array(self.s_vars_full)[
                            self.ocp.bounds._s_needed].tolist())
        self.num_s_vars = len(self.s_vars)

        continuous_vars = (tuple(itertools.chain.from_iterable(p.x_vars
                                                               for p in self.p)) + self.s_vars)
        self.x_vars = continuous_vars
        self.num_vars = len(continuous_vars)
        endpoint_vars = (tuple(itertools.chain.from_iterable(p.x_point_vars
                                                             for p in self.p)) + self.s_vars)
        self.x_point_vars = endpoint_vars
        self.num_point_vars = len(endpoint_vars)
        self.variables = (continuous_vars, endpoint_vars)

        self.phase_y_vars_slices = []
        self.phase_u_vars_slices = []
        self.phase_q_vars_slices = []
        self.phase_t_vars_slices = []
        self.phase_variable_slices = []
        phase_start = 0
        for p in self.p:
            start = phase_start
            stop = start + p.num_y_vars
            p_slice = slice(start, stop)
            self.phase_y_vars_slices.append(p_slice)
            start = stop
            stop = start + p.num_u_vars
            p_slice = slice(start, stop)
            self.phase_u_vars_slices.append(p_slice)
            start = stop
            stop = start + p.num_q_vars
            p_slice = slice(start, stop)
            self.phase_q_vars_slices.append(p_slice)
            start = stop
            stop = start + p.num_t_vars
            p_slice = slice(start, stop)
            self.phase_t_vars_slices.append(p_slice)
            start = stop
            phase_stop = phase_start + p.num_vars
            p_slice = slice(phase_start, phase_stop)
            self.phase_variable_slices.append(p_slice)
            phase_start = phase_stop
        self.s_vars_slice = slice(
            self.num_vars - self.num_s_vars, self.num_vars)
        self.variable_slice = self.s_vars_slice

        self.phase_endpoint_variable_slices = []
        start = 0
        for p in self.p:
            stop = start + p.num_point_vars
            p_slice = slice(start, stop)
            start = stop
            self.phase_endpoint_variable_slices.append(p_slice)
        self.endpoint_variable_slice = slice(
            self.num_point_vars - self.num_s_vars, self.num_point_vars)

    def process_objective_function(self):
        self.J = self.ocp.objective_function.xreplace(self.all_subs_mappings)

    def process_point_constraints(self):
        all_y_vars = []
        all_y_t0_vars = []
        all_y_tF_vars = []
        for p in self.p:
            all_y_vars.extend(list(p.y_vars))
            all_y_t0_vars.extend(list(p.y_t0_vars))
            all_y_tF_vars.extend(list(p.y_tF_vars))
        all_y_vars_set = set(all_y_vars)
        all_y_bnds = []
        for var, x_bnds in zip(self.x_vars, self.bounds.x_bnds):
            if var in all_y_vars_set:
                all_y_bnds.append(x_bnds)
        endpoint_state_constraints = []
        endpoint_state_constraints_bounds = []
        for y_t0_var, y_tF_var, y_bnds, y_t0_bnds, y_tF_bnds in zip(
                all_y_t0_vars, all_y_tF_vars, all_y_bnds, self.bounds.y_t0_bnds, self.bounds.y_tF_bnds):
            if np.any(~np.isclose(np.array(y_t0_bnds), np.array(y_bnds))):
                endpoint_state_constraints.append(y_t0_var)
                endpoint_state_constraints_bounds.append(y_t0_bnds)
            if np.any(~np.isclose(np.array(y_tF_bnds), np.array(y_bnds))):
                endpoint_state_constraints.append(y_tF_var)
                endpoint_state_constraints_bounds.append(y_tF_bnds)
        self.y_beta = tuple(endpoint_state_constraints)
        self.bounds.c_y_bnds = endpoint_state_constraints_bounds
        self.beta = tuple(b_con.xreplace(self.all_subs_mappings)
                          for b_con in self.ocp.endpoint_constraints)
        self.num_c_endpoint = self.ocp.number_endpoint_constraints

    def collect_constraints(self):
        constraints = (tuple(itertools.chain.from_iterable(p.c
                                                           for p in self.p)) + self.beta)

        self.num_c = len(constraints)
        self.phase_defect_constraint_slices = []
        self.phase_path_constraint_slices = []
        self.phase_integral_constraint_slices = []
        self.phase_constraint_slices = []
        phase_start = 0
        for p in self.p:
            start = phase_start
            stop = start + p.num_c_defect
            p_slice = slice(start, stop)
            self.phase_defect_constraint_slices.append(p_slice)
            start = stop
            stop = start + p.num_c_path
            p_slice = slice(start, stop)
            self.phase_path_constraint_slices.append(p_slice)
            start = stop
            stop = start + p.num_c_integral
            p_slice = slice(start, stop)
            self.phase_integral_constraint_slices.append(p_slice)
            start = stop
            phase_stop = phase_start + p.num_c
            p_slice = slice(phase_start, phase_stop)
            self.phase_constraint_slices.append(p_slice)
            phase_start = phase_stop

        start = 0
        stop = self.phase_constraint_slices[-1].stop
        self.c_continuous_slice = slice(start, stop)
        start = stop
        stop = start + self.num_c_endpoint
        self.c_endpoint_slice = slice(start, stop)
        return constraints

    def build_expression_graph(self):
        variables = self.variables
        objective = self.J
        constraints = self.collect_constraints()
        aux_data = dict_merge(self.aux_data, *(p.aux_data for p in self.p), self.bounds.aux_data)
        self.expression_graph = ExpressionGraph(self, variables, objective,
                                                constraints, aux_data)
        self.expression_graph.form_functions_and_derivatives()


class Casadi(BackendABC):

    def __init__(self, ocp):
        raise NotImplementedError


class Sympy(BackendABC):

    def __init__(self, ocp):
        raise NotImplementedError


BACKENDS = Options((PYCOLLO, CASADI, SYMPY), default=CASADI, unsupported=SYMPY,
                   handles=(Pycollo, Casadi, Sympy))
