from abc import (ABC, abstractmethod)
from collections import namedtuple
import itertools

import sympy as sym

from .expression_graph import ExpressionGraph
from .quadrature import Quadrature
from .scaling import Scaling
from .utils import (console_out, dict_merge, fast_sympify, 
	format_multiple_items_for_output)



class PycolloPhaseData:

	def __init__(self, ocp_backend, ocp_phase):

		self.ocp_backend = ocp_backend
		self.ocp_phase = ocp_phase
		self.i = ocp_phase.phase_number
		self.create_variable_symbols()
		self.preprocess_variables()
		self.create_variable_indexes_slices()

	def create_variable_symbols(self):
		self.create_state_variable_symbols()
		self.create_control_variable_symbols()
		self.create_integral_variable_symbols()
		self.create_time_variable_symbols()

	def create_state_variable_symbols(self):
		self.y_vars = tuple(sym.Symbol(f'_y{i_y}_P{self.i}') 
			for i_y, _ in enumerate(self.ocp_phase._y_vars_user))
		self.num_y_vars = len(self.y_vars)
		self.y_t0_vars = tuple(sym.Symbol(f'{y}_t0') for y in self.y_vars)
		self.y_tF_vars = tuple(sym.Symbol(f'{y}_tF') for y in self.y_vars)
		self.y_point_vars = tuple(itertools.chain.from_iterable(y 
			for y in zip(self.y_t0_vars, self.y_tF_vars)))
		self.num_y_point_vars = len(self.y_point_vars)

	def create_control_variable_symbols(self):
		self.u_vars = tuple(sym.Symbol(f'_u{i_u}_P{self.i}') 
			for i_u, _ in enumerate(self.ocp_phase._u_vars_user))
		self.num_u_vars = len(self.u_vars)

	def create_integral_variable_symbols(self):
		self.q_vars = tuple(sym.Symbol(f'_{q}') 
			for q in self.ocp_phase._q_vars_user)
		self.num_q_vars = len(self.q_vars)

	def create_time_variable_symbols(self):
		self.t_vars = (self.ocp_phase._t0, self.ocp_phase._tF)
		self.num_t_vars = len(self.t_vars)
		self.t_norm = self.ocp_phase._STRETCH

	def preprocess_variables(self):
		self.collect_pycollo_variables()
		self.collect_user_variables()

	def collect_pycollo_variables(self):
		self.x_vars = (self.y_vars + self.u_vars + self.q_vars + self.t_vars)
		self.num_vars = len(self.x_vars)
		self.num_each_vars = (self.num_y_vars + self.num_u_vars 
			+ self.num_q_vars + self.num_t_vars)
		self.x_point_vars = (self.y_point_vars + self.q_vars + self.t_vars)
		self.num_point_vars = len(self.x_point_vars)

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
		self.yu_slice = slice(self.y_slice.start, self.u_slice.stop)
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
			self.aux_data[phase_sym] = user_eqn
			self.user_phase_aux_data_mapping[user_sym] = phase_sym

	def check_all_user_phase_aux_data_supplied(self):
		self_aux_user_keys = set(self.user_phase_aux_data_mapping.keys())
		missing_syms = self.ocp_backend.user_aux_data_syms.difference(
			self_aux_user_keys)
		self.missing_user_phase_aux_data_syms = missing_syms

	def collect_variables_substitutions(self):
		vars_subs = dict(zip(self.x_vars_user, self.x_vars))
		point_vars_subs = dict(zip(self.x_point_vars_user, self.x_point_vars))
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
		self.num_c = (self.num_c_defect + self.num_c_path + self.num_c_integral)

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


class Backend(ABC):
	
	def create_scaling(self):
		self.scaling = Scaling(self)

	def create_quadrature(self):
		self.quadrature = Quadrature(self)


class Pycollo(Backend):
	
	def __init__(self, ocp):
		self.ocp = ocp
		self.create_point_variable_symbols()
		self.create_phase_backends()
		self.preprocess_user_problem_aux_data()
		self.preprocess_phase_backends()
		self.collect_variables_substitutions()
		self.preprocess_objective_function()
		self.preprocess_point_constraints()
		self.console_out_variables_constraints_preprocessed()
		self.create_expression_graph()

	def create_point_variable_symbols(self):
		self.create_parameter_variable_symbols()

	def create_parameter_variable_symbols(self):
		self.s_vars = tuple(sym.Symbol(f'_s{i_s}') 
			for i_s, _ in enumerate(self.ocp._s_vars_user))
		self.num_s_vars = len(self.s_vars)
		self.s_vars_subs_mappings = dict(zip(self.ocp._s_vars_user, self.s_vars))

	def create_phase_backends(self):
		self.p = tuple(PycolloPhaseData(self, phase) 
			for phase in self.ocp.phases)
		self.num_phases = len(self.p)
		self.all_phase_vars = {var for p in self.p for var in p.all_user_vars}

	def preprocess_user_problem_aux_data(self):
		self.user_aux_data_syms = {symbol 
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
		if symbol in self.user_aux_data_syms:
			self.aux_data_supplied_in_ocp_and_phase[symbol] = equation
		elif self.user_aux_data_syms.intersection(equation.free_symbols):
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
		elif symbol in self.aux_data_phase_independent or equation is None:
			return False
		elif equation.is_Number or equation in self.s_vars:
			self.new_aux_data_pair_phase_independent(symbol, equation)
			return True
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
		if all(child_is_phase_dependent):
			self.new_aux_data_pair_phase_independent(symbol, equation)
			return True
		else:
			self.new_aux_data_pair_phase_dependent(symbol, equation)
			return False

	def get_child_equation(self, symbol):
		all_symbol_equation_mappings = dict_merge(self.aux_data, 
			self.aux_data_phase_dependent, self.aux_data_phase_independent)
		equation = all_symbol_equation_mappings.get(symbol)
		if equation is None and symbol not in self.all_phase_vars:
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

	def preprocess_objective_function(self):
		self.J = self.ocp.objective_function.xreplace(self.all_subs_mappings)

	def preprocess_point_constraints(self):
		self.beta = tuple(b_con.xreplace(self.all_subs_mappings)
			for b_con in self.ocp.endpoint_constraints)
		self.num_c_endpoint = self.ocp.number_endpoint_constraints
		
	def console_out_variables_constraints_preprocessed(self):
		msg = "Pycollo variables and constraints preprocessed."
		console_out(msg)

	def create_expression_graph(self):
		variables = self.collect_variables()
		objective = self.J
		constraints = self.collect_constraints()
		aux_data = self.collect_aux_data()
		self.expression_graph = ExpressionGraph(self, variables, objective, 
			constraints, aux_data)

	def collect_variables(self):
		continuous_vars = (tuple(itertools.chain.from_iterable(p.x_vars 
			for p in self.p)) + self.s_vars)
		self.num_vars = len(continuous_vars)
		endpoint_vars = (tuple(itertools.chain.from_iterable(p.x_point_vars 
			for p in self.p)) + self.s_vars)
		variables = (continuous_vars, endpoint_vars)
		self.phase_variable_slices = []
		start = 0
		for p in self.p:
			stop = start + p.num_vars
			p_slice = slice(start, stop)
			start = stop
			self.phase_variable_slices.append(p_slice)
		return variables

	def collect_constraints(self):
		constraints = (tuple(itertools.chain.from_iterable(p.c 
			for p in self.p)) + self.beta)
		self.num_c = len(constraints)
		self.phase_constraint_slices = []
		start = 0
		for p in self.p:
			stop = start + p.num_c
			p_slice = slice(start, stop)
			start = stop
			self.phase_constraint_slices.append(p_slice)
		start = 0
		stop = self.phase_constraint_slices[-1].stop
		self.c_continuous_slice = slice(start, stop)
		start = stop
		stop = start + self.num_c_endpoint
		self.c_endpoint_slice = slice(start, stop)
		return constraints

	def collect_aux_data(self):
		aux_data = dict_merge(self.aux_data, *(p.aux_data for p in self.p))
		return aux_data

	




















class Casadi(Backend):

	def __init__(self, ocp):
		raise NotImplementedError



class Sympy(Backend):

	def __init__(self, ocp):
		raise NotImplementedError



backend_dispatcher = {
	'pycollo': Pycollo,
	'casadi': Casadi,
	'sympy': Sympy,
	}