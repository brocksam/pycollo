"""The main way to define and interact with a Pycollo optimal control problem.

This module contains the main class that the user will interact with to define
and run their optimal control problem when working with Pycollo. Terminolgy is 
loosely defined in accordance with "Betts, JT (2010). Practical Methods for 
Optimal Control and Estimiation Using Nonlinear Programming (Second Edition)".
See the ``Notes`` section for a full list of symbols used.

Notes:
------
	* t: independent parameter (time).
	* x = [y, u, q, t0, tf, s]: vector of problem variables.
	* y: vector state variables (which are functions of time).
	* u: vector control variables (which are functions of time).
	* q: vector of integral constraints.
	* t0: the initial time of the (single) phase.
	* tf: the final time of the (single) phase.
	* s: vector of static parameter variables (which are phase-independent).

	* J: objective function.
	* g: gradient of the objective function w.r.t. x.
	* L: Lagrangian of the objective function and constraints.
	* H: Hessian of the Lagrangian.

	* c = [zeta, gamma, rho, beta]: vector of constraints.
	* zeta: vector of defect constraints.
	* gamma: vector of path constraints.
	* rho: vector of integral constraints.
	* beta: vector of endpoint constraints.
	* G: Jacobian of the constaints.

	* n = len(x): number of free variables
	* m = len(c): number of constraints
"""


import itertools
from typing import (AnyStr, Iterable, Optional, Tuple, TypeVar, Union)
from timeit import default_timer as timer

import numba as nb
import numpy as np
from ordered_set import OrderedSet
import scipy.sparse as sparse
import sympy as sym
import sympy.physics.mechanics as me

from .bounds import Bounds
from .expression_graph import ExpressionGraph
from .guess import Guess
from .iteration import Iteration
from .mesh import Mesh
from .numbafy import numbafy
from .phase import Phase
from .quadrature import Quadrature
from .typing import (OptionalSymsType, TupleSymsType)
from .scaling import Scaling
from .settings import Settings
from .utils import (check_sym_name_clash, format_as_named_tuple)


__all__ = ["OptimalControlProblem"]


class OptimalControlProblem():
	"""The main class for Pycollo optimal control problems

	Attributes:
	"""

	_t0_USER = sym.Symbol('t0')
	_tF_USER = sym.Symbol('tF')
	_t0 = sym.Symbol('_t0')
	_tF = sym.Symbol('_tF')

	_STRETCH = 0.5 * (_tF - _t0)
	_SHIFT = 0.5 * (_t0 + _tF)

	_dSTRETCH_dt = np.array([-0.5, 0.5])

	def __init__(self, 
			name, 
			# state_variables=None, 
			# control_variables=None, 
			parameter_variables=None, 
			# state_equations=None, 
			*, 
			# bounds=None, 
			scaling=None, 
			# initial_guess=None, 
			# initial_mesh=None, 
			# path_constraints=None, 
			# integrand_functions=None, 
			# state_endpoint_constraints=None, 
			boundary_constraints=None, 
			objective_function=None, 
			settings=None, 
			auxiliary_data=None,
			):
		"""Initialise the optimal control problem with user-passed objects.

		Args:
			phases (:obj:`Iterable` of :obj:`Phase`, optional): Phases to be 
				associated with the optimal control problem at initialisation. 
				Defaults to None.
			parameter_variables ()
		"""

		# Problem name
		self.name = name

		# Set settings
		self.settings = settings

		# Initialise problem description
		self._init_private_attributes()
		self._init_user_options(
			# state_variables,
			# control_variables, 
			parameter_variables,
			# state_equations,
			# path_constraints, 
			# integrand_functions,
			# state_endpoint_constraints, 
			boundary_constraints,
			objective_function,
			auxiliary_data,
			# bounds,
			scaling,
			# initial_guess,
			)
		# self._init_initial_mesh(initial_mesh)	

	def _init_private_attributes(self):

		# Flags
		self._is_initialised = False
		self._forward_dynamics = False

		# Variables
		# self._y_vars_user = ()
		# self._u_vars_user = ()
		# self._q_vars_user = ()
		# self._t_vars_user = (self._t0_USER, self._tF_USER)
		self._s_vars_user = ()

		# Constraints
		# self._y_eqns_user = ()
		# self._c_cons_user = ()
		# self._q_funcs_user = ()
		# self._y_b_cons_user = ()
		self._b_cons_user = ()

		# Phases
		self._phases = []

	def _init_user_options(self, 
			# state_variables, 
			# control_variables, 
			parameter_variables, 
			# state_equations, 
			# path_constraints, 
			# integrand_functions, 
			# state_endpoint_constraints, 
			boundary_constraints, 
			objective_function, 
			auxiliary_data, 
			# bounds, 
			scaling, 
			# initial_guess,
			):

		# Variables
		# self.state_variables = state_variables
		# self.control_variables = control_variables
		# self.parameter_variables = parameter_variables

		# Constraints
		# self.state_equations = state_equations
		# self.path_constraints = path_constraints
		# self.integrand_functions = integrand_functions
		# self.state_endpoint_constraints = state_endpoint_constraints
		self.boundary_constraints = boundary_constraints

		# Functions
		self.objective_function = objective_function

		self.auxiliary_data = dict(auxiliary_data) if auxiliary_data else {}
		self.scaling = scaling
		# self.bounds = bounds
		# self.initial_guess = initial_guess

	def _init_initial_mesh(self, initial_mesh):
		self._mesh_iterations = []
		if initial_mesh is None:
			initial_mesh = Mesh(optimal_control_problem=self)
		else:
			initial_mesh._ocp = self
		initial_iteration = Iteration(optimal_control_problem=self, 
			iteration_number=1, mesh=initial_mesh)
		self._mesh_iterations
		self._mesh_iterations.append(initial_iteration)

	@property
	def name(self) -> str:
		"""The name associated with the optimal control problem. For setter 
		behaviour, the supplied `name` is cast to a str.

		The name is not strictly needed, however it improves the usefulness of 
		Pycollo console output. This is particularly useful in cases where the 
		user may wish to instantiate multiple :obj:`OptimalControlProblem` 
		objects within a single script, or instantiates other Pycollo objects 
		without providing a valid `optimal_control_problem` argument for them 
		to be linked to at initialisation.
		"""
		return self._name
	
	@name.setter
	def name(self, name: AnyStr):
		self._name = str(name)

	@property
	def phases(self) -> Tuple[Phase, ...]:
		"""A tuple of all phases associated with the optimal control problem.

		Phase numbers (`Phase.number`) are integers beginning at 1 and are 
		ordered corresponding to the order that they were added to the optimal 
		control problem. As Python uses zero-based indexing the phase numbers 
		do not directly map to the indexes of phases within `self.phases`. 
		Phases are however ordered sequentially corresponding to the 
		cronological order they were added to the optimal control problem.
		"""
		return tuple(self._phases)

	def add_phase(self, phase: Iterable[Phase]) -> Phase:
		"""Add an already instantiated `Phase` to this optimal control problem.

		This method is needed as `self.phases` is read only ("private") and 
		therefore users cannot manually add `Phase` objects to an optimal 
		control problem. `self.phases` is required to be read only as it is an 
		iterable of `Phase` objects and must be protected from accidental errors 
		introduced by user interacting with it incorrectly.

		Args:
			phase (Phase): The phase to be added to the optimal control problem

		Returns:
			Phase: the phase that has been added. It is the same 
		"""
		phase.optimal_control_problem = self
		return self.phases[-1]

	def add_phases(self, phases: Iterable[Phase]) -> Tuple[Phase, ...]:
		"""Associate multiple already instantiated `Phase` objects.

		This is a convinience method to allow the user to add multiple `Phase` 
		objects to the optimal control problem in one go.
		"""
		return tuple(self.add_phase(phase) for phase in phases)
	
	def new_phase(self, state_variables: OptionalSymsType = None,
			control_variables: OptionalSymsType = None) -> Phase:
		"""Create a new :obj:`Phase` and add to this optimal control problem.

		Provides the same behaviour as manually creating a :obj:`Phase` called 
		`phase` and calling `self.add_phase(phase)`.
		"""
		new_phase = Phase(optimal_control_problem=self, 
			state_variables=state_variables)
		return new_phase

	def new_phase_like(self, like, *, 
			copy_state_variables=True,
			copy_control_variables=True,
			copy_state_equations=True,
			copy_path_constraints=True,
			copy_integrand_functions=True,
			copy_state_endpoint_constraints=True,
			copy_bounds=True,
			copy_initial_mesh=True,
			copy_scaling=False,
			copy_initial_guess=False,
			):
		new_phase = Phase(optimal_control_problem=self)

		if copy_state_variables:
			new_phase.state_variables = like.state_variables
			if copy_bounds:
				new_phase.bounds.state_variables = like.bounds.state_variables

		if copy_control_variables:
			new_phase.control_variables = like.control_variables
			if copy_bounds:
				new_phase.bounds.control_variables = like.bounds.control_variables

		if copy_state_equations:
			new_phase.state_equations = like.state_equations

		if copy_path_constraints:
			new_phase.path_constraints = like.path_constraints
			if copy_bounds:
				new_phase.bounds.path_constraints = like.bounds.path_constraints

		if copy_integrand_functions:
			new_phase.integrand_functions = like.integrand_functions
			if copy_bounds:
				new_phase.bounds.integral_variables = like.bounds.integral_variables

		if copy_state_endpoint_constraints:
			new_phase.state_endpoint_constraints = like.state_endpoint_constraints
			if copy_bounds:
				new_phase.bounds.state_endpoint_constraints = like.bounds.state_endpoint_constraints

		return new_phase

	def new_phases_like(self, like, number, **kwargs) -> Tuple[Phase, ...]:
		"""Creates multiple new phases like an already instantiated phase.

		For a list of key word arguments and default values see the docstring 
		for the `OptimalControlProblem.new_phase_like` method.

		Returns:
			The newly instantiated and associated phases.
		"""
		new_phases = (self.new_phase_like(like, **kwargs)
			for _ in range(int(number)))
		return new_phases

	@property
	def number_phases(self) -> int:
		"""Number of phases associated with this optimal control problem."""
		return len(self.phases)

	@property
	def time_symbol(self):
		"""
		
		Raises:
			NotImplementedError: Whenever called to inform the user that these 
				types of problem are not currently supported.
		"""
		msg = (f"Pycollo do not currently support dynamic, path or integral "
			f"constraints that are explicit functions of continuous time.")
		raise NotImplementedError(msg)

	# @property
	# def initial_time(self):
	# 	return self._t0_USER

	# @property
	# def final_time(self):
	# 	return self._tF_USER

	# @property
	# def initial_state(self):
	# 	return self._y_t0_user
	
	# @property
	# def final_state(self):
	# 	return self._y_tF_user

	# @property
	# def state_endpoint(self):
	# 	state_endpoint = tuple(itertools.chain.from_iterable(y 
	# 		for y in zip(self._y_t0_user, self._y_tF_user)))
	# 	return state_endpoint

	# @property
	# def state_variables(self):
	# 	return self._y_vars_user

	# @state_variables.setter
	# def state_variables(self, y_vars):
	# 	self._is_initialised = False
	# 	self._y_vars_user = format_as_tuple(y_vars)
	# 	self._y_t0_user = tuple(sym.Symbol(f'{y}(t0)')
	# 		for y in self._y_vars_user)
	# 	self._y_tF_user = tuple(sym.Symbol(f'{y}(tF)')
	# 		for y in self._y_vars_user)
	# 	self._update_vars()
	# 	_ = check_sym_name_clash(self._y_vars_user)

	# @property
	# def number_state_variables(self):
	# 	if self._bounds._bounds_checked:
	# 		return self._bounds._y_needed.sum()
	# 	else:
	# 		return len(self._y_vars_user)
	
	# @property
	# def control_variables(self):
	# 	return self._u_vars_user

	# @control_variables.setter
	# def control_variables(self, u_vars):
	# 	self._is_initialised = False
	# 	self._u_vars_user = format_as_tuple(u_vars)
	# 	self._update_vars()
	# 	_ = check_sym_name_clash(self._u_vars_user)

	# @property
	# def number_control_variables(self):
	# 	if self._bounds._bounds_checked:
	# 		return self._bounds._u_needed.sum()
	# 	else:
	# 		return len(self._u_vars_user)

	# @property
	# def integral_variables(self):
	# 	return self._q_vars_user

	# @property
	# def number_integral_variables(self):
	# 	if self._bounds._bounds_checked:
	# 		return self._bounds._q_needed.sum()
	# 	else:
	# 		return len(self._q_vars_user)

	# @property
	# def time_variables(self):
	# 	return self._t_vars_user

	# @property
	# def number_time_variables(self):
	# 	if self._bounds._bounds_checked:
	# 		return self._bounds._t_needed.sum()
	# 	else:
	# 		return len(self._t_vars_user)
	
	@property
	def parameter_variables(self):
		return self._s_vars_user

	@parameter_variables.setter
	def parameter_variables(self, s_vars):
		self._is_initialised = False
		s_vars = format_as_tuple(s_vars)
		self._s_vars_user = tuple(s_vars)
		self._update_vars()
		_ = check_sym_name_clash(self._s_vars_user)

	@property
	def number_parameter_variables(self):
		if self._bounds._bounds_checked:
			return self._bounds._s_needed.sum()
		else:
			return len(self._s_vars_user)

	# @property
	# def variables(self):
	# 	return self._x_vars_user

	# @property
	# def number_variables(self):
	# 	if self._bounds._bounds_checked:
	# 		return self._bounds._x_needed.sum()
	# 	else:
	# 		return len(self._x_vars_user)

	# def _update_vars(self):
	# 	self._x_vars_user = tuple(self._y_vars_user + self._u_vars_user
	# 		+ self._q_vars_user + self._t_vars_user + self._s_vars_user)
	# 	self._x_b_vars_user = tuple(self.state_endpoint 
	# 		+ self._q_vars_user + self._t_vars_user + self._s_vars_user)

	# @property
	# def state_equations(self):
	# 	return self._y_eqns_user

	# @state_equations.setter
	# def state_equations(self, y_eqns):
	# 	self._is_initialised = False
	# 	y_eqns = format_as_tuple(y_eqns)
	# 	self._y_eqns_user = tuple(y_eqns)

	# @property
	# def number_state_equations(self):
	# 	return len(self._y_eqns_user)

	# @property
	# def path_constraints(self):
	# 	return self._c_cons_user

	# @path_constraints.setter
	# def path_constraints(self, c_cons):
	# 	self._is_initialised = False
	# 	c_cons = format_as_tuple(c_cons)
	# 	self._c_cons_user = tuple(c_cons)

	# @property
	# def number_path_constraints(self):
	# 	return len(self._c_cons_user)

	# @property
	# def integrand_functions(self):
	# 	return self._q_funcs_user

	# @integrand_functions.setter
	# def integrand_functions(self, integrands):
	# 	self._is_initialised = False
	# 	self._q_funcs_user = format_as_tuple(integrands)
	# 	self._q_vars_user = tuple(sym.Symbol(f'_q{i_q}') 
	# 		for i_q, _ in enumerate(self._q_funcs_user))
	# 	self._update_vars()

	# @property
	# def number_integrand_functions(self):
	# 	return len(self._q_funcs_user)

	# @property
	# def state_endpoint_constraints(self):
	# 	return self._y_b_cons_user

	# @state_endpoint_constraints.setter
	# def state_endpoint_constraints(self, y_b_cons):
	# 	self._is_initialised = False
	# 	y_b_cons = format_as_tuple(y_b_cons)
	# 	self._y_b_cons_user = tuple(y_b_cons)

	# @property
	# def number_state_endpoint_constraints(self):
	# 	return len(self._y_b_cons_user)

	@property
	def endpoint_constraints(self):
		return self._b_cons_user

	@endpoint_constraints.setter
	def endpoint_constraints(self, b_cons):
		self._is_initialised = False
		b_cons = format_as_tuple(b_cons)
		self._b_cons_user = tuple(b_cons)

	@property
	def number_endpoint_constraints(self):
		return len(self._b_cons_user)

	@property
	def number_constraints(self):
		return (self.number_state_equations 
			+ self.number_path_constraints 
			+ self.number_integrand_functions 
			+ self.number_state_endpoint_constraints 
			+ self.number_endpoint_constraints)

	@property
	def objective_function(self):
		return self._J_user

	@objective_function.setter
	def objective_function(self, J):
		self._is_initialised = False
		self._J_user = sym.sympify(J)
		self._forward_dynamics = True if self._J_user == 1 else False

	@property
	def auxiliary_data(self):
		return self._aux_data_user
	
	@auxiliary_data.setter
	def auxiliary_data(self, aux_data):
		self._is_initialised = False
		self._aux_data_user = dict(aux_data)

	@property
	def endpoint_bounds(self):
		return self._endpoint_bounds
	
	@endpoint_bounds.setter
	def endpoint_bounds(self, bounds):
		if bounds is None:
			self._endpoint_bounds = EndpointBounds(optimal_control_problem=self)
		else:
			self._endpoint_bounds = bounds
			self._endpoint_bounds._ocp = self

	@property
	def scaling(self):
		return self._scaling
	
	@scaling.setter
	def scaling(self, scaling):
		if scaling is None:
			self._scaling = Scaling(optimal_control_problem=self)
		else:
			self._scaling = scaling
			self._scaling._ocp = self

	# @property
	# def initial_guess(self):
	# 	return self._initial_guess
	
	# @initial_guess.setter
	# def initial_guess(self, guess):
	# 	self._is_initialised = False
	# 	if guess is None:
	# 		self._initial_guess = Guess(optimal_control_problem=self)
	# 	else:
	# 		self._initial_guess = guess
	# 		self._initial_guess._ocp = self

	# @property
	# def initial_mesh(self):
	# 	return self._mesh_iterations[0]._mesh

	@property
	def mesh_iterations(self):
		return self._mesh_iterations
	
	@property
	def num_mesh_iterations(self):
		return len(self._mesh_iterations)

	@property
	def settings(self):
		return self._settings
	
	@settings.setter
	def settings(self, settings):
		self._is_initialised = False
		if settings is None:
			self._settings = Settings(optimal_control_problem=self)
		else:
			self._settings = settings
			self._settings._ocp = self

	@property
	def solution(self):
		return self._mesh_iterations[-1].solution

	@staticmethod
	def _console_out_message_heading(msg):
		msg_len = len(msg)
		seperator = '=' * msg_len
		output_msg = f"\n{seperator}\n{msg}\n{seperator}\n"
		print(output_msg)

	def initialise(self):
		"""Initialise the optimal control problem before solving.

		The initialisation of the optimal control problem involves the 
		following stages:

		- Determine the set of user-defined variables that are allowed in the 
		  continuous and endpoint functions.
		

		"""

		ocp_initialisation_time_start = timer()
		eom_msg = 'Initialising optimal control problem.'
		self._console_out_message_heading(eom_msg)
		self._check_state_equation_for_each_state_variable()
		self._check_user_supplied_bounds()
		self._generate_scaling()
		self._generate_expression_graph()
		self._generate_quadrature()
		self._compile_numba_functions()
		self._check_user_supplied_initial_guess()

		# Initialise the initial mesh iterations
		self._mesh_iterations[0]._initialise_iteration(self.initial_guess)

		ocp_initialisation_time_stop = timer()
		self._ocp_initialisation_time = (ocp_initialisation_time_stop 
			- ocp_initialisation_time_start)
		self._is_initialised = True

	def _check_state_equation_for_each_state_variable(self):
		if self.number_state_variables != self.number_state_equations:
			msg = ("A state equation must be supplied for each state variable. "
				f"Currently {self.number_state_equations} state equations are "
				f"supplied for {self.number_state_variables} state variables.")
			raise ValueError(msg)

	def _check_user_supplied_bounds(self):
		self._bounds._bounds_check()
		print('Bounds checked.')

	def _generate_expression_graph(self):
		self._generate_pycollo_symbols()
		self._create_variable_index_slices()
		self._organise_constraints()
		continuous_needed = self._bounds._x_needed.astype(bool)
		endpoint_needed = self._bounds._x_b_needed.astype(bool)
		continuous_vars_user = tuple(
			np.array(self._x_vars_user)[continuous_needed])
		endpoint_vars_user = tuple(
			np.array(self._x_b_vars_user)[endpoint_needed])
		user_variables = (continuous_vars_user, endpoint_vars_user)
		variables = (self._x_vars, self._x_b_vars)
		aux_data = {**self._aux_data, **self.auxiliary_data}
		constraints = (
			self.state_equations,
			self.path_constraints,
			self.integrand_functions,
			self.state_endpoint_constraints,
			self.endpoint_constraints,
			)
		self._expression_graph = ExpressionGraph(self, user_variables, 
			variables, aux_data, self.objective_function, constraints)

		# print('\n\n\n')
		# for constraint_group in constraints:
		# 	for var in continuous_vars_user:
		# 		print(sym.Matrix(constraint_group).diff(var))
		# print('\n\n\n')
		# raise NotImplementedError

	def _generate_scaling(self):
		self._scaling._generate()
		print('Scaling generated.')

	def _generate_pycollo_symbols(self):
		self._aux_data = {}
		self._generate_pycollo_y_vars()
		self._generate_pycollo_u_vars()
		self._generate_pycollo_q_vars()
		self._generate_pycollo_t_vars()
		self._generate_pycollo_s_vars()
		self._collect_pycollo_x_vars()
		print('Pycollo symbols generated.')

	def _generate_pycollo_y_vars(self):
		y_vars = [sym.Symbol(f'_y{i_y}') 
			for i_y, _ in enumerate(self._y_vars_user)]
		self._y_vars = sym.Matrix([y 
			for y, y_needed in zip(y_vars, self._bounds._y_needed) 
			if y_needed])
		if not self._y_vars:
			self._y_vars = sym.Matrix.zeros(0, 1)
		self._num_y_vars = self._y_vars.shape[0]
		self._y_t0 = sym.Matrix([sym.Symbol(f'{y}_t0') 
			for y in self._y_vars])
		self._y_tF = sym.Matrix([sym.Symbol(f'{y}_tF') 
			for y in self._y_vars])
		self._y_b_vars = sym.Matrix(list(itertools.chain.from_iterable(y 
			for y in zip(self._y_t0, self._y_tF))))
		self._aux_data.update({y: value 
			for y, y_needed, value in zip(
				y_vars, self._bounds._y_needed, self._bounds._y_l) 
			if not y_needed})

	def _generate_pycollo_u_vars(self):
		u_vars = [sym.Symbol(f'_u{i_u}') 
			for i_u, _ in enumerate(self._u_vars_user)]
		self._u_vars = sym.Matrix([u 
			for u, u_needed in zip(u_vars, self._bounds._u_needed) 
			if u_needed])
		if not self._u_vars:
			self._u_vars = sym.Matrix.zeros(0, 1)
		self._num_u_vars = self._u_vars.shape[0]
		self._aux_data.update({u: value 
			for u, u_needed, value in zip(
				u_vars, self._bounds._u_needed, self._bounds._u_l) 
			if not u_needed})

	def _generate_pycollo_q_vars(self):
		q_vars = sym.Matrix(self._q_vars_user)
		self._q_vars = sym.Matrix([q 
			for q, q_needed in zip(q_vars, self._bounds._q_needed) 
			if q_needed])
		if not self._q_vars:
			self._q_vars = sym.Matrix.zeros(0, 1)
		self._num_q_vars = self._q_vars.shape[0]
		self._aux_data.update({q: value 
			for q, q_needed, value in zip(
				q_vars, self._bounds._q_needed, self._bounds._q_l) 
			if not q_needed})

	def _generate_pycollo_t_vars(self):
		t_vars = [self._t0, self._tF]
		self._t_vars = sym.Matrix([t 
			for t, t_needed in zip(t_vars, self._bounds._t_needed) 
			if t_needed])
		if not self._t_vars:
			self._t_vars = sym.Matrix.zeros(0, 1)
		self._num_t_vars = self._t_vars.shape[0]
		if not self._bounds._t_needed[0]:
			self._aux_data.update({self._t0: self._bounds._t0_l})
		if not self._bounds._t_needed[1]:
			self._aux_data.update({self._tF: self._bounds._tF_l})

	def _generate_pycollo_s_vars(self):
		s_vars = [sym.Symbol(f'_s{i_s}') 
			for i_s, _ in enumerate(self._s_vars_user)]
		self._s_vars = sym.Matrix([s 
			for s, s_needed in zip(s_vars, self._bounds._s_needed) 
			if s_needed])
		if not self._s_vars:
			self._s_vars = sym.Matrix.zeros(0, 1)
		self._num_s_vars = self._s_vars.shape[0]
		self._aux_data.update({s: value 
			for s, s_needed, value in zip(
				s_vars, self._bounds._s_needed, self._bounds._s_l) 
			if not s_needed})

	def _collect_pycollo_x_vars(self):
		self._x_vars = sym.Matrix([
			self._y_vars, 
			self._u_vars, 
			self._q_vars, 
			self._t_vars, 
			self._s_vars,
			])
		self._num_vars = self._x_vars.shape[0]
		self._num_vars_tuple = (
			self._num_y_vars, 
			self._num_u_vars, 
			self._num_q_vars, 
			self._num_t_vars, 
			self._num_s_vars,
			)
		self._x_b_vars = sym.Matrix([
			self._y_b_vars, 
			self._q_vars, 
			self._t_vars, 
			self._s_vars,
			])
		self._num_point_vars = self._x_b_vars.shape[0]

	def _check_user_supplied_initial_guess(self):
		self._initial_guess._guess_check()

	def _create_variable_index_slices(self):
		self._y_slice = slice(0, self._num_y_vars)
		self._u_slice = slice(self._y_slice.stop, 
			self._y_slice.stop + self._num_u_vars)
		self._q_slice = slice(self._u_slice.stop, 
			self._u_slice.stop + self._num_q_vars)
		self._t_slice = slice(self._q_slice.stop, 
			self._q_slice.stop + self._num_t_vars)
		self._s_slice = slice(self._t_slice.stop, self._num_vars)
		self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)
		self._yu_qts_split = self._yu_slice.stop

		self._y_b_slice = slice(0, self._num_y_vars*2)
		self._u_b_slice = slice(self._y_b_slice.stop, self._y_b_slice.stop)
		self._q_b_slice = slice(self._u_b_slice.stop, 
			self._u_b_slice.stop + self._num_q_vars)
		self._t_b_slice = slice(self._q_b_slice.stop, 
			self._q_b_slice.stop + self._num_t_vars)
		self._s_b_slice = slice(self._t_b_slice.stop, 
			self._t_b_slice.stop + self._num_s_vars)
		self._qts_b_slice = slice(self._q_b_slice.start, self._s_b_slice.stop)
		self._y_b_qts_b_split = self._y_b_slice.stop

	def _organise_constraints(self):
		self._num_c = self.number_constraints

		self._c_defect_slice = slice(0, self.number_state_equations)
		self._c_path_slice = slice(self._c_defect_slice.stop, 
			self._c_defect_slice.stop + self.number_path_constraints)
		self._c_integral_slice = slice(self._c_path_slice.stop, 
			self._c_path_slice.stop + self.number_integrand_functions)
		self._c_state_endpoint_slice = slice(self._c_integral_slice.stop,
			self._c_integral_slice.stop 
				+ self.number_state_endpoint_constraints)
		self._c_endpoint_slice = slice(self._c_state_endpoint_slice.start, 
			self._c_state_endpoint_slice.stop 
				+ self.number_endpoint_constraints)
		self._c_continuous_slice = slice(0, 
			self._num_c - self.number_state_endpoint_constraints 
			- self.number_endpoint_constraints)
		self._c_boundary_slice = slice(self._c_state_endpoint_slice.start, 
			self._c_endpoint_slice.stop)

	def _generate_quadrature(self):
		self._quadrature = Quadrature(optimal_control_problem=self)
		print('Quadrature scheme initialised.')

	def _compile_numba_functions(self):

		self._compile_reshape()
		self._compile_objective()
		self._compile_objective_gradient()
		self._compile_constraints()
		self._compile_jacobian_constraints()
		if self.settings.derivative_level == 2:
			self._compile_hessian_lagrangian()

	def _compile_reshape(self):

		def reshape_x(x, num_yu, yu_qts_split):
			x = np.array(x)
			yu = x[:yu_qts_split].reshape(num_yu, -1)
			qts = x[yu_qts_split:].reshape(len(x) - yu_qts_split, )
			x_tuple = (*yu, *qts)
			return x_tuple

		def reshape_x_point(x, x_endpoint_indices):
			x = np.array(x)
			x_tuple = x[x_endpoint_indices]
			return x_tuple

		self._x_reshape_lambda = reshape_x
		self._x_reshape_lambda_point = reshape_x_point
		print('Variable reshape functions compiled.')

	def _compile_objective(self):

		def objective_lambda(x_reshaped_point):
			J = J_lambda(*x_reshaped_point)
			return J

		expr_graph = self._expression_graph

		J_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.J,
			precomputable_nodes=expr_graph.J_precomputable,
			dependent_tiers=expr_graph.J_dependent_tiers,
			parameters=self._x_b_vars,
			)
		self._J_lambda = objective_lambda
		print('Objective function compiled.')

	def _compile_objective_gradient(self):

		def objective_gradient_lambda(x_tuple_point, N):
			g = dJ_dxb_lambda(*x_tuple_point, N)
			return g

		expr_graph = self._expression_graph

		dJ_dxb_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.dJ_dxb,
			precomputable_nodes=expr_graph.dJ_dxb_precomputable,
			dependent_tiers=expr_graph.dJ_dxb_dependent_tiers,
			parameters=self._x_b_vars,
			return_dims=1, 
			N_arg=True, 
			endpoint=True, 
			ocp_num_vars=self._num_vars_tuple
			)

		self._g_lambda = objective_gradient_lambda
		print('Objective gradient function compiled.')

	def _compile_constraints(self):
		
		def defect_constraints_lambda(y, dy, A, D, stretch):
			return (D.dot(y.T) + stretch*A.dot(dy.T)).flatten(order='F')

		def path_constraints_lambda(p):
			return p

		def integral_constraints_lambda(q, g, W, stretch):
			return q - stretch*np.matmul(g, W) if g.size else q

		def endpoint_constraints_lambda(b):
			return b

		def constraints_lambda(x_tuple, x_tuple_point, N, ocp_y_slice, 
				ocp_q_slice, num_c, dy_slice, p_slice, g_slice, defect_slice, 
				path_slice, integral_slice, boundary_slice, A, D, W):
			
			y = np.vstack(x_tuple[ocp_y_slice])
			q = np.array(x_tuple[ocp_q_slice])

			stretch = t_stretch_lambda(*x_tuple)
			c_continuous = c_continuous_lambda(*x_tuple, N)
			c_endpoint = c_endpoint_lambda(*x_tuple_point, N)

			dy = c_continuous[dy_slice].reshape((-1, N))
			p = c_continuous[p_slice]
			g = c_continuous[g_slice]
			b = c_endpoint

			c = np.empty(num_c)
			c[defect_slice] = defect_constraints_lambda(y, dy, A, D, stretch)
			c[path_slice] = path_constraints_lambda(p)
			c[integral_slice] = integral_constraints_lambda(q, g, W, stretch)
			c[boundary_slice] = endpoint_constraints_lambda(b)

			return c

		expr_graph = self._expression_graph

		t_stretch_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.t_norm, 
			precomputable_nodes=expr_graph.t_norm_precomputable,
			dependent_tiers=expr_graph.t_norm_dependent_tiers,
			parameters=self._x_vars, 
			)

		c_continuous_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.c,
			expression_nodes=expr_graph.c_nodes,
			precomputable_nodes=expr_graph.c_precomputable,
			dependent_tiers=expr_graph.c_dependent_tiers,
			parameters=self._x_vars,
			return_dims=2, 
			N_arg=True, 
			ocp_num_vars=self._num_vars_tuple,
			)

		c_endpoint_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.b,
			precomputable_nodes=expr_graph.b_precomputable,
			dependent_tiers=expr_graph.b_dependent_tiers,
			parameters=self._x_b_vars,
			return_dims=1,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		self._t_stretch_lambda = t_stretch_lambda
		self._dstretch_dt = [val 
			for val, t_needed in zip(self._dSTRETCH_dt, self._bounds._t_needed) 
			if t_needed]
		self._c_continuous_lambda = c_continuous_lambda
		self._c_lambda = constraints_lambda
		print('Constraints function compiled.')

		# print(expr_graph.c)
		# print('\n\n\n')
		# raise NotImplementedError

	def _compile_jacobian_constraints(self):

		def A_x_dot_sparse(A_sparse, ddy_dx, num_y, num_nz, stretch, 
			return_array):
			for i, row in enumerate(ddy_dx):
				start = i*num_nz
				stop = start + num_nz
				entries = stretch*A_sparse.multiply(row).data
				return_array[start:stop] = entries
			return return_array

		def G_dzeta_dy_lambda(ddy_dy, stretch, A, D, A_row_col_array, num_y, 
			dzeta_dy_D_nonzero, dzeta_dy_slice):
			num_nz = A_row_col_array.shape[1]
			return_array = np.empty(num_y**2 * num_nz)
			G_dzeta_dy = A_x_dot_sparse(A, ddy_dy, num_y, num_nz, stretch, 
				return_array)
			D_data = (D.data * np.ones((num_y, 1))).flatten()
			G_dzeta_dy[dzeta_dy_D_nonzero] += D_data
			return G_dzeta_dy

		def G_dzeta_du_lambda(ddy_du, stretch, A, A_row_col_array, num_y, 
			num_u):
			num_nz = A_row_col_array.shape[1]
			return_array = np.empty(num_u*num_y*num_nz)
			G_dzeta_du = A_x_dot_sparse(A, ddy_du, num_u, num_nz, stretch, 
				return_array)
			return G_dzeta_du

		def G_dzeta_dt_lambda(dy, A):
			A_dy_flat = (A.dot(dy.T).flatten(order='F'))
			product = np.outer(self._dstretch_dt, A_dy_flat)
			G_dzeta_dt = product.flatten(order='F')
			return G_dzeta_dt

		def G_dzeta_ds_lambda(ddy_ds, stretch, A):
			G_dzeta_ds = stretch*A.dot(ddy_ds.T).flatten(order='F')
			return G_dzeta_ds

		def G_dgamma_dy_lambda(dgamma_dy):
			if dgamma_dy.size:
				return dgamma_dy.flatten()
			else:
				return []

		def G_dgamma_du_lambda(dgamma_du):
			if dgamma_du.size:
				return dgamma_du.flatten()
			else:
				return []

		def G_dgamma_dt_lambda(p):
			if p.size:
				return np.zeros_like(p).flatten()
			else:
				return []

		def G_dgamma_ds_lambda(dgamma_ds):
			if dgamma_ds.size:
				return dgamma_ds.flatten()
			else:
				return []

		def G_drho_dy_lambda(drho_dy, stretch, W):

			if drho_dy.size:
				G_drho_dy = (- stretch * drho_dy * W).flatten()
				return G_drho_dy
			else:
				return []

		def G_drho_du_lambda(drho_du, stretch, W):
			if drho_du.size:
				G_drho_du = (- stretch * drho_du * W).flatten()
				return G_drho_du
			else:
				return []

		def G_drho_dt_lambda(g, W):
			if g.size > 0:
				product = np.outer(self._dstretch_dt, np.matmul(g, W))
				return - product.flatten(order='F')
			else:
				return []

		def G_drho_ds_lambda(drho_ds, stretch, W):
			if drho_ds.size:
				G_drho_ds = (- stretch * np.matmul(drho_ds, W))
				return G_drho_ds
			else:
				return []

		def jacobian_lambda(x_tuple, x_tuple_point, N, num_G_nonzero, 
			num_x_ocp, A, D, W, A_row_col_array, dy_slice, 
			p_slice, g_slice, dzeta_dy_D_nonzero, 
			dzeta_dy_slice, dzeta_du_slice, 
			dzeta_dt_slice, dzeta_ds_slice, dgamma_dy_slice, dgamma_du_slice, 
			dgamma_dt_slice, dgamma_ds_slice, drho_dy_slice, drho_du_slice, 
			drho_dq_slice, drho_dt_slice, drho_ds_slice, dbeta_dxb_slice):

			stretch = t_stretch_lambda(*x_tuple)
			dstretch_dt = self._dstretch_dt
			c_continuous = c_continuous_lambda(*x_tuple, N)
			dc_dx = dc_dx_lambda(*x_tuple, N)
			dy = c_continuous[dy_slice].reshape((-1, N))
			p = c_continuous[p_slice]
			g = c_continuous[g_slice]

			dzeta_dy = dc_dx[dc_dx_slice.zeta_y].reshape(dc_dx_shape.zeta_y, N)
			dzeta_du = dc_dx[dc_dx_slice.zeta_u].reshape(dc_dx_shape.zeta_u, N)
			dzeta_dt = None
			dzeta_ds = dc_dx[dc_dx_slice.zeta_s].reshape(dc_dx_shape.zeta_s, N)
			dgamma_dy = dc_dx[dc_dx_slice.gamma_y].reshape(dc_dx_shape.gamma_y, N)
			dgamma_du = dc_dx[dc_dx_slice.gamma_u].reshape(dc_dx_shape.gamma_u, N)
			dgamma_dt = None
			dgamma_ds = dc_dx[dc_dx_slice.gamma_s].reshape(dc_dx_shape.gamma_s, N)
			drho_dy = dc_dx[dc_dx_slice.rho_y].reshape(dc_dx_shape.rho_y, N)
			drho_du = dc_dx[dc_dx_slice.rho_u].reshape(dc_dx_shape.rho_u, N)
			drho_dt = None
			drho_ds = dc_dx[dc_dx_slice.rho_s].reshape(dc_dx_shape.rho_s, N)

			G = np.empty(num_G_nonzero)

			if num_x_ocp.y:
				G[dzeta_dy_slice] = G_dzeta_dy_lambda(dzeta_dy, stretch, A, D, 
					A_row_col_array, num_x_ocp.y, dzeta_dy_D_nonzero, 
					dzeta_dy_slice)
				G[dgamma_dy_slice] = G_dgamma_dy_lambda(dgamma_dy)
				G[drho_dy_slice] = G_drho_dy_lambda(drho_dy, stretch, W)

			if num_x_ocp.u:
				G[dzeta_du_slice] = G_dzeta_du_lambda(dzeta_du, stretch, A, 
					A_row_col_array, num_x_ocp.y, num_x_ocp.u)
				G[dgamma_du_slice] = G_dgamma_du_lambda(dgamma_du)
				G[drho_du_slice] = G_drho_du_lambda(drho_du, stretch, W)

			if num_x_ocp.q:
				G[drho_dq_slice] = 1

			if num_x_ocp.t:
				G[dzeta_dt_slice] = G_dzeta_dt_lambda(dy, A)
				G[dgamma_dt_slice] = G_dgamma_dt_lambda(p)
				G[drho_dt_slice] = G_drho_dt_lambda(g, W)

			if num_x_ocp.s:
				G[dzeta_ds_slice] = G_dzeta_ds_lambda(dzeta_ds, stretch, A)
				G[dgamma_ds_slice] = G_dgamma_ds_lambda(dgamma_ds)
				G[drho_ds_slice] = G_drho_ds_lambda(drho_ds, stretch, W)

			G[dbeta_dxb_slice] = db_dxb_lambda(*x_tuple_point, N)

			return G

		expr_graph = self._expression_graph

		t_stretch_lambda = self._t_stretch_lambda
		dstretch_dt = None

		c_continuous_lambda = self._c_continuous_lambda

		dc_dx_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.dc_dx,
			expression_nodes=expr_graph.dc_dx_nodes,
			precomputable_nodes=expr_graph.dc_dx_precomputable,
			dependent_tiers=expr_graph.dc_dx_dependent_tiers,
			parameters=self._x_vars, 
			return_dims=3, 
			N_arg=True, 
			ocp_num_vars=self._num_vars_tuple,
			)

		dc_dx_slice = pu.dcdxInfo(
			zeta_y=(self._c_defect_slice, self._y_slice),
			zeta_u=(self._c_defect_slice, self._u_slice),
			zeta_s=(self._c_defect_slice, self._s_slice),
			gamma_y=(self._c_path_slice, self._y_slice),
			gamma_u=(self._c_path_slice, self._u_slice),
			gamma_s=(self._c_path_slice, self._s_slice),
			rho_y=(self._c_integral_slice, self._y_slice),
			rho_u=(self._c_integral_slice, self._u_slice),
			rho_s=(self._c_integral_slice, self._s_slice),
			)

		dc_dx_shape = pu.dcdxInfo(
			zeta_y=(self.number_state_equations*self.number_state_variables),
			zeta_u=(self.number_state_equations*self.number_control_variables),
			zeta_s=(self.number_state_equations*self.number_parameter_variables),
			gamma_y=(self.number_path_constraints*self.number_state_variables),
			gamma_u=(self.number_path_constraints*self.number_control_variables),
			gamma_s=(self.number_path_constraints*self.number_parameter_variables),
			rho_y=(self.number_integrand_functions*self.number_state_variables),
			rho_u=(self.number_integrand_functions*self.number_control_variables),
			rho_s=(self.number_integrand_functions*self.number_parameter_variables),
			)

		db_dxb_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.db_dxb,
			precomputable_nodes=expr_graph.db_dxb_precomputable,
			dependent_tiers=expr_graph.db_dxb_dependent_tiers,
			parameters=self._x_b_vars, 
			return_dims=1, 
			N_arg=True, 
			ocp_num_vars=self._num_vars_tuple,
			)

		self._G_lambda = jacobian_lambda
		print('Jacobian function compiled.')

	def _compile_hessian_lagrangian(self):

		def H_point_lambda(ddL_dxbdxb):
			vals = np.tril(ddL_dxbdxb).flatten()
			H = vals[vals != 0]
			return H

		def H_defect_lambda(ddL_dxdx, A, sum_flag):
			H = np.array([])
			for matrix, flag in zip(ddL_dxdx, sum_flag):
				if flag:
					vals = A.multiply(matrix).sum().flatten()
				else:
					vals = np.array(A.multiply(matrix).sum(axis=0)).flatten()
				H = np.concatenate([H, vals])
			return H

		def H_path_lambda(ddL_dxdx):
			H = np.array([])
			for matrix in ddL_dxdx:
				vals = np.diag(matrix).flatten()
				H = np.concatenate([H, vals])
			return H

		def H_integral_lambda(ddL_dxdx, W, sum_flag):
			H = np.array([])
			for row, flag in zip(ddL_dxdx, sum_flag):
				if flag:
					vals = np.array([-np.dot(W, row)])
				else:
					vals = -np.multiply(W, row).flatten()
				H = np.concatenate([H, vals])
			return H

		def H_lambda(x_tuple, x_tuple_point, sigma, zeta_lagrange, 
			gamma_lagrange, rho_lagrange, beta_lagrange, N, num_nonzero, 
			objective_index, defect_index, path_index, integral_index,
			endpoint_index, A, W, defect_sum_flag, integral_sum_flag):

			ddL_objective_dxbdxb = ddL_objective_dxbdxb_lambda(*x_tuple_point, 
				sigma, N)
			ddL_defect_dxdx = ddL_defect_dxdx_lambda(*x_tuple, 
				*zeta_lagrange, N)
			ddL_path_dxdx = ddL_path_dxdx_lambda(*x_tuple, *gamma_lagrange, N)
			ddL_integral_dxdx = ddL_integral_dxdx_lambda(*x_tuple,
				*rho_lagrange, N)
			ddL_endpoint_dxbdxb = ddL_endpoint_dxbdxb_lambda(*x_tuple_point, 
				*beta_lagrange, N)

			H = np.zeros(num_nonzero)

			H[objective_index] += H_point_lambda(ddL_objective_dxbdxb)
			H[defect_index] += H_defect_lambda(ddL_defect_dxdx, A, 
				defect_sum_flag)
			if self.number_path_constraints and path_index:
				H[path_index] += H_path_lambda(ddL_path_dxdx)
			H[integral_index] += H_integral_lambda(ddL_integral_dxdx, W, 
				integral_sum_flag)
			H[endpoint_index] += H_point_lambda(ddL_endpoint_dxbdxb)

			return H

		expr_graph = self._expression_graph
		L_sigma = expr_graph.lagrange_syms[0]
		L_syms = expr_graph.lagrange_syms[1:]

		ddL_objective_dxbdxb_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.ddL_J_dxbdxb,
			expression_nodes=expr_graph.ddL_J_dxbdxb_nodes,
			precomputable_nodes=expr_graph.ddL_J_dxbdxb_precomputable,
			dependent_tiers=expr_graph.ddL_J_dxbdxb_dependent_tiers,
			parameters=self._x_b_vars,
			lagrange_parameters=L_sigma,
			return_dims=2,
			endpoint=True,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		ddL_defect_dxdx_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.ddL_zeta_dxdx,
			expression_nodes=expr_graph.ddL_zeta_dxdx_nodes,
			precomputable_nodes=expr_graph.ddL_zeta_dxdx_precomputable,
			dependent_tiers=expr_graph.ddL_zeta_dxdx_dependent_tiers,
			parameters=self._x_vars,
			lagrange_parameters=L_syms[self._c_defect_slice],
			return_dims=2,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		ddL_path_dxdx_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.ddL_gamma_dxdx,
			expression_nodes=expr_graph.ddL_gamma_dxdx_nodes,
			precomputable_nodes=expr_graph.ddL_gamma_dxdx_precomputable,
			dependent_tiers=expr_graph.ddL_gamma_dxdx_dependent_tiers,
			parameters=self._x_vars,
			lagrange_parameters=L_syms[self._c_path_slice],
			return_dims=2,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		ddL_integral_dxdx_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.ddL_rho_dxdx,
			expression_nodes=expr_graph.ddL_rho_dxdx_nodes,
			precomputable_nodes=expr_graph.ddL_rho_dxdx_precomputable,
			dependent_tiers=expr_graph.ddL_rho_dxdx_dependent_tiers,
			parameters=self._x_vars,
			lagrange_parameters=L_syms[self._c_integral_slice],
			return_dims=2,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		ddL_endpoint_dxbdxb_lambda = numbafy(
			expression_graph=expr_graph,
			expression=expr_graph.ddL_b_dxbdxb,
			expression_nodes=expr_graph.ddL_b_dxbdxb_nodes,
			precomputable_nodes=expr_graph.ddL_b_dxbdxb_precomputable,
			dependent_tiers=expr_graph.ddL_b_dxbdxb_dependent_tiers,
			parameters=self._x_b_vars,
			lagrange_parameters=L_syms[self._c_endpoint_slice],
			return_dims=2,
			endpoint=True,
			N_arg=True,
			ocp_num_vars=self._num_vars_tuple,
			)

		ocp_defect_slice = self._c_defect_slice
		ocp_path_slice = self._c_path_slice
		ocp_integral_slice = self._c_integral_slice
		ocp_endpoint_slice = self._c_endpoint_slice

		self._H_lambda = H_lambda
		print('Hessian function compiled.')

		# print(self._x_vars)
		# print(expr_graph.ddL_gamma_dxdx)
		# print('\n\n\n')
		# raise NotImplementedError

	def solve(self, display_progress=False):
		"""Solve the optimal control problem.

		If the initialisation flag is not set to True then the initialisation 
		method is called to initialise the optimal control problem. 

		Parameters:
		-----------
		display_progress : bool
			Option for whether progress updates should be outputted to the 
			console during solving. Defaults to False.
		"""

		self._set_solve_options(display_progress)
		self._check_if_initialisation_required_before_solve()

		# Solve the transcribed NLP on the initial mesh
		new_iteration_mesh, new_iteration_guess = self._mesh_iterations[0]._solve()

		mesh_iterations_met = self._settings.max_mesh_iterations == 1
		if new_iteration_mesh is None:
				mesh_tolerance_met = True
		else:
			mesh_tolerance_met = False

		while not mesh_iterations_met and not mesh_tolerance_met:
			new_iteration = Iteration(optimal_control_problem=self, iteration_number=self.num_mesh_iterations+1, mesh=new_iteration_mesh)
			self._mesh_iterations.append(new_iteration)
			self._mesh_iterations[-1]._initialise_iteration(new_iteration_guess)
			new_iteration_mesh, new_iteration_guess = self._mesh_iterations[-1]._solve()
			if new_iteration_mesh is None:
				mesh_tolerance_met = True
				print(f'Mesh tolerance met in mesh iteration {len(self._mesh_iterations)}.\n')
			elif self.num_mesh_iterations >= self._settings.max_mesh_iterations:
				mesh_iterations_met = True
				print(f'Maximum number of mesh iterations reached. pycollo exiting before mesh tolerance met.\n')
	
		_ = self._final_output()

	def _set_solve_options(self, display_progress):
		self._display_progress = display_progress

	def _check_if_initialisation_required_before_solve(self):
		if self._is_initialised == False:
			self.initialise()

	def _final_output(self):

		def solution_results():
			J_msg = (f'Final Objective Function Evaluation: {self.mesh_iterations[-1]._solution._J:.4f}\n')
			print(J_msg)

		def mesh_results():
			section_msg = (f'Final Number of Mesh Sections:       {self.mesh_iterations[-1]._mesh._K}')
			node_msg = (f'Final Number of Collocation Nodes:   {self.mesh_iterations[-1]._mesh._N}\n')
			print(section_msg)
			print(node_msg)

		def time_results():
			ocp_init_time_msg = (f'Total OCP Initialisation Time:       {self._ocp_initialisation_time:.4f} s')
			print(ocp_init_time_msg)

			self._iteration_initialisation_time = np.sum(np.array([iteration._initialisation_time for iteration in self._mesh_iterations]))

			iter_init_time_msg = (f'Total Iteration Initialisation Time: {self._iteration_initialisation_time:.4f} s')
			print(iter_init_time_msg)

			self._nlp_time = np.sum(np.array([iteration._nlp_time for iteration in self._mesh_iterations]))

			nlp_time_msg = (f'Total NLP Solver Time:               {self._nlp_time:.4f} s')
			print(nlp_time_msg)	

			self._process_results_time = np.sum(np.array([iteration._process_results_time for iteration in self._mesh_iterations]))

			process_results_time_msg = (f'Total Mesh Refinement Time:          {self._process_results_time:.4f} s')
			print(process_results_time_msg)

			total_time_msg = (f'\nTotal Time:                          {self._ocp_initialisation_time + self._iteration_initialisation_time + self._nlp_time + self._process_results_time:.4f} s')
			print(total_time_msg)
			print('\n\n')

		solved_msg = ('Optimal control problem sucessfully solved.')
		self._console_out_message_heading(solved_msg)

		solution_results()
		mesh_results()
		time_results()

	def __str__(self):
		return self.name

	def __repr__(self):
		return f"OptimalControlProblem('{self.name}')"






def kill():
		print('\n\n')
		raise ValueError

def cout(*args):
	print('\n\n')
	for arg in args:
		print(f'{arg}\n')



		


















