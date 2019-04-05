import collections

import numpy as np
import sympy as sym

from opytimise.auxiliary_data import AuxiliaryData
from opytimise.bounds import Bounds
from opytimise.guess import Guess
from opytimise.iteration import Iteration
from opytimise.mesh import Mesh
from opytimise.settings import Settings

"""
Parameters are definied in accordance with Betts, JT (2010). Practical Methods for Optimal Control and Estimiation Using Nonlinear Programming (Second Edition).

Parameters:
	t: independent parameter (time).
	y: state variables vector.
	u: control variables vector.
	J: objective function.
	g: gradient of the objective function w.r.t. x.
	G: Jacobian matrix.
	H: Hessian matrix.
	b: vector of event constraints (equal to zero).
	c: vector of path constraint equations (equal to zero).
	s: free parameters.
	q: vector of integral constraints.

	N: number of temporal nodes in collocation.

Notes:
	x = [y, u, q, t, s]
	n = len(x): number of free variables
	m = len([c, b]): number of constraints

"""

class OptimalControlProblem():

	def __init__(self, state_variables=None, control_variables=None, parameter_variables=[], state_equations=None, *, bounds=None, initial_guess=None, initial_mesh=None, path_constraints=None, integrand_functions=None, objective_function=None, boundary_constraints=None, settings=None, auxiliary_data=None):

		# Set settings
		self.settings = settings

		# Auxiliary data
		self.auxiliary_data = auxiliary_data

		# Initialise problem description
		self._y_vars = sym.Matrix.zeros(0, 1)
		self._u_vars = sym.Matrix.zeros(0, 1)
		self._q_vars = sym.Matrix.zeros(0, 1)
		self._t_vars = sym.Matrix.zeros(0, 1)
		self._s_vars = sym.Matrix.zeros(0, 1)
		self._y_eqns = sym.Matrix.zeros(0, 1)

		self._c_cons = sym.Matrix.zeros(0, 1)
		self._b_cons = sym.Matrix.zeros(0, 1)

		# Variables
		self.state_variables = state_variables
		self.control_variables = control_variables
		self.parameter_variables = parameter_variables
		self.state_equations = state_equations

		# Continuous constraints and functions
		self.integrand_functions = integrand_functions
		self.path_constraints = path_constraints

		# Endpoint constraints and functions
		self.objective_function = objective_function
		self.boundary_constraints = boundary_constraints

		# Set bounds
		self.bounds = bounds

		# Set initial guess
		self.initial_guess = initial_guess

		# Set initial mesh
		self._mesh_iterations = []
		if initial_mesh is None:
			initial_mesh = Mesh(optimal_control_problem=self)
		else:
			initial_mesh._ocp = self
		initial_iteration = Iteration(optimal_control_problem=self, iteration_number=1, mesh=initial_mesh)
		self._mesh_iterations
		self._mesh_iterations.append(initial_iteration)

	@staticmethod
	def _format_as_np_array(iterable):
		if iterable is None:
			iterable = np.array([])
		try:
			iter(iterable)
		except TypeError:
			iterable = (iterable, )
		return np.array(iterable)

	@staticmethod
	def _format_as_sym_matrix(iterable):
		if not iterable:
			iterable = sym.Matrix.zeros(0, 1)
		try:
			iter(iterable)
		except TypeError:
			iterable = (iterable, )
		return sym.Matrix(iterable)

	@staticmethod
	def _check_sym_name_clash(syms):
		pass

	@property
	def state_variables(self):
		return self._y_vars

	@state_variables.setter
	def state_variables(self, y_vars):
		# self._y_vars = self._format_as_np_array(y_vars)
		self._y_vars = self._format_as_sym_matrix(y_vars)
		self._num_y_vars = self._y_vars.shape[0]
		self._update_vars()
		self._y_t0 = [sym.symbols(str(y)[:-3] + '(_t0)') for y in self._y_vars]
		self._y_tF = [sym.symbols(str(y)[:-3] + '(_tF)') for y in self._y_vars]

	@property
	def initial_state(self):
		return self._y_t0
		# y_t0 = [sym.symbols(str(y)[:-3] + '(t0)') for y in self._y_vars]
		# return np.array(y_t0)
	
	@property
	def final_state(self):
		return self._y_tF
		# y_tF = [sym.symbols(str(y)[:-3] + '(tF)') for y in self._y_vars]
		# return np.array(y_tF)
	
	@property
	def control_variables(self):
		return self._u_vars

	@control_variables.setter
	def control_variables(self, u_vars):
		self._u_vars = self._format_as_sym_matrix(u_vars)
		self._num_u_vars = self._u_vars.shape[0]
		self._update_vars()

	@property
	def integral_variables(self):
		return self._q_vars

	@property
	def integrand_functions(self):
		return self._q_funcs

	@integrand_functions.setter
	def integrand_functions(self, integrands):
		integrands_dict = collections.OrderedDict()
		integrand_functions = self._format_as_sym_matrix(integrands)
		for index, integrand in enumerate(integrand_functions):
			symbol = sym.symbols("q_{}".format(index))
			integrands_dict.update({symbol: integrand})
		self._q_vars = sym.Matrix(list(integrands_dict.keys()))
		self._q_funcs = sym.Matrix(list(integrands_dict.values()))
		self._num_q_vars = self._q_vars.shape[0]
		self._update_vars()

		# integrands_dict = collections.OrderedDict()
		# if integrands:
		# 	integrand_functions = self._format_as_np_array(integrands)
		# 	for index, integrand in enumerate(integrand_functions):
		# 		symbol = sym.symbols("q_{}".format(index))
		# 		integrands_dict.update({symbol: integrand})
		# 	self._q_vars = np.array(list(integrands_dict.keys()))
		# 	self._q_funcs = np.array(list(integrands_dict.values()))
		# 	self._num_q_vars = len(self._q_vars)
		# else:
		# 	self._q_vars = np.array([])
		# 	self._q_funcs = np.array([])
		# 	self._num_q_vars = 0
		# self._q_func_map = integrands_dict
		# self._update_vars()

	@property
	def time_variables(self):
		return self._t_vars
	
	@property
	def parameter_variables(self):
		return self._s_vars

	@parameter_variables.setter
	def parameter_variables(self, s_vars):
		self._s_vars = self._format_as_sym_matrix(s_vars)
		self._num_s_vars = self._s_vars.shape[0]
		self._update_vars()

	@property
	def variables(self):
		return self._x_vars

	@property
	def _num_vars(self):
		return self._x_vars

	def _update_vars(self):

		# Variable index slices
		self._x_vars = sym.Matrix([self._y_vars, self._u_vars, self._q_vars, self._t_vars, self._s_vars])
		self._num_x_vars = self._x_vars.shape[0]

		# self._aux_data._y_map = dict()
		# for index, state in enumerate(self._y_vars):
		# 	self._aux_data._y_map.update({state: index})

		# self._aux_data._u_map = dict()
		# for index, control in enumerate(self._u_vars):
		# 	self._aux_data._u_map.update({control: index})

		# self._aux_data._q_map = dict()
		# for index, integral in enumerate(self._q_vars):
		# 	self._aux_data._q_map.update({integral: index})

		# self._aux_data._t_map = dict()
		# for index, time in enumerate(self._t_vars):
		# 	self._aux_data._t_map.update({time: index})

		# self._aux_data._s_map = dict()
		# for index, parameter in enumerate(self._s_vars):
		# 	self._aux_data._s_map.update({parameter: index})

		# self._aux_data._x_map = dict()
		# for index, variable in enumerate(self._x_vars):
		# 	self._aux_data._x_map.update({variable: index})

	@property
	def state_equations(self):
		return self._y_eqns

	@state_equations.setter
	def state_equations(self, y_eqns):
		self._y_eqns = self._format_as_sym_matrix(y_eqns)
		self._num_y_eqns = self._y_eqns.shape[0]

	@property
	def bounds(self):
		return self._bounds
	
	@bounds.setter
	def bounds(self, bounds):
		if bounds is None:
			self._bounds = Bounds(optimal_control_problem=self)
		else:
			self._bounds = bounds
			self._bounds._ocp = self

	@property
	def initial_guess(self):
		return self._initial_guess
	
	@initial_guess.setter
	def initial_guess(self, guess):
		if guess is None:
			self._initial_guess = Guess(optimal_control_problem=self)
		else:
			self._initial_guess = guess
			self._initial_guess._ocp = self

	@property
	def mesh_iterations(self):
		return self._mesh_iterations
	
	@property
	def num_mesh_iterations(self):
		return len(self._mesh_iterations)

	@property
	def objective_function(self):
		return self._J
	
	@objective_function.setter
	def objective_function(self, J):
		self._J = J

	@property
	def path_constraints(self):
		return self._c_cons

	@path_constraints.setter
	def path_constraints(self, c_cons):
		self._c_cons = self._format_as_sym_matrix(c_cons)
		self._num_c_cons = self._c_cons.shape[0]
	
	@property
	def boundary_constraints(self):
		return self._b_cons

	@boundary_constraints.setter
	def boundary_constraints(self, b_cons):
		self._b_cons = self._format_as_sym_matrix(b_cons)
		self._num_b_cons = self._b_cons.shape[0]

	@property
	def settings(self):
		return self._settings
	
	@settings.setter
	def settings(self, settings):
		if settings is None:
			self._settings = Settings(optimal_control_problem=self)
		else:
			self._settings = settings
			self._settings._ocp = self

	@property
	def auxiliary_data(self):
		return self._aux_data
	
	@auxiliary_data.setter
	def auxiliary_data(self, aux_data):
		if aux_data is None:
			self._aux_data = AuxiliaryData(optimal_control_problem=self)
		else:
			self._aux_data = aux_data
			self._aux_data._optimal_control_problem = self

	@property
	def nlp_problem(self):
		return self._nlp

	@profile
	def _generate_cons_and_derivs(self):

		# Variables index slices
		self._y_slice = slice(0, self._num_y_vars)
		self._u_slice = slice(self._y_slice.stop, self._y_slice.stop + self._num_u_vars)
		self._q_slice = slice(self._u_slice.stop, self._u_slice.stop + self._num_q_vars)
		self._t_slice = slice(self._q_slice.stop, self._q_slice.stop + self._num_t_vars)
		self._s_slice = slice(self._t_slice.stop, self._num_vars)
		self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)

		# # Objective derivatives
		self._J_subbed = self._J.subs(self._aux_data._consts_vals)
		self._dJ_dx = self._J_subbed.diff(self._x_vars)

		# Constraints
		self._c = sym.Matrix([self._y_eqns, self._c_cons, self._q_funcs, self._b_cons])
		self._c_subbed = self._c.subs(self._aux_data._consts_vals)
		self._num_c = self._c.shape[0]

		# Constraints index slices
		self._c_defect_slice = slice(0, self._num_y_vars)
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_cons)
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_q_vars)
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + self._num_b_cons)
		self._c_continuous_slice = slice(0, self._num_c - self._num_b_cons)

		# # Constraints derivatives Jacobian
		self._dc_dx = self._c[self._c_continuous_slice, :].jacobian(self._x_vars)

		# Initial and final time boundary derivatives
		self._db_dy0 = self._b_cons.jacobian(self._y_t0)
		self._db_dyF = self._b_cons.jacobian(self._y_tF)

	def solve(self):

		# Error checking
		if self._num_y_eqns != self._num_y_vars:
			msg = ("A differential equation must be supplied for each state variable. {1} differential equations were supplied for {0} state variables.")
			raise ValueError(msg.format(self._num_y_vars, self._num_y_eqns))
		self.bounds._bounds_check()
		self.initial_guess._guess_check()

		# Time variables
		t_vars = []
		if self._bounds._t0_l != self._bounds._t0_u:
			t_vars.append(sym.symbols('_t0'))
		if self._bounds._tF_l != self._bounds._tF_u:
			t_vars.append(sym.symbols('_tF'))
		self._t_vars = self._format_as_sym_matrix(t_vars)
		self._num_t_vars = self._t_vars.shape[0]
		self._update_vars()

		# Constraints and derivatives
		self._generate_cons_and_derivs()

		def print_formatted(name, arg):
			try:
				print("\n{}: {} {}".format(name, type(arg), arg.shape), arg, sep='\n')
			except AttributeError:
				print("\n{}: {}".format(name, type(arg)), arg, sep='\n')

		# Problem setup
		if False:
			print_formatted('State Variables', self._y_vars)
			print_formatted('Control Variables', self._u_vars)
			print_formatted('Integral Variables', self._q_vars)
			print_formatted('Time Variables', self._t_vars)
			print_formatted('Parameter Variables', self._s_vars)
			print_formatted('Variables', self._x_vars)

			print_formatted('State Equations', self._y_eqns)
			
			print_formatted('Defect Constraints', self._y_eqns)
			print_formatted('Path Constraints', self._c_cons)
			print_formatted('Integral Constraints', self._q_funcs)
			print_formatted('Boundary Constraints', self._b_cons)
			print_formatted('Constraints', self._c)

			print_formatted('Objective Function', self._J_subbed)
			print_formatted('Objective Gradient', self._dJ_dx)

			print_formatted('Continuous Jacobian', self._dc_dx)
			print_formatted('Boundary Jacobian Initial', self._db_dy0)
			print_formatted('Boundary Jacobian Initial', self._db_dyF)

		# Initial iteration setup
		# self._mesh_iterations[0]._initialise_iteration(self.initial_guess)

		# Display generated problem initialisation
		# print("\nJ:")
		# print(self.objective_function)
		# print("\ng:")
		# print(self.objective_gradient)
		# print("\nc:")
		# print(self.constraints)
		# print("\nG:")
		# print(self.constraints_jacobian)
		# print('\n\n\n')
		# print(self.mesh_iterations[0].mesh.N)
		# print(self.mesh_iterations[0].mesh.t)
		# print(self.mesh_iterations[0].mesh.h)
		# print('\n\n\n')
		# print("Completed successfully.")
		# print('\n\n\n')

		# raise NotImplementedError

		# Solve
		# self._mesh_iterations[0]._solve()