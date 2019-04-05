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
		self._y_vars = np.array([])
		self._u_vars = np.array([])
		self._q_vars = np.array([])
		self._t_vars = np.array([])
		self._s_vars = np.array([])
		self._y_eqns = np.array([])

		self._c_cons = np.array([])
		self._b_cons = np.array([])

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
	def _format_as_np_array(variables):
		if variables is None:
			variables = np.array([])
		try:
			iter(variables)
		except TypeError:
			variables = (variables, )
		return np.array(variables)

	@property
	def state_variables(self):
		return self._y_vars

	@state_variables.setter
	def state_variables(self, y_vars):
		self._y_vars = self._format_as_np_array(y_vars)
		self._num_y_vars = len(self._y_vars)
		self._update_vars()
		self._y_t0 = [sym.symbols(str(y)[:-3] + '(t0)') for y in self._y_vars]
		self._y_tF = [sym.symbols(str(y)[:-3] + '(tF)') for y in self._y_vars]

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
		self._u_vars = self._format_as_np_array(u_vars)
		self._num_u_vars = len(self._u_vars)
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
		if integrands:
			integrand_functions = self._format_as_np_array(integrands)
			for index, integrand in enumerate(integrand_functions):
				symbol = sym.symbols("q_{}".format(index))
				integrands_dict.update({symbol: integrand})
			self._q_vars = np.array(list(integrands_dict.keys()))
			self._q_funcs = np.array(list(integrands_dict.values()))
			self._num_q_vars = len(self._q_vars)
		else:
			self._q_vars = np.array([])
			self._q_funcs = np.array([])
			self._num_q_vars = 0
		self._q_func_map = integrands_dict
		self._update_vars()

	@property
	def time_variables(self):
		return self._t_vars
	
	@property
	def parameter_variables(self):
		return self._s_vars

	@parameter_variables.setter
	def parameter_variables(self, s_vars):
		self._s_vars = self._format_as_np_array(s_vars)
		self._num_s_vars = len(self._s_vars)
		self._update_vars()

	@property
	def variables(self):
		return self._x_vars

	@property
	def _num_vars(self):
		return len(self._x_vars)

	def _update_vars(self):
		self._x_vars = np.concatenate((self._y_vars, self._u_vars, self._q_vars, self._t_vars, self._s_vars))

		self._aux_data._y_map = dict()
		for index, state in enumerate(self._y_vars):
			self._aux_data._y_map.update({state: index})

		self._aux_data._u_map = dict()
		for index, control in enumerate(self._u_vars):
			self._aux_data._u_map.update({control: index})

		self._aux_data._q_map = dict()
		for index, integral in enumerate(self._q_vars):
			self._aux_data._q_map.update({integral: index})

		self._aux_data._t_map = dict()
		for index, time in enumerate(self._t_vars):
			self._aux_data._t_map.update({time: index})

		self._aux_data._s_map = dict()
		for index, parameter in enumerate(self._s_vars):
			self._aux_data._s_map.update({parameter: index})

		self._aux_data._x_map = dict()
		for index, variable in enumerate(self._x_vars):
			self._aux_data._x_map.update({variable: index})

	@property
	def state_equations(self):
		return self._y_eqns

	@state_equations.setter
	def state_equations(self, y_eqns):
		self._y_eqns = self._format_as_np_array(y_eqns)
		if self._y_eqns is not None:
			self._num_y_eqns = len(self._y_eqns)

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
		self._c_cons = self._format_as_np_array(c_cons)
		self._num_c_cons = len(self._c_cons)
	
	@property
	def boundary_constraints(self):
		return self._b_cons

	@boundary_constraints.setter
	def boundary_constraints(self, b_cons):
		self._b_cons = self._format_as_np_array(b_cons)
		self._num_b_cons = len(self._b_cons)

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

		# Variable index slices
		self._y_slice = slice(0, self._num_y_vars)
		self._u_slice = slice(self._y_slice.stop, self._y_slice.stop + self._num_u_vars)
		self._q_slice = slice(self._u_slice.stop, self._u_slice.stop + self._num_q_vars)
		self._t_slice = slice(self._q_slice.stop, self._q_slice.stop + self._num_t_vars)
		self._s_slice = slice(self._t_slice.stop, self._num_vars)
		self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)

		# Constraints index slices
		self._c_defect_slice = slice(0, self._num_y_vars)
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_cons)
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_q_vars)
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + self._num_b_cons)

		# Substitute constants into functions
		self._J_subbed = self._J.subs(self._aux_data._consts_vals)
		sym_subs_vectorised = np.vectorize(lambda sym: sym.subs(self._aux_data._consts_vals))
		self._y_eqns_subbed = sym_subs_vectorised(self._y_eqns)
		if self._c_cons.any():
			self._c_cons_subbed = sym_subs_vectorised(self._c_cons)
		else:
			self._c_cons_subbed = self._c_cons
		if self._q_funcs.any():
			self._q_funcs_subbed = sym_subs_vectorised(self._q_funcs)
		else:
			self._q_funcs_subbed = self._q_funcs
		if self._b_cons.any():
			self._b_cons_subbed = sym_subs_vectorised(self._b_cons)
		else:
			self._b_cons_subbed = self._b_cons

		# Constraints
		self._c = np.concatenate((self._y_eqns_subbed, self._c_cons_subbed, self._q_funcs_subbed, self._b_cons_subbed))
		self._num_c = len(self._c)

		# Objective derivatives
		self._dJ_dx = np.array(self._J_subbed.diff(self._x_vars))

		# Constraints derivatives Jacobian
		self._dc_dx = np.vectorize(lambda c: c.diff(self._x_vars).tolist())(self._c).tolist()
		self._dc_dx = np.array(self._dc_dx)

		c = sym.Matrix(self._c)

		print(type(c))
		print(c.shape)

		G = c.jacobian(self._x_vars)

		print(G)


		# Initial time boundary derivatives
		self._G_dbdy0 = np.empty((self._num_b_cons, self._num_y_vars), dtype=object)
		for row_i, c in enumerate(self._b_cons):
			for col_i, y in enumerate(self._y_t0):
				self._G_dbdy0[row_i, col_i] = c.diff(y)

		# Final time boundary derivatives
		self._G_dbdyF = np.empty((self._num_b_cons, self._num_y_vars), dtype=object)
		for row_i, c in enumerate(self._b_cons):
			for col_i, y in enumerate(self._y_tF):
				self._G_dbdyF[row_i, col_i] = c.diff(y)


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
			t_vars.append(sym.symbols('t0'))
		if self._bounds._tF_l != self._bounds._tF_u:
			t_vars.append(sym.symbols('tF'))
		self._t_vars = self._format_as_np_array(t_vars)
		self._num_t_vars = len(self._t_vars)
		self._update_vars

		# Constraints and derivatives
		self._generate_cons_and_derivs()

		# Problem setup
		# print("\nVariables:\n", self._x_vars)
		# print("\nState Equations:\n", self._y_eqns)

		# print("\nDefect Constraints:\n", self._y_eqns)
		# print("\nPath Constraints:\n", self._c_cons)
		# print("\nIntegral Constraints:\n", self._q_funcs)
		# print("\nBoundary Constraints:\n", self._b_cons)
		# print("\nConstraints:\n", self._c)

		# print("\nObjective Function:\n", self._J)
		# print("\nObjective Gradient wrt y:\n", self._g_dJdy)
		# print("\nObjective Gradient wrt u:\n", self._g_dJdu)
		# print("\nObjective Gradient wrt q:\n", self._g_dJdq)
		# print("\nObjective Gradient wrt t:\n", self._g_dJdt)
		# print("\nObjective Gradient wrt s:\n", self._g_dJds)

		# print("\nDefect Jacobian wrt y:\n", self._G_dddy)
		# print("\nDefect Jacobian wrt u:\n", self._G_dddu)
		# print("\nDefect Jacobian wrt q:\n", self._G_dddq)
		# print("\nDefect Jacobian wrt t:\n", self._G_dddt)
		# print("\nDefect Jacobian wrt s:\n", self._G_ddds)
		# print("\n\n\n\n")

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