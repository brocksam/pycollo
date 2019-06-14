import itertools

import numba as nb
import numpy as np
import scipy.sparse as sparse
import sympy as sym
import sympy.physics.mechanics as me

from pycollo.bounds import Bounds
from pycollo.guess import Guess
from pycollo.iteration import Iteration
from pycollo.mesh import Mesh
from pycollo.quadrature import Quadrature
from pycollo.settings import Settings
import pycollo.utils as pu

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
	delta: defect constraints.
	rho: integral constraints.
	beta: boundary constraints.

	N: number of temporal nodes in collocation.

Notes:
	x = [y, u, q, t, s]
	n = len(x): number of free variables
	m = len([c, b]): number of constraints

"""

class OptimalControlProblem():

	_t0_USER = sym.Symbol('t0')
	_tF_USER = sym.Symbol('tF')
	_t0 = sym.Symbol('_t0')
	_tF = sym.Symbol('_tF')

	_STRETCH = 0.5 * (_tF - _t0)
	_SHIFT = 0.5 * (_t0 + _tF)

	_dSTRETCH_dt = np.array([-0.5, 0.5])

	def __init__(self, state_variables=None, control_variables=None, parameter_variables=None, state_equations=None, *, bounds=None, initial_guess=None, initial_mesh=None, path_constraints=None, integrand_functions=None, state_endpoint_constraints=None, boundary_constraints=None, objective_function=None, settings=None, auxiliary_data=None):

		# Set settings
		self.settings = settings

		# Initialise problem description
		
		self._y_vars_user = ()
		self._u_vars_user = ()
		self._q_vars_user = ()
		self._t_vars_user = (self._t0_USER, self._tF_USER)
		self._s_vars_user = ()

		self._y_eqns_user = ()
		self._c_cons_user = ()
		self._q_funcs_user = ()
		self._y_b_cons_user = ()
		self._b_cons_user = ()
		
		# Variables, constraints and functions
		self.state_variables = state_variables
		self.control_variables = control_variables
		self.parameter_variables = parameter_variables

		self.state_equations = state_equations
		self.path_constraints = path_constraints
		self.integrand_functions = integrand_functions
		self.state_endpoint_constraints = state_endpoint_constraints
		self.boundary_constraints = boundary_constraints

		self.objective_function = objective_function

		self.auxiliary_data = dict(auxiliary_data) if auxiliary_data else {}

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

		# Initialistion flag
		self._initialised = False

	@property
	def initial_time(self):
		return self._t0_user

	@property
	def final_time(self):
		return self._tF_user

	@property
	def initial_state(self):
		return self._y_t0_user
	
	@property
	def final_state(self):
		return self._y_tF_user

	@property
	def state_endpoint(self):
		return self._y_t0_user + self._y_tF_user

	@property
	def state_variables(self):
		return self._y_vars_user

	@state_variables.setter
	def state_variables(self, y_vars):
		self._initialised = False
		self._y_vars_user = pu.format_as_tuple(y_vars)
		self._y_t0_user = tuple(sym.symbols(f'{y}(t0)') for y in self._y_vars_user)
		self._y_tF_user = tuple(sym.symbols(f'{y}(tF)') for y in self._y_vars_user)
		_ = self._update_vars()
		_ = pu.check_sym_name_clash(self._y_vars_user)

	@property
	def number_state_variables(self):
		return len(self._y_vars_user)
	
	@property
	def control_variables(self):
		return self._u_vars_user

	@control_variables.setter
	def control_variables(self, u_vars):
		self._initialised = False
		self._u_vars_user = pu.format_as_tuple(u_vars)
		_ = self._update_vars()
		_ = pu.check_sym_name_clash(self._u_vars_user)

	@property
	def number_control_variables(self):
		return len(self._u_vars_user)

	@property
	def integral_variables(self):
		return self._q_vars_user

	@property
	def number_integral_variables(self):
		return len(self._q_vars_user)

	@property
	def time_variables(self):
		return self._t_vars_user

	@property
	def number_time_variables(self):
		return len(self._t_vars_user)
	
	@property
	def parameter_variables(self):
		return self._s_vars_user

	@parameter_variables.setter
	def parameter_variables(self, s_vars):
		self._initialised = False
		s_vars = pu.format_as_tuple(s_vars)
		self._s_vars_user = tuple(s_vars)
		_ = self._update_vars()
		_ = pu.check_sym_name_clash(self._s_vars_user)

	@property
	def number_parameter_variables(self):
		return len(self._s_vars_user)

	@property
	def variables(self):
		return self._x_vars_user

	@property
	def number_variables(self):
		return len(self._x_vars_user)

	def _update_vars(self):
		self._x_vars_user = tuple(self._y_vars_user + self._u_vars_user + self._q_vars_user + self._t_vars_user + self._s_vars_user)
		self._x_b_vars_user = tuple(self._y_t0_user + self._y_tF_user + self._q_vars_user + self._t_vars_user + self._s_vars_user)
		return None

	@property
	def state_equations(self):
		return self._y_eqns_user

	@state_equations.setter
	def state_equations(self, y_eqns):
		self._initialised = False
		y_eqns = pu.format_as_tuple(y_eqns)
		self._y_eqns_user = tuple(y_eqns)

	@property
	def path_constraints(self):
		return self._c_cons_user

	@path_constraints.setter
	def path_constraints(self, c_cons):
		self._initialised = False
		c_cons = pu.format_as_tuple(c_cons)
		self._c_cons_user = tuple(c_cons)

	@property
	def integrand_functions(self):
		return self._q_funcs_user

	@integrand_functions.setter
	def integrand_functions(self, integrands):
		self._initialised = False
		self._q_funcs_user = pu.format_as_tuple(integrands)
		self._q_vars_user = tuple(sym.symbols(f'_q{i_q}') for i_q, _ in enumerate(self._q_funcs_user))
		_ = self._update_vars()

	@property
	def state_endpoint_constraints(self):
		return self._y_b_cons_user

	@state_endpoint_constraints.setter
	def state_endpoint_constraints(self, y_b_cons):
		self._initialised = False
		y_b_cons = pu.format_as_tuple(y_b_cons)
		self._y_b_cons_user = tuple(y_b_cons)

	@property
	def boundary_constraints(self):
		return self._b_cons_user

	@boundary_constraints.setter
	def boundary_constraints(self, b_cons):
		self._initialised = False
		b_cons = pu.format_as_tuple(b_cons)
		self._b_cons_user = tuple(b_cons)

	@property
	def objective_function(self):
		return self._J_user

	@objective_function.setter
	def objective_function(self, J):
		self._initialised = False
		self._J_user = sym.sympify(J)

	@property
	def auxiliary_data(self):
		return self._aux_data_user
	
	@auxiliary_data.setter
	def auxiliary_data(self, aux_data):
		self._initialised = False
		self._aux_data_user = dict(aux_data)

	@property
	def bounds(self):
		return self._bounds
	
	@bounds.setter
	def bounds(self, bounds):
		self._initialised = False
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
		self._initialised = False
		if guess is None:
			self._initial_guess = Guess(optimal_control_problem=self)
		else:
			self._initial_guess = guess
			self._initial_guess._ocp = self

	@property
	def initial_mesh(self):
		return self._mesh_iterations[0]._mesh

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
		self._initialised = False
		if settings is None:
			self._settings = Settings(optimal_control_problem=self)
		else:
			self._settings = settings
			self._settings._ocp = self

	def _compile_numba_functions(self):

		def reshape_x(x, num_yu, yu_qts_split):
			x = np.array(x)
			yu = x[:yu_qts_split].reshape(num_yu, -1)
			qts = x[yu_qts_split:].reshape(len(x) - yu_qts_split, )
			x_tuple = (*yu, *qts)
			return x_tuple

		def reshape_x_point(x, x_endpoint_indices):#num_y, y_u_split, u_q_split):
			x = np.array(x)
			# yu_t0 = x[:y_u_split].reshape(num_y, -1)[:, 0]
			# yu_tF = x[:y_u_split].reshape(num_y, -1)[:, -1]
			# qts = x[u_q_split:].reshape(len(x) - u_q_split,)
			# x_tuple = (*yu_t0, *yu_tF, *qts)
			x_tuple = x[x_endpoint_indices]
			return x_tuple

		self._x_reshape_lambda = reshape_x
		self._x_reshape_lambda_point = reshape_x_point

		self._J_lambda = pu.numbafy(expression=self._J, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=0)

		dJ_dxb_lambda = pu.numbafy(expression=self._dJ_dxb_chain, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, endpoint=True, ocp_num_vars=self._num_vars_tuple)

		def g_lambda(x_tuple_point, N):
			g = dJ_dxb_lambda(*x_tuple_point, N)
			return g

		self._g_lambda = g_lambda

		t_stretch_lambda = pu.numbafy(expression=self._STRETCH, parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=0)
		self._dstretch_dt = [val for val, t_needed in zip(self._dSTRETCH_dt, self._bounds._t_needed) if t_needed]

		dy_lambda = pu.numbafy(expression=self._y_eqns, parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		self._dy_lambda = dy_lambda

		def c_defect_lambda(x_tuple, N, ocp_y_slice, A, D):
			y = np.vstack(x_tuple[ocp_y_slice])
			dy = dy_lambda(*x_tuple, N)
			stretch = t_stretch_lambda(*x_tuple)
			c_zeta = (D.dot(y.T) + stretch*A.dot(dy.T)).flatten(order='F')
			return c_zeta

		def c_path_lambda(x_tuple, N):
			return 0

		rho_lambda = pu.numbafy(expression=self._q_funcs, parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		def c_integral_lambda(x_tuple, N, q_slice, W):
			q = np.array(x_tuple[q_slice])
			g = rho_lambda(*x_tuple, N)
			stretch = t_stretch_lambda(*x_tuple)
			if g.size:
				return q - stretch*np.matmul(g, W)
			else:
				return q

		beta_lambda = pu.numbafy(expression=self._b_cons, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		def c_boundary_lambda(x_tuple_point, N):
			c = beta_lambda(*x_tuple_point, N)
			return c

		# @profile
		def c_lambda(x_tuple, x_tuple_point, N, ocp_y_slice, ocp_q_slice, num_c, defect_slice, path_slice, integral_slice, boundary_slice, A, D, W):
			c = np.empty(num_c)
			c[defect_slice] = c_defect_lambda(x_tuple, N, ocp_y_slice, A, D)
			c[path_slice] = c_path_lambda(x_tuple, N)
			c[integral_slice] = c_integral_lambda(x_tuple, N, ocp_q_slice, W)
			c[boundary_slice] = c_boundary_lambda(x_tuple_point, N)
			return c

		self._c_lambda = c_lambda

		ddy_dy_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_defect_slice, self._y_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		ddy_du_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_defect_slice, self._u_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		ddy_ds_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_defect_slice, self._s_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		drho_dy_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_integral_slice, self._y_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		drho_du_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_integral_slice, self._u_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		drho_ds_lambda = pu.numbafy(expression=self._dc_dx_chain[self._c_integral_slice, self._s_slice], parameters=self._x_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=2, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		# dbeta_dy0_lambda = pu.numbafy(expression=self._db_dy0, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		# dbeta_dyF_lambda = pu.numbafy(expression=self._db_dyF, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		# dbeta_dqts_lambda = pu.numbafy(expression=self._db_dqts, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		dbeta_dxb_lambda = pu.numbafy(expression=self._db_dxb, parameters=self._x_b_vars, constants=self._aux_data, substitutions=self._aux_subs, return_dims=1, N_arg=True, ocp_num_vars=self._num_vars_tuple)

		# print(self._db_dxb_chain)
		# print('\n\n\n')
		# raise NotImplementedError

		def A_x_dot_sparse(A_sparse, ddy_dx, num_y, num_nz, stretch, return_array):
			for i, row in enumerate(ddy_dx):
				start = i*num_nz
				stop = start + num_nz
				entries = stretch*A_sparse.multiply(row).data
				return_array[start:stop] = entries
			return return_array

		# @profile
		def G_dzeta_dy_lambda(x_tuple, N, A, D, A_row_col_array, num_y, dzeta_dy_D_nonzero, dzeta_dy_slice):
			ddy_dy = ddy_dy_lambda(*x_tuple, N)
			stretch = t_stretch_lambda(*x_tuple)
			num_nz = A_row_col_array.shape[1]
			return_array = np.empty(num_y**2 * num_nz)
			G_dzeta_dy = A_x_dot_sparse(A, ddy_dy, num_y, num_nz, stretch, return_array)
			D_data = (D.data * np.ones((num_y, 1))).flatten()
			G_dzeta_dy[dzeta_dy_D_nonzero] += D_data
			return G_dzeta_dy

		# @profile
		def G_dzeta_du_lambda(x_tuple, N, A, A_row_col_array, num_y, num_u):
			ddy_du = ddy_du_lambda(*x_tuple, N)
			stretch = t_stretch_lambda(*x_tuple)
			num_nz = A_row_col_array.shape[1]
			return_array = np.empty(num_u*num_y*num_nz)
			G_dzeta_du = A_x_dot_sparse(A, ddy_du, num_u, num_nz, stretch, return_array)
			return G_dzeta_du

		# @profile
		def G_dzeta_dt_lambda(x_tuple, N, A):
			dy = dy_lambda(*x_tuple, N)
			G_dzeta_dt = np.outer(self._dstretch_dt, (A.dot(dy.T).flatten(order='F'))).flatten(order='F')
			return G_dzeta_dt

		# @profile
		def G_dzeta_ds_lambda(x_tuple, N, A):
			ddy_ds = ddy_ds_lambda(*x_tuple, N)
			stretch = t_stretch_lambda(*x_tuple)
			G_dzeta_ds = stretch*A.dot(ddy_ds.T).flatten(order='F')
			return G_dzeta_ds

		def G_dgamma_dy_lambda(x_tuple):
			return np.zeros((0,))

		def G_dgamma_du_lambda(x_tuple):
			return np.zeros((0,))

		def G_dgamma_dt_lambda(x_tuple):
			return np.zeros((0,))

		def G_dgamma_ds_lambda(x_tuple):
			return np.zeros((0,))

		def G_drho_dy_lambda(x_tuple, N, W):
			drho_dy = drho_dy_lambda(*x_tuple, N)
			if drho_dy.size:
				stretch = t_stretch_lambda(*x_tuple)
				G_drho_dy = (- stretch * drho_dy * W).flatten()
				return G_drho_dy
			else:
				return []

		def G_drho_du_lambda(x_tuple, N, W):
			drho_du = drho_du_lambda(*x_tuple, N)
			if drho_du.size:
				stretch = t_stretch_lambda(*x_tuple)
				G_drho_du = (- stretch * drho_du * W).flatten()
				return G_drho_du
			else:
				return []

		def G_drho_dt_lambda(x_tuple, N, W):
			g = rho_lambda(*x_tuple, N)
			if g.size > 0:
				return - np.outer(self._dstretch_dt, np.matmul(g, W)).flatten(order='F')
			else:
				return []

		def G_drho_ds_lambda(x_tuple, N, W):
			drho_ds = drho_ds_lambda(*x_tuple, N)
			if drho_ds.size:
				stretch = t_stretch_lambda(*x_tuple)
				G_drho_ds = (- stretch * np.matmul(drho_ds, W))
				return G_drho_ds
			else:
				return []

		def G_dbeta_dxb_lambda(x_tuple_point, N):
			return dbeta_dxb_lambda(*x_tuple_point, N)

		def G_lambda(x_tuple, x_tuple_point, N, num_G_nonzero, num_x_ocp, A, D, W, A_row_col_array, dzeta_dy_D_nonzero, dzeta_dy_slice, dzeta_du_slice, dzeta_dt_slice, dzeta_ds_slice, dgamma_dy_slice, dgamma_du_slice, dgamma_dt_slice, dgamma_ds_slice, drho_dy_slice, drho_du_slice, drho_dq_slice, drho_dt_slice, drho_ds_slice, dbeta_dxb_slice):
			G = np.empty(num_G_nonzero)

			if num_x_ocp.y:
				G[dzeta_dy_slice] = G_dzeta_dy_lambda(x_tuple, N, A, D, A_row_col_array, num_x_ocp.y, dzeta_dy_D_nonzero, dzeta_dy_slice)
				G[dgamma_dy_slice] = G_dgamma_dy_lambda(x_tuple)
				G[drho_dy_slice] = G_drho_dy_lambda(x_tuple, N, W)

			if num_x_ocp.u:
				G[dzeta_du_slice] = G_dzeta_du_lambda(x_tuple, N, A,  A_row_col_array, num_x_ocp.y, num_x_ocp.u)
				G[dgamma_du_slice] = G_dgamma_du_lambda(x_tuple)
				G[drho_du_slice] = G_drho_du_lambda(x_tuple, N, W)

			if num_x_ocp.q:
				G[drho_dq_slice] = 1

			if num_x_ocp.t:
				G[dzeta_dt_slice] = G_dzeta_dt_lambda(x_tuple, N, A)
				# G[dgamma_dt_slice] = G_dgamma_dt_lambda(x_tuple)
				G[drho_dt_slice] = G_drho_dt_lambda(x_tuple, N, W)

			if num_x_ocp.s:
				G[dzeta_ds_slice] = G_dzeta_ds_lambda(x_tuple, N, A)
				# G[dgamma_ds_slice] = G_dgamma_ds_lambda(x_tuple)
				G[drho_ds_slice] = G_drho_ds_lambda(x_tuple, N, W)

			G[dbeta_dxb_slice] = G_dbeta_dxb_lambda(x_tuple_point, N)

			# if sum(num_x_ocp[2:]):
			# 	G[dbeta_dqts_slice] = G_dbeta_dqts_lambda(x_tuple_point)

			return G

		self._G_lambda = G_lambda

		return None

	def initialise(self):

		# User-defined symbols allowed in continuous and endpoint functions
		user_var_syms_endpoint = set(self._x_b_vars_user)
		user_var_syms_continuous = set.union(set(self._y_vars_user), set(self._u_vars_user), {sym.Symbol('t')}, set(self._s_vars_user))
		user_var_syms = set.union(user_var_syms_endpoint, user_var_syms_continuous)
		self._allowed_endpoint_set = set.union(user_var_syms_endpoint, set(self._aux_data_user.keys()))
		self._allowed_continuous_set =set.union(user_var_syms_continuous, set(self._aux_data_user.keys()))
		
		# Check auxiliary data
		disallowed_syms_set = set(self._aux_data_user.keys()).intersection(set(user_var_syms))
		if disallowed_syms_set:
			disallowed_syms = ', '.join(f'{symbol}' for symbol in disallowed_syms_set)
			msg = (f"Additional information about {disallowed_syms} cannot be supplied as auxiliary data as these are variables in the optimal control problem.")
			raise ValueError(msg)

		aux_data_temp = {}
		aux_subs_temp = {}

		for (k, v) in self._aux_data_user.items():
			try:
				aux_data_temp[k] = float(v)
			except (ValueError, TypeError):
				aux_subs_temp[k] = v

		for k, v in aux_subs_temp.items():
			v_free_syms = v.free_symbols
			v_var_syms = v_free_syms.intersection(user_var_syms)
			v_data_syms = v_free_syms.intersection(set(aux_data_temp.keys()))
			v_subs_syms = v_free_syms.intersection(set(aux_subs_temp.keys()))
			v_extra_syms = v_free_syms.difference(set.union(v_var_syms, v_data_syms, v_subs_syms))
			if v_extra_syms:
				disallowed_syms = ', '.join(f'{symbol}' for symbol in v_extra_syms)
				msg = (f"Additional information for {k} cannot be provided as auxiliary data in its current form as it is a function of {disallowed_syms}. These symbols have not been defined elsewhere in the optimal control problem. Please supply numerical values for them as auxiliary data.")
				raise ValueError(msg)
			for i in range(100):
				v_free_syms = v.free_symbols
				v_subs_syms = v_free_syms.intersection(set(aux_subs_temp.keys()))
				if v_subs_syms:
					v = v.subs(aux_subs_temp)
				else:
					break
			else:
				msg = (f'Substitution dependency chain for {k} appears to be too deep.')
			if v.free_symbols:
				aux_subs_temp[k] = v
			else:
				aux_data_temp[k] = float(v)

		# Check state equations
		if len(self._y_eqns_user) != len(self._y_vars_user):
			msg = (f"A differential state equation must be supplied for each state variable. {len(self._y_eqns_user)} differential equations were supplied for {len(self._y_vars_user)} state variables.")
			raise ValueError(msg)
		for i_y, eqn in enumerate(self._y_eqns_user):
			eqn_syms = set.union(me.find_dynamicsymbols(eqn), eqn.free_symbols)
			if not eqn_syms.issubset(self._allowed_continuous_set):
				disallowed_syms = ', '.join(f'{symbol}' for symbol in eqn_syms.difference(self._allowed_continuous_set))
				msg = (f"State equation #{i_y+1}: {eqn}, cannot be a function of {disallowed_syms}.")
				raise ValueError(msg)

		# Check path constraints
		for i_c, con in enumerate(self._c_cons_user):
			con_syms = set.union(me.find_dynamicsymbols(con), con.free_symbols)
			if not con_syms.issubset(self._allowed_continuous_set):
				disallowed_syms = ', '.join(f'{symbol}' for symbol in con_syms.difference(self._allowed_continuous_set))
				msg = (f"Path constraint #{i_c+1}: {con}, cannot be a function of {disallowed_syms}.")
				raise ValueError(msg)

		# Check integrand functions
		for i_q, func in enumerate(self._q_funcs_user):
			func_syms = set.union(me.find_dynamicsymbols(func), func.free_symbols)
			if not func_syms.issubset(self._allowed_continuous_set):
				disallowed_syms = ', '.join(f'{symbol}' for symbol in func_syms.difference(self._allowed_continuous_set))
				msg = (f"Integrand function #{i_q+1}: {func}, cannot be a function of {disallowed_syms}.")
				raise ValueError(msg)

		# Check state endpoint constraints
		for i_y_b, con in enumerate(self._y_b_cons_user):
			if con not in self.state_endpoint:
				msg = (f"State endpoint constraints #{i_y_b}: {con} should not be supplied as a state endpoint constraint as it more than a function of a single state endpoint variable. Please resupply this constraint as a boundary constraint using `OptimalControlProblem.boundary_constraints`.")
				raise ValueError(msg)

		# Check endpoint constraints
		for i_b, con in enumerate(self._b_cons_user):
			if con in self.state_endpoint:
				msg = (f"Boundary constraint #{i_b}: {con} should not be supplied as a boundary constraint as it only contains a single state endpoint variable. Please resupply this constraint as a state endpoint constraint using `OptimalControlProblem.state_endpoint_constraints`.")
				raise ValueError(msg)
			con_syms = set.union(me.find_dynamicsymbols(con), con.free_symbols)
			if not con_syms.issubset(self._allowed_endpoint_set):
				disallowed_syms = ', '.join(f'{symbol}' for symbol in con_syms.difference(self._allowed_endpoint_set))
				msg = (f"Boundary constraint #{i_b+1}: {con}, cannot be a function of {disallowed_syms}.")
				raise ValueError(msg)

		# Check objective function
		if self._J_user is not None:
			if not self._J_user.free_symbols:
				msg = (f"The declared objective function {J} is invalid as it doesn't contain any Mayer terms (functions of the initial and final times and states) or any Lagrange terms (integrals of functions of the state and control variables with respect to time between the limits '_t0' and '_tF').")
				raise ValueError(msg)
			if isinstance(self._J_user, sym.Add):
				bolza_set = set(self._J_user.args)
			else:
				bolza_set = {self._J_user}
			for term in bolza_set:
				term_syms = set.union(me.find_dynamicsymbols(term), term.free_symbols)
				if not term_syms.issubset(self._allowed_endpoint_set):
					disallowed_syms = ', '.join(f'{symbol}' for symbol in term_syms.difference(self._allowed_endpoint_set))
					msg = (f"The objective function cannot be a function of {disallowed_syms}.")
					raise ValueError(msg)
		else:
			allowed_syms = ', '.join(f'{symbol}' for symbol in self._x_b_vars_user)
			msg = (f"User must supply an objective function as a function of {allowed_syms}.")
			raise ValueError(msg)

		# Generate pycollo symbols and functions
		self._aux_data = {sym.Symbol(f'_a{i_a}'): value for i_a, (_, value) in enumerate(aux_data_temp.items())}
		a_subs_dict = dict(zip(aux_data_temp.keys(), self._aux_data.keys()))

		# Check user-supplied bounds
		self._bounds._bounds_check(aux_data=aux_data_temp, aux_subs=aux_subs_temp)

		# State variables
		y_vars = [sym.Symbol(f'_y{i_y}') for i_y, _ in enumerate(self._y_vars_user)]
		self._y_vars = sym.Matrix([y for y, y_needed in zip(y_vars, self._bounds._y_needed) if y_needed])
		if not self._y_vars:
			self._y_vars = sym.Matrix.zeros(0, 1)
		self._num_y_vars = self._y_vars.shape[0]
		self._y_t0 = sym.Matrix([sym.Symbol(f'{y}_t0') for y in self._y_vars])
		self._y_tF = sym.Matrix([sym.Symbol(f'{y}_tF') for y in self._y_vars])
		self._y_b_vars = sym.Matrix(list(itertools.chain.from_iterable(y for y in zip(self._y_t0, self._y_tF))))
		self._aux_data.update({y: value for y, y_needed, value in zip(y_vars, self._bounds._y_needed, self._bounds._y_l) if not y_needed})
		y_subs_dict = dict(zip(self._y_vars_user, y_vars))
		y_endpoint_subs_dict = {**dict(zip(self._y_t0_user, self._y_t0)), ** dict(zip(self._y_tF_user, self._y_tF))}

		# Control variables
		u_vars = [sym.Symbol(f'_u{i_u}') for i_u, _ in enumerate(self._u_vars_user)]
		self._u_vars = sym.Matrix([u for u, u_needed in zip(u_vars, self._bounds._u_needed) if u_needed])
		if not self._u_vars:
			self._u_vars = sym.Matrix.zeros(0, 1)
		self._num_u_vars = self._u_vars.shape[0]
		self._aux_data.update({u: value for u, u_needed, value in zip(u_vars, self._bounds._u_needed, self._bounds._u_l) if not u_needed})
		u_subs_dict = dict(zip(self._u_vars_user, u_vars))

		# Integral variables
		q_vars = sym.Matrix(self._q_vars_user)
		self._q_vars = sym.Matrix([q for q, q_needed in zip(q_vars, self._bounds._q_needed) if q_needed])
		if not self._q_vars:
			self._q_vars = sym.Matrix.zeros(0, 1)
		self._num_q_vars = self._q_vars.shape[0]
		self._aux_data.update({q: value for q, q_needed, value in zip(q_vars, self._bounds._q_needed, self._bounds._q_l) if not q_needed})

		# Time variables
		t_vars = [self._t0, self._tF]
		self._t_vars = sym.Matrix([t for t, t_needed in zip(t_vars, self._bounds._t_needed) if t_needed])
		if not self._t_vars:
			self._t_vars = sym.Matrix.zeros(0, 1)
		self._num_t_vars = self._t_vars.shape[0]
		if not self._bounds._t_needed[0]:
			self._aux_data.update({self._t0: self._bounds._t0_l})
		if not self._bounds._t_needed[1]:
			self._aux_data.update({self._tF: self._bounds._tF_l})
		t_subs_dict = dict(zip(self._t_vars_user, t_vars))

		# Parameter variables
		s_vars = [sym.Symbol(f'_s{i_s}') for i_s, _ in enumerate(self._s_vars_user)]
		self._s_vars = sym.Matrix([s for s, s_needed in zip(s_vars, self._bounds._s_needed) if s_needed])
		if not self._s_vars:
			self._s_vars = sym.Matrix.zeros(0, 1)
		self._num_s_vars = self._s_vars.shape[0]
		self._aux_data.update({s: value for s, s_needed, value in zip(s_vars, self._bounds._s_needed, self._bounds._s_l) if not s_needed})
		s_subs_dict = dict(zip(self._s_vars_user, s_vars))

		# Variables set
		self._x_vars = sym.Matrix([self._y_vars, self._u_vars, self._q_vars, self._t_vars, self._s_vars])
		self._num_vars = self._x_vars.shape[0]
		self._num_vars_tuple = (self._num_y_vars, self._num_u_vars, self._num_q_vars, self._num_t_vars, self._num_s_vars)
		self._x_b_vars = sym.Matrix([self._y_b_vars, self._q_vars, self._t_vars, self._s_vars])
		self._user_subs_dict = {**y_subs_dict, **y_endpoint_subs_dict, **u_subs_dict, **t_subs_dict, **s_subs_dict, **a_subs_dict}

		# Substitutions
		self._aux_subs = {sym.Symbol(f'_e{i_e}'): value.subs(self._user_subs_dict) for i_e, (_, value) in enumerate(aux_subs_temp.items())}
		e_subs_dict = dict(zip(aux_subs_temp.keys(), self._aux_subs.keys()))
		self._user_subs_dict.update(e_subs_dict)

		# State equations
		self._y_eqns = sym.Matrix(self._y_eqns_user).subs(self._user_subs_dict) if self._y_eqns_user else sym.Matrix.zeros(0, 1)

		# Path constraints
		self._c_cons = sym.Matrix(self._c_cons_user).subs(self._user_subs_dict) if self._c_cons_user else sym.Matrix.zeros(0, 1)
		self._num_c_cons = self._c_cons.shape[0]

		# Integrand functions
		self._q_funcs = sym.Matrix(self._q_funcs_user).subs(self._user_subs_dict) if self._q_funcs_user else sym.Matrix.zeros(0, 1)

		# Boundary constraints
		self._y_b_cons = sym.Matrix(self._y_b_cons_user).subs(self._user_subs_dict) if self._y_b_cons_user else sym.Matrix.zeros(0, 1)
		self._b_end_cons = sym.Matrix(self._b_cons_user).subs(self._user_subs_dict) if self._b_cons_user else sym.Matrix.zeros(0, 1)
		self._b_cons = sym.Matrix([self._y_b_cons, self._b_end_cons])
		self._num_b_cons = self._b_cons.shape[0]

		# Auxiliary substitutions
		self._e_syms = sym.Matrix([eqn for eqn in self._aux_subs.keys()]) if self._aux_subs else sym.Matrix.zeros(0, 1)
		self._e_subs = sym.Matrix([eqn for eqn in self._aux_subs.values()]) if self._aux_subs else sym.Matrix.zeros(0, 1)
		self._num_e_subs = self._e_subs.shape[0]

		# Objective function
		self._J = self._J_user.subs(self._user_subs_dict)

		# Check user-defined initial guess
		self._initial_guess._guess_check()

		# Generate constraint and derivative functions
		# Variables index slices
		self._y_slice = slice(0, self._num_y_vars)
		self._u_slice = slice(self._y_slice.stop, self._y_slice.stop + self._num_u_vars)
		self._q_slice = slice(self._u_slice.stop, self._u_slice.stop + self._num_q_vars)
		self._t_slice = slice(self._q_slice.stop, self._q_slice.stop + self._num_t_vars)
		self._s_slice = slice(self._t_slice.stop, self._num_vars)
		self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)
		self._yu_qts_split = self._yu_slice.stop

		# Auxiliary substitutions derivatives
		self._de_dx = self._e_subs.jacobian(self._x_vars) if self._e_subs else sym.Matrix.zeros(0, 1)
		self._de_dxb = self._e_subs.jacobian(self._x_b_vars) if self._e_subs else sym.Matrix.zeros(0, 1)

		# Objective derivatives
		self._dJ_de = self._J.diff(self._e_syms)
		self._dJ_dxb = self._J.diff(self._x_b_vars)

		# Constraints
		self._c = sym.Matrix([self._y_eqns, self._c_cons, self._q_funcs, self._b_cons])
		self._num_c = self._c.shape[0]

		# Constraints index slices
		self._c_defect_slice = slice(0, self._num_y_vars)
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_cons)
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_q_vars)
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + self._num_b_cons)
		self._c_continuous_slice = slice(0, self._num_c - self._num_b_cons)

		# # Constraints derivatives Jacobian
		self._dc_dx = self._c[self._c_continuous_slice, :].jacobian(self._x_vars)
		self._dc_de = self._c[self._c_continuous_slice, :].jacobian(self._e_syms)

		# Initial and final time boundary derivatives
		self._db_dxb = self._b_cons.jacobian(self._x_b_vars)
		self._db_de = self._b_cons.jacobian(self._e_syms)
		

		if self._num_e_subs:
			self._dJ_dxb_chain = self._dJ_dxb + self._de_dxb.T*self._dJ_de
			self._dc_dx_chain = self._dc_dx + self._dc_de*self._de_dx
			self._db_dxb_chain = self._db_dxb + self._db_de*self._de_dxb
		else:
			self._dJ_dxb_chain = self._dJ_dxb
			self._dc_dx_chain = self._dc_dx
			self._db_dxb_chain = self._db_dxb 

		# Quadrature computations
		self._quadrature = Quadrature(optimal_control_problem=self)

		# Compile numba numerical functions
		_ = self._compile_numba_functions()

		# Initialise the initial mesh iterations
		self._mesh_iterations[0]._initialise_iteration(self.initial_guess)

		# Set the initialisation flag
		return True

	def solve(self):
		
		if self._initialised == False:
			self._initialised = self.initialise()

		# Solve the transcribed NLP on the initial mesh
		self._mesh_iterations[0]._solve()

