from numbers import Number

import numpy as np
import scipy.optimize as optimize
import sympy as sym

import pycollo.utils as pu

class Bounds():

	INF = 1e19
	
	def __init__(self, optimal_control_problem, *, initial_time=None, initial_time_lower=None, initial_time_upper=None, final_time=None, final_time_lower=None, final_time_upper=None, state=None, state_lower=None, state_upper=None, control=None, control_lower=None, control_upper=None, integral=None, integral_lower=None, integral_upper=None, parameter=None, parameter_lower=None, parameter_upper=None, path=None, path_lower=None, path_upper=None, state_endpoint=None, state_endpoint_lower=None, state_endpoint_upper=None, boundary=None, boundary_lower=None, boundary_upper=None, default_inf=False, infer_bounds=False):

		# Optimal Control Problem
		self._ocp = optimal_control_problem
		self.default_inf=default_inf
		self.infer_bounds=infer_bounds

		if initial_time is not None:
			_ = self.kwarg_conflict_check(initial_time_lower, initial_time_upper, 'initial_time')
			self.initial_time = initial_time
		else:
			self.initial_time_lower = initial_time_lower
			self.initial_time_upper = initial_time_upper

		if final_time is not None:
			_ = self.kwarg_conflict_check(final_time_lower, final_time_upper, 'final_time')
			self.final_time = final_time
		else:
			self.final_time_lower = final_time_lower
			self.final_time_upper = final_time_upper

		if state is not None:
			_ = self.kwarg_conflict_check(state_lower, state_upper, 'state')
			self.state = state
		else:
			self.state_lower = state_lower
			self.state_upper = state_upper

		if control is not None:
			_ = self.kwarg_conflict_check(control_lower, control_upper, 'control')
			self.control = control
		else:
			self.control_lower = control_lower
			self.control_upper = control_upper

		if integral is not None:
			_ = self.kwarg_conflict_check(integral_lower, integral_upper, 'integral')
			self.integral = integral
		else:
			self.integral_lower = integral_lower
			self.integral_upper = integral_upper

		if parameter is not None:
			_ = self.kwarg_conflict_check(parameter_lower, parameter_upper, 'parameter')
			self.parameter = parameter
		else:
			self.parameter_lower = parameter_lower
			self.parameter_upper = parameter_upper

		if path is not None:
			_ = self.kwarg_conflict_check(path_lower, path_upper, 'path')
			self.path = path
		else:
			self.path_lower = path_lower
			self.path_upper = path_upper

		if state_endpoint is not None:
			_ = self.kwarg_conflict_check(state_endpoint_lower, state_endpoint_upper, 'state_endpoint')
			self.state_endpoint = state_endpoint
		else:
			self.state_endpoint_lower = state_endpoint_lower
			self.state_endpoint_upper = state_endpoint_upper

		if boundary is not None:
			_ = self.kwarg_conflict_check(boundary_lower, boundary_upper, 'boundary')
			self.boundary = boundary
		else:
			self.boundary_lower = boundary_lower
			self.boundary_upper = boundary_upper

	@property
	def infer_bounds(self):
		return self._infer_bounds
	
	@infer_bounds.setter
	def infer_bounds(self, val):
		self._infer_bounds = bool(val)

	@property
	def default_inf(self):
		return self._default_inf
	
	@default_inf.setter
	def default_inf(self, val):
		self._default_inf = bool(val)

	@property
	def initial_time(self):
		return self.initial_time_lower, self.initial_time_upper
	
	@initial_time.setter
	def initial_time(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'initial_time', vec=False)
		self.initial_time_lower = l
		self.initial_time_upper = u

	@property
	def initial_time_lower(self):
		return self._t0_l
	
	@initial_time_lower.setter
	def initial_time_lower(self, t0_l):
		self._t0_l = self.parse_single_bounds(t0_l, 'initial_time_lower', vec=False)

	@property
	def initial_time_upper(self):
		return self._t0_u
	
	@initial_time_upper.setter
	def initial_time_upper(self, t0_u):
		self._t0_u = self.parse_single_bounds(t0_u, 'initial_time_upper', vec=False)

	@property
	def final_time(self):
		return (self.final_time_lower, self.final_time_upper)
	
	@final_time.setter
	def final_time(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'final_time', vec=False)
		self.final_time_lower = l
		self.final_time_upper = u

	@property
	def final_time_lower(self):
		return self._tF_l
	
	@final_time_lower.setter
	def final_time_lower(self, tF_l):
		self._tF_l = self.parse_single_bounds(tF_l, 'final_time_lower', vec=False)

	@property
	def final_time_upper(self):
		return self._tF_u
	
	@final_time_upper.setter
	def final_time_upper(self, tF_u):
		self._tF_u = self.parse_single_bounds(tF_u, 'final_time_upper', vec=False)

	@property
	def state(self):
		return self.return_user_bounds(self._y_l, self._y_u, 'state')
	
	@state.setter
	def state(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'state', self._ocp._y_vars_user)
		self.state_lower = l
		self.state_upper = u

	@property
	def state_lower(self):
		return self._y_l
	
	@state_lower.setter
	def state_lower(self, y_l):
		self._y_l = self.parse_single_bounds(y_l, 'state_lower', self._ocp._y_vars_user)

	@property
	def state_upper(self):
		return self._y_u
	
	@state_upper.setter
	def state_upper(self, y_u):
		self._y_u = self.parse_single_bounds(y_u, 'state_upper', self._ocp._y_vars_user)

	@property
	def control(self):
		return self.return_user_bounds(self._u_l, self._u_u, 'control')
	
	@control.setter
	def control(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'control', self._ocp._u_vars_user)
		self.control_lower = l
		self.control_upper = u

	@property
	def control_lower(self):
		return self._u_l
	
	@control_lower.setter
	def control_lower(self, u_l):
		self._u_l = self.parse_single_bounds(u_l, 'control_lower', self._ocp._u_vars_user)

	@property
	def control_upper(self):
		return self._u_u
	
	@control_upper.setter
	def control_upper(self, u_u):
		self._u_u = self.parse_single_bounds(u_u, 'control_upper', self._ocp._u_vars_user)

	@property
	def integral(self):
		return self.return_user_bounds(self._q_l, self._q_u, 'integral')
	
	@integral.setter
	def integral(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'integral', self._ocp._q_vars_user)
		self.integral_lower = l
		self.integral_upper = u

	@property
	def integral_lower(self):
		return self._q_l
	
	@integral_lower.setter
	def integral_lower(self, q_l):
		self._q_l = self.parse_single_bounds(q_l, 'integral_lower', self._ocp._q_vars_user)

	@property
	def integral_upper(self):
		return self._q_u
	
	@integral_upper.setter
	def integral_upper(self, q_u):
		self._q_u = self.parse_single_bounds(q_u, 'integral_upper', self._ocp._q_vars_user)

	@property
	def time(self):
		return np.array([self.initial_time, self.final_time])

	# Include properties for time variables so that initial and final times can be set in the same manner as all other state, control, integral and parameter variables.

	@property
	def parameter(self):
		return self.return_user_bounds(self._s_l, self._s_u, 'parameter')
	
	@parameter.setter
	def parameter(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'parameter', self._ocp._s_vars_user)
		self.parameter_lower = l
		self.parameter_upper = u

	@property
	def parameter_lower(self):
		return self._s_l
	
	@parameter_lower.setter
	def parameter_lower(self, s_l):
		self._s_l = self.parse_single_bounds(s_l, 'parameter_lower', self._ocp._s_vars_user)

	@property
	def parameter_upper(self):
		return self._s_u
	
	@parameter_upper.setter
	def parameter_upper(self, s_u):
		self._s_u = self.parse_single_bounds(s_u, 'parameter_upper', self._ocp._s_vars_user)

	@property
	def path(self):
		return self.return_user_bounds(self._c_l, self._c_u, 'path')
	
	@path.setter
	def path(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'path', self._ocp._c_cons_user)
		self.path_lower = l
		self.path_upper = u

	@property
	def path_lower(self):
		return self._c_l
	
	@path_lower.setter
	def path_lower(self, c_l):
		self._c_l = self.parse_single_bounds(c_l, 'path_lower', self._ocp._c_cons_user)

	@property
	def path_upper(self):
		return self._c_u
	
	@path_upper.setter
	def path_upper(self, c_u):
		self._c_u = self.parse_single_bounds(c_u, 'path_upper', self._ocp._c_cons_user)

	@property
	def state_endpoint(self):
		return self.return_user_bounds(self._y_b_l, self._y_b_u, 'state_endpoint')
	
	@state_endpoint.setter
	def state_endpoint(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'state_endpoint', self._ocp._y_b_cons_user)
		self.state_endpoint_lower = l
		self.state_endpoint_upper = u

	@property
	def state_endpoint_lower(self):
		return self._y_b_l
	
	@state_endpoint_lower.setter
	def state_endpoint_lower(self, b_l):
		self._y_b_l = self.parse_single_bounds(b_l, 'state_endpoint_lower', self._ocp._y_b_cons_user)

	@property
	def state_endpoint_upper(self):
		return self._y_b_u
	
	@state_endpoint_upper.setter
	def state_endpoint_upper(self, b_u):
		self._y_b_u = self.parse_single_bounds(b_u, 'state_endpoint_upper', self._ocp._y_b_cons_user)

	@property
	def boundary(self):
		return self.return_user_bounds(self._b_l, self._b_u, 'boundary')
	
	@boundary.setter
	def boundary(self, bnd):
		l, u = self.parse_both_bounds(bnd, 'boundary', self._ocp._b_cons_user)
		self.boundary_lower = l
		self.boundary_upper = u

	@property
	def boundary_lower(self):
		return self._b_l
	
	@boundary_lower.setter
	def boundary_lower(self, b_l):
		self._b_l = self.parse_single_bounds(b_l, 'boundary_lower', self._ocp._b_cons_user)

	@property
	def boundary_upper(self):
		return self._b_u
	
	@boundary_upper.setter
	def boundary_upper(self, b_u):
		self._b_u = self.parse_single_bounds(b_u, 'boundary_upper', self._ocp._b_cons_user)

	def _bounds_check(self, aux_data=None, aux_subs=None):

		def compare_lower_and_upper(lower, upper, name):
			if not np.logical_or(np.less(lower, upper), np.isclose(lower, upper)).all():
				msg = (f"Lower bound of {lower} for {name} must be less than or equal to upper bound of {upper}.")
				raise ValueError(msg)
			vars_needed = np.ones(len(lower), dtype=int)
			for i, (l, u) in enumerate(zip(lower, upper)):
				if np.isclose(l, u):
					vars_needed[i] = 0
			return vars_needed

		def parse_bounds_to_np_array(bnds, syms, bnd_str):
			if isinstance(bnds, np.ndarray):
				num_y = len(syms)
				if bnds.shape[0] < num_y:
					msg = (f"Insufficient number of bounds have been supplied for `Bounds.{bnd_str}`. Currently {bnds.shape[0]} pairs of bounds are supplied where {num_y} pairs are needed.")
					raise ValueError(msg)
				elif bnds.shape[0] > num_y:
					msg = (f"Excess bounds have been supplied for `Bounds.{bnd_str}`. Currently {bnds.shape[0]} pairs of bounds are supplied where only {num_y} pairs are needed.")
					raise ValueError(msg)
				return_bnds = bnds
			elif isinstance(bnds, dict):
				extra_keys = set(bnds.keys()).difference(set(syms))
				if extra_keys:
					raise ValueError
				return_bnds = np.array([bnds.get(sym, (None, None)) for sym in syms], dtype=np.float64)
			return replace_non_float_vals(return_bnds, syms)

		def replace_non_float_vals(bnds, syms):
			bnds[bnds < -self.INF] = -self.INF
			bnds[bnds > self.INF] = self.INF
			if self._default_inf:
				bnds[bnds == -np.nan] = -self.INF
				bnds[bnds == np.nan] = self.INF
			elif np.isnan(np.sum(bnds)):
				for bnd_pair, symb in zip(bnds, syms):
					l, u = bnd_pair
					if np.isnan(l) and np.isnan(u):
						msg = (f"Both lower and upper bounds are missing for {symb}.")
						raise ValueError(msg)
					elif np.isnan(l):
						msg = (f"A lower bound is missing for {symb}.")
						raise ValueError(msg)
					elif np.isnan(u):
						msg = (f"An upper bound is missing for {symb}.")
						raise ValueError(msg)
			return bnds[:, 0], bnds[:, 1]

		def infer_bound(bnds, syms, eqns, aux_data, aux_subs, bnd_str):

			def q_unpack(x):
				q = q_lambda(*x)
				return q

			def J_unpack(x):
				J = np.array(J_lambda(*x).tolist()).squeeze()
				return J

			if isinstance(bnds, np.ndarray):
				if bnds.size == 0:
					bnds = np.empty((len(syms), 2))
					bnds.fill(np.nan)
				elif bnds.shape == (len(syms), 2):
					pass
				else:
					msg = (f"Bounds for `Bounds.{bnd_str}` could not be automatically inferred as lower and upper bounds have been supplied as an iterable with length different to the number of {bnd_str} symbols in the optimal control problem. The supplied bounds could not be mapped to their corresponding symbols. Please resupply bounds as a {dict}.")
					raise ValueError(msg)
			elif isinstance(bnds, dict):
				bnds = np.array([bnds.get(s, (None, None)) for s in syms], dtype=np.float64)

			x_lb = np.concatenate((self._y_l, self._u_l, np.zeros(len(self._ocp._q_vars_user) + 2), self._s_l))
			x_ub = np.concatenate((self._y_u, self._u_u, np.zeros(len(self._ocp._q_vars_user) + 2), self._s_u))
			x_bnds = np.array([x_lb, x_ub]).transpose()
			q_bnds = []
			for symb, eqn, bnd_user in zip(syms, eqns, bnds):

				if np.isnan(bnd_user).any():

					# Prepare SQP problem
					eqn = eqn.subs(aux_subs)
					eqn = eqn.subs(aux_data)
					min_args = []
					min_bnds = []
					for x, bnd_pair in zip(self._ocp._x_vars_user, x_bnds):
						if x in eqn.free_symbols:
							min_args.append(x)
							min_bnds.append(bnd_pair.tolist())
					min_bnds = np.array(min_bnds)
					x0 = min_bnds[:, 0] + np.random.random(len(min_args))*(min_bnds[:, 1] - min_bnds[:, 0])

					# Minimize
					if np.isnan(bnd_user[0]):
						q_lambda = sym.lambdify(min_args, eqn, modules='numpy')
						jacobian = eqn.diff(sym.Matrix(min_args))
						J_lambda = sym.lambdify(min_args, jacobian, modules='numpy')
						rslts = optimize.minimize(q_unpack, x0, method='SLSQP', jac=J_unpack, bounds=min_bnds)
						q_lb = q_unpack(rslts.x)
					else:
						q_lb = bnd_user[0]

					# Maximize
					if np.isnan(bnd_user[1]):
						q_lambda = sym.lambdify(min_args, -eqn, modules='numpy')
						jacobian = eqn.diff(sym.Matrix(min_args))
						J_lambda = sym.lambdify(min_args, -jacobian, modules='numpy')
						rslts = optimize.minimize(q_unpack, x0, method='SLSQP', jac=J_unpack, bounds=min_bnds)
						q_ub = -q_unpack(rslts.x)
					else:
						q_ub = bnd_user[1]
				else:
					q_lb, q_ub = bnd_user

				q_bnds.append([q_lb, q_ub])

			return np.array(q_bnds)

		# State
		self._y_l, self._y_u = parse_bounds_to_np_array(self.state, self._ocp._y_vars_user, 'state')
		self._y_needed = compare_lower_and_upper(self._y_l, self._y_u, 'state')

		self._y_l_needed = self._y_l[self._y_needed.astype(bool)]
		self._y_u_needed = self._y_u[self._y_needed.astype(bool)]

		# Control
		self._u_l, self._u_u = parse_bounds_to_np_array(self.control, self._ocp._u_vars_user, 'control')
		self._u_needed = compare_lower_and_upper(self._u_l, self._u_u, 'control')

		self._u_l_needed = self._u_l[self._u_needed.astype(bool)]
		self._u_u_needed = self._u_u[self._u_needed.astype(bool)]

		# Time
		_ = parse_bounds_to_np_array(np.array([[self._t0_l, self._tF_l], [self._t0_u, self._tF_u]], dtype=np.float64), self._ocp._t_vars_user, 'time')
		if self._t0_l > self._t0_u:
			msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be greater than the lower bound ({self._t0_l}).")
			raise ValueError(msg)
		elif self._tF_l > self._tF_u:
			msg = (f"The upper bound for `Bounds.final_time` ({self._tF_u}) must be greater than the lower bound ({self._tF_l}).")
			raise ValueError(msg)
		elif self._t0_l > self._tF_l:
			msg = (f"The lower bound for `Bounds.initial_time` ({self._t0_l}) must be less than the lower bound for `Bounds.final_time` ({self._tF_l}).")
			raise ValueError(msg)
		elif self._t0_u > self._tF_u:
			msg = (f"The upper bound for `Bounds.initial_time` ({self._t0_u}) must be less than the upper bound for `Bounds.final_time` ({self._tF_u}).")
			raise ValueError(msg)
		else:
			self._t0_l = self.initial_time_lower
			self._t0_u = self.initial_time_upper
			self._tF_l = self.final_time_lower
			self._tF_u = self.final_time_upper
			self._t_l = np.array([self._t0_l, self._tF_l]).flatten()
			self._t_u = np.array([self._t0_u, self._tF_u]).flatten()
		self._t_needed = compare_lower_and_upper([self._t0_l, self._tF_l], [self._t0_u, self._tF_u], 'time')

		self._t_l_needed = self._t_l[self._t_needed.astype(bool)]
		self._t_u_needed = self._t_u[self._t_needed.astype(bool)]

		# print('\n\n\n')

		# print(self._t_u)


		# print('\n\n\n')
		# raise ValueError

		

		# Parameter
		self._s_l, self._s_u = parse_bounds_to_np_array(self.parameter, self._ocp._s_vars_user, 'parameter')
		self._s_needed = compare_lower_and_upper(self._s_l, self._s_u, 'parameter')

		self._s_l_needed = self._s_l[self._s_needed.astype(bool)]
		self._s_u_needed = self._s_u[self._s_needed.astype(bool)]

		# Integral
		if self.infer_bounds:
			self.integral = infer_bound(self.integral, self._ocp._q_vars_user, self._ocp._q_funcs_user, aux_data, aux_subs, 'integral')
		self._q_l, self._q_u = parse_bounds_to_np_array(self.integral, self._ocp._q_vars_user, 'integral')
		self._q_needed = compare_lower_and_upper(self._q_l, self._q_u, 'integral')

		self._q_l_needed = self._q_l[self._q_needed.astype(bool)]
		self._q_u_needed = self._q_u[self._q_needed.astype(bool)]

		# Path
		if self.infer_bounds:
			if self._ocp._c_cons_user:
				raise NotImplementedError
		self._c_l, self._c_u = parse_bounds_to_np_array(self.path, self._ocp._c_cons_user, 'path constraint')
		_ = compare_lower_and_upper(self._c_l, self._c_u, 'path constraint')

		# Boundary
		self._y_b_l, self._y_b_u = parse_bounds_to_np_array(self.state_endpoint, self._ocp._y_b_cons_user, 'state_endpoint_constraint')
		_ = compare_lower_and_upper(self._b_l, self._b_u, 'state_endpoint_constraint')

		# Boundary
		self._b_l, self._b_u = parse_bounds_to_np_array(self.boundary, self._ocp._b_cons_user, 'boundary constraint')
		_ = compare_lower_and_upper(self._b_l, self._b_u, 'boundary constraint')

	@staticmethod
	def kwarg_conflict_check(lower, upper, kwarg_str):
		if lower is not None or upper is not None:
			msg = (f"If the key-word argument `{kwarg_str}` is used then the key-word arguments `{kwarg_str}_lower` and `{kwarg_str}_upper` cannot be used.")
			raise TypeError(msg)
		return None

	@staticmethod
	def split_both_bounds_dict(bnds, bnd_str, syms, vec):
		l = {}
		u = {}
		for bnd_sym, bnd_pair in bnds.items():
			if isinstance(bnd_pair, pu.supported_iter_types):
				if len(bnd_pair) != 2:
					msg = (f"When values for lower and upper bounds are supplied together using a {dict} and the `Bounds.{bnd_str}` attribute, the  values must be a iterables of length 2: (`lower_bound`, `upper_bound`).")
					raise ValueError(msg)
			else:
				try:
					bnd_pair = np.array([bnd_pair, bnd_pair], dtype=np.float64)
				except ValueError:
					msg = (f"When values for lower and upper bounds are supplied together using a {dict} and the `Bounds.{bnd_str}` attribute, the  values must be a iterables of length 2: (`lower_bound`, `upper_bound`).")
					raise ValueError(msg)
			l.update({bnd_sym: bnd_pair[0]})
			u.update({bnd_sym: bnd_pair[1]})
		return l, u

	@staticmethod
	def split_both_bounds_iter(bnds, bnd_str, syms, vec):
		
		if vec is False:
			if len(bnds) != 2:
				msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be of length 2.")
				raise ValueError(msg)
			return float(bnds[0]), float(bnds[1])
		else:
			bnds = np.array(bnds)
			if bnds.ndim == 2:
				if bnds.shape[1] != 2:
					msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be a two-dimensional iterable with first dimension of length 2. Current dimensions are: {bnds.shape}.")
					raise ValueError(msg)
			elif bnds.ndim > 2:
				msg = (f"Bounds supplied to `Bounds.{bnd_str}` of type {type(bnds)} must be a two-dimensional iterable. The currently supplied iterable's dimensions are: {bnds.shape}.")
				raise ValueError(msg)
			bnds = np.array(bnds, dtype=float).flatten().reshape((-1, 2), order='C')
			return bnds[:, 0], bnds[:, 1]
		
	@staticmethod
	def parse_single_bounds(bnds, bnd_str, syms=None, vec=True):

		if vec is True:
			if bnds is None:
				return np.array([])
			elif isinstance(bnds, dict):
				for bnd_sym in bnds.keys():
					if bnd_sym not in syms:
						msg = (f"{bnd_sym} cannot be supplied as a key value for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` as it is not a {bnd_str} variables in the optimal control problem.")
						raise ValueError(msg)
				try:
					return {bnd_sym: float(bnd) for bnd_sym, bnd in bnds.items()}
				except TypeError:
					msg = (f"Dictionary values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` must be convertable into {float} objects.")
					raise TypeError(msg)
			elif isinstance(bnds, pu.supported_iter_types):
				return np.array(bnds, dtype=np.float64)
			else:
				return np.array([bnds], dtype=np.float64)

		else:
			if isinstance(bnds, dict):
				msg = (f"Value for `Bounds.{bnd_str}` cannot be of type {dict}.")
				raise ValueError(msg)
			else:
				return None if bnds is None else float(bnds)

	@classmethod
	def parse_both_bounds(cls, bnds, bnd_str, syms=None, vec=True):

		if vec is True:
			if bnds is None:
				return None, None
			elif isinstance(bnds, dict):
				return cls.split_both_bounds_dict(bnds, bnd_str, syms, vec)
			elif isinstance(bnds, pu.supported_iter_types):
				return cls.split_both_bounds_iter(bnds, bnd_str, syms, vec)
			else:
				return np.array([bnds], dtype=np.float64), np.array([bnds], dtype=np.float64)

		else:
			if isinstance(bnds, dict):
				msg = (f"Value for `Bounds.{bnd_str}` cannot be of type {dict}.")
				raise ValueError(msg)
			elif isinstance(bnds, pu.supported_iter_types):
				return cls.split_both_bounds_iter(bnds, bnd_str, syms, vec)
			else:
				return float(bnds), float(bnds)

	@staticmethod
	def return_user_bounds(l, u, bnd_str):
		if type(l) is type(u):
			if isinstance(l, np.ndarray):
				try:
					return np.array([l, u], dtype=np.float64).transpose()
				except ValueError:
					msg = (f"Values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` have been supplied as iterables of different length. Pycollo cannot infer which lower and upper bounds correspond to which {bnd_str} variables.")
					raise ValueError(msg)
			elif isinstance(l, dict):
				merged = {}
				sym_keys = set.union(set(l.keys()), set(u.keys()))
				for k in sym_keys:
					merged.update({k: (l.get(k), u.get(k))})
				return merged
		else:
			msg = (f"Values for `Bounds.{bnd_str}_lower` and `Bounds.{bnd_str}_upper` have been supplied as iterables of different types. If lower and upper bounds for {bnd_str} are being supplied separately, please ensure that these are both either of type {dict} or type {np.ndarray}.")
			raise TypeError(msg)
	
		
		