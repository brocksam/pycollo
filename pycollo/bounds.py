import numpy as np

class Bounds():

	INF = 1e19
	
	def __init__(self, *, optimal_control_problem=None, initial_time=None, initial_time_lower=None, initial_time_upper=None, final_time=None, final_time_lower=None, final_time_upper=None, state=None, state_lower=None, state_upper=None, control=None, control_lower=None, control_upper=None, integral=None, integral_lower=None, integral_upper=None, parameter=None, parameter_lower=None, parameter_upper=None, path=None, path_lower=None, path_upper=None, boundary=None, boundary_lower=None, boundary_upper=None):

		# Optimal Control Problem
		self._ocp = optimal_control_problem

		self.initial_time = initial_time
		self.initial_time_lower = initial_time_lower
		self.initial_time_upper = initial_time_upper
		self.final_time = final_time
		self.final_time_lower = final_time_lower
		self.final_time_upper = final_time_upper

		self.state = state
		self.state_lower = state_lower
		self.state_upper = state_upper

		self.control = control
		self.control_lower = control_lower
		self.control_upper = control_upper

		self.integral = integral
		self.integral_lower = integral_lower
		self.integral_upper = integral_upper

		self.parameter = parameter
		self.parameter_lower = parameter_lower
		self.parameter_upper = parameter_upper

		self.path = path
		self.path_lower = path_lower
		self.path_upper = path_upper

		self.boundary = boundary
		self.boundary_lower = boundary_lower
		self.boundary_upper = boundary_upper

	def _bounds_check(self):

		def parse_bounds_input(bounds, bounds_l, bounds_u, syms, var_name):
			if len(syms) == 0:
				return np.array([]), np.array([])

			if bounds is not None:
				if bounds_l is not None or bounds_u is not None:
					msg = (f"Bounds for {var_name} were supplied multiple times. Either supply bounds using the key-word argument `{var_name}` or using the key-word arguments `{var_name}_lower` and `{var_name}_upper`.")
					raise ValueError(msg)
				try:
					bounds_pairs = bounds.values()
				except AttributeError:
					bounds = np.array(bounds, dtype=float).flatten().reshape((-1, 2), order='C')
				else:
					for bound_pair in bounds_pairs:
						try:
							bound_pair_array = np.array(bound_pair, dtype=float)
						except TypeError:
							msg = (f"Bound values for {var_name} must be supplied as a one-dimensional iterable.")
							raise TypeError(msg)
						if bound_pair_array.shape != (2, ):
							msg = (f"Bound values for {var_name} must be supplied as a one-dimensional iterables of two values.")
							raise ValueError(msg)

			elif bounds_l is not None and bounds_u is not None:
				try:
					keys_l = bounds_l.keys()
				except AttributeError:
					try:
						bounds = np.array([bounds_l, bounds_u], dtype=float)
					except ValueError:
						try: 
							keys_u = bounds_u.keys()
						except AttributeError:
							bounds_l_len = len(bounds_l)
							bounds_u_len = len(bounds_u)
							if bounds_l_len != bounds_u_len:
								msg = (f"Lower and upper bounds must be provided for all variables in {var_name}. Currently {bounds_l_len} lower bounds and {bounds_u_len} upper bounds are provided.")
								raise ValueError(msg)
						else:
							msg = (f"Upper and lower bounds for {var_name} must be passed as the same type. Currently the lower bounds for {var_name} are {type(bounds_l)} and the supper bounds are {type(bounds_u)}. Please provide both either as an iterable or as a dict with keys as symbols in the optimal control problem.")
							raise TypeError(msg)
					else:
						bounds = bounds.flatten().reshape((-1, 2), order='F')
				else:
					try:
						keys_u = bounds_u.keys()
					except AttributeError:
						msg = (f"Upper and lower bounds for {var_name} must be passed as the same type. Currently the lower bounds for {var_name} are {type(bounds_l)} and the supper bounds are {type(bounds_u)}. Please provide both either as an iterable or as a dict with keys as symbols in the optimal control problem.")
						raise TypeError(msg)
					else:
						keys_l_set = set(keys_l)
						keys_u_set = set(keys_u)
						if keys_l_set != keys_u_set:
							msg = (f"Both lower and upper bound dicts for {var_name} must contain the same set of keys. `{var_name}_lower` contains the keys: {keys_l_set}, while `{var_name}_upper` contains the keys: {keys_u_set}.")
							raise KeyError(msg)
						try:
							bounds = {key: np.array([bounds_l[key], bounds_u[key]], dtype=float) for key in keys_l}
						except ValueError:
							msg = (f"Values in {var_name} lower and upper bounds dicts must be single numbers.")
							raise ValueError(msg)

			else:
				if bounds_l is None and bounds is None:
					msg = (f"No bounds have been supplied for {var_name}.")
					raise ValueError(msg)
				else:
					msg = (f"If supplying upper and lower bounds for {var_name} separately, bounds must be supplied using both the key-word arguments `{var_name}_lower` and `{var_name}_upper`.")
					raise ValueError(msg)

			try:
				len_bounds = len(bounds.keys())
			except AttributeError:
				bounds_array = np.array(bounds, dtype=float)
				if bounds_array.ndim == 1:
					bounds_array = bounds_array.reshape(-1, 2)
				if bounds_array.shape != (len(syms), 2):
					msg = (f"Bounds for {var_name} must be passed as a two-dimensional iterable; an iterable of length iterables of pairs of lower and upper bounds. Currently {bounds_array.shape[0]} pairs of bounds are supplied where {len(syms)} are needed.")
					raise ValueError(msg)
			else:
				if len(syms) != len_bounds:
					missing_bounds = ', '.join(f'{symbol}' for symbol in set(syms).difference(set(bounds.keys())))
					msg = (f"Bounds must be provided for each {var_name} variable. Please also provide bounds for: {missing_bounds}.")
					raise ValueError(msg)
				bounds_array = np.empty((len(syms), 2), dtype=float)
				for i_sym, symbol in enumerate(syms):
					bounds_array[i_sym, :] = bounds[symbol]

			bounds_array[bounds_array == -np.inf] = -self.INF
			bounds_array[bounds_array == np.inf] = self.INF

			return bounds_array[:, 0], bounds_array[:, 1]

		def compare_lower_and_upper(lower, upper, name):
			if not np.logical_or(np.less(lower, upper), np.isclose(lower, upper)).all():
				msg = (f"Lower bound of {lower} for {name} must be less than or equal to upper bound of {upper}.")
				raise ValueError(msg)
			vars_needed = np.ones(len(lower), dtype=int)
			for i, (l, u) in enumerate(zip(lower, upper)):
				if np.isclose(l, u):
					vars_needed[i] = 0
			return vars_needed

		# State
		self._y_l, self._y_u = parse_bounds_input(self.state, self.state_lower, self.state_upper, self._ocp._y_vars_user, 'state')
		self._y_needed = compare_lower_and_upper(self._y_l, self._y_u, 'state')

		self._y_l_needed = self._y_l[self._y_needed.astype(bool)]
		self._y_u_needed = self._y_u[self._y_needed.astype(bool)]

		# Control
		self._u_l, self._u_u = parse_bounds_input(self.control, self.control_lower, self.control_upper, self._ocp._u_vars_user, 'control')
		self._u_needed = compare_lower_and_upper(self._u_l, self._u_u, 'control')

		self._u_l_needed = self._u_l[self._u_needed.astype(bool)]
		self._u_u_needed = self._u_u[self._u_needed.astype(bool)]

		# Integral
		self._q_l, self._q_u = parse_bounds_input(self.integral, self.integral_lower, self.integral_upper, self._ocp._q_vars_user, 'integral')
		self._q_needed = compare_lower_and_upper(self._q_l, self._q_u, 'integral')

		self._q_l_needed = self._q_l[self._q_needed.astype(bool)]
		self._q_u_needed = self._q_u[self._q_needed.astype(bool)]

		# Time
		self._t0_l, self._t0_u = parse_bounds_input(self.initial_time, self.initial_time_lower, self.initial_time_upper, [self._ocp._t0_USER], 'initial time')
		self._tF_l, self._tF_u = parse_bounds_input(self.final_time, self.final_time_lower, self.final_time_upper, [self._ocp._tF_USER], 'final time')
		self._t_l = np.array([self._t0_l, self._tF_l]).flatten()
		self._t_u = np.array([self._t0_u, self._tF_u]).flatten()
		self._t_needed = compare_lower_and_upper([self._t0_l, self._tF_l], [self._t0_u, self._tF_u], 'time')

		self._t_l_needed = self._t_l[self._t_needed.astype(bool)]
		self._t_u_needed = self._t_u[self._t_needed.astype(bool)]

		# Parameter
		self._s_l, self._s_u = parse_bounds_input(self.parameter, self.parameter_lower, self.parameter_upper, self._ocp._s_vars_user, 'parameter')
		self._s_needed = compare_lower_and_upper(self._s_l, self._s_u, 'parameter')

		self._s_l_needed = self._s_l[self._s_needed.astype(bool)]
		self._s_u_needed = self._s_u[self._s_needed.astype(bool)]

		# Path
		self._c_l, self._c_u = parse_bounds_input(self.path, self.path_lower, self.path_upper, self._ocp._c_cons_user, 'path constraint')
		self._c_needed = compare_lower_and_upper(self._c_l, self._c_u, 'path constraint')

		# Boundary
		self._b_l, self._b_u = parse_bounds_input(self.boundary, self.boundary_lower, self.boundary_upper, self._ocp._b_cons_user, 'boundary constraint')
		self._b_needed = compare_lower_and_upper(self._b_l, self._b_u, 'boundary constraint')
		