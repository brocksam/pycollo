import numpy as np

class Bounds():

	INF = 10e19
	
	def __init__(self, *, optimal_control_problem=None, initial_time=None, initial_time_lower=0.0, initial_time_upper=0.0, final_time=None, final_time_lower=None, final_time_upper=None, state=None, state_lower=None, state_upper=None, control=None, control_lower=None, control_upper=None, integral=None, integral_lower=None, integral_upper=None, parameter=None, parameter_lower=None, parameter_upper=None, path=None, path_lower=None, path_upper=None, boundary=None, boundary_lower=None, boundary_upper=None, by_var=True):

		# Optimal Control Problem
		self._ocp = optimal_control_problem

		# Initial time
		self._check_upper_lower_supplied_once(initial_time, initial_time_lower, initial_time_upper, 'initial time')
		if initial_time:
			self.initial_time = initial_time
		else:
			self.initial_time_lower = initial_time_lower
			self.initial_time_upper = initial_time_upper

		# Final time
		self._check_upper_lower_supplied_once(final_time, final_time_lower, final_time_upper, 'final time')
		if final_time:
			self.final_time = final_time
		else:
			self.final_time_lower = final_time_lower
			self.final_time_upper = final_time_upper

		# State
		self._check_upper_lower_supplied_once(state, state_lower, state_upper, 'state')
		if state:
			self.state = state
		else:
			self.state_lower = state_lower
			self.state_upper = state_upper

		# Control
		self._check_upper_lower_supplied_once(control, control_lower, control_upper, 'control')
		if control:
			self.control = control
		else:
			self.control_lower = control_lower
			self.control_upper = control_upper

		# Integral
		self._check_upper_lower_supplied_once(integral, integral_lower, integral_upper, 'integral')
		if integral:
			self.integral = integral
		else:
			self.integral_lower = integral_lower
			self.integral_upper = integral_upper

		# Parameter
		self._check_upper_lower_supplied_once(parameter, parameter_lower, parameter_upper, 'parameter')
		if parameter:
			self.parameter = parameter
		else:
			self.parameter_lower = parameter_lower
			self.parameter_upper = parameter_upper

		# Path
		self._check_upper_lower_supplied_once(path, path_lower, path_upper, 'path')
		if path:
			self.path = path
		else:
			self.path_lower = path_lower
			self.path_upper	= path_upper

		# Boundary
		self._check_upper_lower_supplied_once(boundary, boundary_lower, boundary_upper, 'boundary')
		if boundary:
			self.boundary = boundary
		else:
			self.boundary_lower = boundary_lower
			self.boundary_upper = boundary_upper

	@staticmethod
	def _check_upper_lower_supplied_once(both, lower, upper, name):
		if both and (lower or upper):
			msg = ("Bounds for {0} were supplied multiple times. Either supply bounds using the key-word argument `{0}` or using the key-word arguments `{0}_lower` and `{0}_upper`.".format(name))
			raise ValueError(msg)

	# @property
	# def optimal_control_problem(self):
	# 	return self._optimal_control_problem

	@property
	def initial_time(self):
		return np.array([self._initial_time_lower, self._initial_time_upper])

	@initial_time.setter
	def initial_time(self, initial_time):
		for index, row in enumerate(initial_time.copy()):
			initial_time[index] = np.array(row)
		initial_time = np.array(initial_time)
		self.initial_time_lower = initial_time[:, 0]
		self.initial_time_upper = initial_time[:, 1]

	@property
	def initial_time_lower(self):
		return self._t0_l
	
	@initial_time_lower.setter
	def initial_time_lower(self, t0_l):
		self._t0_l = self._format_as_single_val(t0_l)

	@property
	def initial_time_upper(self):
		return self._t0_u
	
	@initial_time_upper.setter
	def initial_time_upper(self, t0_u):
		self._t0_u = self._format_as_single_val(t0_u)

	@property
	def final_time(self):
		return np.array([self._final_time_lower, self._final_time_upper])

	@final_time.setter
	def final_time(self, final_time):
		for index, row in enumerate(final_time.copy()):
			final_time[index] = np.array(row)
		final_time = np.array(final_time)
		self.final_time_lower = final_time[:, 0]
		self.final_time_upper = final_time[:, 1]

	@property
	def final_time_lower(self):
		return self._tF_l
	
	@final_time_lower.setter
	def final_time_lower(self, tF_l):
		self._tF_l = self._format_as_single_val(tF_l)

	@property
	def final_time_upper(self):
		return self._tF_u
	
	@final_time_upper.setter
	def final_time_upper(self, tF_u):
		self._tF_u = self._format_as_single_val(tF_u)

	@property
	def state(self):
		return np.array([self._y_l, self._y_u])

	@state.setter
	def state(self, state):
		for index, row in enumerate(state.copy()):
			state[index] = np.array(row)
		state = np.array(state)
		self.state_lower = state[:, 0]
		self.state_upper = state[:, 1]

	@property
	def state_lower(self):
		return self._y_l

	@state_lower.setter
	def state_lower(self, y_l):
		self._y_l = self._format_as_np_array(y_l)

	@property
	def state_upper(self):
		return self._y_u

	@state_upper.setter
	def state_upper(self, y_u):
		self._y_u = self._format_as_np_array(y_u)

	@property
	def control_lower(self):
		return self._u_l

	@control_lower.setter
	def control_lower(self, u_l):
		self._u_l = self._format_as_np_array(u_l)

	@property
	def control_upper(self):
		return self._u_u

	@control_upper.setter
	def control_upper(self, u_u):
		self._u_u = self._format_as_np_array(u_u)

	@property
	def integral_lower(self):
		return self._q_l

	@integral_lower.setter
	def integral_lower(self, q_l):
		self._q_l = self._format_as_np_array(q_l)

	@property
	def integral_upper(self):
		return self._q_u

	@integral_upper.setter
	def integral_upper(self, q_u):
		self._q_u = self._format_as_np_array(q_u)

	@property
	def parameter_lower(self):
		return self._s_l

	@parameter_lower.setter
	def parameter_lower(self, s_l):
		self._s_l = self._format_as_np_array(s_l)

	@property
	def parameter_upper(self):
		return self._s_u

	@parameter_upper.setter
	def parameter_upper(self, s_u):
		self._s_u = self._format_as_np_array(s_u)

	@property
	def path_lower(self):
		return self._c_l

	@path_lower.setter
	def path_lower(self, c_l):
		self._c_l = self._format_as_np_array(c_l)

	@property
	def path_upper(self):
		return self._c_u

	@path_upper.setter
	def path_upper(self, c_u):
		self._c_u = self._format_as_np_array(c_u)

	@property
	def boundary_lower(self):
		return self._b_l

	@boundary_lower.setter
	def boundary_lower(self, b_l):
		self._b_l = self._format_as_np_array(b_l)

	@property
	def boundary_upper(self):
		return self._b_u

	@boundary_upper.setter
	def boundary_upper(self, b_u):
		self._b_u = self._format_as_np_array(b_u)

	@classmethod
	def _format_as_single_val(cls, bound):
		if bound is not None:
			bound = float(bound)
		if bound == np.inf:
			bound = cls.INF
		elif bound == -np.inf:
			bound = -cls.INF
		return bound

	@classmethod
	def _format_as_np_array(cls, bounds):
		try:
			float(bounds)
		except TypeError:
			try:
				iter(bounds)
			except TypeError:
				bounds = []
			else:
				bounds = np.array(bounds).astype(np.float64)
		else:
			bounds = [bounds]
		for index, bound in enumerate(bounds):
			bound = float(bound)
			if bound == np.inf:
				bounds[index] = cls.INF
			elif bound == -np.inf:
				bounds[index] = -cls.INF
		bounds_as_np_array = np.array(bounds)
		if bounds_as_np_array.ndim is not 1:
			msg = ("Bounds must be passed either as a single value or as a single-dimensional iterable of values.")
			raise ValueError(msg)
		return bounds_as_np_array

	def _bounds_check(self):

		def lower_less_than_upper_single_val(lower, upper, name):
			if lower > upper:
				msg = ("Lower bound of {0} for {2} must be less than or equal to upper bound of {1}.")
				raise ValueError(msg.format(lower, upper, name))

		def lower_less_than_upper_np_array(lowers, uppers, name):
			for index, (lower, upper) in enumerate(zip(lowers, uppers)):
				if lower > upper:
					msg = ("Lower bound of {0} for {2} {3} must be less than or equal to upper bound of {1}.")
					raise ValueError(msg.format(lower, upper, name, index))

		def lower_and_upper_same_length(lower, upper, parent_property, name,endpoint_name=None):
			lower_len = len(lower)
			upper_len = len(upper)
			if lower_len != upper_len:
				msg = ("Lower and upper bounds must be provided for all variables in {2}. Currently {0} lower bounds and {1} upper bounds are provided.")
				raise ValueError(msg.format(lower_len, upper_len, name))
			if parent_property is None:
				parent_property = lower_len
			else:
				if lower_len != parent_property:
					msg = ("The same number of bounds for the {3} must be provided as there are {2} variables for the optimal control problem ({1}). {0} variable bounds have been provided for the {3}.")
					raise ValueError(msg.format(lower_len, parent_property, name, endpoint_name))

		def endpoint_bounds_within_state(lowers, uppers, name):
			for index, (lower, upper) in enumerate(zip(lowers, uppers)):
				if lower < self.state_lower[index]:
					msg = ("{2} lower bound of {0} for state {3} must be greater than or equal to the state {3} lower bound of {1}")
					raise ValueError(msg.format(lower, self.state_lower[index], name, index))

				if upper > self.state_upper[index]:
					msg = ("{2} upper bound of {0} for state {3} must be less than or equal to the state {3} upper bound of {1}")
					raise ValueError(msg.format(upper, self.state_upper[index], name, index))

		# Time
		lower_less_than_upper_single_val(self._t0_l, self._t0_u, 'initial time')
		lower_less_than_upper_single_val(self._tF_l, self._tF_u, 'final time')

		# State
		lower_and_upper_same_length(self._y_l, self._y_u, self._ocp._num_y_vars, 'state')
		lower_less_than_upper_np_array(self._y_l, self._y_u, 'state')

		# Control
		lower_and_upper_same_length(self._u_l, self._u_u, self._ocp._num_u_vars, 'control')
		lower_less_than_upper_np_array(self._u_l, self._u_u, 'control')

		# Integral
		lower_and_upper_same_length(self._q_l, self._q_u, self._ocp._num_q_vars, 'integrals')
		lower_less_than_upper_np_array(self._q_l, self._q_u, 'integrals')

		# Parameter
		lower_and_upper_same_length(self._s_l, self._s_u, self._ocp._num_s_vars, 'parameter')
		lower_less_than_upper_np_array(self._s_l, self._s_u, 'parameter')

		# Path
		lower_and_upper_same_length(self.path_lower, self.path_upper, self._ocp._num_c_cons, 'path constraint')
		lower_less_than_upper_np_array(self.path_lower, self.path_upper, 'path constraint')

		# Event
		lower_and_upper_same_length(self.boundary_lower, self.boundary_upper, self._ocp._num_b_cons, 'boundary constraint')
		lower_less_than_upper_np_array(self.boundary_lower, self.boundary_upper, 'boundary constraint')

		