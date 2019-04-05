import numpy as np

class Guess():
	
	def __init__(self, *, optimal_control_problem=None, time=None, state=None, control=None, parameter=None, integral=None):

		# Set optimal control problem
		self._ocp = optimal_control_problem

		# Number of nodes in guess
		self._num_nodes = None

		# Guess
		self._t = time
		self._y = state
		self._u = control
		self._q = integral
		self._s = parameter
		# self._path = path
		# self._boundary = boundary

	@property
	def _x(self):
		x = np.hstack([self._y.flatten(), self._u.flatten(), self._q, self._s])
		return x

	# @property
	# def optimal_control_problem(self):
	# 	return self._optimal_control_problem

	# @property
	# def time(self):
	# 	return self._t

	# @property
	# def initial_time(self):
	# 	return self._t[0]

	# @property
	# def final_time(self):
	# 	return self._t[-1]
	
	# @property
	# def state(self):
	# 	return self._y
	
	# @property
	# def control(self):
	# 	return self._u

	# @property
	# def integral(self):
	# 	return self._q

	# @property
	# def parameter(self):
	# 	return self._s
	
	# @property
	# def path(self):
	# 	return self._path
	
	# @property
	# def boundary(self):
	# 	return self._boundary

	def _format_as_np_array(self, guess, dims):
		if guess is None:
			if dims == 1:
				guess_as_np_array = np.array([])
			elif dims == 2:
				list_of_lists = []
				for _ in range(self._num_nodes):
					list_of_lists.append([])
				guess_as_np_array = np.array(list_of_lists)
		else:
			guess_as_np_array = np.array(guess)
		return guess_as_np_array

	def _guess_check(self):

		def check_is_within_bounds(val, lower, upper):
			if val < lower or val > upper:
				msg = (".")
				raise ValueError(msg.format())

		def check_shape(guess, num_vars, name):
			try:
				shape = guess.shape
			except TypeError:
				raise NotImplementedError
			else:
				ndim = guess.ndim
				if ndim == 1:
					if guess.size != num_vars:
						msg = ("The number of variables in {2} is {0} and the number of variables in the guess is {1}.")
						raise ValueError(msg.format(num_vars, guess.size, name))
				elif ndim == 2:
					if shape != (num_vars, self._num_nodes):
						msg = ("The number of variables in {2} is {0} and the number of nodes in the guess is {1}.")
						raise ValueError(msg.format(num_vars, self._num_nodes, name))
				else:
					msg = (".")
					raise ValueError(msg.format())

		bounds = self._ocp._bounds

		# Time
		check_is_within_bounds(self._t[0], bounds.initial_time_lower, bounds.initial_time_upper)
		check_is_within_bounds(self._t[-1], bounds.final_time_lower, bounds.final_time_upper)
		t_prev = self._t[0]
		for t in self._t[1:]:
			if t <= t_prev:
				msg = (".")
				raise ValueError(msg.format())
		self._num_nodes = len(self._t)
		self._t = self._format_as_np_array(self._t, 2)

		# State
		if self._ocp._num_y_vars:
			self._y = self._format_as_np_array(self._y, 2)
			check_shape(self._y, self._ocp._num_y_vars, 'state')
		else:
			self._y = np.array([])

		# Control
		if self._ocp._num_u_vars:
			self._u = self._format_as_np_array(self._u, 2)
			check_shape(self._u, self._ocp._num_u_vars, 'control')
		else:
			self._u = np.array([])

		# Integral
		if self._ocp._num_q_vars:
			self._q = self._format_as_np_array(self._q, 1)
			check_shape(self._q, self._ocp._num_q_vars, 'integral')
		else:
			self._q = np.array([])

		# Parameter
		if self._ocp._num_s_vars:
			self._s = self._format_as_np_array(self._s, 1)
			check_shape(self._s, self._ocp._num_s_vars, 'parameter')
		else:
			self._s = np.array([])

		# # Path
		# if self.optimal_control_problem.num_path_constraints:
		# 	self._path = self._format_as_np_array(self._path, 2)
		# 	check_shape(self.path, self.optimal_control_problem.num_path_constraints, 'path')
		# else:
		# 	self._path = None

		# # Boundary
		# if self.optimal_control_problem.num_boundary_constraints:
		# 	self._boundary = self._format_as_np_array(self._boundary, 1)
		# 	check_shape(self.boundary, self.optimal_control_problem.num_boundary_constraints, 'event')
		# else:
		# 	self._boundary = None

		