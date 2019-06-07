import numpy as np

class Guess():
	
	def __init__(self, *, optimal_control_problem=None, time=None, state=None, control=None, integral=None, parameter=None):

		# Set optimal control problem
		self._ocp = optimal_control_problem
		self._iteration = None
		self._mesh = None

		# Guess
		self.time = time
		self.state = state
		self.control = control
		self.integral = integral
		self.parameter = parameter

	@property
	def _x(self):
		t = np.array([t for t, t_needed in zip([self._time[0], self._time[-1]], self._ocp._bounds._t_needed) if t_needed])
		x = np.hstack([self._y.flatten(), self._u.flatten(), self._q, t, self._s])
		return x

	@property
	def time(self):
		return self._t_user

	@time.setter
	def time(self, t):
		self._t_user = None if t is None else np.array(t)
		if t is None:
			self._N = None 
		else:
			try:
				self._N = len(t.squeeze())
			except AttributeError:
				self._N = len(t)

	@property
	def initial_time(self):
		return self._t_user[0]

	@property
	def final_time(self):
		return self._t_user[-1]

	def _guess_check(self):

		def check_is_within_bounds(val, lower, upper):
			if val < lower or val > upper:
				msg = (f"The current guess lies outside of the bounds for the variable supplied by the user.")
				raise ValueError(msg)
			return val

		def check_shape(guess, num_vars, name):
			if guess.ndim == 1:
				if guess.size != num_vars:
					msg = (f"The number of variables in {name} is {num_vars} and the number of variables in the guess is {guess.size}.")
					raise ValueError(msg)
			elif guess.ndim == 2:
				if guess.shape != (num_vars, self._N):
					msg = (f"The number of variables in {name} is {num_vars} and the number of nodes in the guess is {self._N}.")
					raise ValueError(msg)
			else:
				msg = (f".")
				raise ValueError(msg)

		def parse_guess_input(guess, syms, var_name, t_func=False):
			if guess is None:
				return np.array([])
			dim_2 = self._N if t_func else 1
			try:
				iter(guess)
			except TypeError:
				guess = np.array([guess])
			try:
				len_guess = len(guess.keys())
			except AttributeError:
				guess_array = np.array(guess, dtype=float)
				if guess_array.ndim == 1:
					guess_array = guess_array.reshape(len(syms), dim_2)
				if guess_array.shape != (len(syms), dim_2):
					msg = (f"Guess for {var_name} must be passed as a two-dimensional iterable; an iterable of {len(syms)} iterables for each {var_name} variable of length {dim_2}. Currently guesses for {guess_array.shape[0]} {var_name} variables are supplied at {guess_array.shape[1]} nodes.")
					raise ValueError(msg)
			else:
				if len(syms) != len_guess:
					missing_guess = ', '.join(f'{symbol}' for symbol in set(syms).difference(set(guess.keys())))
					msg = (f"Guess must be provided for each {var_name} variable. Please also provide guess for: {missing_guess}.")
					raise ValueError(msg)
				guess_array = np.empty((len(syms), dim_2), dtype=float)
				for i_sym, symbol in enumerate(syms):
					guess_array[i_sym, :] = guess[symbol]
			return guess_array if t_func else guess_array.flatten()

		bounds = self._ocp._bounds

		# Time
		self._t0 = check_is_within_bounds(self._t_user[0], bounds._t0_l, bounds._t0_u)
		self._tF = check_is_within_bounds(self._t_user[-1], bounds._tF_l, bounds._tF_u)
		self._stretch = (self._tF - self._t0)/2
		self._shift = (self._t0 + self._tF)/2
		t_prev = self._t0
		for t in self._t_user[1:]:
			if t <= t_prev:
				msg = (f"Guesses must be supplied in chronological order.")
				raise ValueError(msg)
		self._time = np.array(self._t_user)

		# State
		y_user = parse_guess_input(self.state, self._ocp._y_vars_user, 'state', t_func=True)
		self._y = np.array([y for y, y_needed in zip(y_user, self._ocp._bounds._y_needed) if y_needed])
		check_shape(self._y, self._ocp._num_y_vars, 'state')

		# Control
		u_user = parse_guess_input(self.control, self._ocp._u_vars_user, 'control', t_func=True)
		self._u = np.array([u for u, u_needed in zip(u_user, self._ocp._bounds._u_needed) if u_needed])
		check_shape(self._u, self._ocp._num_u_vars, 'control')

		# Integral
		q_user = parse_guess_input(self.integral, self._ocp._q_vars_user, 'integral')
		self._q = np.array([q for q, q_needed in zip(q_user, self._ocp._bounds._q_needed) if q_needed])
		check_shape(self._q, self._ocp._num_q_vars, 'integral')

		# Time
		t_user = np.array([self._t0, self._tF])
		self._t = np.array([t for t, t_needed in zip(t_user, self._ocp._bounds._t_needed) if t_needed])
		check_shape(self._t, self._ocp._num_t_vars, 'time')

		# Parameter
		s_user = parse_guess_input(self.parameter, self._ocp._s_vars_user, 'parameter')
		self._s = np.array([s for s, s_needed in zip(s_user, self._ocp._bounds._s_needed) if s_needed])
		check_shape(self._s, self._ocp._num_s_vars, 'parameter')

		