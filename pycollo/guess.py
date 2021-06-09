""""""

from numbers import Number

import numpy as np
from pyproprop import processed_property
from typing import Optional


__all__ = ["EndpointGuess", "PhaseGuess"]


def assert_increasing(a):
    """Utility function for enforcing ascending values.

    This function's handle can be supplied as :py:kwarg:`post_method` to a
    :py:func:`processed_proprty <pyproprop>` to enforce values within a
    :py:type:`ndarray <numpy>` are in ascending order. This is useful for
    enforcing time guesses to be sequential.

    """
    if (a is not None) and (not np.all(np.diff(a) >= 0)):
        msg = f"Elements in {a} must be in ascending numerical order."
        raise ValueError(msg)
    return a


class PhaseGuess:
    """Data class for holding phase-specific user-supplied guess information.

    Attributes
    ----------
    phase : Phase
        The :py:class:`Phase` object with which this guess is assocaited.
    time : Optional[numpy.ndarray], None
        A sequential list of the time points at which the guess occur for this
        phase. These must be in ascending order otherwise an error will be
        raised. If it has a value of None then it has not been set and will
        raise an error during the preprocessing of the guesses during the
        problem initialisation.
    state_variables : Optional[numpy.ndarray], None
        An array of values, one each for each state variable in the optimal
        control problem and for every temporal node in the time guess. This
        guess must therefore be square.
    control_variables : Optional[numpy.ndarray], None
        An array of values, one each for each control variable in the optimal
        control problem and for every temporal node in the time guess. This
        guess must therefore be square.
    integral_variables : Optional[numpy.ndarray], None
        An array of values, one each for each integral variable in the optimal
        control problem. If there is only a single integral variable in the
        optimal control problem then a single `float` can be supplied as the
        `pyproprop.processed_property` will automatically convert this to a
        one-dimensional `numpy.ndarray`.
    state_endpoint_override : bool, True
        An option which when is set to True will automatically override the
        user-supplied guess for state endpoint variables with constrained state
        endpoints where the optimal control problem has them. This option is
        recommended to be left as True.
    auto_bound : bool, True
        An option which when is set to True will override any user-supplied
        guess values with the upper-/lower-bound for the appropriate variable
        in any cases where the user-supplied guess does not fall within the
        user-supplied bounds for that variable. This option is useful in
        situations where small floating point inaccuaries may cause a bound not
        to be met by a very small amount.

    """

    phase = processed_property("phase", read_only=True)
    time = processed_property("time", description="phase time guess", 
        type=np.ndarray, cast=True, optional=True, method=assert_increasing)
    state_variables = processed_property("state_variables",
        description="phase state variables guess", type=np.ndarray, cast=True,
        optional=True)
    control_variables = processed_property("control_variables",
        description="phase control variables guess", type=np.ndarray,
        cast=True, optional=True)
    integral_variables = processed_property("integral_variables",
        description="phase integral variables guess", type=np.ndarray,
        cast=True, optional=True, method=np.ndarray.flatten)

    def __init__(self,
                 phase: "Phase",
                 time=None,
                 state_variables=None,
                 control_variables=None,
                 integral_variables=None
                 ):
        self.phase = phase
        self.time = time
        self.state_variables = state_variables
        self.control_variables = control_variables
        self.integral_variables = integral_variables
        self.state_endpoints_override = True
        self.auto_bound = True


class EndpointGuess:

    optimal_control_problem = processed_property("optimal_control_problem",
                                                 read_only=True)
    parameter_variables = processed_property(
        "parameter_variables",
        description="static parameter variables guess",
        type=np.ndarray,
        cast=True,
        optional=True,
        method=np.ndarray.flatten
    )

    def __init__(self, optimal_control_problem: "OptimalControlProblem",
                 parameter_variables=None):
        self.optimal_control_problem = optimal_control_problem
        self.parameter_variables = parameter_variables


class Guess:

    def __init__(self, backend, phase_guesses, endpoint_guess):
        self.backend = backend
        self.p = phase_guesses
        self.endpoint = endpoint_guess
        self.settings = self.backend.ocp.settings
        self.generate()

    def generate(self):
        self.num_tau = []
        self.tau = []
        self.t0 = []
        self.tF = []
        self.y = []
        self.u = []
        self.q = []
        self.t = []
        for p, p_guess in zip(self.backend.p, self.p):
            num_tau = self.check_time_guess(p_guess.time)
            self.num_tau.append(num_tau)
            data = self.generate_single_phase(p, p_guess, num_tau)
            tau, t0, tF, y, u, q, t = data
            self.tau.append(tau)
            self.t0.append(t0)
            self.tF.append(tF)
            self.y.append(y[p.ocp_phase.bounds._y_needed])
            self.u.append(u[p.ocp_phase.bounds._u_needed])
            self.q.append(q[p.ocp_phase.bounds._q_needed])
            self.t.append(t[p.ocp_phase.bounds._t_needed])
        s = self.check_guess(self.endpoint.parameter_variables,
                             self.backend.num_s_var)
        self.s = s[self.backend.ocp.bounds._s_needed]

        # print(self.tau)
        # print(self.y)
        # print(self.u)
        # print(self.q)
        # print(self.t)
        # print(self.s)
        # print('\n\n\n')
        # raise NotImplementedError

    def check_time_guess(self, t_guess):
        if t_guess.ndim != 1:
            msg = (f"Time guess must be a 1d array.")
            raise ValueError(msg)
        return t_guess.size

    def generate_single_phase(self, p, p_guess, num_tau):
        time = p_guess.time.astype(np.float64)
        t0 = time[0]
        tF = time[-1]
        stretch = 0.5 * (tF - t0)
        shift = 0.5 * (t0 + tF)
        tau = np.array(time - shift) / stretch
        y = self.check_guess(p_guess.state_variables, p.num_y_var, num_tau)
        u = self.check_guess(p_guess.control_variables, p.num_u_var, num_tau)
        q = self.check_guess(p_guess.integral_variables, p.num_q_var)
        t = np.array([t0, tF])
        data = (tau, t0, tF, y, u, q, t)
        return data

    def check_guess(self, guess, num_var, num_t=None):
        if guess is None or num_var == 0:
            if num_var != 0:
                msg = "A guess must be supplied."
                raise ValueError(msg)
            if num_t is not None:
                return np.empty((0, num_t))
            else:
                return np.empty((0, ))
        shape = guess.shape
        if num_t is not None:
            if guess.shape != (num_var, num_t):
                msg = "A guess must be supplied for every symbol and time."
                raise ValueError(msg)
            pass
        elif guess.shape != (num_var, ):
            msg = "A guess must be supplied for every symbol."
            raise ValueError(msg)
        guess = self.backend.substitute_pycollo_sym(guess)
        guess = self.backend.expr_as_numeric(guess).reshape(shape)
        return guess

    # @property
    # def guess_type(self):
    #   return self._guess_type

    # @guess_type.setter
    # def guess_type(self, guess_type):
    #   valid_guess_args = ('default', 'user-supplied', 'inverse-dynamics', 'forward-dynamics')
    #   if guess_type not in valid_guess_args:
    #       spacer = f"', '"
    #       msg = (f"`guess_type` must be one of: '{spacer.join(valid_guess_args)}'. {repr(guess_type)} is an invalid argument.")
    #       raise ValueError(msg)
    #   elif guess_type == valid_guess_args[0]:
    #       self._guess_generator = self._default_guess_generator
    #   elif guess_type == valid_guess_args[1]:
    #       self._guess_generator = self._user_supplied_guess_generator
    #   elif guess_type == valid_guess_args[2]:
    #       msg = (f"Inverse dynamics guess generation is not currently supported.")
    #       raise NotImplementedError(msg)
    #   elif guess_type == valid_guess_args[3]:
    #       msg = (f"Forward dynamics guess generation is not currently supported.")
    #       raise NotImplementedError(msg)

    # @property
    # def state_endpoints_override(self):
    #   return self._state_endpoints_override

    # @state_endpoints_override.setter
    # def state_endpoints_override(self, bool_val):
    #   self._state_endpoints_override = bool(bool_val)

    # @property
    # def auto_bound(self):
    #   return self._auto_bound

    # @auto_bound.setter
    # def auto_bound(self, bool_val):
    #   self._auto_bound = bool(bool_val)

    # @property
    # def _x(self):
    #   t = np.array([t for t, t_needed in zip([self._time[0], self._time[-1]], self._ocp._bounds._t_needed) if t_needed])
    #   x = np.hstack([self._y.flatten(), self._u.flatten(), self._q, t, self._s])
    #   return x

    # @property
    # def time(self):
    #   return self._time_user

    # @time.setter
    # def time(self, t):
    #   self._time_user = np.array(t, dtype=np.float64)
    #   if t is None:
    #       self._time_user = None
    #       self._N = None
    #   else:
    #       self._time_user = np.array(t, dtype=np.float64)
    #       if self._time_user.ndim != 1:
    #           msg = (f"Guesses for time need to be supplied as an iterable with a single dimension. The current guess for time ({self.time}) is invalid.")
    #           raise ValueError(msg)
    #       self._N = len(self._time_user)
    #       if self._N <= 1:
    #           msg = (f"Guesses need to be given for at least two time points and must include the initial time point and final time point as the first and last values respectively. The current guess for time ({self.time}) is invalid.")
    #           raise ValueError(msg)
    #       if np.isnan(self._time_user).any():
    #           msg = (f"Guesses for time must contain only numerical values and these must be in chronological order. The current guess for time ({self.time}) is invalid.")
    #           raise ValueError(msg)
    #       t_prev = self.initial_time
    #       for t in self._time_user[1:]:
    #           if t <= t_prev:
    #               msg = (f"Guesses for time must be supplied in chronological order. The current guess for time ({self.time}) is invalid.")
    #               raise ValueError(msg)
    #           t_prev = t

    # @property
    # def initial_time(self):
    #   return self._time_user[0] if self.time is not None else None

    # @property
    # def final_time(self):
    #   return self._time_user[-1] if self.time is not None else None

    # def _default_guess_generator(self):

    #   if self.auto_bound:
    #       raise NotImplementedError

    #   def lower_upper_mean(bounds):
    #       return 0.5 * sum(bounds)

    #   def parse_val(val, lower, upper):
    #       if np.isnan(val):
    #           return lower_upper_mean((lower, upper))
    #       else:
    #           return val

    #   def check_val(val, lower, upper):
    #       if np.isinf(val):
    #           msg = (f"Cannot have infinite guesses.")
    #           raise ValueError(msg)
    #       else:
    #           return val

    #   def parse_check_val(val, lower, upper):
    #       val = check_val(val, lower, upper)
    #       return parse_val(val, lower, upper)

    #   if self.time is None:
    #       t0 = lower_upper_mean(self._ocp._bounds.initial_time)
    #       tF = lower_upper_mean(self._ocp._bounds.final_time)
    #       self.time = [t0, tF]

    #   y_guess = np.empty((self._ocp.number_state_variables, self._N), dtype=np.float64)
    #   y_guess.fill(np.nan)

    #   for y_b, lower, upper in zip(self._ocp.state_endpoint_constraints, self._ocp._bounds._y_b_l, self._ocp._bounds._y_b_u):
    #       try:
    #           row_ind = self._ocp.initial_state.index(y_b)
    #       except ValueError:
    #           pass
    #       else:
    #           col_ind = 0
    #       try:
    #           row_ind = self._ocp.final_state.index(y_b)
    #       except ValueError:
    #           pass
    #       else:
    #           col_ind = -1
    #       y_guess[row_ind, col_ind] = lower_upper_mean((lower, upper))

    #   if self._y_user is None:
    #       self._y_user = np.empty([self._ocp.number_state_variables, self._N])
    #       self._y_user.fill(np.nan)
    #   for i, (y_var, guess, lower, upper) in enumerate(zip(self._ocp._y_vars, self._y_user, self._ocp._bounds._y_l, self._ocp._bounds._y_u)):
    #       for j, val in enumerate(guess):
    #           _ = check_val(val, lower, upper)
    #           if not np.isnan(y_guess[i, j]):
    #               if self.state_endpoints_override or np.isclose(y_guess[i, j], lower) or np.isclose(y_guess[i, j], upper):
    #                   pass
    #               else:
    #                   if np.isnan(val):
    #                       pass
    #                   else:
    #                       msg = (f"Values from state endpoint constraints cannot be overriden by guesses. The problem is occuring for state variables #{i+1} at time node #{j+1}.")
    #                       raise ValueError(msg)
    #           else:
    #               y_guess[i, j] = parse_val(val, lower, upper)
    #   self._y_user = y_guess

    #   if self._u_user is None:
    #       self._u_user = np.empty([self._ocp.number_control_variables, self._N])
    #       self._u_user.fill(np.nan)
    #   u_guess = np.empty((self._ocp.number_control_variables, self._N), dtype=np.float64)
    #   for i, (u_var, guess, lower, upper) in enumerate(zip(self._ocp._u_vars, self._u_user, self._ocp._bounds._u_l, self._ocp._bounds._u_u)):
    #       for j, val in enumerate(guess):
    #           val = check_val(val, lower, upper)
    #           u_guess[i, j] = parse_val(val, lower, upper)
    #   self._u_user = u_guess

    #   if self._q_user is None:
    #       self._q_user = np.empty(self._ocp.number_integral_variables)
    #       self._q_user.fill(np.nan)
    #   self._q_user = np.array([parse_check_val(val, l, u) for val, l, u in zip(self._q_user, self._ocp._bounds._q_l, self._ocp._bounds._q_u)])

    #   self._t_user = [self._time_user[0], self._time_user[-1]]

    #   if self._s_user is None:
    #       self._s_user = np.empty(self._ocp.number_parameter_variables)
    #       self._s_user.fill(np.nan)
    #   self._s_user = np.array([parse_check_val(val, l, u) for val, l, u in zip(self._s_user, self._ocp._bounds._s_l, self._ocp._bounds._s_u)])

    #   if False:
    #       print('\nBoundary constraints:')
    #       print(self._ocp.state_endpoint_constraints)
    #       print('\nTime:')
    #       print(self._time_user)
    #       print('\nState variables:')
    #       print(self._y_user)
    #       print('\nControl variables:')
    #       print(self._u_user)
    #       print('\nIntegral variables:')
    #       print(self._q_user)
    #       print('\nTime variables:')
    #       print(self._t_user)
    #       print(self._ocp._bounds._t_needed)
    #       print('\nParameter variables:')
    #       print(self._s_user)

    # def _user_supplied_guess_generator(self):
    #   pass

    # def _check_user_supplied_shape(self):

    #   def check_t_dep_is_none(var, var_str):
    #       if var is not None:
    #           msg = (f"No guess has been supplied for time and therefore the supplied time-dependent guesses for {var_str} variables cannot be parsed.")
    #           ValueError(msg)

    #   def check_t_dep_shape(var, var_str, num_var):

    #       error_str = (f"The guess for {var_str} must be supplied as a two-dimensional iterable with the first dimension being the number of {var_str} variables in the optimal control problem ({len(num_var)}), and the second dimension being the number of nodes in `Guess.time` ({self._N}).")

    #       try:
    #           var = np.array(var, dtype=np.float64)
    #       except ValueError:
    #           raise ValueError(error_str)
    #       except TypeError:
    #           msg = (f"All elements in the guess for {var_str} must be either numberical types or `None`. {type(var)} objects are not allowed.")
    #           raise TypeError(msg)

    #       if len(num_var) == 1:
    #           var = np.expand_dims(var, axis=0)

    #       if var.shape != (len(num_var), self._N):
    #           msg = (f"{error_str} The shape should be: ({len(num_var)}, {self._N}). Currently the shape of the guess for {var_str} is: {var.shape}")
    #           raise ValueError(msg)

    #       return var

    #   def check_t_ind_shape(var, var_str, num_var):

    #       if isinstance(var, Number):
    #           var = [var]

    #       error_str = (f"The guess for {var_str} must be supplied as an iterable of length equally the number of {var_str} variables in the optimal control problem ({len(num_var)}).")

    #       try:
    #           var = np.array(var, dtype=np.float64)
    #       except ValueError:
    #           raise ValueError(error_str)
    #       except TypeError:
    #           msg = (f"All elements in the guess for {var_str} must be either numberical types or `None`. {type(var)} objects are not allowed.")
    #           raise TypeError(msg)

    #       if var.shape != (len(num_var), ):
    #           msg = (f"{error_str} The shape should be: ({len(num_var)},). Currently the shape of the guess for {var_str} is: {var.shape}")
    #           raise ValueError(msg)

    #       return var

    #   if False:
    #       print('\nTime:')
    #       print(self.time)
    #       print('\nState variables:')
    #       print(self.state)
    #       print('\nControl variables:')
    #       print(self.control)
    #       print('\nIntegral variables:')
    #       print(self.integral)
    #       print('\nTime variables:')
    #       print(self.initial_time, self.final_time)
    #       print(self._ocp._bounds._t_needed)
    #       print('\nParameter variables:')
    #       print(self.parameter)

    #   if self._N is None:
    #       self._y_user = check_t_dep_is_none(self.state, 'state')
    #       self._u_user = check_t_dep_is_none(self.control, 'control')
    #   else:
    #       self._y_user = check_t_dep_shape(self.state, 'state', self._ocp._y_vars_user) if self.state is not None else None
    #       self._u_user = check_t_dep_shape(self.control, 'control', self._ocp._u_vars_user) if self.control is not None else None

    #   self._q_user = check_t_ind_shape(self.integral, 'ingetal', self._ocp._q_vars_user) if self.integral is not None else None
    #   self._s_user = check_t_ind_shape(self.parameter, 'parameter', self._ocp._s_vars_user) if self.parameter is not None else None

    #   return None

    # def _guess_check(self):

    #   def check_is_within_bounds(val, lower, upper):
    #       if val < lower or val > upper:
    #           msg = (f"The current guess lies outside of the bounds for the variable supplied by the user.")
    #           raise ValueError(msg)
    #       return val

    #   # def parse_guess_input(guess, syms, var_name, t_func=False):
    #   #   if guess is None:
    #   #       return np.array([])
    #   #   dim_2 = self._N if t_func else 1
    #   #   try:
    #   #       iter(guess)
    #   #   except TypeError:
    #   #       guess = np.array([guess])
    #   #   try:
    #   #       len_guess = len(guess.keys())
    #   #   except AttributeError:
    #   #       guess_array = np.array(guess, dtype=float)
    #   #       if guess_array.ndim == 1:
    #   #           guess_array = guess_array.reshape(len(syms), dim_2)
    #   #       if guess_array.shape != (len(syms), dim_2):
    #   #           msg = (f"Guess for {var_name} must be passed as a two-dimensional iterable; an iterable of {len(syms)} iterables for each {var_name} variable of length {dim_2}. Currently guesses for {guess_array.shape[0]} {var_name} variables are supplied at {guess_array.shape[1]} nodes.")
    #   #           raise ValueError(msg)
    #   #   else:
    #   #       if len(syms) != len_guess:
    #   #           missing_guess = ', '.join(f'{symbol}' for symbol in set(syms).difference(set(guess.keys())))
    #   #           msg = (f"Guess must be provided for each {var_name} variable. Please also provide guess for: {missing_guess}.")
    #   #           raise ValueError(msg)
    #   #       guess_array = np.empty((len(syms), dim_2), dtype=float)
    #   #       for i_sym, symbol in enumerate(syms):
    #   #           guess_array[i_sym, :] = guess[symbol]
    #   #   return guess_array if t_func else guess_array.flatten()

    #   _ = self._check_user_supplied_shape()
    #   _ = self._guess_generator()

    #   bounds = self._ocp._bounds

    #   # Time
    #   self._t0 = check_is_within_bounds(self._time_user[0], bounds._t0_l, bounds._t0_u)
    #   self._tF = check_is_within_bounds(self._time_user[-1], bounds._tF_l, bounds._tF_u)
    #   self._stretch = 0.5 * (self._tF - self._t0)
    #   self._shift = 0.5 * (self._t0 + self._tF)
    #   self._tau = np.array(self._time_user - self._shift)/self._stretch

    #   # State
    #   # y_user = parse_guess_input(self.state, self._ocp._y_vars_user, 'state', t_func=True)
    #   self._y = np.array([y for y, y_needed in zip(self._y_user, self._ocp._bounds._y_needed) if y_needed])
    #   # _ = check_shape(self._y, self._ocp._num_y_vars, 'state', self._N)

    #   # Control
    #   # u_user = parse_guess_input(self.control, self._ocp._u_vars_user, 'control', t_func=True)
    #   self._u = np.array([u for u, u_needed in zip(self._u_user, self._ocp._bounds._u_needed) if u_needed])
    #   # check_shape(self._u, self._ocp._num_u_vars, 'control', self._N)

    #   # Integral
    #   # q_user = parse_guess_input(self.integral, self._ocp._q_vars_user, 'integral')
    #   self._q = np.array([q for q, q_needed in zip(self._q_user, self._ocp._bounds._q_needed) if q_needed])
    #   # check_shape(self._q, self._ocp._num_q_vars, 'integral')

    #   # Time
    #   # t_user = np.array([self._t0, self._tF])
    #   self._t = np.array([t for t, t_needed in zip(self._t_user, self._ocp._bounds._t_needed) if t_needed])
    #   # check_shape(self._t, self._ocp._num_t_vars, 'time')

    #   # Parameter
    #   # s_user = parse_guess_input(self.parameter, self._ocp._s_vars_user, 'parameter')
    #   self._s = np.array([s for s, s_needed in zip(self._s_user, self._ocp._bounds._s_needed) if s_needed])
    #   # check_shape(self._s, self._ocp._num_s_vars, 'parameter')

    #   if False:
    #       print('\nTime:')
    #       print(self._tau)
    #       print(self._t0, self._tF)
    #       print('\nState variables:')
    #       print(self._y)
    #       print('\nControl variables:')
    #       print(self._u)
    #       print('\nIntegral variables:')
    #       print(self._q)
    #       print('\nTime variables:')
    #       print(self._t)
    #       print('\nParameter variables:')
    #       print(self._s)

    # def _mesh_refinement_bypass_init(self):
    #   self._t0 = self._time_user[0]
    #   self._tF = self._time_user[-1]
    #   self._stretch = 0.5 * (self._tF - self._t0)
    #   self._shift = 0.5 * (self._t0 + self._tF)
    #   self._tau = np.array(self._time_user - self._shift)/self._stretch

    #   self._y = self.state
    #   self._u = self.control
    #   self._q = self.integral
    #   self._t = np.array([t for t, t_needed in zip([self._t0, self._tF], self._ocp._bounds._t_needed) if t_needed])
    #   self._s = self.parameter

    #   return None
