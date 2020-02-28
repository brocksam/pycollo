"""Everything needed for defining phases within an optimal control problem.

Classes:
	Phase: 
"""


import itertools
from typing import (Optional, )

import sympy as sym

from .processed_property import processed_property
from .typing import (OptionalSymsType, TupleSymsType)
from .utils import (check_sym_name_clash, format_as_tuple)


__all__ = ["Phase"]


class Phase:
	"""

	"""

	def __init__(self, *, 
			optimal_control_problem=None, 
			state_variables=None, 
			control_variables=None,
			state_equations=None,
			integrand_functions=None,
			path_constraints=None,
			state_endpoint_constraints=None,
			bounds=None,
			scaling=None,
			initial_guess=None,
			initial_mesh=None,
			):

		self._ocp = None
		self._phase_number = None
		self._phase_suffix = "X"

		self._y_vars_user = ()
		self._u_vars_user = ()
		self._q_vars_user = ()
		self._t_vars_user = ()

		self._y_eqns_user = ()
		self._c_cons_user = ()
		self._q_funcs_user = ()
		self._y_b_cons_user = ()

		self._all_internal_syms = ()

		if optimal_control_problem is not None:
			self.optimal_control_problem = optimal_control_problem

		self.state_variables = state_variables
		self.control_variables = control_variables

		self.state_equations = state_equations
		self.integrand_functions = integrand_functions
		self.path_constraints = path_constraints
		self.state_endpoint_constraints = state_endpoint_constraints

		self.bounds = bounds
		self.scaling = scaling
		self.initial_guess = initial_guess
		self.initial_mesh = initial_mesh

	@classmethod
	def new_like(cls, phase_for_copying):
		pass

	@property
	def optimal_control_problem(self) -> Optional["OptimalControlProblem"]:
		"""The optimal control problem with which this phase is associated.

		There are two allowable scenarios. In the first scenario a phase may be 
		instantiated without being associated with an optimal control problem. 
		If this is the case then the default values of `None` for the phase 
		number and 'X' for the phase suffix remain. 

		In the second scenario a phase is instantiated with an associated 
		optimal control problem or is associated with an optimal control 
		problem after the first type of instantiation. In this case the phase 
		is appended to the protected `_phases` attribute of the 
		:obj:`OptimalControlProblem`, the phase number is set according to its 
		position in the order of addition to the optimal controls problem's 
		phases, and its phase suffix is set as a string version of the phase 
		number. Finally a replacement of any symbols that may have been used in 
		supplementary information about the phase that contained the placeholder 
		'X' phase suffix are renamed and substituted.

		No checking is done to see whether the phase is already associated with 
		the optimal control problem in question or any other optimal control 
		problem. The reason being that if the setter method for this property is 
		accessed after having already been set then an `AttributeError` is 
		raised (see below). The reason this class works like that is to avoid 
		having to allow phases to be disassociated from an 
		:obj:`OptimalControlProblem` and thus having to handled the complexities 
		that would come with the phase renumbering and substitution of any 
		phase-related information that has already been given to the optimal 
		control problem.

		Raises:
			AttributeError: If an :obj:`OptimalControlProblem` has already been 
				associated with `self`. If a argument of any type other than 
				:obj:`OptimalControlProblem` is passed to the 
				`optimal_control_problem` property setter.
			TypeError: If an argument of any type other than 
				:obj:`OptimalControlProblem` is passed to the 
				`optimal_control_problem` property setter and that type happens 
				to have protected attribute called `_phases` that is of a type 
				that doesn't have an `append` method.
		"""
		return self._ocp

	@optimal_control_problem.setter
	def optimal_control_problem(self, ocp):
		if self._ocp is not None:
			msg = ('Optimal control problem is already set for this phase and '
				'cannot be reset.')
			raise AttributeError(msg)

		try:
			ocp._phases.append(self)
		except (AttributeError, TypeError) as error:
			msg = ('Phase can only be associated with a '
				'`Pycollo.OptimalControlProblem` object.')
			raise error(msg)

		self._ocp = ocp
		self._phase_number = self._ocp.number_phases
		self._phase_suffix = str(self.phase_number)

		self.state_variables = self.state_variables
		self.integrand_functions = self.integrand_functions

		# old_state = self._y_vars_user
		# old_control = self._u_vars_user
		# self.state_variables = old_state
		# self.control_variables = old_control
		# self._y_vars_user + self._u_vars_user
		# subs_mapping = dict(zip(
		# 	(old_state + old_control), (self._y_vars_user + self._u_vars_user)))
		# self.state_equations = [y_eqn.subs(subs_mapping)
		# 	for y_eqn in self.state_equations]
		# self.path_constraints = [c_con.subs(subs_mapping)
		# 	for c_con in self.path_constraints]
		# self.integrand_functions = [q_func.subs(subs_mapping)
		# 	for q_func in self.integrand_functions]

	@property
	def phase_number(self) -> Optional[int]:
		"""The integer numerical identifier for the phase.

		If this phase has not yet been associated with an optimal control 
		problem then None is returned.

		Corresponds to the chronological order in which it was associated with 
		the optimal control problem in question.
		"""
		return self._phase_number

	@property
	def initial_time_variable(self) -> sym.Symbol:
		"""Symbol for the time at which this phase begins."""
		try:
			return self._t0_USER
		except AttributeError:
			msg = ("Can't access initial time until associated with an optimal "
				"control problem.")
			raise AttributeError(msg)

	@property
	def final_time_variable(self) -> sym.Symbol:
		"""Symbol for the time at which this phase begins."""
		try:
			return self._tF_USER
		except AttributeError:
			msg = ("Can't access final time until associated with an optimal "
				"control problem.")
			raise AttributeError(msg)

	@property
	def initial_state_variables(self) -> TupleSymsType:
		"""Symbols for this phase's state variables at the initial time.

		Raises:
			AttributeError: If `optimal_control_problem` property has not yet 
				been set to a not None value. See docstring for 
				`state_variables` for details about why.
		"""
		try:
			return self._y_t0_user
		except AttributeError:
			msg = ("Can't access initial state until associated with an optimal "
				"control problem.")
			raise AttributeError(msg)
	
	@property
	def final_state_variables(self) -> TupleSymsType:
		"""Symbols for this phase's state variables at the final time.

		Raises:
			AttributeError: If `optimal_control_problem` property has not yet 
				been set to a not None value. See docstring for 
				`state_variables` for details about why.
		"""
		try:
			return self._y_tF_user
		except AttributeError:
			msg = ("Can't access initial state until associated with an optimal "
				"control problem.")
			raise AttributeError(msg)

	@property
	def state_variables(self) -> TupleSymsType:
		"""Symbols for this phase's state variables in order added by user.

		The user may supply either a single symbol or an iterable of symbols. 
		The supplied argument is handled by the `format_as_tuple` method from 
		the `utils` module. Additional protected attributes `_y_t0_user` and 
		`_y_tF_user` are set by post-appending either '_PX(t0)' or '_PX(tF)' to 
		the user supplied symbols where the X is replaced by the phase suffix. 
		As such if this phase has not yet been associated with an optimal 
		control problem yet then `self` will not have attributes `_y_t0_user` 
		and `_y_tF_user` and accessing either the `initial_state` or 
		`final_state` property will raise an AttributeError.
		"""
		return self._y_vars_user

	@state_variables.setter
	def state_variables(self, y_vars: OptionalSymsType):

		self._y_vars_user = format_as_tuple(y_vars)

		# Generate the state endpoint variable symbols only if phase has number
		if self.optimal_control_problem is not None:
			self._t0_USER = sym.Symbol(f't0_P{self._phase_suffix}')
			self._tF_USER = sym.Symbol(f'tF_P{self._phase_suffix}')
			self._t0 = sym.Symbol(f'_t0_P{self._phase_suffix}')
			self._tF = sym.Symbol(f'_tF_P{self._phase_suffix}')
			self._STRETCH = 0.5 * (self._tF - self._t0)
			self._SHIFT = 0.5 * (self._t0 + self._tF)
			self._t_vars_user = (self._t0_USER, self._tF_USER)

			self._y_t0_user = tuple(sym.Symbol(f'{y}_P{self._phase_suffix}(t0)')
				for y in self._y_vars_user)
			self._y_tF_user = tuple(sym.Symbol(f'{y}_P{self._phase_suffix}(tF)')
				for y in self._y_vars_user)
		
		check_sym_name_clash(self._y_vars_user)

	@property
	def number_state_variables(self):
		return len(self._y_vars_user)

	@property
	def control_variables(self):
		return self._u_vars_user

	@control_variables.setter
	def control_variables(self, u_vars):
		self._u_vars_user = format_as_tuple(u_vars)
		_ = check_sym_name_clash(self._u_vars_user)

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
	def state_equations(self):
		return self._y_eqns_user

	@state_equations.setter
	def state_equations(self, y_eqns):
		self._y_eqns_user = format_as_tuple(y_eqns)

	@property
	def number_state_equations(self):
		return len(self._y_eqns_user)

	@property
	def path_constraints(self):
		return self._c_cons_user

	@path_constraints.setter
	def path_constraints(self, c_cons):
		self._c_cons_user = format_as_tuple(c_cons)

	@property
	def number_path_constraints(self):
		return len(self._c_cons_user)

	@property
	def integrand_functions(self):
		return self._q_funcs_user

	@integrand_functions.setter
	def integrand_functions(self, integrands):
		self._q_funcs_user = format_as_tuple(integrands)
		self._q_vars_user = tuple(sym.Symbol(f'q{i_q}_P{self._phase_suffix}') 
			for i_q, _ in enumerate(self._q_funcs_user))

	@property
	def number_integrand_functions(self):
		return len(self._q_funcs_user)

	@property
	def state_endpoint_constraints(self):
		return self._y_b_cons_user

	@state_endpoint_constraints.setter
	def state_endpoint_constraints(self, y_b_cons):
		self._is_initialised = False
		y_b_cons = format_as_tuple(y_b_cons)
		self._y_b_cons_user = tuple(y_b_cons)

	@property
	def number_state_endpoint_constraints(self):
		return len(self._y_b_cons_user)

	def __str__(self):
		string = (f"Phase {self.phase_number} of {self.optimal_control_problem}")
		return string

	def __repr__(self):
		string = (f"Phase({repr(self.optimal_control_problem)}, "
			f"phase_number={self.phase_number})")
		return string



