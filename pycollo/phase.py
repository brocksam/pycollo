from typing import (Optional, )

import sympy as sym

from .processed_property import processed_property
from .utils import (check_sym_name_clash, format_as_tuple)


__all__ = ["Phase"]


class Phase:

	def __init__(self, *, optimal_control_problem=None, state_variables=None):

		self._init_private_attributes()
		self.optimal_control_problem = optimal_control_problem
		self.state_variables = state_variables

	@property
	def optimal_control_problem(self) -> "OptimalControlProblem":
		return self._optimal_control_problem
	
	@optimal_control_problem.setter
	def optimal_control_problem(self, ocp: Optional["OptimalControlProblem"]):
		from .optimal_control_problem import OptimalControlProblem

		if ocp is None:
			self._optimal_control_problem = None
			self._phase_number = None
			self._phase_suffix = 'X'
		elif not isinstance(ocp, OptimalControlProblem):
			raise TypeError
		else:
			self._optimal_control_problem = self._optimal_control_problem_setter(ocp)

	def _optimal_control_problem_setter(self, ocp):

		ocp._phases.append(self)
		self._phase_number = ocp.phases.index(self)
		self._phase_suffix = str(self._phase_number)
		self._init_time_symbols()

		return ocp

		# Set self._optimal_control_problem
		# Set self._phase_number
		# Create phase-specific symbols
		# Replace phase specific symbols

	def _init_private_attributes(self):
		self._y_vars_user = ()
		self._u_vars_user = ()
		self._q_vars_user = ()
		self._t_vars_user = ()

		self._y_eqns_user = ()
		self._c_cons_user = ()
		self._q_funcs_user = ()
		self._y_b_cons_user = ()

		self._all_internal_syms = ()

	def _init_time_symbols(self):

		self._t0_USER = sym.Symbol(f't0_P{self._phase_suffix}')
		self._tF_USER = sym.Symbol(f'tF_P{self._phase_suffix}')
		self._t0 = sym.Symbol(f'_t0_P{self._phase_suffix}')
		self._tF = sym.Symbol(f'_tF_P{self._phase_suffix}')

		self._STRETCH = 0.5 * (self._tF - self._t0)
		self._SHIFT = 0.5 * (self._t0 + self._tF)

		self._t_vars_user = (self._t0_USER, self._tF_USER)

		self._all_internal_syms = (
			self._t0_USER, 
			self._tF_USER, 
			self._t0, 
			self._tF,
			)

	def _internal_symbol_replace(self):
		pass

	@property
	def state_variables(self):
		return self._y_vars_user

	@state_variables.setter
	def state_variables(self, y_vars):
		try:
			self._initialised = False
		except AttributeError:
			pass

		self._y_vars_user = format_as_tuple(y_vars)
		self._y_t0_user = tuple(sym.Symbol(f'{y}_P{self._phase_suffix}(t0)')
			for y in self._y_vars_user)
		self._y_tF_user = tuple(sym.Symbol(f'{y}_P{self._phase_suffix}(tF)')
			for y in self._y_vars_user)
		# self._update_vars()
		_ = check_sym_name_clash(self._y_vars_user)

	@property
	def number_state_variables(self):
		if self._bounds._bounds_checked:
			return self._bounds._y_needed.sum()
		else:
			return len(self._y_vars_user)

	# def _update_vars(self):
	# 	self._x_vars_user = tuple(self._y_vars_user + self._u_vars_user
	# 		+ self._q_vars_user + self._t_vars_user + self._s_vars_user)
	# 	self._x_b_vars_user = tuple(self.state_endpoint 
	# 		+ self._q_vars_user + self._t_vars_user + self._s_vars_user)



