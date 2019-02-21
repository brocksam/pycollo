import collections

import ipopt
import numpy as np
import scipy.interpolate as interpolate
import sympy as sym

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
	p: free parameters.
	q: vector of integral constraints.

	N: number of temporal nodes in collocation.

Notes:
	x = [y, u, p]
	n = len(x): number of free variables
	m = len([c, b]): number of constraints

"""

class OptimalControlProblem():

	def __init__(self, state_variables=None, control_variables=None, parameter_variables=None, differential_equations=None, *, bounds=None, initial_guess=None, initial_mesh=None, path_constraints=None, integrand_functions=None, objective_function=None, event_constraints=None, integral_constraints=None, settings=None, auxiliary_data=None):

		# Set settings
		self.settings = settings

		# Auxiliary data
		self.auxiliary_data = auxiliary_data

		# Initialise problem description
		self._state_variables = np.array([])
		self._control_variables = np.array([])
		self._parameter_variables = np.array([])
		self._differential_equations = np.array([])
		self.state_variables = state_variables
		self.control_variables = control_variables
		self.parameter_variables = parameter_variables
		self.differential_equations = differential_equations
		self._num_parameter_variables = None
		self._num_integrals = None
		self._num_defect_constraints = None
		self._num_path_constraints = None
		self._num_state_endpoint_constraints = None
		self._num_event_constraints = None
		self._num_integral_constraints = None

		# Set bounds
		self.bounds = bounds

		# Set initial guess
		self.initial_guess = initial_guess

		# Set initial mesh
		self._mesh_iterations = []
		if initial_mesh is None:
			first_mesh = Mesh(optimal_control_problem=self)
		else:
			first_mesh = mesh
			first_mesh._optimal_control_problem = self
		first_iteration = Iteration(optimal_control_problem=self, iteration_number=1, mesh=first_mesh)
		self._mesh_iterations
		self._mesh_iterations.append(first_iteration)

		# Continuous constraints and functions
		self.integrand_functions = integrand_functions
		self.path_constraints = path_constraints

		# Endpoint constraints and functions
		self.objective_function = objective_function
		self.integrand_functions = integrand_functions
		self.event_constraints = event_constraints
		self.integral_constraints = integral_constraints

	@staticmethod
	def _format_as_np_array(variables):
		if not variables:
			raise ValueError
		try:
			iter(variables)
		except TypeError:
			variables = (variables, )
		return np.array(variables)

	@classmethod
	def _format_constraint_as_np_array(cls, constraint):
		if not constraint:
			return None
		else:
			return cls._format_as_np_array(constraint)

	@property
	def state_variables(self):
		return self._state_variables

	@state_variables.setter
	def state_variables(self, variables):
		self._state_variables = self._format_as_np_array(variables)
		self._num_state_variables = len(self._state_variables)
		self._update_variables()
	
	@property
	def control_variables(self):
		return self._control_variables

	@control_variables.setter
	def control_variables(self, variables):
		self._control_variables = self._format_as_np_array(variables)
		self._num_control_variables = len(self._control_variables)
		self._update_variables()

	@property
	def variables(self):
		return self._variables

	def _update_variables(self):
		self._variables = np.concatenate((self._state_variables, self._control_variables))
		self.auxiliary_data._state_mapping = dict()
		for index, state in enumerate(self._state_variables):
			self.auxiliary_data.state_mapping.update({state: index})
		self.auxiliary_data._control_mapping = dict()
		for index, control in enumerate(self._control_variables):
			self.auxiliary_data.control_mapping.update({control: index})
		self.auxiliary_data._variable_mapping = dict()
		for index, variable in enumerate(self._variables):
			self.auxiliary_data.variable_mapping.update({variable: index})

	@property
	def differential_equations(self):
		return self._differential_equations

	@differential_equations.setter
	def differential_equations(self, diff_eqns):
		self._differential_equations = self._format_constraint_as_np_array(diff_eqns)
		if self._differential_equations is not None:
			self._num_differential_equations = len(self._differential_equations)

	@property
	def bounds(self):
		return self._bounds
	
	@bounds.setter
	def bounds(self, bounds):
		if bounds is None:
			self._bounds = Bounds(optimal_control_problem=self)
		else:
			self._bounds = bounds
			self._bounds._optimal_control_problem = self

	@property
	def initial_guess(self):
		return self._initial_guess
	
	@initial_guess.setter
	def initial_guess(self, guess):
		if guess is None:
			self._initial_guess = Guess(optimal_control_problem=self)
		else:
			self._initial_guess = guess
			self._initial_guess._optimal_control_problem = self

	@property
	def mesh_iterations(self):
		return self._mesh_iterations
	
	@property
	def num_mesh_iterations(self):
		return len(self._mesh_iterations)

	@property
	def objective_function(self):
		return self._objective_function
	
	@objective_function.setter
	def objective_function(self, J):
		self._objective_function = J

	@property
	def objective_gradient(self):
		return self._objective_gradient

	@property
	def objective_hessian(self):
		return self._objective_hessian

	@property
	def integrand_functions(self):
		return self._integrand_functions

	@integrand_functions.setter
	def integrand_functions(self, integrands):
		if integrands:
			integrand_functions = self._format_as_np_array(integrands)
			integrands = collections.OrderedDict()
			for index, integrand in enumerate(integrand_functions):
				symbol = sym.symbols("q_{}".format(index))
				integrands.update({symbol: integrand})
			self._integrals = list(integrands.keys())
			self._integrand_functions = list(integrands.values())
			self._num_integrals = len(self._integrals)
			self._integral_integrand_function_mapping = integrands
		else:
			self._integrals = None
			self._integrand_functions = None
			self._num_integrands = None
			self._integral_integrand_function_mapping = None
	
	@property
	def integrals(self):
		return self._integrals

	@property
	def path_constraints(self):
		return self._path_constraints

	@path_constraints.setter
	def path_constraints(self, paths):
		self._path_constraints = self._format_constraint_as_np_array(paths)
	
	@property
	def state_endpoint_constraints(self):
		return self._state_endpoint_constraints

	@state_endpoint_constraints.setter
	def state_endpoint_constraints(self, endpoints):
		self._state_endpoint_constraints = self._format_constraint_as_np_array(endpoints)
	
	@property
	def event_constraints(self):
		return self._event_constraints

	@event_constraints.setter
	def event_constraints(self, events):
		self._event_constraints = self._format_constraint_as_np_array(events)

	@property
	def integral_constraints(self):
		return self._integral_constraints

	@integral_constraints.setter
	def integral_constraints(self, integrals):
		self._integral_constraints = self._format_constraint_as_np_array(integrals)

	@property
	def constraints(self):
		return self._constraints

	@property
	def constraints_jacobian(self):
		return self._constraints_jacobian
	
	@property
	def constraints_jacobian_sparsity(self):
		return self._constraints_jacobian_sparsity
	
	@property
	def constraints_hessian(self):
		return self._constraints_hessian

	@property
	def settings(self):
		return self._settings
	
	@settings.setter
	def settings(self, settings):
		if settings is None:
			self._settings = Settings(optimal_control_problem=self)
		else:
			self._settings = settings
			self._settings._optimal_control_problem = self

	@property
	def auxiliary_data(self):
		return self._auxiliary_data
	
	@auxiliary_data.setter
	def auxiliary_data(self, aux_data):
		if aux_data is None:
			self._auxiliary_data = AuxiliaryData(optimal_control_problem=self)
		else:
			self._auxiliary_data = aux_data
			self._auxiliary_data._optimal_control_problem = self

	@property
	def num_state_variables(self):
		if self._num_state_variables:
			return self._num_state_variables
		else:
			return 0

	@property
	def num_control_variables(self):
		if self._num_control_variables:
			return self._num_control_variables
		else:
			return 0

	@property
	def num_parameter_variables(self):
		if self._num_parameter_variables:
			return self._num_parameter_variables
		else:
			return 0

	@property
	def num_differential_equations(self):
		return self._num_differential_equations

	@property
	def num_variables(self):
		return self._num_state_variables + self._num_control_variables + self._num_parameter_variables

	@property
	def num_integrals(self):
		return self._num_integrals
	
	@property
	def num_defect_constraints(self):
		return self._num_defect_constraints
	
	@property
	def num_path_constraints(self):
		return self._num_path_constraints

	@property
	def num_state_endpoint_constraints(self):
		return self._num_state_endpoint_constraints
	
	@property
	def num_event_constraints(self):
		return self._num_event_constraints
	
	@property
	def num_integral_constraints(self):
		return self._num_integral_constraints

	@property
	def num_constraints(self):
		return self._num_constraints

	@property
	def nlp_problem(self):
		return self._nlp

	@property
	def nlp_problem(self):
		return self._nlp_problem

	def _generate_state_endpoint_constraints(self):
		c = []
		for state in self._state_variables:
			c.append(state)
		for state in self._state_variables:
			c.append(state)
		self._endpoint_constraints = c

	def solve(self):
		# Error checking
		if self.num_differential_equations != self.num_state_variables:
			msg = ("A differential equation must be supplied for each state variable. {1} differential equations were supplied for {0} state variables.")
			raise ValueError(msg.format(self.num_state_variables, self.num_differential_equations))
		self.bounds._bounds_check()
		self.initial_guess._guess_check()
		self._generate_state_endpoint_constraints()

		# Problem setup
		self._mesh_iterations[0]._initialise_iteration(previous_guess=self.initial_guess)

		print("\nJ:")
		print(self.objective_function)
		print("\ng:")
		print(self.objective_gradient)
		print("\nc:")
		print(self.constraints)
		print("\nG:")
		print(self.constraints_jacobian)
		print('\n\n\n')
		print(self.mesh_iterations[0].mesh.N)
		print(self.mesh_iterations[0].mesh.t)
		print(self.mesh_iterations[0].mesh.h)
		print('\n\n\n')
		print("Completed successfully.")
		print('\n\n\n')

		raise NotImplementedError

		# Solve
		self.nlp_problem.solve()