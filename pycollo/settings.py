from typing import Optional

class Settings():

	_COLLOCATION_MATRIX_FORMS = {'differential', 'integral'}
	_SCALING_OPTIONS = {None, 'user', 'guess', 'bounds'}
	_NLP_SOLVERS = {'ipopt'}
	_LINEAR_SOLVERS = {'mumps', 'ma57'}
	_QUADRATURE_METHODS = {'gauss', 'lobatto', 'radau'}
	_DERIVATIVE_LEVELS = {1, 2}

	def __init__(self, *, optimal_control_problem=None, 
		collocation_matrix_form='integral', nlp_solver='ipopt', 
		linear_solver='mumps', nlp_tolerance=1e-10, max_nlp_iterations=2000, 
		quadrature_method='lobatto', derivative_level=2, max_mesh_iterations=10, 
		mesh_tolerance=1e-8, collocation_points_min=4, collocation_points_max=10, 
		display_mesh_refinement_info=True, display_mesh_result_info=False, 
		display_mesh_result_graph=False, scaling_method='bounds', 
		update_scaling=True, number_scaling_samples=0, scaling_weight=0.8):

		# Optimal Control Problem
		self.ocp = optimal_control_problem

		# NLP solver
		self.nlp_solver = nlp_solver
		self.linear_solver = linear_solver
		self.nlp_tolerance = nlp_tolerance
		self.max_nlp_iterations = max_nlp_iterations

		# Collocation and quadrature
		self.quadrature_method = quadrature_method
		self.derivative_level = derivative_level

		# Mesh refinement
		self.collocation_points_min = collocation_points_min
		self.collocation_points_max = collocation_points_max
		self.mesh_tolerance = mesh_tolerance
		self.max_mesh_iterations = max_mesh_iterations

		# Scaling
		self.scaling_method = scaling_method
		self.update_scaling = update_scaling
		self.number_scaling_samples = number_scaling_samples
		self.scaling_weight = scaling_weight

		# Output information
		self.display_mesh_refinement_info = display_mesh_refinement_info
		self.display_mesh_result_info = display_mesh_result_info
		self.display_mesh_result_graph = display_mesh_result_graph

	@property
	def scaling_method(self) -> Optional[str]:
		return self._scaling_method

	@scaling_method.setter
	def scaling_method(self, method: Optional[str]):
		if method is not None:
			method = method.casefold()
		if method not in self._SCALING_OPTIONS:
			msg = (f"{method} is not a valid scaling option.")
			raise ValueError(msg)
		self._scaling_method = method

	@property
	def update_scaling(self):
		return self._update_scaling
	
	@update_scaling.setter
	def update_scaling(self, do_update):
		do_update = bool(do_update)
		self._update_scaling = do_update

	@property
	def scaling_weight(self):
		return self._scaling_weight
	
	@scaling_weight.setter
	def scaling_weight(self, weight):
		weight = float(weight)
		if weight < 0 or weight > 1:
			msg = (f'Scaling weight must be a number between 0 and 1. {weight} '
				'is invalid.')
			raise ValueError(msg)
		self._scaling_weight = weight

	@property
	def number_scaling_samples(self):
		return self._number_scaling_samples
	
	@number_scaling_samples.setter
	def number_scaling_samples(self, num):
		num = int(num)
		if num < 0:
			msg = (f'Number of scaling samples must be an integer greater than '
				f'or equal to 0. {num} is invalid.')
		self._number_scaling_samples = num

	@property
	def collocation_matrix_form(self):
		return self._col_mat_form
	
	@collocation_matrix_form.setter
	def collocation_matrix_form(self, form):
		form = form.casefold()
		if form not in self._COLLOCATION_MATRIX_FORMS:
			msg = (f"{form} is not a valid collocation matrix form.")
			raise ValueError(msg)
		elif form == 'derivative':
			msg = ("Derivative matrix form is not currently supported. Please "
				"use integral matrix form.")
			raise ValueError(msg)
		self._col_mat_form = form

	@property
	def nlp_solver(self):
		return self._nlp_solver

	@nlp_solver.setter
	def nlp_solver(self, nlp_solver):
		if nlp_solver not in self._NLP_SOLVERS:
			msg = (f"{nlp_solver} is not a valid NLP solver.")
			raise ValueError(msg)
		self._nlp_solver = nlp_solver

	@property
	def linear_solver(self):
		return self._linear_solver
	
	@linear_solver.setter
	def linear_solver(self, linear_solver):
		if self.nlp_solver == 'ipopt':
			if linear_solver not in self._LINEAR_SOLVERS:
				msg = ("{} is not a valid linear solver.")
				raise ValueError(msg.format(linear_solver))
			self._linear_solver = linear_solver
		elif self.nlp_solver == 'snopt':
			msg = ("SNOPT is not currently supported. Please use IPOPT as the NLP solver.")
			raise NotImplementedError(msg)

	@property
	def nlp_tolerance(self):
		return self._nlp_tolerance
	
	@nlp_tolerance.setter
	def nlp_tolerance(self, tolerance):
		if tolerance <= 0:
			msg = ("Tolerance for the NLP must be a postive real number. {} is invalid.")
			raise ValueError(msg.format(tolerance))
		self._nlp_tolerance = tolerance

	@property
	def max_nlp_iterations(self):
		return self._max_nlp_iterations
	
	@max_nlp_iterations.setter
	def max_nlp_iterations(self, max_nlp_iterations):
		if max_nlp_iterations <= 0:
			msg = (f"Maximum number of NLP iterations must be a positive integer. {max_nlp_iterations} is invalid.")
			raise ValueError(msg)
		self._max_nlp_iterations = int(max_nlp_iterations)

	@property
	def quadrature_method(self):
		return self._quadrature_method
	
	@quadrature_method.setter
	def quadrature_method(self, method):
		method = method.casefold()
		if method not in self._QUADRATURE_METHODS:
			msg = ("The quadrature method of '{}' is not a valid argument string.")
			raise ValueError(msg.format(method))
		self._quadrature_method = method

	@property
	def derivative_level(self):
		return self._derivative_level
	
	@derivative_level.setter
	def derivative_level(self, deriv_level):
		deriv_level = int(deriv_level)
		if deriv_level not in self._DERIVATIVE_LEVELS:
			msg = (f"Derivative level must be set to either 1 (which uses the gradient vector of the objective function and the Jacobian matrix of the constraints vector) or 2 (which also uses the Hessian matrix of the Lagrangian of the constraints).")
			raise ValueError(msg)
		self._derivative_level = deriv_level

	@property
	def mesh_tolerance(self):
		return self._mesh_tolerance
	
	@mesh_tolerance.setter
	def mesh_tolerance(self, tolerance):
		if tolerance <= 0:
			msg = ("Tolerance for the mesh must be a postive real number. {} is invalid.")
			raise ValueError(msg.format(tolerance))
		self._mesh_tolerance = tolerance

	@property
	def max_mesh_iterations(self):
		return self._max_mesh_iterations
	
	@max_mesh_iterations.setter
	def max_mesh_iterations(self, max_mesh_iterations):
		if max_mesh_iterations <= 0:
			msg = (f"Maximum number of mesh iterations must be a positive integer. {max_mesh_iterations} is invalid.")
			raise ValueError(msg)
		self._max_mesh_iterations = int(max_mesh_iterations)

	@property
	def collocation_points_min(self):
		return self._col_points_min
	
	@collocation_points_min.setter
	def collocation_points_min(self, points_min):
		points_min = int(points_min)
		self._col_points_min = points_min
		if points_min < 2:
			msg = ("The minimum number of collocation points must be great than 2.")
			raise ValueError(msg)
		if points_min > 4:
			msg = ("It is recommended that a minimum of 2, 3 or 4 collocation points is used per mesh section to allow for efficient computation.")
			raise ValueError(msg)

	@property
	def collocation_points_max(self):
		return self._col_points_max
	
	@collocation_points_max.setter
	def collocation_points_max(self, points_max):
		points_max = int(points_max)
		if points_max < self._col_points_min:
			msg = ("The maximum number of collocation points must be greater than or equal to {}, the minimum number of collocation points.")
			raise ValueError(msg.format(self._col_points_min))
		if points_max > 10:
			msg = ("The maximum number of collocation points recommended in a single mesh sections is 10 due to the numerical instabilty of Lagrange polynomial interpolation above this threshold.")
			raise ValueError(msg)
		self._col_points_max = points_max

	@property
	def display_mesh_refinement_info(self):
		return self._display_mesh_refinement_info
	
	@display_mesh_refinement_info.setter
	def display_mesh_refinement_info(self, val):
		self._display_mesh_refinement_info = bool(val)

	@property
	def display_mesh_result_info(self):
		return self._display_mesh_result_info
	
	@display_mesh_result_info.setter
	def display_mesh_result_info(self, val):
		self._display_mesh_result_info = bool(val)

	@property
	def display_mesh_result_graph(self):
		return self._display_mesh_result_graph
	
	@display_mesh_result_graph.setter
	def display_mesh_result_graph(self, val):
		self._display_mesh_result_graph = bool(val)


