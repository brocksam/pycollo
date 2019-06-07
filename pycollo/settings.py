class Settings():

	def __init__(self, *, optimal_control_problem=None, collocation_matrix_form='integral', nlp_solver='ipopt', linear_solver='mumps', nlp_tolerance=1e-7, max_iterations=2000, quadrature_method='lobatto', mesh_refinement_method=None, mesh_tolerance=1e-7, collocation_points_min=2, collocation_points_max=10):

		# Optimal Control Problem
		self._ocp = optimal_control_problem

		# NLP solver
		self.nlp_solver = nlp_solver
		self.linear_solver = linear_solver
		self.nlp_tolerance = nlp_tolerance
		self.max_iterations = max_iterations

		# Collocation and quadrature
		self.quadrature_method = quadrature_method

		# Mesh refinement
		# self.mesh_refinement_method = mesh_refinement_method
		self.mesh_tolerance = mesh_tolerance
		self.collocation_points_min = collocation_points_min
		self.collocation_points_max = collocation_points_max

	@property
	def collocation_matrix_form(self):
		return self._col_mat_form
	
	@collocation_matrix_form.setter
	def collocation_matrix_form(self, form):
		form = form.casefold()
		if form not in {'differential', 'integral'}:
			msg = ("{} is not a valid collocation matrix form.")
			raise ValueError(msg.format(form))
		elif form == 'integral':
			msg = ("Integral matrix form is not currently supported. Please use integral matrix form.")
		self._col_mat_form = form

	@property
	def nlp_solver(self):
		return self._nlp_solver

	@nlp_solver.setter
	def nlp_solver(self, nlp_solver):
		if nlp_solver not in {'ipopt'}:
			msg = ("{} is not a valid NLP solver.")
			raise ValueError(msg.format(nlp_solver))
		self._nlp_solver = nlp_solver

	@property
	def linear_solver(self):
		return self._linear_solver
	
	@linear_solver.setter
	def linear_solver(self, linear_solver):
		if self.nlp_solver == 'ipopt':
			if linear_solver not in {'mumps', 'ma57'}:
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
	def max_iterations(self):
		return self._max_iterations
	
	@max_iterations.setter
	def max_iterations(self, max_iterations):
		if max_iterations <= 0:
			msg = ("Maximum number of iterations must be a positive integer. {} is invalid.")
			raise ValueError(msg.format(max_iterations))
		self._max_iterations = int(max_iterations)

	@property
	def quadrature_method(self):
		return self._quadrature_method
	
	@quadrature_method.setter
	def quadrature_method(self, method):
		method = method.casefold()
		if method not in {'gauss', 'lobatto', 'radau'}:
			msg = ("The quadrature method of '{}' is not a valid argument string.")
			raise ValueError(msg.format(method))
		self._quadrature_method = method

	@property
	def mesh_refinement_method(self):
		return self._mesh_refinement_method
	
	@mesh_refinement_method.setter
	def mesh_refinement_method(self, method):
		# method = method.casefold()
		if method not in {None}:
			msg = ("The mesh refinement method of '{}' is not a valid argument string.")
			raise ValueError(msg.format(method))
		self._mesh_refinement_method = method

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
	def collocation_points_min(self):
		return self._col_points_min
	
	@collocation_points_min.setter
	def collocation_points_min(self, points_min):
		points_min = int(points_min)
		self._col_points_min = points_min
		if points_min < 2:
			msg = ("The minimum number of collocation points must be great than 2.")
			raise ValueError(msg)
		if points_min > 3:
			msg = ("It is recommended that a minimum of 2 or 3 collocation points is used per mesh section to allow for efficient computation.")
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

		