import numpy as np

class Mesh():
	"""
	A mesh class describing the temporal transcription of the problem.

	Long mesh description.

	Attributes:
		
	"""

	def __init__(self, *, optimal_control_problem=None, mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=2):

		# Optimal Control Problem
		self._ocp = optimal_control_problem

		# Mesh settings
		self.mesh_sections = mesh_sections
		self.mesh_section_fractions = mesh_section_fractions
		self.mesh_collocation_points = mesh_collocation_points

	# @property
	# def optimal_control_problem(self):
	# 	return self._optimal_control_problem

	@property
	def mesh_sections(self):
		return self._mesh_secs
	
	@mesh_sections.setter
	def mesh_sections(self, mesh_sections):
		self._mesh_secs = int(mesh_sections)

	@property
	def _K(self):
		return self._mesh_secs

	@property
	def _Kplus1(self):
		return self._mesh_secs + 1

	@property
	def mesh_section_fractions(self):
		return self._mesh_sec_fracs
	
	@mesh_section_fractions.setter
	def mesh_section_fractions(self, fracs):
		if fracs is None:
			fracs = np.ones(self._mesh_secs) / self._mesh_secs
		if len(fracs) != self._mesh_secs:
			msg = ("Mesh section fractions must be an iterable of length {} (i.e. matching the number of mesh sections).")
			raise ValueError(msg.format(self._mesh_secs))
		fracs = np.array(fracs)
		fracs = fracs / fracs.sum()
		self._mesh_sec_fracs = fracs

	@property
	def mesh_collocation_points(self):
		return self._mesh_col_points
	
	@mesh_collocation_points.setter
	def mesh_collocation_points(self, col_points):
		try:
			col_points = int(col_points)
		except TypeError:
			col_points = np.array([int(val) for val in col_points], dtype=int)
		else:
			col_points = np.ones(self._mesh_secs, dtype=int) * col_points
		if len(col_points) != self._mesh_secs:
			msg = ("Mesh collocation points must be an iterable of length {} (i.e. matching the number of mesh sections).")
			raise ValueError(msg.format(self._mesh_secs))
		self._mesh_col_points = col_points

	@property
	def t(self):
		return self._t

	@t.setter
	def t (self, t):
		self._t = t
		self._h = np.diff(self.t)
		self._N = len(self.t)
		self._mesh_index_boundaries = np.insert(np.cumsum(self._mesh_col_points), 0, 0)
		self._mesh_index_boundaries_offset = self._mesh_index_boundaries - np.array(range(self._Kplus1))
		self._hK = np.array([np.sum(self._h[i:ip1]) for i, ip1 in zip(self._mesh_index_boundaries_offset[:-1], self._mesh_index_boundaries_offset[1:])])

	# @property
	# def h(self):
	# 	return self._h
	
	# @property
	# def N(self):
	# 	return self._N

	def _generate_mesh(self, t0, tF):

		# Check that the number of collocation points in each mesh sections is bounded by the minimum and maximum values set in settings.
		for section, col_points in enumerate(self._mesh_col_points):
			if col_points < self._ocp._settings._col_points_min:
				msg = ("The number of collocation points, {0}, in mesh section {1} must be greater than or equal to {2}.")
				raise ValueError(msg.format(col_points, section, self.ocp.settings.col_points_min))
			if col_points > self._ocp._settings.collocation_points_max:
				msg = ("The number of collocation points, {0}, in mesh section {1} must be less than or equal to {2}.")
				raise ValueError(msg.format(col_points, section, self._ocp._settings.col_points_max))

		# Generate the mesh based on using Lobatto methods for the collocation.
		if self._ocp._settings._quadrature_method == 'lobatto':
			section_boundaries = [t0]
			for index, fraction in enumerate(self._mesh_sec_fracs):
				step = (tF - t0) * fraction
				section_boundaries.append(section_boundaries[index] + step)
			section_boundaries = np.array(section_boundaries)
			section_lengths = np.diff(section_boundaries)
			mesh = []
			for section_number, boundary_point in enumerate(section_boundaries):
				mesh.append(boundary_point)
				if section_number < self._mesh_secs:
					num_interior_points = self._mesh_col_points[section_number] - 2
					coefficients = [0]*(num_interior_points)
					coefficients.append(1)
					legendre_polynomial = np.polynomial.legendre.Legendre(coefficients, domain=[boundary_point, section_boundaries[section_number+1]])
					lobatto_points = legendre_polynomial.roots()
					for lobatto_point in lobatto_points:
						mesh.append(lobatto_point)
			mesh[-1] = tF
			self.t = np.array(mesh)

		# Generate the mesh based on using other methods for the collocation.
		elif self._ocp._settings._quadrature_method == 'radau':
			raise NotImplementedError

		elif self._ocp._settings._quadrature_method == 'gauss':
			raise NotImplementedError