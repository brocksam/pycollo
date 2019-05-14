import numpy as np

class Mesh():
	"""
	A mesh class describing the temporal transcription of the problem.

	Long mesh description.

	Attributes:
		
	"""

	_TAU_0 = -1
	_TAU_F = 1
	_PERIOD = _TAU_F - _TAU_0

	def __init__(self, *, optimal_control_problem=None, mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=2):

		# Optimal Control Problem
		self._ocp = optimal_control_problem

		# Mesh settings
		self.mesh_sections = mesh_sections
		self.mesh_section_fractions = mesh_section_fractions
		self.mesh_collocation_points = mesh_collocation_points

		self._stretch = None
		self._shift = None

	@property
	def mesh_sections(self):
		return self._K
	
	@mesh_sections.setter
	def mesh_sections(self, mesh_sections):
		self._K = int(mesh_sections)
		self._Kplus1 = self._K + 1

	@property
	def mesh_section_fractions(self):
		return self._mesh_sec_fracs
	
	@mesh_section_fractions.setter
	def mesh_section_fractions(self, fracs):
		if fracs is None:
			fracs = np.ones(self._K) / self._K
		if len(fracs) != self._K:
			msg = (f"Mesh section fractions must be an iterable of length {self._mesh_secs} (i.e. matching the number of mesh sections).")
			raise ValueError(msg)
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
			col_points = np.ones(self._K, dtype=int) * col_points
		if len(col_points) != self._K:
			msg = (f"Mesh collocation points must be an iterable of length {self._mesh_secs} (i.e. matching the number of mesh sections).")
			raise ValueError(msg)
		self._mesh_col_points = col_points

	@property
	def period(self):
		return self._T

	@property
	def _quadrature(self):
		return self._ocp._quadrature
	
	def _generate_mesh(self, t0, tF):

		# Check that the number of collocation points in each mesh sections is bounded by the minimum and maximum values set in settings.
		for section, col_points in enumerate(self._mesh_col_points):
			if col_points < self._ocp._settings._col_points_min:
				msg = (f"The number of collocation points, {col_points}, in mesh section {section} must be greater than or equal to {self.ocp.settings._col_points_min}.")
				raise ValueError(msg)
			if col_points > self._ocp._settings._col_points_max:
				msg = (f"The number of collocation points, {col_points}, in mesh section {section} must be less than or equal to {self._ocp._settings._col_points_max}.")
				raise ValueError(msg)

		# Generate the mesh based on using the quadrature method defined by the problem's `Settings` class.

		section_boundaries = [self._TAU_0]
		for index, fraction in enumerate(self._mesh_sec_fracs):
			step = self._PERIOD * fraction
			section_boundaries.append(section_boundaries[index] + step)
		section_boundaries = np.array(section_boundaries)
		section_lengths = np.diff(section_boundaries)

		mesh = []
		for section_num, (sec_start, sec_end, sec_num_points) in enumerate(zip(section_boundaries[:-1], section_boundaries[1:], self._mesh_col_points)):
			points = self._quadrature.quadrature_point(sec_num_points, domain=[sec_start, sec_end])
			if self._ocp._settings._quadrature_method == 'lobatto':
				points = points[:-1]
			mesh.extend(list(points))
		mesh.append(self._TAU_F)

		self._tau = np.array(mesh)
		self._H = np.diff(self._tau)
		self._N = len(self._tau)
		self._stretch_num = (tF - t0)/2
		self._shift_num = (t0 + tF)/2

		self._t = self._stretch_num * self._tau + self._shift_num
		self._T = self._t[-1] - self._t[0]
		self._h = np.diff(self._t)
		
		self._mesh_index_boundaries = np.insert(np.cumsum(self._mesh_col_points - 1), 0, 0)
		self._H_K = np.diff(self._tau[self._mesh_index_boundaries])
		self._hK = np.diff(self._t[self._mesh_index_boundaries])

		block_starts = self._mesh_index_boundaries[:-1]
		num_rows = self._mesh_index_boundaries[-1]
		num_cols = self._mesh_index_boundaries[-1] + 1
		matrix_dims = (num_rows, num_cols)

		self._D_matrix = np.zeros(matrix_dims)
		self._A_matrix = np.zeros(matrix_dims)
		self._W_matrix = np.zeros(num_cols)

		for block_size, H_K, block_start in zip(self._mesh_col_points, self._H_K, block_starts):
			row_slice = slice(block_start, block_start+block_size-1)
			col_slice = slice(block_start, block_start+block_size)
			self._A_matrix[row_slice, col_slice] = self._quadrature.A_matrix(block_size) * H_K
			self._D_matrix[row_slice, col_slice] = self._quadrature.D_matrix(block_size)
			self._W_matrix[col_slice] += self._quadrature.quadrature_weight(block_size) * H_K
		self._num_c_boundary_per_y = self._D_matrix.shape[0]

