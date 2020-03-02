from typing import (Iterable, Optional, )

import numpy as np
import scipy.sparse as sparse


class PhaseMesh:

	_DEFAULT_NUMBER_MESH_SECTIONS = 10
	_DEFAULT_MESH_SECTION_SIZES = None
	_DEFAULT_NUMBER_MESH_SECTION_NODES = 4
	
	def __init__(self, phase: "Phase", *, 
			number_mesh_sections: Optional[int] = None,
			mesh_section_sizes: Optional[Iterable[float]] = None,
			number_mesh_section_nodes: Optional[int] = None,
			):

		self.phase = phase

		self._mesh_sec_fracs = None

		if number_mesh_sections is None:
			try:
				self.number_mesh_sections = (
					self.phase.optimal_control_problem.settings.default_number_mesh_sections)
			except AttributeError:
				self.number_mesh_sections = self._DEFAULT_NUMBER_MESH_SECTIONS
		else:
			self.number_mesh_sections = number_mesh_sections

		if mesh_section_sizes is None:
			try:
				self.mesh_section_sizes = (
					self.phase.optimal_control_problem.settings.default_mesh_section_sizes)
			except AttributeError:
				self.mesh_section_sizes = self._DEFAULT_MESH_SECTION_SIZES
		else:
			self.mesh_section_sizes = mesh_section_sizes

		if number_mesh_section_nodes is None:
			try:
				self.number_mesh_section_nodes = (
					self.phase.optimal_control_problem.settings.default_number_mesh_section_nodes)
			except AttributeError:
				self.number_mesh_section_nodes = self._DEFAULT_NUMBER_MESH_SECTION_NODES
		else:
			self.number_mesh_section_nodes = number_mesh_section_nodes

	@property
	def number_mesh_sections(self):
		return self._num_mesh_secs
	
	@number_mesh_sections.setter
	def number_mesh_sections(self, num_mesh_secs):
		self._num_mesh_secs = int(num_mesh_secs)
		self._num_mesh_endpoints = self._num_mesh_secs + 1

		if (self._mesh_sec_fracs is not None 
				and (len(self._mesh_sec_fracs) != self._num_mesh_secs)):
			self.mesh_section_fractions = None

	@property
	def mesh_section_sizes(self):
		return self._mesh_sec_sizes
	
	@mesh_section_sizes.setter
	def mesh_section_sizes(self, sizes):
		if sizes is None:
			sizes = np.ones(self._num_mesh_secs) / self._num_mesh_secs
		if len(sizes) != self._num_mesh_secs:
			msg = (f"Mesh section sizes must be an iterable of length "
				f"{self._num_mesh_secs} (i.e. matching the number of mesh "
				f"sections).")
			raise ValueError(msg)
		sizes = np.array(sizes)
		sizes = sizes / sizes.sum()
		self._mesh_sec_sizes = sizes

	@property
	def number_mesh_section_nodes(self):
		return self._num_mesh_sec_nodes
	
	@number_mesh_section_nodes.setter
	def number_mesh_section_nodes(self, num_nodes):
		try:
			num_nodes = int(num_nodes)
		except TypeError:
			num_nodes = np.array([int(val) for val in num_nodes], dtype=int)
		else:
			num_nodes = np.ones(self._num_mesh_secs, dtype=int) * num_nodes
		if len(num_nodes) != self._num_mesh_secs:
			msg = (f"Number of mesh section nodes must be an interable of "
				f"length {self._num_mesh_secs} (i.e. matching the number of "
				f"mesh sections).")
			raise ValueError(msg)
		self._num_mesh_sec_nodes = num_nodes

	def __repr__(self):
		string = (f"PhaseMesh(number_mesh_sections={self._num_mesh_secs}, "
			f"mesh_section_sizes={self._mesh_sec_sizes}, "
			f"number_mesh_section_nodes={self._num_mesh_sec_nodes})")
		print(string)



class Mesh:
	"""
	A mesh class describing the temporal transcription of the problem.

	Long mesh description.

	Attributes:
		
	"""

	_TAU_0 = -1
	_TAU_F = 1
	_PERIOD = _TAU_F - _TAU_0

	def __init__(self, *, optimal_control_problem=None, mesh_sections=10, mesh_section_fractions=None, mesh_collocation_points=4):

		# Optimal Control Problem
		self._ocp = optimal_control_problem
		self._iteration = None
		self._guess = None

		self._mesh_sec_fracs = None

		# Mesh settings
		self.mesh_sections = mesh_sections
		self.mesh_section_fractions = mesh_section_fractions
		self.mesh_collocation_points = mesh_collocation_points

		self._stretch = None
		self._shift = None

	@property
	def ocp(self):
		return self._ocp

	@property
	def mesh_sections(self):
		return self._K
	
	@mesh_sections.setter
	def mesh_sections(self, mesh_sections):
		self._K = int(mesh_sections)
		self._Kplus1 = self._K + 1

		if self._mesh_sec_fracs is not None and len(self._mesh_sec_fracs) != self._K:
			self.mesh_section_fractions = None

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
	
	# @profile
	def _generate_mesh(self):#, t0, tF):

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
		self._h = np.diff(self._tau)
		self._N = len(self._tau)
		
		self._mesh_index_boundaries = np.insert(np.cumsum(self._mesh_col_points - 1), 0, 0)
		self._h_K = np.diff(self._tau[self._mesh_index_boundaries])

		block_starts = self._mesh_index_boundaries[:-1]
		self._num_c_boundary_per_y = int(self._mesh_index_boundaries[-1])
		num_cols = self._mesh_index_boundaries[-1] + 1
		matrix_dims = (self._num_c_boundary_per_y, num_cols)

		self._W_matrix = np.zeros(num_cols)

		A_vals = []
		A_row_inds = []
		A_col_inds = []

		D_vals = []
		D_row_inds = []
		D_col_inds = []

		A_index_array = []
		D_index_array = []

		num_A_nonzero = 0

		for block_size, h_K, block_start in zip(self._mesh_col_points, self._h_K, block_starts):
			row_slice = slice(block_start, block_start+block_size-1)
			col_slice = slice(block_start, block_start+block_size)
			A_block = self._quadrature.A_matrix(block_size) * h_K
			A_vals_entry = A_block.flatten().tolist()
			A_row_inds_entry = np.repeat(np.array(range(block_start, block_start+block_size-1)), block_size)
			A_col_inds_entry = np.tile(np.array(range(block_start, block_start+block_size)), block_size-1)
			A_vals.extend(A_vals_entry)
			A_row_inds.extend(A_row_inds_entry)
			A_col_inds.extend(A_col_inds_entry)
			A_indicies = self._quadrature.A_index_array(block_size)
			A_index_array.extend(A_indicies + num_A_nonzero)

			D_block = self._quadrature.D_matrix(block_size)
			nonzero = np.flatnonzero(D_block)
			D_vals_entry = D_block.flatten()[nonzero].tolist()
			D_row_inds_entry = np.repeat(np.array(range(block_start, block_start+block_size-1)), block_size)[nonzero]
			D_col_inds_entry = np.tile(np.array(range(block_start, block_start+block_size)), block_size-1)[nonzero]
			D_vals.extend(D_vals_entry)
			D_row_inds.extend(D_row_inds_entry)
			D_col_inds.extend(D_col_inds_entry)
			D_index_array.extend(self._quadrature.D_index_array(block_size) + num_A_nonzero)

			self._W_matrix[col_slice] += self._quadrature.quadrature_weight(block_size) * h_K

			num_A_nonzero = len(A_index_array)

		self._sA_matrix = sparse.coo_matrix((A_vals, (A_row_inds, A_col_inds)), shape=matrix_dims).tocsr()
		self._sD_matrix = sparse.coo_matrix((D_vals, (D_row_inds, D_col_inds)), shape=matrix_dims).tocsr()

		self._A_index_array = np.array(A_index_array)
		self._D_index_array = np.array(D_index_array)

