from typing import (Iterable, Optional, )

import numpy as np
import scipy.sparse as sparse


class PhaseMesh:

    _DEFAULT_NUMBER_MESH_SECTIONS = 10
    _DEFAULT_MESH_SECTION_SIZES = None

    def __init__(self, phase: "Phase", *,
                 number_mesh_sections: Optional[int] = None,
                 mesh_section_sizes: Optional[Iterable[float]] = None,
                 number_mesh_section_nodes: Optional[int] = None,
                 ):

        self.phase = phase

        self._mesh_sec_sizes = None

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
            settings = self.phase.optimal_control_problem.settings
            self.number_mesh_section_nodes = settings.collocation_points_min
        else:
            self.number_mesh_section_nodes = number_mesh_section_nodes

    @property
    def number_mesh_sections(self):
        return self._num_mesh_secs

    @number_mesh_sections.setter
    def number_mesh_sections(self, num_mesh_secs):
        self._num_mesh_secs = int(num_mesh_secs)
        self._num_mesh_endpoints = self._num_mesh_secs + 1

        if (self._mesh_sec_sizes is not None
                and (len(self._mesh_sec_sizes) != self._num_mesh_secs)):
            self.mesh_section_sizes = None
            if max(self.number_mesh_section_nodes) == min(self.number_mesh_section_nodes):
                self.number_mesh_section_nodes = self.number_mesh_section_nodes[0]
            else:
                msg = "Mismatch between mesh section sizes and mesh section nodes."
                raise ValueError(msg)

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
        return string


class Mesh:
    """
    A mesh class describing the temporal transcription of the problem.

    Long mesh description.

    Attributes:

    """

    _TAU_0 = -1
    _TAU_F = 1
    _PERIOD = _TAU_F - _TAU_0

    def __init__(self, backend, phase_meshes):

        self.backend = backend
        self.p = phase_meshes
        self.settings = self.backend.ocp.settings
        self.quadrature = self.backend.quadrature

        # # Optimal Control Problem
        # self._ocp = optimal_control_problem
        # self._iteration = None
        # self._guess = None

        # self._mesh_sec_fracs = None

        # # Mesh settings
        # self.mesh_sections = mesh_sections
        # self.mesh_section_fractions = mesh_section_fractions
        # self.mesh_collocation_points = mesh_collocation_points

        # self._stretch = None
        # self._shift = None

        self.generate()

    # @property
    # def ocp(self):
    # 	return self._ocp

    # @property
    # def mesh_sections(self):
    # 	return self._K

    # @mesh_sections.setter
    # def mesh_sections(self, mesh_sections):
    # 	self._K = int(mesh_sections)
    # 	self._Kplus1 = self._K + 1

    # 	if self._mesh_sec_fracs is not None and len(self._mesh_sec_fracs) != self._K:
    # 		self.mesh_section_fractions = None

    # @property
    # def mesh_section_fractions(self):
    # 	return self._mesh_sec_fracs

    # @mesh_section_fractions.setter
    # def mesh_section_fractions(self, fracs):
    # 	if fracs is None:
    # 		fracs = np.ones(self._K) / self._K
    # 	if len(fracs) != self._K:
    # 		msg = (f"Mesh section fractions must be an iterable of length {self._mesh_secs} (i.e. matching the number of mesh sections).")
    # 		raise ValueError(msg)
    # 	fracs = np.array(fracs)
    # 	fracs = fracs / fracs.sum()
    # 	self._mesh_sec_fracs = fracs

    # @property
    # def mesh_collocation_points(self):
    # 	return self._mesh_col_points

    # @mesh_collocation_points.setter
    # def mesh_collocation_points(self, col_points):
    # 	try:
    # 		col_points = int(col_points)
    # 	except TypeError:
    # 		col_points = np.array([int(val) for val in col_points], dtype=int)
    # 	else:
    # 		col_points = np.ones(self._K, dtype=int) * col_points
    # 	if len(col_points) != self._K:
    # 		msg = (f"Mesh collocation points must be an iterable of length {self._mesh_secs} (i.e. matching the number of mesh sections).")
    # 		raise ValueError(msg)
    # 	self._mesh_col_points = col_points

    # @property
    # def period(self):
    # 	return self._T

    # @property
    # def _quadrature(self):
    # 	return self._ocp._quadrature

    def generate(self):
        self.tau = []
        self.h = []
        self.N = []
        self.K = []
        self.mesh_index_boundaries = []
        self.h_K = []
        self.h_K_expanded = []
        self.N_K = []
        self.num_c_defect_per_y = []
        self.W_matrix = []
        self.sI_matrix = []
        self.sA_matrix = []
        self.A_index_array = []
        self.D_index_array = []
        for p in self.p:
            data = self.generate_single_phase(p)
            self.tau.append(data[0])
            self.h.append(data[1])
            self.N.append(data[2])
            self.K.append(data[3])
            self.mesh_index_boundaries.append(data[4])
            self.h_K.append(data[5])
            self.h_K_expanded.append(data[6])
            self.N_K.append(data[7])
            self.num_c_defect_per_y.append(data[8])
            self.W_matrix.append(data[9])
            self.sI_matrix.append(data[10])
            self.sA_matrix.append(data[11])
            self.A_index_array.append(data[12])
            self.D_index_array.append(data[13])

    def generate_single_phase(self, p):

        # Check that the number of collocation points in each mesh sections is bounded by the minimum and maximum values set in settings.
        for i_sec, col_points in enumerate(p.number_mesh_section_nodes):
            if col_points < self.settings.collocation_points_min:
                msg = (f"The number of collocation points, {col_points}, in mesh section {i_sec} must be greater than or equal to {self.ocp.settings._col_points_min}.")
                raise ValueError(msg)
            if col_points > self.settings.collocation_points_max:
                msg = (f"The number of collocation points, {col_points}, in mesh section {i_sec} must be less than or equal to {self.ocp._settings._col_points_max}.")
                raise ValueError(msg)

        # Generate the mesh based on using the quadrature method defined by the problem's `Settings` class.
        section_boundaries = [self._TAU_0]
        for index, fraction in enumerate(p.mesh_section_sizes):
            step = self._PERIOD * fraction
            section_boundaries.append(section_boundaries[index] + step)
        section_boundaries = np.array(section_boundaries)
        section_lengths = np.diff(section_boundaries)

        mesh = []
        for section_num, (sec_start, sec_end, sec_num_points) in enumerate(zip(section_boundaries[:-1], section_boundaries[1:], p.number_mesh_section_nodes)):
            points = self.quadrature.quadrature_point(
                sec_num_points, domain=[sec_start, sec_end])
            # if self.settings.quadrature_method == "lobatto":
            # 	points = points[:-1]
            # mesh.extend(list(points))
            mesh.extend(list(points[:-1]))
        mesh.append(self._TAU_F)

        tau = np.array(mesh)
        h = np.diff(tau)
        N = len(tau)
        K = p.number_mesh_sections

        mesh_index_boundaries = np.insert(
            np.cumsum(p.number_mesh_section_nodes - 1), 0, 0)
        h_K = np.diff(tau[mesh_index_boundaries])
        N_K = p.number_mesh_section_nodes

        block_starts = mesh_index_boundaries[:-1]
        num_c_defect_per_y = int(mesh_index_boundaries[-1])
        num_cols = mesh_index_boundaries[-1] + 1
        matrix_dims = (num_c_defect_per_y, num_cols)

        W_matrix = np.zeros(num_cols)

        A_vals = []
        A_row_inds = []
        A_col_inds = []

        D_vals = []
        D_row_inds = []
        D_col_inds = []

        A_index_array = []
        D_index_array = []

        num_A_nonzero = 0

        h_K_expanded = []

        for block_size, h_k, block_start in zip(p.number_mesh_section_nodes, h_K, block_starts):
            row_slice = slice(block_start, block_start + block_size - 1)
            col_slice = slice(block_start, block_start + block_size)
            A_block = self.quadrature.A_matrix(block_size) * h_k
            A_vals_entry = A_block.flatten().tolist()
            A_row_inds_entry = np.repeat(
                np.array(range(block_start, block_start + block_size - 1)), block_size)
            A_col_inds_entry = np.tile(
                np.array(range(block_start, block_start + block_size)), block_size - 1)
            A_vals.extend(A_vals_entry)
            A_row_inds.extend(A_row_inds_entry)
            A_col_inds.extend(A_col_inds_entry)
            A_indicies = self.quadrature.A_index_array(block_size)
            A_index_array.extend(A_indicies + num_A_nonzero)

            D_block = self.quadrature.D_matrix(block_size)
            nonzero = np.flatnonzero(D_block)
            D_vals_entry = D_block.flatten()[nonzero].tolist()
            D_row_inds_entry = np.repeat(np.array(
                range(block_start, block_start + block_size - 1)), block_size)[nonzero]
            D_col_inds_entry = np.tile(np.array(
                range(block_start, block_start + block_size)), block_size - 1)[nonzero]
            D_vals.extend(D_vals_entry)
            D_row_inds.extend(D_row_inds_entry)
            D_col_inds.extend(D_col_inds_entry)
            D_index_array.extend(self.quadrature.D_index_array(
                block_size) + num_A_nonzero)

            W_matrix[col_slice] += self.quadrature.quadrature_weight(
                block_size) * h_k

            num_A_nonzero = len(A_index_array)

            h_K_expanded.extend([h_k] * (block_size - 1))

        sA_matrix = sparse.coo_matrix(
            (A_vals, (A_row_inds, A_col_inds)), shape=matrix_dims).tocsr()
        sD_matrix = sparse.coo_matrix(
            (D_vals, (D_row_inds, D_col_inds)), shape=matrix_dims).tocsr()

        A_index_array = np.array(A_index_array)
        D_index_array = np.array(D_index_array)

        h_K_expanded = np.array(h_K_expanded)

        data = (tau,
                h,
                N,
                K,
                mesh_index_boundaries,
                h_K,
                h_K_expanded,
                N_K,
                num_c_defect_per_y,
                W_matrix,
                sA_matrix,
                sD_matrix,
                A_index_array,
                D_index_array)
        return data
