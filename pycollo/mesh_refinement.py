"""Module for mesh refinement including algorithms and mesh error calculation.

Attributes
----------
DEFAULT_MESH_TOLERANCE : float
    Default value for py:class:`Settings` for the default minimum acceptable
    mesh error tolerance.
DEFAULT_MAX_MESH_ITERATIONS : int
    Default value for py:class:`Settings` for the default maximum number of
    mesh iterations that Pycollo should conduct before terminating the OCP
    solve if the mesh error has not been met.
PATTERSON_RAO : str
    String keyword identifier for the Patterson-Rao mesh refinement algorithm.

"""


from abc import ABC, abstractmethod
import itertools

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from pyproprop import Options

from .mesh import Mesh, PhaseMesh
from .utils import casadi_substitute, dict_merge, symbol_name
from .vis.plot import plot_mesh


DEFAULT_MESH_TOLERANCE = 1e-7
DEFAULT_MAX_MESH_ITERATIONS = 10
PATTERSON_RAO = "patterson-rao"
chain_from_iter = itertools.chain.from_iterable


class MeshRefinementABC(ABC):

    def __init__(self, solution):
        self.sol = solution
        self.it = solution.it
        self.ocp = solution.ocp
        self.backend = solution.backend
        self.mesh_error()

    @abstractmethod
    def mesh_error(self):
        pass

    @abstractmethod
    def phase_mesh_error(self):
        pass

    @abstractmethod
    def next_iteration_mesh(self):
        pass

    @abstractmethod
    def next_iteration_phase_mesh(self):
        pass


class PattersonRaoMeshRefinement(MeshRefinementABC):

    def mesh_error(self):
        self.absolute_mesh_errors = []
        self.relative_mesh_errors = []
        self.maximum_relative_mesh_errors = []
        self.ph_mesh = self.create_ph_mesh()
        self.dy_ph_callables = self.generate_dy_ph_callables()
        x_ph, y_ph, u_ph = self.construct_x_ph()
        zipped = zip(self.backend.p, self.sol.phase_data, y_ph, u_ph)
        for args in zipped:
            self.phase_mesh_error(*args, x_ph)
        self.next_iter_mesh = self.next_iteration_mesh()

    def create_ph_mesh(self):
        phase_ph_meshes = []
        for p in self.backend.p:
            K = self.it.mesh.K[p.i]
            h_k = self.it.mesh.h_K[p.i]
            N_k = (self.it.mesh.N_K[p.i] + 1)
            phase_ph_mesh = PhaseMesh(phase=p.ocp_phase,
                                      number_mesh_sections=K,
                                      mesh_section_sizes=h_k,
                                      number_mesh_section_nodes=N_k)
            phase_ph_meshes.append(phase_ph_mesh)
        return Mesh(self.backend, phase_ph_meshes)

    def generate_dy_ph_callables(self):

        def make_all_phase_mapping(mesh):
            """Create mapping from OCP symbol to iteration symbols."""
            all_phase_mapping = {}
            for p, N in zip(self.backend.p, mesh.N):
                phase_mapping = {}
                for i in range(N):
                    mapping = {}
                    for y in p.y_var:
                        mapping[y] = ocp_ph_sym_mapping[y][i]
                    for u in p.u_var:
                        mapping[u] = ocp_ph_sym_mapping[u][i]
                    phase_mapping[i] = mapping
                all_phase_mapping[p] = phase_mapping
            return all_phase_mapping

        def make_state_derivatives(p, all_phase_mapping):
            """Construct all state derivatives for the mesh iteration."""
            dy = []
            phase_mapping = all_phase_mapping[p]
            for y_eqn in p.y_eqn:
                y_eqn = expand_eqn_to_vec(y_eqn, phase_mapping)
                dy.append(y_eqn)
            return ca.vertcat(*dy)

        def expand_eqn_to_vec(eqn, phase_mapping):
            """Convert an equation in OCP base to vector iteration base."""
            vec = []
            for mapping in phase_mapping.values():
                vec.append(casadi_substitute(eqn, mapping))
            return ca.vertcat(*vec)

        mapping = {}
        mapping_point = {}
        for p, N in zip(self.backend.p, self.ph_mesh.N):
            mapping.update({y: self.backend.sym(symbol_name(y), N)
                            for y in p.y_var})
            mapping_point.update({y_t0: mapping[y][0]
                                  for y, y_t0 in zip(p.y_var, p.y_t0_var)})
            mapping_point.update({y_tF: mapping[y][-1]
                                  for y, y_tF in zip(p.y_var, p.y_tF_var)})
            mapping.update({u: self.backend.sym(symbol_name(u), N)
                            for u in p.u_var})
            mapping_point.update({q: q for q in p.q_var})
            mapping_point.update({t: t for t in p.t_var})
        mapping_point.update({s: s for s in self.backend.s_var})
        ocp_ph_sym_mapping = mapping
        ocp_ph_sym_point_mapping = mapping_point
        var_ph = []
        for x in self.backend.x_var:
            var = ocp_ph_sym_mapping.get(x)
            if var is not None:
                var_ph.append(var)
            var = ocp_ph_sym_point_mapping.get(x)
            if var is not None:
                var_ph.append(var)
        x_var_ph = ca.vertcat(*var_ph)

        all_phase_mapping = make_all_phase_mapping(self.ph_mesh)
        subs = dict_merge(ocp_ph_sym_point_mapping,
                          {V: 1 for V in self.backend.V_sym_val_mapping_iter},
                          {r: 0 for r in self.backend.r_sym_val_mapping_iter},
                          self.backend.bounds.aux_data)
        dy_ph_fncs = []
        for p in self.backend.p:
            dy_phase = make_state_derivatives(p, all_phase_mapping)
            dy_ph_phase = casadi_substitute(dy_phase, subs)
            dy_ph_fnc = ca.Function(f"dy_P{p.i}", [x_var_ph], [dy_ph_phase])
            dy_ph_fncs.append(dy_ph_fnc)
        return tuple(dy_ph_fncs)

    def construct_x_ph(self):

        def get_y_ph(p, p_data):
            y_ph = np.zeros((p.num_y_var, self.ph_mesh.N[p.i]))
            y_slice = self.it.mesh.mesh_index_boundaries[p.i]
            y_slice_ph = self.ph_mesh.mesh_index_boundaries[p.i]
            y_ph[:, y_slice_ph] = p_data.y[:, y_slice]
            y_polys = self.sol.phase_polys[p.i].y
            return eval_polynomials(y_polys, self.ph_mesh, y_ph)

        def get_u_ph(p, p_data):
            if not p.num_u_var:
                return np.array([])
            u_ph = np.zeros((p.num_u_var, self.ph_mesh.N[p.i]))
            u_slice = self.it.mesh.mesh_index_boundaries[p.i]
            u_slice_ph = self.ph_mesh.mesh_index_boundaries[p.i]
            u_ph[:, u_slice_ph] = p_data.u[:, u_slice]
            u_polys = self.sol.phase_polys[p.i].u
            return eval_polynomials(u_polys, self.ph_mesh, u_ph)

        def eval_polynomials(polys, mesh, vals):
            sec_bnd_inds = mesh.mesh_index_boundaries[p.i]
            for i_var, poly_row in enumerate(polys):
                zipped = zip(poly_row, sec_bnd_inds[:-1], sec_bnd_inds[1:])
                for i_k, (poly, i_start, i_stop) in enumerate(zipped):
                    sec_slice = slice(i_start + 1, i_stop)
                    vals[i_var, sec_slice] = poly(mesh.tau[p.i][sec_slice])
            return vals

        y_ph_all = []
        u_ph_all = []
        x_ph_all = []
        for p, p_data in zip(self.backend.p, self.sol.phase_data):
            y_ph = get_y_ph(p, p_data)
            u_ph = get_u_ph(p, p_data)
            y_ph_all.append(y_ph)
            u_ph_all.append(u_ph)
            x_ph = chain_from_iter(((y for y in y_ph),
                                    (u for u in u_ph),
                                    (np.array([q]) for q in p_data.q),
                                    (np.array([t]) for t in p_data.t)))
            x_ph_all.append(np.concatenate(tuple(x_ph)))
        x_ph_all.append([np.array(s) for s in self.sol._s])
        x_ph_all = np.concatenate(x_ph_all)
        return x_ph_all, y_ph_all, u_ph_all

    def phase_mesh_error(self, p, p_data, y_ph, u_ph, x_ph):
        dy_ph = np.array(self.dy_ph_callables[p.i](x_ph))
        dy_ph = dy_ph.reshape((-1, p.num_y_var), order="F")
        I_dy_ph = p_data.stretch * self.ph_mesh.sI_matrix[p.i].dot(dy_ph)
        dim_1 = self.it.mesh.K[p.i]
        dim_2 = p.num_y_var
        dim_3 = max(self.ph_mesh.N_K[p.i]) - 1
        mesh_error = np.zeros((dim_1, dim_2, dim_3))
        rel_error_scale_factor = np.zeros((self.it.mesh.K[p.i], p.num_y_var))

        zipped = zip(self.ph_mesh.mesh_index_boundaries[p.i][:-1],
                     self.ph_mesh.N_K[p.i] - 1)
        for i_k, (i_start, m_k) in enumerate(zipped):
            y_k = y_ph[:, i_start]
            Y_ph_k = (y_k + I_dy_ph[i_start:i_start + m_k]).T
            Y_k = y_ph[:, i_start + 1:i_start + 1 + m_k]
            mesh_error[i_k, :, :m_k] = Y_ph_k - Y_k
            rel_error_scale_factor[i_k, :] = np.max(np.abs(Y_k), axis=1) + 1

        absolute_mesh_error = np.abs(mesh_error)
        self.absolute_mesh_errors.append(absolute_mesh_error)

        relative_mesh_error = np.zeros_like(absolute_mesh_error)
        for i_k in range(self.ph_mesh.K[p.i]):
            for i_y in range(p.num_y_var):
                for i_m in range(self.ph_mesh.N_K[p.i][i_k] - 1):
                    denominator = (1 + rel_error_scale_factor[i_k, i_y])
                    val = absolute_mesh_error[i_k, i_y, i_m] / denominator
                    relative_mesh_error[i_k, i_y, i_m] = val
        self.relative_mesh_errors.append(relative_mesh_error)

        max_relative_error = np.zeros(self.it.mesh.K[p.i])
        for i_k in range(self.ph_mesh.K[p.i]):
            max_relative_error[i_k] = np.max(relative_mesh_error[i_k, :, :])
        self.maximum_relative_mesh_errors.append(max_relative_error)

    def next_iteration_mesh(self):
        phase_meshes = []
        for p in self.backend.p:
            phase_mesh = self.next_iteration_phase_mesh(p)
            phase_meshes.append(phase_mesh)
        new_mesh = Mesh(self.backend, phase_meshes)
        return new_mesh

    def next_iteration_phase_mesh(self, p):

        def merge_sections(new_mesh_sec_sizes,
                           new_num_mesh_sec_nodes,
                           merge_group):
            merge_group = np.array(merge_group)
            P_q = merge_group[:, 0]
            h_q = merge_group[:, 1]
            p_q = merge_group[:, 2]
            N = np.sum(p_q)
            T = np.sum(h_q)
            collocation_points_min = self.ocp.settings.collocation_points_min
            merge_ratio = p_q / (collocation_points_min - P_q)
            mesh_secs_needed = np.ceil(np.sum(merge_ratio)).astype(int)
            if mesh_secs_needed == 1:
                new_mesh_secs = np.array([T])
            else:
                required_reduction = np.divide(h_q, merge_ratio)
                weighting_factor = np.reciprocal(np.sum(required_reduction))
                reduction_factor = weighting_factor * required_reduction
                knot_locations = np.cumsum(h_q) / T
                current_density = np.cumsum(reduction_factor)
                density_func = interpolate.interp1d(knot_locations,
                                                    current_density,
                                                    bounds_error=False,
                                                    fill_value="extrapolate")
                new_density = np.linspace(1 / mesh_secs_needed,
                                          1,
                                          mesh_secs_needed)
                new_knots = np.concatenate([np.array([0]),
                                            density_func(new_density)])
                new_mesh_secs = T * np.diff(new_knots)
            new_mesh_sec_sizes.extend(new_mesh_secs.tolist())
            new_node_counts = [collocation_points_min] * mesh_secs_needed
            new_num_mesh_sec_nodes.extend(new_node_counts)
            return new_mesh_sec_sizes, new_num_mesh_sec_nodes

        def subdivide_sections(new_mesh_sec_sizes,
                               new_num_mesh_sec_nodes,
                               subdivide_group):
            subdivide_group = np.array(subdivide_group)
            subdivide_required = subdivide_group[:, 0].astype(bool)
            subdivide_factor = subdivide_group[:, 1].astype(int)
            reduction_tol = subdivide_group[:, 2]
            P_q = subdivide_group[:, 3]
            h_q = subdivide_group[:, 4]
            p_q = subdivide_group[:, 5]

            is_node_reduction = P_q <= 0

            predicted_nodes = P_q + p_q
            predicted_nodes[is_node_reduction] = np.ceil(
                P_q[is_node_reduction] * reduction_tol[is_node_reduction]) + p_q[is_node_reduction]

            next_mesh_nodes = np.ones_like(
                predicted_nodes, dtype=int) * col_points_min
            next_mesh_nodes[np.invert(
                subdivide_required)] = predicted_nodes[np.invert(subdivide_required)]
            next_mesh_nodes_lower_than_min = next_mesh_nodes < col_points_min
            next_mesh_nodes[next_mesh_nodes_lower_than_min] = col_points_min

            for h, k, n in zip(h_q, subdivide_factor, next_mesh_nodes):
                new_mesh_sec_sizes.extend([h / k] * k)
                new_num_mesh_sec_nodes.extend([n] * k)

            return new_mesh_sec_sizes, new_num_mesh_sec_nodes

        mesh_tol = self.ocp.settings.mesh_tolerance
        col_points_min = self.ocp.settings.collocation_points_min
        col_points_max = self.ocp.settings.collocation_points_max
        max_rel_mesh_errs = self.maximum_relative_mesh_errors[p.i]

        if np.max(max_rel_mesh_errs) > mesh_tol:

            error_to_tolerance_ratio = max_rel_mesh_errs / mesh_tol
            log_error_to_tolerance_ratio = np.log(error_to_tolerance_ratio)
            log_base = np.log(self.it.mesh.N_K[p.i])
            P_q = np.ceil(np.divide(log_error_to_tolerance_ratio, log_base))
            P_q_zero = P_q <= 0
            P_q_reduced = P_q[P_q_zero]
            P_q[P_q_zero] = P_q_reduced + np.ceil((np.log(-P_q_reduced + 1)))
            predicted_nodes = P_q + self.it.mesh.N_K[p.i]

            MERGE_TOLERANCE_FACTOR = 0
            log_tolerance = np.log(np.divide(mesh_tol, max_rel_mesh_errs))
            merge_tolerance = MERGE_TOLERANCE_FACTOR / log_tolerance
            merge_required = predicted_nodes < merge_tolerance

            reduction_tolerance = 1 + np.reciprocal(log_tolerance)
            reduction_tolerance_lt_zero = reduction_tolerance < 0
            reduction_tolerance[reduction_tolerance_lt_zero] = 0

            subdivide_required = predicted_nodes >= col_points_max
            subdivide_level = np.ones_like(predicted_nodes)
            subdivide_level[subdivide_required] = np.ceil(
                predicted_nodes[subdivide_required] / col_points_min)

            merge_group = []
            subdivide_group = []
            new_mesh_sec_sizes = []
            new_num_mesh_sec_nodes = []
            zipped = zip(merge_required,
                         subdivide_required,
                         subdivide_level,
                         reduction_tolerance,
                         P_q,
                         self.it.mesh.h_K[p.i],
                         self.it.mesh.N_K[p.i])
            for need_merge, need_subdivide, subdivide_factor, tol, P, h, N_k in zipped:
                if need_merge:
                    if subdivide_group != []:
                        new_mesh_sec_sizes, new_num_mesh_sec_nodes = subdivide_sections(
                            new_mesh_sec_sizes, new_num_mesh_sec_nodes, subdivide_group)
                        subdivide_group = []
                    merge = [P, h, N_k]
                    merge_group.append(merge)
                else:
                    if merge_group != []:
                        new_mesh_sec_sizes, new_num_mesh_sec_nodes = merge_sections(
                            new_mesh_sec_sizes, new_num_mesh_sec_nodes, merge_group)
                        merge_group = []
                    subdivide = [need_subdivide,
                                 subdivide_factor,
                                 tol,
                                 P,
                                 h,
                                 N_k]
                    subdivide_group.append(subdivide)
            else:
                if merge_group != []:
                    new_mesh_sec_sizes, new_num_mesh_sec_nodes = merge_sections(
                        new_mesh_sec_sizes, new_num_mesh_sec_nodes, merge_group)
                elif subdivide_group != []:
                    new_mesh_sec_sizes, new_num_mesh_sec_nodes = subdivide_sections(
                        new_mesh_sec_sizes, new_num_mesh_sec_nodes, subdivide_group)
            new_number_mesh_secs = len(new_mesh_sec_sizes)
            new_mesh = PhaseMesh(phase=p.ocp_phase,
                                 number_mesh_sections=new_number_mesh_secs,
                                 mesh_section_sizes=new_mesh_sec_sizes,
                                 number_mesh_section_nodes=new_num_mesh_sec_nodes)
            return new_mesh
        else:
            return p.ocp_phase.mesh


MESH_REFINEMENT_ALGORITHMS = Options((PATTERSON_RAO, ),
                                     default=PATTERSON_RAO,
                                     handles=(PattersonRaoMeshRefinement, ))
