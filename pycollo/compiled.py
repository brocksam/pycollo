import numpy as np
import scipy.sparse as sparse
import sympy as sym
from pyproprop import Options

from .numbafy import numbafy
from .numbafy_hessian import (
    numbafy_endpoint_hessian, numbafy_continuous_hessian)
from .utils import console_out


DIFFERENTIAL = "differential"
INTEGRAL = "integral"
COLLOCATION_MATRIX_FORMS = Options((DIFFERENTIAL, INTEGRAL), default=INTEGRAL,
                                   unsupported=DIFFERENTIAL)


class CompiledFunctions:

    def __init__(self, ocp_backend):
        self.ocp_backend = ocp_backend
        self.ocp = ocp_backend.ocp

        self.console_out_compiling_nlp_functions()
        self.compile_reshape()
        self.compile_objective()
        self.compile_objective_gradient()
        self.compile_constraints()
        self.compile_jacobian_constraints()
        if self.ocp.settings.derivative_level == 2:
            self.compile_hessian_lagrangian()

    def console_out_compiling_nlp_functions(self):
        msg = ("Beginning NLP function compilation.")
        console_out(msg)

    def compile_reshape(self):

        def reshape_x(x, y_slices, u_slices, q_slices, t_slices, s_slice, p_N):
            x = np.array(x)
            x_phase_tuples = []
            for y_slice, u_slice, q_slice, t_slice, N in zip(y_slices, u_slices, q_slices, t_slices, p_N):
                y = x[y_slice].reshape(-1, N)
                u = x[u_slice].reshape(-1, N)
                q = x[q_slice]
                t = x[t_slice]
                x_phase_tuple = (*y, *u, *q, *t)
                x_phase_tuples.extend(x_phase_tuple)
            s = x[s_slice]
            x_tuple = (*x_phase_tuples, *s)
            return x_tuple

        def reshape_x_point(x, x_endpoint_indices):
            x = np.array(x)
            x_tuple = x[x_endpoint_indices]
            return x_tuple

        self.x_reshape_lambda = reshape_x
        self.x_reshape_lambda_point = reshape_x_point
        print('Variable reshape functions compiled.')

    def compile_objective(self):

        def objective_lambda(x_reshaped_point):
            J = J_lambda(*x_reshaped_point)
            return J

        expr_graph = self.ocp_backend.expression_graph

        J_lambda = numbafy(
            expression_graph=expr_graph,
            expression=expr_graph.J,
            precomputable_nodes=expr_graph.J_precomputable,
            dependent_tiers=expr_graph.J_dependent_tiers,
            parameters=self.ocp_backend.x_point_vars,
        )
        self.J_lambda = objective_lambda
        print('Objective function compiled.')

    def compile_objective_gradient(self):

        def objective_gradient_lambda(x_tuple_point, p_N):
            g_phases = []
            for dJ_dxb, N in zip(dJ_dxb_phase_lambdas, p_N):
                g_phases.append(dJ_dxb(*x_tuple_point, N))
            g_endpoint = dJ_dxb_endpoint_lambda(*x_tuple_point)
            g = np.concatenate(g_phases + [g_endpoint])
            return g

        expr_graph = self.ocp_backend.expression_graph
        dJ_dxb = expr_graph.dJ_dxb
        dJ_dxb = sym.Matrix(sym.SparseMatrix(*dJ_dxb.shape, dJ_dxb.entries))

        dJ_dxb_phase_lambdas = []
        for p, p_slice in zip(self.ocp_backend.p, self.ocp_backend.phase_endpoint_variable_slices):

            dJ_dxb_phase_lambda = numbafy(
                expression_graph=expr_graph,
                expression=dJ_dxb[p_slice],
                precomputable_nodes=expr_graph.dJ_dxb_precomputable,
                dependent_tiers=expr_graph.dJ_dxb_dependent_tiers,
                parameters=self.ocp_backend.x_point_vars,
                return_dims=1,
                N_arg=True,
                endpoint=True,
                ocp_num_vars=p.num_each_vars,
            )
            dJ_dxb_phase_lambdas.append(dJ_dxb_phase_lambda)

        dJ_dxb_endpoint_lambda = numbafy(
            expression_graph=expr_graph,
            expression=dJ_dxb[self.ocp_backend.endpoint_variable_slice],
            precomputable_nodes=expr_graph.dJ_dxb_precomputable,
            dependent_tiers=expr_graph.dJ_dxb_dependent_tiers,
            parameters=self.ocp_backend.x_point_vars,
            return_dims=1,
            N_arg=False,
            endpoint=False,
        )

        self.g_lambda = objective_gradient_lambda
        print('Objective gradient function compiled.')

    def compile_constraints(self):

        def phase_constraints_lambda(x_tuple, t_stretch_lambda, c_continuous_lambda, A, D, W, N, y_slice, q_slice, dy_slice, p_slice, g_slice):
            stretch = t_stretch_lambda(*x_tuple)
            c_continuous = c_continuous_lambda(*x_tuple, N)
            y = np.vstack(x_tuple[y_slice])
            q = np.array(x_tuple[q_slice])
            dy = c_continuous[dy_slice].reshape((-1, N))
            p = c_continuous[p_slice]
            g = c_continuous[g_slice].reshape((-1, N))
            c_defect = defect_constraints_lambda(y, dy, A, D, stretch)
            c_path = path_constraints_lambda(p)
            c_integral = integral_constraints_lambda(q, g, W, stretch)
            c = np.concatenate([c_defect, c_path, c_integral])
            return c

        def defect_constraints_lambda(y, dy, A, D, stretch):
            return (D.dot(y.T) + stretch * A.dot(dy.T)).flatten(order='F')

        def path_constraints_lambda(p):
            return p

        def integral_constraints_lambda(q, g, W, stretch):
            return q - stretch * np.matmul(g, W) if g.size else q

        def constraints_lambda(x_tuple, x_tuple_point, p_A, p_D, p_W, p_N, ocp_phase_y_slice, ocp_phase_q_slice, ocp_phase_dy_slice, ocp_phase_p_slice, ocp_phase_g_slice):
            c = np.array([])
            for t_stretch_lambda, c_continuous_lambda, A, D, W, N, y_slice, q_slice, dy_slice, p_slice, g_slice in zip(t_stretch_lambdas, c_continuous_lambdas, p_A, p_D, p_W, p_N, ocp_phase_y_slice, ocp_phase_q_slice, ocp_phase_dy_slice, ocp_phase_p_slice, ocp_phase_g_slice):
                c_phase = phase_constraints_lambda(
                    x_tuple, t_stretch_lambda, c_continuous_lambda, A, D, W, N, y_slice, q_slice, dy_slice, p_slice, g_slice)
                c = np.concatenate([c, c_phase])
            c_endpoint = c_endpoint_lambda(*x_tuple_point)
            c = np.concatenate([c, c_endpoint])
            return c

        expr_graph = self.ocp_backend.expression_graph

        t_stretch_lambdas = []
        for p in self.ocp_backend.p:
            t_norm = getattr(expr_graph, f"t_norm_P{p.i}")
            t_norm_precomputable = getattr(expr_graph, f"t_norm_P{p.i}_precomputable")
            t_norm_dependent_tiers = getattr(expr_graph, f"t_norm_P{p.i}_dependent_tiers")
            t_stretch_lambda = numbafy(
                expression_graph=expr_graph,
                expression=t_norm,
                precomputable_nodes=t_norm_precomputable,
                dependent_tiers=t_norm_dependent_tiers,
                parameters=self.ocp_backend.x_vars,
            )
            t_stretch_lambdas.append(t_stretch_lambda)

        c_continuous_lambdas = []
        for p, p_slice in zip(self.ocp_backend.p, self.ocp_backend.phase_constraint_slices):
            c = sym.Matrix(expr_graph.c[p_slice])
            c_nodes = expr_graph.c_nodes[p_slice]
            c_continuous_lambda = numbafy(
                expression_graph=expr_graph,
                expression=c,
                expression_nodes=c_nodes,
                precomputable_nodes=expr_graph.c_precomputable,
                dependent_tiers=expr_graph.c_dependent_tiers,
                parameters=self.ocp_backend.x_vars,
                return_dims=2,
                N_arg=True,
                ocp_num_vars=p.num_each_vars,
            )
            c_continuous_lambdas.append(c_continuous_lambda)

        c_endpoint_lambda = numbafy(
            expression_graph=expr_graph,
            expression=expr_graph.b,
            precomputable_nodes=expr_graph.b_precomputable,
            dependent_tiers=expr_graph.b_dependent_tiers,
            parameters=self.ocp_backend.x_point_vars,
            return_dims=1,
            N_arg=False,
        )

        self._t_stretch_lambdas = t_stretch_lambdas
        self._t_dstretch_dt_lambdas = []
        for p in self.ocp_backend.p:
            dstretch_dt = np.array([val
                                    for val, t_needed in zip(np.array([-0.5, 0.5]), p.ocp_phase.bounds._t_needed)
                                    if t_needed])
            self._t_dstretch_dt_lambdas.append(dstretch_dt)
        self.c_continuous_lambdas = c_continuous_lambdas
        self.c_endpoint_lambda = c_endpoint_lambda
        self.c_lambda = constraints_lambda
        print('Constraints function compiled.')

    def compile_jacobian_constraints(self):

        def phase_jacobian_lambda(G_shape, x_tuple, t_stretch_lambda, dstretch_dt, c_continuous_lambda, dc_dx_lambda, A, D, W, N, phase_row_offset, phase_col_offset, y_slice, u_slice, q_slice, t_slice, dy_slice, p_slice, g_slice, c_defect_slice, c_path_slice, c_integral_slice, num_y, num_u, num_q, num_t, num_x, num_s, num_c_defect, num_c_path, num_c_integral, num_c, phase_y_slice, phase_u_slice, phase_q_slice, phase_t_slice, phase_c_defect_slice, phase_c_path_slice, phase_c_integral_slice):

            ocp_num_y = int(num_y / N)
            ocp_num_u = int(num_u / N)

            ocp_num_c_path = int(num_c_path / N)

            stretch = t_stretch_lambda(*x_tuple)
            c_continuous = c_continuous_lambda(*x_tuple, N)
            y = np.vstack(x_tuple[y_slice])
            q = np.array(x_tuple[q_slice])
            dy = c_continuous[dy_slice].reshape((-1, N))
            p = c_continuous[p_slice]
            g = c_continuous[g_slice].reshape((-1, N))

            dc_dx = dc_dx_lambda(*x_tuple, N).astype(float)
            dzeta_dy = dc_dx[phase_c_defect_slice,
                             phase_y_slice, :].reshape(-1, N)
            dzeta_du = dc_dx[phase_c_defect_slice,
                             phase_u_slice, :].reshape(-1, N)
            dzeta_ds = dc_dx[phase_c_defect_slice,
                             phase_t_slice.stop:, :].reshape(-1, N)
            dgamma_dy = dc_dx[phase_c_path_slice,
                              phase_y_slice, :].reshape(-1, N)
            dgamma_du = dc_dx[phase_c_path_slice,
                              phase_u_slice, :].reshape(-1, N)
            dgamma_ds = dc_dx[phase_c_path_slice,
                              phase_t_slice.stop:, :].reshape(-1, N)
            drho_dy = dc_dx[phase_c_integral_slice,
                            phase_y_slice, :].reshape(-1, N)
            drho_du = dc_dx[phase_c_integral_slice,
                            phase_u_slice, :].reshape(-1, N)
            drho_ds = dc_dx[phase_c_integral_slice,
                            phase_t_slice.stop:, :].reshape(-1, N)

            G_dzeta_dy = phase_G_dzeta_dy_lambda(
                dzeta_dy, stretch, A, D, ocp_num_y)
            G_dzeta_du = phase_G_dzeta_du_lambda(
                dzeta_du, stretch, A, ocp_num_y, ocp_num_u) if num_u else None
            G_dzeta_dq = None
            G_dzeta_dt = phase_G_dzeta_dt_lambda(
                dy, A, dstretch_dt, ocp_num_y, num_c_defect, num_t) if num_t else None
            G_dzeta_ds = phase_G_dzeta_ds_lambda(
                dzeta_ds, stretch, A, ocp_num_y, num_c_defect, num_s) if num_s else sparse.csr_matrix((num_c_defect, num_s))
            G_dgamma_dy = phase_G_dgamma_dy_lambda(
                dgamma_dy, ocp_num_c_path, ocp_num_y) if num_c_path else None
            G_dgamma_du = phase_G_dgamma_du_lambda(
                dgamma_du, ocp_num_c_path, ocp_num_u) if (num_c_path and num_u) else None
            G_dgamma_dq = None
            G_dgamma_dt = None
            G_dgamma_ds = phase_G_dgamma_ds_lambda(dgamma_ds, num_c_path, num_s) if (
                num_c_path and num_s) else sparse.csr_matrix((num_c_path, num_s))
            G_drho_dy = phase_G_drho_dy_lambda(
                drho_dy, stretch, W, num_c_integral, num_y) if num_c_integral else None
            G_drho_du = phase_G_drho_du_lambda(drho_du, stretch, W, num_c_integral, num_u) if (
                num_c_integral and num_u) else None
            G_drho_dq = phase_G_drho_dq_lambda(num_q) if (
                num_c_integral and num_q) else None
            G_drho_dt = phase_G_drho_dt_lambda(g, dstretch_dt, W, num_c_integral, num_t) if (
                num_c_integral and num_t) else None
            G_drho_ds = phase_G_drho_ds_lambda(drho_ds, stretch, W, num_c_integral, num_s) if (
                num_c_integral and num_s) else sparse.csr_matrix((num_c_integral, num_s))

            G_phase = sparse.bmat([
                [G_dzeta_dy, G_dzeta_du, G_dzeta_dq, G_dzeta_dt],
                [G_dgamma_dy, G_dgamma_du, G_dgamma_dq, G_dgamma_dt],
                [G_drho_dy, G_drho_du, G_drho_dq, G_drho_dt],
            ])

            G_parameter = sparse.vstack(
                [G_dzeta_ds, G_dgamma_ds, G_drho_ds]).tocoo()

            phase_data = G_phase.data
            phase_row = G_phase.row + phase_row_offset
            phase_col = G_phase.col + phase_col_offset
            G_phase = sparse.coo_matrix(
                (phase_data, (phase_row, phase_col)), shape=G_shape)

            parameter_row_offset = phase_row_offset
            parameter_col_offset = G_shape[1] - num_s
            parameter_data = G_parameter.data
            parameter_row = G_parameter.row + parameter_row_offset
            parameter_col = G_parameter.col + parameter_col_offset
            G_parameter = sparse.coo_matrix(
                (parameter_data, (parameter_row, parameter_col)), shape=G_shape)

            G_continuous = G_phase + G_parameter

            return G_continuous

        def phase_G_dzeta_dy_lambda(ddy_dy, stretch, A, D, ocp_num_y):
            G_dzeta_dy = []
            for row in ddy_dy:
                G_dzeta_dy.append(stretch * A * sparse.diags(row))
            G_dzeta_dy = sparse.bmat(np.array(G_dzeta_dy).reshape(
                ocp_num_y, ocp_num_y), dtype=float)
            D_data = sparse.block_diag([D] * ocp_num_y)
            G_dzeta_dy += D_data
            return G_dzeta_dy

        def phase_G_dzeta_du_lambda(ddy_du, stretch, A, ocp_num_y, ocp_num_u):
            G_dzeta_du = []
            for row in ddy_du:
                block = sparse.dia_matrix(row)
                G_dzeta_du.append(stretch * A.multiply(block))
            G_dzeta_du = sparse.bmat(
                np.array(G_dzeta_du).reshape(ocp_num_y, ocp_num_u))
            return G_dzeta_du

        def phase_G_dzeta_dt_lambda(dy, A, dstretch_dt, ocp_num_y, num_c_defect, num_t):
            A_dy_flat = (A.dot(dy.T).flatten(order="F"))
            G_dzeta_dt = np.outer(dstretch_dt, A_dy_flat).T
            G_dzeta_dt = sparse.csr_matrix(G_dzeta_dt)
            return G_dzeta_dt

        def phase_G_dzeta_ds_lambda(ddy_ds, stretch, A, ocp_num_y, num_c_defect, num_s):
            G_dzeta_ds = stretch * A.dot(ddy_ds.T)
            G_dzeta_ds = np.vstack(np.split(G_dzeta_ds, ocp_num_y, axis=1))
            G_dzeta_ds = sparse.csr_matrix(G_dzeta_ds)
            return G_dzeta_ds

        def phase_G_dgamma_dy_lambda(dgamma_dy, ocp_num_c_path, ocp_num_y):
            G_dgamma_dy = []
            for row in dgamma_dy:
                G_dgamma_dy.append(sparse.diags(row))
            G_dgamma_dy = sparse.bmat(
                np.array(G_dgamma_dy).reshape(ocp_num_c_path, ocp_num_y))
            return G_dgamma_dy

        def phase_G_dgamma_du_lambda(dgamma_du, ocp_num_c_path, ocp_num_u):
            G_dgamma_du = []
            for row in dgamma_du:
                G_dgamma_du.append(sparse.diags(row))
            G_dgamma_du = sparse.bmat(
                np.array(G_dgamma_du).reshape(ocp_num_c_path, ocp_num_u))
            return G_dgamma_du

        def phase_G_dgamma_ds_lambda(dgamma_ds, num_c_path, num_s):
            G_dgamma_ds = sparse.csr_matrix(
                dgamma_ds.reshape((num_c_path, num_s), order="F"))
            return G_dgamma_ds

        def phase_G_drho_dy_lambda(drho_dy, stretch, W, num_c_integral, num_y):
            G_drho_dy = sparse.csr_matrix(
                (- stretch * drho_dy * W).reshape(num_c_integral, num_y))
            return G_drho_dy

        def phase_G_drho_du_lambda(drho_du, stretch, W, num_c_integral, num_u):
            G_drho_du = sparse.csr_matrix(
                (- stretch * drho_du * W).reshape(num_c_integral, num_u))
            return G_drho_du

        def phase_G_drho_dq_lambda(num_q):
            G_drho_dq = sparse.eye(num_q)
            return G_drho_dq

        def phase_G_drho_dt_lambda(g, dstretch_dt, W, num_c_integral, num_t):
            G_drho_dt = sparse.csr_matrix(-np.outer(dstretch_dt,
                                                    np.matmul(g, W)).T)
            return G_drho_dt

        def phase_G_drho_ds_lambda(drho_ds, stretch, W, num_c_integral, num_s):
            G_drho_ds = sparse.csr_matrix(
                (- stretch * np.matmul(drho_ds, W)).reshape((num_c_integral), num_s, order="F"))
            return G_drho_ds

        def endpoint_jacobian_lambda(G_shape, x_point_tuple, db_dxb_phase_lambdas, db_ds_lambda, p_N, num_y_per_phase, num_q_per_phase, num_t_per_phase, num_x_per_phase, num_s, num_c_endpoint):
            row_offset = 0
            db_dxb_phases = []
            for db_dxb_lambda, N, num_y, num_q, num_t, num_x in zip(db_dxb_phase_lambdas, p_N, num_y_per_phase, num_q_per_phase, num_t_per_phase, num_x_per_phase):
                ocp_num_y = int(num_y / N)
                db_dxb_phase = db_dxb_lambda(*x_point_tuple, N).flatten()
                y_row_indices = np.array(
                    [(i * N, (i + 1) * N - 1) for i in range(ocp_num_y)]).flatten()
                q_row_indices = np.array(range(num_x - num_q - num_t, num_x))
                row_indices = np.repeat(
                    np.array(range(num_c_endpoint)), ocp_num_y * 2 + num_q + num_t)
                col_indices = np.tile(np.concatenate(
                    [y_row_indices, q_row_indices]), num_c_endpoint)
                db_dxb_phase = sparse.coo_matrix(
                    (db_dxb_phase, (row_indices, col_indices)), shape=(num_c_endpoint, num_x))
                db_dxb_phases.append(db_dxb_phase)
            db_ds = sparse.csr_matrix(db_ds_lambda(
                *x_point_tuple).reshape(num_c_endpoint, num_s))
            db_dxb_blocks = db_dxb_phases + [db_ds]
            G_endpoint = sparse.hstack(db_dxb_blocks)
            G_continuous_shape = (G_shape[0] - num_c_endpoint, G_shape[1])
            G_continuous = sparse.csr_matrix(G_continuous_shape)
            G_endpoint = sparse.vstack([G_continuous, G_endpoint])
            return G_endpoint

        def jacobian_lambda(G_shape, x_tuple, x_point_tuple, p_A, p_D, p_W, p_N, phase_row_offsets, phase_col_offsets, ocp_phase_y_slice, ocp_phase_u_slice, ocp_phase_q_slice, ocp_phase_t_slice, ocp_phase_dy_slice, ocp_phase_p_slice, ocp_phase_g_slice, ocp_phase_c_defect_slice, ocp_phase_c_path_slice, ocp_phase_c_integral_slice, num_y_per_phase, num_u_per_phase, num_q_per_phase, num_t_per_phase, num_x_per_phase, num_s, num_c_defect_per_phase, num_c_path_per_phase, num_c_integral_per_phase, num_c_per_phase, num_c_endpoint, phase_y_slices, phase_u_slices, phase_q_slices, phase_t_slices, phase_c_defect_slices, phase_c_path_slices, phase_c_integral_slices):
            G = sparse.csr_matrix(G_shape)
            for t_stretch_lambda, dstretch_dt, c_continuous_lambda, dc_dx_lambda, A, D, W, N, row_offset, col_offset, y_slice, u_slice, q_slice, t_slice, dy_slice, p_slice, g_slice, c_defect_slice, c_path_slice, c_integral_slice, num_y, num_u, num_q, num_t, num_x, num_c_defect, num_c_path, num_c_integral, num_c, phase_y_slice, phase_u_slice, phase_q_slice, phase_t_slice, phase_c_defect_slice, phase_c_path_slice, phase_c_integral_slice in zip(self._t_stretch_lambdas, self._t_dstretch_dt_lambdas, self.c_continuous_lambdas, dc_dx_lambdas, p_A, p_D, p_W, p_N, phase_row_offsets, phase_col_offsets, ocp_phase_y_slice, ocp_phase_u_slice, ocp_phase_q_slice, ocp_phase_t_slice, ocp_phase_dy_slice, ocp_phase_p_slice, ocp_phase_g_slice, ocp_phase_c_defect_slice, ocp_phase_c_path_slice, ocp_phase_c_integral_slice, num_y_per_phase, num_u_per_phase, num_q_per_phase, num_t_per_phase, num_x_per_phase, num_c_defect_per_phase, num_c_path_per_phase, num_c_integral_per_phase, num_c_per_phase, phase_y_slices, phase_u_slices, phase_q_slices, phase_t_slices, phase_c_defect_slices, phase_c_path_slices, phase_c_integral_slices):
                G += phase_jacobian_lambda(G_shape, x_tuple, t_stretch_lambda, dstretch_dt, c_continuous_lambda, dc_dx_lambda, A, D, W, N, row_offset, col_offset, y_slice, u_slice, q_slice, t_slice, dy_slice, p_slice, g_slice, c_defect_slice, c_path_slice,
                                           c_integral_slice, num_y, num_u, num_q, num_t, num_x, num_s, num_c_defect, num_c_path, num_c_integral, num_c, phase_y_slice, phase_u_slice, phase_q_slice, phase_t_slice, phase_c_defect_slice, phase_c_path_slice, phase_c_integral_slice)
            G += endpoint_jacobian_lambda(G_shape, x_point_tuple, db_dxb_phase_lambdas, db_ds_lambda, p_N,
                                          num_y_per_phase, num_q_per_phase, num_t_per_phase, num_x_per_phase, num_s, num_c_endpoint)
            return G

        expr_graph = self.ocp_backend.expression_graph
        dc_dx = expr_graph.dc_dx
        dc_dx = sym.Matrix(sym.SparseMatrix(*dc_dx.shape, dc_dx.entries))
        dc_dx_nodes = np.zeros(expr_graph.dc_dx_nodes.shape, dtype=object)
        for k, v in expr_graph.dc_dx_nodes.entries.items():
            dc_dx_nodes[k] = v

        dc_dx_lambdas = []
        for p, p_x_slice, p_c_slice in zip(self.ocp_backend.p, self.ocp_backend.phase_variable_slices, self.ocp_backend.phase_constraint_slices):
            dc_dx_phase = dc_dx[p_c_slice, p_x_slice]
            dc_dx_parameter = dc_dx[p_c_slice, self.ocp_backend.variable_slice]
            dc_dx_joined = sym.Matrix(
                sym.BlockMatrix([dc_dx_phase, dc_dx_parameter]))
            dc_dx_nodes_phase = dc_dx_nodes[p_c_slice, p_x_slice]
            dc_dx_nodes_parameter = dc_dx_nodes[p_c_slice,
                                                self.ocp_backend.variable_slice]
            dc_dx_nodes_joined = np.hstack(
                [dc_dx_nodes_phase, dc_dx_nodes_parameter]).flatten().tolist()
            dc_dx_lambda = numbafy(
                expression_graph=expr_graph,
                expression=dc_dx_joined,
                expression_nodes=dc_dx_nodes_joined,
                precomputable_nodes=expr_graph.dc_dx_precomputable,
                dependent_tiers=expr_graph.dc_dx_dependent_tiers,
                parameters=self.ocp_backend.x_vars,
                return_dims=3,
                N_arg=True,
                ocp_num_vars=p.num_each_vars,
            )
            dc_dx_lambdas.append(dc_dx_lambda)

        db_dxb_phase_lambdas = []
        for p, p_slice in zip(self.ocp_backend.p, self.ocp_backend.phase_endpoint_variable_slices):
            db_dxb = expr_graph.db_dxb
            db_dxb_phase = sym.Matrix(sym.SparseMatrix(*db_dxb.shape, db_dxb.entries))[:, p_slice]
            db_dxb_phase_lambda = numbafy(
                expression_graph=expr_graph,
                expression=db_dxb_phase,
                precomputable_nodes=expr_graph.db_dxb_precomputable,
                dependent_tiers=expr_graph.db_dxb_dependent_tiers,
                parameters=self.ocp_backend.x_point_vars,
                return_dims=1,
                N_arg=True,
                ocp_num_vars=p.num_each_vars,
            )
            db_dxb_phase_lambdas.append(db_dxb_phase_lambda)

        db_dxb = expr_graph.db_dxb
        db_ds = sym.Matrix(sym.SparseMatrix(*db_dxb.shape, db_dxb.entries))[:, self.ocp_backend.endpoint_variable_slice]
        db_ds_lambda = numbafy(
            expression_graph=expr_graph,
            expression=db_ds,
            precomputable_nodes=expr_graph.db_dxb_precomputable,
            dependent_tiers=expr_graph.db_dxb_dependent_tiers,
            parameters=self.ocp_backend.x_point_vars,
            return_dims=1,
            N_arg=False,
            endpoint=False,
        )

        self.G_lambda = jacobian_lambda
        print('Jacobian function compiled.')

    def compile_hessian_lagrangian(self):

        def endpoint_hessian_lambda(H_shape, x_point_tuple, sigma, lagrange, H_indices):
            data = ddL_dxbdxb_lambda(*x_point_tuple, sigma, *lagrange)
            H = sparse.coo_matrix((data, H_indices), shape=H_shape)
            return H

        def continuous_hessian_lambda(H_shape, x_tuple, lagrange, H_indices_and_flags):
            phase_data = ddL_dxdx_lambda(*x_tuple, *lagrange)
            H = sparse.dok_matrix(H_shape)
            for data, ((root_row_index, root_col_index), flag) in zip(phase_data, H_indices_and_flags):
                if flag == 1:
                    for offset, datum in enumerate(data):
                        index = (root_row_index + offset,
                                 root_col_index + offset)
                        H[index] = datum
                elif flag == 2:
                    for offset, datum in enumerate(data):
                        index = (root_row_index, root_col_index + offset)
                        H[index] = datum
                else:
                    index = (root_row_index, root_col_index)
                    H[index] = np.sum(data)
            return H

        def hessian_lambda(H_shape, x_tuple, x_point_tuple, sigma, lagrange, H_continuous_indices, H_endpoint_indices):
            H = endpoint_hessian_lambda(
                H_shape, x_point_tuple, sigma, lagrange, H_endpoint_indices)
            H += continuous_hessian_lambda(H_shape,
                                           x_tuple, lagrange, H_continuous_indices)
            return H

        expr_graph = self.ocp_backend.expression_graph
        L_sigma = expr_graph.lagrange_syms[0]
        L_syms = expr_graph.lagrange_syms[1:]

        ddL_dxbdxb_lambda = numbafy_endpoint_hessian(
            expression_graph=expr_graph,
            expression=expr_graph.ddL_dxbdxb,
            expression_nodes=expr_graph.ddL_dxbdxb_nodes,
            precomputable_nodes=expr_graph.ddL_dxbdxb_precomputable,
            dependent_tiers=expr_graph.ddL_dxbdxb_dependent_tiers,
            parameters=self.ocp_backend.x_point_vars,
            objective_factor=L_sigma,
            lagrange_multipliers=L_syms,
        )

        ddL_dxdx_lambda = numbafy_continuous_hessian(
            expression_graph=expr_graph,
            expression=expr_graph.ddL_dxdx,
            expression_nodes=expr_graph.ddL_dxdx_nodes,
            precomputable_nodes=expr_graph.ddL_dxdx_precomputable,
            dependent_tiers=expr_graph.ddL_dxdx_dependent_tiers,
            parameters=self.ocp_backend.x_vars,
            lagrange_multipliers=L_syms,
            summing_nodes=expr_graph.ddL_dxdx_sum_nodes,
        )

        # print(expr_graph.ddL_dxdx.entries)
        # input()

        self.H_lambda = hessian_lambda
        print('Hessian function compiled.')
