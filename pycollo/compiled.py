import numpy as np

from .numbafy import numbafy
from .utils import console_out

class CompiledFunctions:

	def __init__(self, ocp_backend):
		self.ocp_backend = ocp_backend
		self.ocp = ocp_backend.ocp

		self.console_out_compiling_nlp_functions()
		self.compile_reshape()
		self.compile_objective()
		# self.compile_objective_gradient()
		# self.compile_constraints()
		# self.compile_jacobian_constraints()
		# if self.ocp.settings.derivative_level == 2:
		# 	self.compile_hessian_lagrangian()

	def console_out_compiling_nlp_functions(self):
		msg = ("Beginning NLP function compilation.")
		console_out(msg)

	def compile_reshape(self):

		def reshape_x(x, num_yu, yu_qts_split):
			x = np.array(x)
			yu = x[:yu_qts_split].reshape(num_yu, -1)
			qts = x[yu_qts_split:].reshape(len(x) - yu_qts_split, )
			x_tuple = (*yu, *qts)
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

	# def _compile_objective_gradient(self):

	# 	def objective_gradient_lambda(x_tuple_point, N):
	# 		g = dJ_dxb_lambda(*x_tuple_point, N)
	# 		return g

	# 	expr_graph = self._expression_graph

	# 	dJ_dxb_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.dJ_dxb,
	# 		precomputable_nodes=expr_graph.dJ_dxb_precomputable,
	# 		dependent_tiers=expr_graph.dJ_dxb_dependent_tiers,
	# 		parameters=self._x_b_vars,
	# 		return_dims=1, 
	# 		N_arg=True, 
	# 		endpoint=True, 
	# 		ocp_num_vars=self._num_vars_tuple
	# 		)

	# 	self._g_lambda = objective_gradient_lambda
	# 	print('Objective gradient function compiled.')

	# def _compile_constraints(self):
		
	# 	def defect_constraints_lambda(y, dy, A, D, stretch):
	# 		return (D.dot(y.T) + stretch*A.dot(dy.T)).flatten(order='F')

	# 	def path_constraints_lambda(p):
	# 		return p

	# 	def integral_constraints_lambda(q, g, W, stretch):
	# 		return q - stretch*np.matmul(g, W) if g.size else q

	# 	def endpoint_constraints_lambda(b):
	# 		return b

	# 	def constraints_lambda(x_tuple, x_tuple_point, N, ocp_y_slice, 
	# 			ocp_q_slice, num_c, dy_slice, p_slice, g_slice, defect_slice, 
	# 			path_slice, integral_slice, boundary_slice, A, D, W):
			
	# 		y = np.vstack(x_tuple[ocp_y_slice])
	# 		q = np.array(x_tuple[ocp_q_slice])

	# 		stretch = t_stretch_lambda(*x_tuple)
	# 		c_continuous = c_continuous_lambda(*x_tuple, N)
	# 		c_endpoint = c_endpoint_lambda(*x_tuple_point, N)

	# 		dy = c_continuous[dy_slice].reshape((-1, N))
	# 		p = c_continuous[p_slice]
	# 		g = c_continuous[g_slice]
	# 		b = c_endpoint

	# 		c = np.empty(num_c)
	# 		c[defect_slice] = defect_constraints_lambda(y, dy, A, D, stretch)
	# 		c[path_slice] = path_constraints_lambda(p)
	# 		c[integral_slice] = integral_constraints_lambda(q, g, W, stretch)
	# 		c[boundary_slice] = endpoint_constraints_lambda(b)

	# 		return c

	# 	expr_graph = self._expression_graph

	# 	t_stretch_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.t_norm, 
	# 		precomputable_nodes=expr_graph.t_norm_precomputable,
	# 		dependent_tiers=expr_graph.t_norm_dependent_tiers,
	# 		parameters=self._x_vars, 
	# 		)

	# 	c_continuous_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.c,
	# 		expression_nodes=expr_graph.c_nodes,
	# 		precomputable_nodes=expr_graph.c_precomputable,
	# 		dependent_tiers=expr_graph.c_dependent_tiers,
	# 		parameters=self._x_vars,
	# 		return_dims=2, 
	# 		N_arg=True, 
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	c_endpoint_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.b,
	# 		precomputable_nodes=expr_graph.b_precomputable,
	# 		dependent_tiers=expr_graph.b_dependent_tiers,
	# 		parameters=self._x_b_vars,
	# 		return_dims=1,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	self._t_stretch_lambda = t_stretch_lambda
	# 	self._dstretch_dt = [val 
	# 		for val, t_needed in zip(self._dSTRETCH_dt, self._bounds._t_needed) 
	# 		if t_needed]
	# 	self._c_continuous_lambda = c_continuous_lambda
	# 	self._c_lambda = constraints_lambda
	# 	print('Constraints function compiled.')

	# 	# print(expr_graph.c)
	# 	# print('\n\n\n')
	# 	# raise NotImplementedError

	# def _compile_jacobian_constraints(self):

	# 	def A_x_dot_sparse(A_sparse, ddy_dx, num_y, num_nz, stretch, 
	# 		return_array):
	# 		for i, row in enumerate(ddy_dx):
	# 			start = i*num_nz
	# 			stop = start + num_nz
	# 			entries = stretch*A_sparse.multiply(row).data
	# 			return_array[start:stop] = entries
	# 		return return_array

	# 	def G_dzeta_dy_lambda(ddy_dy, stretch, A, D, A_row_col_array, num_y, 
	# 		dzeta_dy_D_nonzero, dzeta_dy_slice):
	# 		num_nz = A_row_col_array.shape[1]
	# 		return_array = np.empty(num_y**2 * num_nz)
	# 		G_dzeta_dy = A_x_dot_sparse(A, ddy_dy, num_y, num_nz, stretch, 
	# 			return_array)
	# 		D_data = (D.data * np.ones((num_y, 1))).flatten()
	# 		G_dzeta_dy[dzeta_dy_D_nonzero] += D_data
	# 		return G_dzeta_dy

	# 	def G_dzeta_du_lambda(ddy_du, stretch, A, A_row_col_array, num_y, 
	# 		num_u):
	# 		num_nz = A_row_col_array.shape[1]
	# 		return_array = np.empty(num_u*num_y*num_nz)
	# 		G_dzeta_du = A_x_dot_sparse(A, ddy_du, num_u, num_nz, stretch, 
	# 			return_array)
	# 		return G_dzeta_du

	# 	def G_dzeta_dt_lambda(dy, A):
	# 		A_dy_flat = (A.dot(dy.T).flatten(order='F'))
	# 		product = np.outer(self._dstretch_dt, A_dy_flat)
	# 		G_dzeta_dt = product.flatten(order='F')
	# 		return G_dzeta_dt

	# 	def G_dzeta_ds_lambda(ddy_ds, stretch, A):
	# 		G_dzeta_ds = stretch*A.dot(ddy_ds.T).flatten(order='F')
	# 		return G_dzeta_ds

	# 	def G_dgamma_dy_lambda(dgamma_dy):
	# 		if dgamma_dy.size:
	# 			return dgamma_dy.flatten()
	# 		else:
	# 			return []

	# 	def G_dgamma_du_lambda(dgamma_du):
	# 		if dgamma_du.size:
	# 			return dgamma_du.flatten()
	# 		else:
	# 			return []

	# 	def G_dgamma_dt_lambda(p):
	# 		if p.size:
	# 			return np.zeros_like(p).flatten()
	# 		else:
	# 			return []

	# 	def G_dgamma_ds_lambda(dgamma_ds):
	# 		if dgamma_ds.size:
	# 			return dgamma_ds.flatten()
	# 		else:
	# 			return []

	# 	def G_drho_dy_lambda(drho_dy, stretch, W):

	# 		if drho_dy.size:
	# 			G_drho_dy = (- stretch * drho_dy * W).flatten()
	# 			return G_drho_dy
	# 		else:
	# 			return []

	# 	def G_drho_du_lambda(drho_du, stretch, W):
	# 		if drho_du.size:
	# 			G_drho_du = (- stretch * drho_du * W).flatten()
	# 			return G_drho_du
	# 		else:
	# 			return []

	# 	def G_drho_dt_lambda(g, W):
	# 		if g.size > 0:
	# 			product = np.outer(self._dstretch_dt, np.matmul(g, W))
	# 			return - product.flatten(order='F')
	# 		else:
	# 			return []

	# 	def G_drho_ds_lambda(drho_ds, stretch, W):
	# 		if drho_ds.size:
	# 			G_drho_ds = (- stretch * np.matmul(drho_ds, W))
	# 			return G_drho_ds
	# 		else:
	# 			return []

	# 	def jacobian_lambda(x_tuple, x_tuple_point, N, num_G_nonzero, 
	# 		num_x_ocp, A, D, W, A_row_col_array, dy_slice, 
	# 		p_slice, g_slice, dzeta_dy_D_nonzero, 
	# 		dzeta_dy_slice, dzeta_du_slice, 
	# 		dzeta_dt_slice, dzeta_ds_slice, dgamma_dy_slice, dgamma_du_slice, 
	# 		dgamma_dt_slice, dgamma_ds_slice, drho_dy_slice, drho_du_slice, 
	# 		drho_dq_slice, drho_dt_slice, drho_ds_slice, dbeta_dxb_slice):

	# 		stretch = t_stretch_lambda(*x_tuple)
	# 		dstretch_dt = self._dstretch_dt
	# 		c_continuous = c_continuous_lambda(*x_tuple, N)
	# 		dc_dx = dc_dx_lambda(*x_tuple, N)
	# 		dy = c_continuous[dy_slice].reshape((-1, N))
	# 		p = c_continuous[p_slice]
	# 		g = c_continuous[g_slice]

	# 		dzeta_dy = dc_dx[dc_dx_slice.zeta_y].reshape(dc_dx_shape.zeta_y, N)
	# 		dzeta_du = dc_dx[dc_dx_slice.zeta_u].reshape(dc_dx_shape.zeta_u, N)
	# 		dzeta_dt = None
	# 		dzeta_ds = dc_dx[dc_dx_slice.zeta_s].reshape(dc_dx_shape.zeta_s, N)
	# 		dgamma_dy = dc_dx[dc_dx_slice.gamma_y].reshape(dc_dx_shape.gamma_y, N)
	# 		dgamma_du = dc_dx[dc_dx_slice.gamma_u].reshape(dc_dx_shape.gamma_u, N)
	# 		dgamma_dt = None
	# 		dgamma_ds = dc_dx[dc_dx_slice.gamma_s].reshape(dc_dx_shape.gamma_s, N)
	# 		drho_dy = dc_dx[dc_dx_slice.rho_y].reshape(dc_dx_shape.rho_y, N)
	# 		drho_du = dc_dx[dc_dx_slice.rho_u].reshape(dc_dx_shape.rho_u, N)
	# 		drho_dt = None
	# 		drho_ds = dc_dx[dc_dx_slice.rho_s].reshape(dc_dx_shape.rho_s, N)

	# 		G = np.empty(num_G_nonzero)

	# 		if num_x_ocp.y:
	# 			G[dzeta_dy_slice] = G_dzeta_dy_lambda(dzeta_dy, stretch, A, D, 
	# 				A_row_col_array, num_x_ocp.y, dzeta_dy_D_nonzero, 
	# 				dzeta_dy_slice)
	# 			G[dgamma_dy_slice] = G_dgamma_dy_lambda(dgamma_dy)
	# 			G[drho_dy_slice] = G_drho_dy_lambda(drho_dy, stretch, W)

	# 		if num_x_ocp.u:
	# 			G[dzeta_du_slice] = G_dzeta_du_lambda(dzeta_du, stretch, A, 
	# 				A_row_col_array, num_x_ocp.y, num_x_ocp.u)
	# 			G[dgamma_du_slice] = G_dgamma_du_lambda(dgamma_du)
	# 			G[drho_du_slice] = G_drho_du_lambda(drho_du, stretch, W)

	# 		if num_x_ocp.q:
	# 			G[drho_dq_slice] = 1

	# 		if num_x_ocp.t:
	# 			G[dzeta_dt_slice] = G_dzeta_dt_lambda(dy, A)
	# 			G[dgamma_dt_slice] = G_dgamma_dt_lambda(p)
	# 			G[drho_dt_slice] = G_drho_dt_lambda(g, W)

	# 		if num_x_ocp.s:
	# 			G[dzeta_ds_slice] = G_dzeta_ds_lambda(dzeta_ds, stretch, A)
	# 			G[dgamma_ds_slice] = G_dgamma_ds_lambda(dgamma_ds)
	# 			G[drho_ds_slice] = G_drho_ds_lambda(drho_ds, stretch, W)

	# 		G[dbeta_dxb_slice] = db_dxb_lambda(*x_tuple_point, N)

	# 		return G

	# 	expr_graph = self._expression_graph

	# 	t_stretch_lambda = self._t_stretch_lambda
	# 	dstretch_dt = None

	# 	c_continuous_lambda = self._c_continuous_lambda

	# 	dc_dx_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.dc_dx,
	# 		expression_nodes=expr_graph.dc_dx_nodes,
	# 		precomputable_nodes=expr_graph.dc_dx_precomputable,
	# 		dependent_tiers=expr_graph.dc_dx_dependent_tiers,
	# 		parameters=self._x_vars, 
	# 		return_dims=3, 
	# 		N_arg=True, 
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	dc_dx_slice = pu.dcdxInfo(
	# 		zeta_y=(self._c_defect_slice, self._y_slice),
	# 		zeta_u=(self._c_defect_slice, self._u_slice),
	# 		zeta_s=(self._c_defect_slice, self._s_slice),
	# 		gamma_y=(self._c_path_slice, self._y_slice),
	# 		gamma_u=(self._c_path_slice, self._u_slice),
	# 		gamma_s=(self._c_path_slice, self._s_slice),
	# 		rho_y=(self._c_integral_slice, self._y_slice),
	# 		rho_u=(self._c_integral_slice, self._u_slice),
	# 		rho_s=(self._c_integral_slice, self._s_slice),
	# 		)

	# 	dc_dx_shape = pu.dcdxInfo(
	# 		zeta_y=(self.number_state_equations*self.number_state_variables),
	# 		zeta_u=(self.number_state_equations*self.number_control_variables),
	# 		zeta_s=(self.number_state_equations*self.number_parameter_variables),
	# 		gamma_y=(self.number_path_constraints*self.number_state_variables),
	# 		gamma_u=(self.number_path_constraints*self.number_control_variables),
	# 		gamma_s=(self.number_path_constraints*self.number_parameter_variables),
	# 		rho_y=(self.number_integrand_functions*self.number_state_variables),
	# 		rho_u=(self.number_integrand_functions*self.number_control_variables),
	# 		rho_s=(self.number_integrand_functions*self.number_parameter_variables),
	# 		)

	# 	db_dxb_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.db_dxb,
	# 		precomputable_nodes=expr_graph.db_dxb_precomputable,
	# 		dependent_tiers=expr_graph.db_dxb_dependent_tiers,
	# 		parameters=self._x_b_vars, 
	# 		return_dims=1, 
	# 		N_arg=True, 
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	self._G_lambda = jacobian_lambda
	# 	print('Jacobian function compiled.')

	# def _compile_hessian_lagrangian(self):

	# 	def H_point_lambda(ddL_dxbdxb):
	# 		vals = np.tril(ddL_dxbdxb).flatten()
	# 		H = vals[vals != 0]
	# 		return H

	# 	def H_defect_lambda(ddL_dxdx, A, sum_flag):
	# 		H = np.array([])
	# 		for matrix, flag in zip(ddL_dxdx, sum_flag):
	# 			if flag:
	# 				vals = A.multiply(matrix).sum().flatten()
	# 			else:
	# 				vals = np.array(A.multiply(matrix).sum(axis=0)).flatten()
	# 			H = np.concatenate([H, vals])
	# 		return H

	# 	def H_path_lambda(ddL_dxdx):
	# 		H = np.array([])
	# 		for matrix in ddL_dxdx:
	# 			vals = np.diag(matrix).flatten()
	# 			H = np.concatenate([H, vals])
	# 		return H

	# 	def H_integral_lambda(ddL_dxdx, W, sum_flag):
	# 		H = np.array([])
	# 		for row, flag in zip(ddL_dxdx, sum_flag):
	# 			if flag:
	# 				vals = np.array([-np.dot(W, row)])
	# 			else:
	# 				vals = -np.multiply(W, row).flatten()
	# 			H = np.concatenate([H, vals])
	# 		return H

	# 	def H_lambda(x_tuple, x_tuple_point, sigma, zeta_lagrange, 
	# 		gamma_lagrange, rho_lagrange, beta_lagrange, N, num_nonzero, 
	# 		objective_index, defect_index, path_index, integral_index,
	# 		endpoint_index, A, W, defect_sum_flag, integral_sum_flag):

	# 		ddL_objective_dxbdxb = ddL_objective_dxbdxb_lambda(*x_tuple_point, 
	# 			sigma, N)
	# 		ddL_defect_dxdx = ddL_defect_dxdx_lambda(*x_tuple, 
	# 			*zeta_lagrange, N)
	# 		ddL_path_dxdx = ddL_path_dxdx_lambda(*x_tuple, *gamma_lagrange, N)
	# 		ddL_integral_dxdx = ddL_integral_dxdx_lambda(*x_tuple,
	# 			*rho_lagrange, N)
	# 		ddL_endpoint_dxbdxb = ddL_endpoint_dxbdxb_lambda(*x_tuple_point, 
	# 			*beta_lagrange, N)

	# 		H = np.zeros(num_nonzero)

	# 		H[objective_index] += H_point_lambda(ddL_objective_dxbdxb)
	# 		H[defect_index] += H_defect_lambda(ddL_defect_dxdx, A, 
	# 			defect_sum_flag)
	# 		if self.number_path_constraints and path_index:
	# 			H[path_index] += H_path_lambda(ddL_path_dxdx)
	# 		H[integral_index] += H_integral_lambda(ddL_integral_dxdx, W, 
	# 			integral_sum_flag)
	# 		H[endpoint_index] += H_point_lambda(ddL_endpoint_dxbdxb)

	# 		return H

	# 	expr_graph = self._expression_graph
	# 	L_sigma = expr_graph.lagrange_syms[0]
	# 	L_syms = expr_graph.lagrange_syms[1:]

	# 	ddL_objective_dxbdxb_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.ddL_J_dxbdxb,
	# 		expression_nodes=expr_graph.ddL_J_dxbdxb_nodes,
	# 		precomputable_nodes=expr_graph.ddL_J_dxbdxb_precomputable,
	# 		dependent_tiers=expr_graph.ddL_J_dxbdxb_dependent_tiers,
	# 		parameters=self._x_b_vars,
	# 		lagrange_parameters=L_sigma,
	# 		return_dims=2,
	# 		endpoint=True,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	ddL_defect_dxdx_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.ddL_zeta_dxdx,
	# 		expression_nodes=expr_graph.ddL_zeta_dxdx_nodes,
	# 		precomputable_nodes=expr_graph.ddL_zeta_dxdx_precomputable,
	# 		dependent_tiers=expr_graph.ddL_zeta_dxdx_dependent_tiers,
	# 		parameters=self._x_vars,
	# 		lagrange_parameters=L_syms[self._c_defect_slice],
	# 		return_dims=2,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	ddL_path_dxdx_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.ddL_gamma_dxdx,
	# 		expression_nodes=expr_graph.ddL_gamma_dxdx_nodes,
	# 		precomputable_nodes=expr_graph.ddL_gamma_dxdx_precomputable,
	# 		dependent_tiers=expr_graph.ddL_gamma_dxdx_dependent_tiers,
	# 		parameters=self._x_vars,
	# 		lagrange_parameters=L_syms[self._c_path_slice],
	# 		return_dims=2,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	ddL_integral_dxdx_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.ddL_rho_dxdx,
	# 		expression_nodes=expr_graph.ddL_rho_dxdx_nodes,
	# 		precomputable_nodes=expr_graph.ddL_rho_dxdx_precomputable,
	# 		dependent_tiers=expr_graph.ddL_rho_dxdx_dependent_tiers,
	# 		parameters=self._x_vars,
	# 		lagrange_parameters=L_syms[self._c_integral_slice],
	# 		return_dims=2,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	ddL_endpoint_dxbdxb_lambda = numbafy(
	# 		expression_graph=expr_graph,
	# 		expression=expr_graph.ddL_b_dxbdxb,
	# 		expression_nodes=expr_graph.ddL_b_dxbdxb_nodes,
	# 		precomputable_nodes=expr_graph.ddL_b_dxbdxb_precomputable,
	# 		dependent_tiers=expr_graph.ddL_b_dxbdxb_dependent_tiers,
	# 		parameters=self._x_b_vars,
	# 		lagrange_parameters=L_syms[self._c_endpoint_slice],
	# 		return_dims=2,
	# 		endpoint=True,
	# 		N_arg=True,
	# 		ocp_num_vars=self._num_vars_tuple,
	# 		)

	# 	ocp_defect_slice = self._c_defect_slice
	# 	ocp_path_slice = self._c_path_slice
	# 	ocp_integral_slice = self._c_integral_slice
	# 	ocp_endpoint_slice = self._c_endpoint_slice

	# 	self._H_lambda = H_lambda
	# 	print('Hessian function compiled.')

	# 	# print(self._x_vars)
	# 	# print(expr_graph.ddL_gamma_dxdx)
	# 	# print('\n\n\n')
	# 	# raise NotImplementedError

