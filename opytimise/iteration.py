import collections

import ipopt
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import sympy as sym

from opytimise.guess import Guess
from opytimise.utils import numbafy

class Iteration:

	def __init__(self, optimal_control_problem=None, iteration_number=None, *, mesh=None, guess=None):

		# Optimal control problem
		self._ocp = optimal_control_problem

		# Iteration number
		self._iteration_number = int(iteration_number)

		# Mesh
		self._mesh = mesh

		# Guess
		self._guess = guess
		
		# Result
		self._result = None

	@property
	def iteration_number(self):
		return self._iteration_number

	@property
	def mesh(self):
		return self._mesh
	
	@property
	def guess(self):
		return self._guess
	
	@property
	def result(self):
		return self._result

	@property
	def x(self):
		return self._x

	@property
	def _y(self):
		return self._x[self._y_slice, :].reshape(self._ocp._num_y_vars, self._mesh._N)
		
	@property
	def _u(self):
		return self._x[self._u_slice, :].reshape(self._ocp._num_u_vars, self._mesh._N)
	
	@property
	def _q(self):
		return self._x[self._q_slice, :]

	@property
	def _t(self):
		return self._x[self._t_slice, :]

	@property
	def _s(self):
		return self._x[self._s_slice, :]

	def _initialise_iteration(self, prev_guess):

		def interpolate_to_new_mesh(num_vars, prev):
			new_guess = np.empty((num_vars, self._mesh._N))
			for index, row in enumerate(prev):
				interp_func = interpolate.interp1d(prev_guess._t, row)
				new_guess[index, :] = interp_func(self._mesh._t)
			return new_guess

		# Mesh
		t0 = prev_guess._t[0]
		tF = prev_guess._t[-1]
		self._mesh._generate_mesh(t0, tF)

		# Guess
		if self._ocp._num_y_vars:
			new_y = interpolate_to_new_mesh(self._ocp._num_y_vars, prev_guess._y)
		else:
			new_y = np.array([])

		if self._ocp._num_u_vars:
			new_u = interpolate_to_new_mesh(self._ocp._num_u_vars, prev_guess._u)
		else:
			new_u = np.array([])

		self._guess = Guess(
			optimal_control_problem=self._ocp,
			time=self._mesh._t,
			state=new_y,
			control=new_u,
			integral=prev_guess._q,
			parameter=prev_guess._s)

		# Variables
		self._num_y = self._ocp._num_y_vars * self._mesh._N
		self._num_u = self._ocp._num_u_vars * self._mesh._N
		self._num_q = self._ocp._num_q_vars
		self._num_t = self._ocp._num_t_vars
		self._num_s = self._ocp._num_s_vars
		self._num_x = self._num_y + self._num_u + self._num_q + self._num_t + self._num_s

		self._y_slice = slice(0, self._num_y)
		self._u_slice = slice(self._y_slice.stop, self._y_slice.stop + self._num_u)
		self._q_slice = slice(self._u_slice.stop, self._u_slice.stop + self._num_q)
		self._t_slice = slice(self._q_slice.stop, self._q_slice.stop + self._num_t)
		self._s_slice = slice(self._t_slice.stop, self._num_x)

		self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)

		# Constraints
		self._num_c_defect = self._ocp._num_y_vars * self._mesh._num_c_boundary_per_y
		self._num_c_path = self._ocp._num_c_cons
		self._num_c_integral = self._ocp._num_q_vars
		self._num_c_boundary = self._ocp._num_b_cons
		self._num_c = self._num_c_defect + self._num_c_path + self._num_c_integral + self._num_c_boundary

		self._c_defect_slice = slice(0, self._num_c_defect)
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_path)
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_c_integral)
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._num_c)

		# Jacobian
		G_nonzero_row = []
		G_nonzero_col = []
		dzeta_dy_D_nonzero = []

		# A_row_col_ind = list(zip(*np.nonzero(self._mesh._A_matrix)))
		# D_row_col_ind = list(zip(*np.nonzero(self._mesh._D_matrix)))
		# A_and_D_row_col_ind = sorted(set.union(set(A_row_col_ind), set(D_row_col_ind)))
		# A_and_D_row_col_array = np.array(A_and_D_row_col_ind).T
		# A_ind_array = np.array([A_and_D_row_col_ind.index(ind) for ind in A_row_col_ind])
		# D_ind_array = np.array([A_and_D_row_col_ind.index(ind) for ind in D_row_col_ind])
		# num_A_nonzero = len(A_ind_array)
		# num_D_nonzero = len(D_ind_array)
		# num_A_and_D_nonzero = max(max(A_ind_array), max(D_ind_array)) + 1

		A_row_col_ind = list(zip(*np.nonzero(self._mesh._A_matrix)))
		D_row_col_ind = list(zip(*np.nonzero(self._mesh._D_matrix)))
		A_row_col_array = np.array(A_row_col_ind).T
		A_ind_array = np.array([i for i, _ in enumerate(A_row_col_ind)])
		D_ind_array = np.array([A_row_col_ind.index(ind) for ind in D_row_col_ind])
		num_A_nonzero = A_ind_array.shape[0]
		num_D_nonzero = D_ind_array.shape[0]

		for i_c in range(self._ocp._num_y_vars):
			for i_y in range(self._ocp._num_y_vars):
				row_offset = i_c * self._mesh._num_c_boundary_per_y
				col_offset = i_y * self._mesh._N
				ind_offset = len(G_nonzero_row)
				G_nonzero_row.extend(list(A_row_col_array[0] + row_offset))
				G_nonzero_col.extend(list(A_row_col_array[1] + col_offset))
				if i_c == i_y:
					# dzeta_dy_A_nonzero.extend(list(A_ind_array + ind_offset))
					dzeta_dy_D_nonzero.extend(list(D_ind_array + ind_offset))
					# G_nonzero_row.extend(list(A_and_D_row_col_array[0] + row_offset))
					# G_nonzero_col.extend(list(A_and_D_row_col_array[1] + col_offset))
				# else:
					# dzeta_dy_A_nonzero.extend(list(np.array(range(num_A_nonzero)) + ind_offset))
					# G_nonzero_row.extend(list(A_and_D_row_col_array[0][A_ind_array] + row_offset))
					# G_nonzero_col.extend(list(A_and_D_row_col_array[1][A_ind_array] + col_offset))
				# print(self._mesh._num_c_boundary_per_y)
				# print(row_offset)
				# print(col_offset)
				# print(ind_offset)
				# print('')

		dzeta_dy_slice = slice(0, len(G_nonzero_row))

		# print(G_nonzero_row)
		# print(G_nonzero_col)
		# print(dzeta_dy_D_nonzero)
		# print('\n\n\n')
		# raise NotImplementedError

		for i_c in range(self._ocp._num_y_vars):
			for i_u in range(self._ocp._num_u_vars):
				row_offset = i_c * self._mesh._num_c_boundary_per_y
				col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
				G_nonzero_row.extend(list(A_row_col_array[0] + row_offset))
				G_nonzero_col.extend(list(A_row_col_array[1] + col_offset))
				# row_offset = i_c * self._mesh._num_c_boundary_per_y
				# col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
				# G_nonzero_row.extend(list(A_and_D_row_col_array[0][A_ind_array] + row_offset))
				# G_nonzero_col.extend(list(A_and_D_row_col_array[1][A_ind_array] + col_offset))
		dzeta_du_slice = slice(dzeta_dy_slice.stop, len(G_nonzero_row))
		dzeta_dt_slice = slice(dzeta_du_slice.stop, len(G_nonzero_row))
		dzeta_ds_slice = slice(dzeta_dt_slice.stop, len(G_nonzero_row))

		dgamma_dy_slice = slice(dzeta_ds_slice.stop, len(G_nonzero_row))
		dgamma_du_slice = slice(dgamma_dy_slice.stop, len(G_nonzero_row))
		dgamma_dt_slice = slice(dgamma_du_slice.stop, len(G_nonzero_row))
		dgamma_ds_slice = slice(dgamma_dt_slice.stop, len(G_nonzero_row))

		for i_c in range(self._ocp._num_q_vars):
			for i_y in range(self._ocp._num_y_vars):
				row_offset = (self._ocp._num_y_vars + self._ocp._num_c_cons) * self._mesh._num_c_boundary_per_y + i_c
				col_offset = i_y * self._mesh._N
				G_nonzero_row.extend(list(row_offset*np.ones(self._mesh._N, dtype=int)))
				G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		drho_dy_slice = slice(dgamma_ds_slice.stop, len(G_nonzero_row))

		for i_c in range(self._ocp._num_q_vars):
			for i_u in range(self._ocp._num_u_vars):
				row_offset = (self._ocp._num_y_vars + self._ocp._num_c_cons) * self._mesh._num_c_boundary_per_y + i_c
				col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
				G_nonzero_row.extend(list(row_offset*np.ones(self._mesh._N, dtype=int)))
				G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		drho_du_slice = slice(drho_dy_slice.stop, len(G_nonzero_row))

		for i_c in range(self._ocp._num_q_vars):
			row_offset = (self._ocp._num_y_vars + self._ocp._num_c_cons) * self._mesh._num_c_boundary_per_y + i_c
			col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars + i_c) * self._mesh._N
			G_nonzero_row.extend(list(row_offset*np.ones(self._ocp._num_q_vars, dtype=int)))
			G_nonzero_col.extend(list(range(col_offset, self._ocp._num_q_vars + col_offset)))
		drho_dq_slice = slice(drho_du_slice.stop, len(G_nonzero_row))
		drho_dt_slice = slice(drho_dq_slice.stop, len(G_nonzero_row))
		drho_ds_slice = slice(drho_dt_slice.stop, len(G_nonzero_row))

		for i_c in range(self._ocp._num_b_cons):
			for i_y in range(self._ocp._num_y_vars):
				row_offset = (self._ocp._num_y_vars + self._ocp._num_c_cons) * self._mesh._num_c_boundary_per_y + self._ocp._num_q_vars + i_c
				col_offset = i_y * self._mesh._N
				G_nonzero_row.extend([row_offset])
				G_nonzero_col.extend([col_offset])
		dbeta_dy0_slice = slice(drho_dt_slice.stop, len(G_nonzero_row))

		for i_c in range(self._ocp._num_b_cons):
			for i_y in range(self._ocp._num_y_vars):
				row_offset = (self._ocp._num_y_vars + self._ocp._num_c_cons) * self._mesh._num_c_boundary_per_y + self._ocp._num_q_vars + i_c
				col_offset = (i_y + 1) * self._mesh._N - 1
				G_nonzero_row.extend([row_offset])
				G_nonzero_col.extend([col_offset])
		dbeta_dyF_slice = slice(dbeta_dy0_slice.stop, len(G_nonzero_row))
		dbeta_dq_slice = slice(dbeta_dyF_slice.stop, len(G_nonzero_row))
		dbeta_dt_slice = slice(dbeta_dq_slice.stop, len(G_nonzero_row))
		dbeta_ds_slice = slice(dbeta_dt_slice.stop, len(G_nonzero_row))

		self._G_nonzero_row = G_nonzero_row
		self._G_nonzero_col = G_nonzero_col
		self._num_G_nonzero = len(G_nonzero_row)

		# Lambda to prepare x from IPOPT for numba funcs
		def reshape_x(x):
			num_yu = self._ocp._num_y_vars + self._ocp._num_u_vars
			yu_qts_split = self._q_slice.start
			x_tuple = self._ocp._x_reshape_lambda(x, num_yu, yu_qts_split)
			return x_tuple

		def reshape_x_point(x):
			num_yu = self._ocp._num_y_vars + self._ocp._num_u_vars
			yu_qts_split = self._q_slice.start
			x_tuple = self._ocp._x_reshape_lambda_point(x, num_yu, yu_qts_split)
			return x_tuple

		# Generate objective function lambda
		def objective(x):
			x_tuple = reshape_x(x)
			J = self._ocp._J_lambda(*x_tuple)
			return J

		self._objective_lambda = objective

		# Generate objective function gradient lambda
		def gradient(x):
			x_tuple = reshape_x(x)
			g = self._ocp._g_lambda(x_tuple, self._num_x, self._yu_slice, self._qts_slice)
			return g

		self._gradient_lambda = gradient

		# Generate constraint lambdas
		def constraint(x):
			x_tuple = reshape_x(x)
			x_tuple_point = reshape_x_point(x)
			c = self._ocp._c_lambda(x_tuple, x_tuple_point, self._ocp._y_slice, self._ocp._q_slice, self._num_c, self._c_defect_slice, self._c_path_slice, self._c_integral_slice, self._c_boundary_slice, self._mesh._A_matrix, self._mesh._D_matrix, self._mesh._W_matrix)
			return c

		self._constraint_lambda = constraint

		def jacobian(x):
			x_tuple = reshape_x(x)
			x_tuple_point = reshape_x_point(x)
			G = self._ocp._G_lambda(x_tuple, x_tuple_point, self._num_G_nonzero, self._mesh._N, self._mesh._A_matrix, self._mesh._D_matrix, self._mesh._W_matrix, A_row_col_array, dzeta_dy_D_nonzero, dzeta_dy_slice, dzeta_du_slice, dzeta_dt_slice, dzeta_ds_slice, dgamma_dy_slice, dgamma_du_slice, dgamma_dt_slice, dgamma_ds_slice, drho_dy_slice, drho_du_slice, drho_dq_slice, drho_dt_slice, drho_ds_slice, dbeta_dy0_slice, dbeta_dyF_slice, dbeta_dq_slice, dbeta_dt_slice, dbeta_ds_slice)
			return G

		self._jacobian_lambda = jacobian

		def jacobian_structure():
			return (self._G_nonzero_row, self._G_nonzero_col)

		self._jacobian_structure_lambda = jacobian_structure

		# print('A:')
		# print(self._mesh._A_matrix, '\n')
		# print('D:')
		# print(self._mesh._D_matrix, '\n')
		# # print(self._ocp._y_eqns, '\n')
		# # print(self._ocp._x_vars, '\n')
		# # print(self._ocp._y_eqns.jacobian(self._ocp._x_vars), '\n')
		# # print(self._ocp._dc_dx[self._ocp._c_defect_slice, self._ocp._yu_slice], '\n')

		# print(G_nonzero_row[:], '\n')
		# print(G_nonzero_col[:], '\n')
		# # print(len(G_nonzero_row[dbeta_dq_slice]), '\n')
		# # print(dzeta_du_nonzero, '\n')
		# # print(dzeta_dy_A_nonzero, '\n')
		# # print(dzeta_dy_D_nonzero, '\n')
		# # print(self._mesh._A_matrix, '\n')
		# # print(self._mesh._D_matrix, '\n')

		for i in range(1):
			# x_data = np.zeros(self._num_x)
			x_data = np.ones(self._num_x)
			# x_data = np.array(range(self._num_x))
			# x_data = np.random.rand(self._num_x, 1)
			print('x:')
			print(x_data, '\n')
			result_J = self._objective_lambda(x_data)
			result_g = self._gradient_lambda(x_data)
			result_c = self._constraint_lambda(x_data)
			result_G = self._jacobian_lambda(x_data)
			print('J:')
			print(result_J, '\n')
			print('g:')
			print(result_g, '\n')
			print('c:')
			print(result_c, '\n')
			print('G:')
			print(result_G, '\n')

		# Generate bounds
		self._x_bnd_l, self._x_bnd_u = self._generate_x_bounds()
		self._c_bnd_l, self._c_bnd_u = self._generate_c_bounds()

		

		# print('\n\n\n\n')
		# raise NotImplementedError


		# Initialise the NLP problem
		self._initialise_nlp()


















	# 	# Generate state derivatives lambdas
	# 	def state_derivatives(x):
	# 		return self._dy_lambda(*self._reshape_x_for_numba(x))

	# 	self._dy_lambda = numbafy(expression=self._ocp._y_eqns, parameters=self._ocp._x_vars, constants=self._ocp._aux_data._consts_vals, num_N=self._mesh._N, ocp_vars=self._ocp._x_vars, ocp_yu_slice=self._ocp._yu_slice, ocp_qts_slice=self._ocp._qts_slice)
	# 	self._dy_numba = state_derivatives

	# 	# Generate integrand lambdas
	# 	def integrands(x):
	# 		return self._g_lambda(*self._reshape_x_for_numba(x))

	# 	self._g_lambda = numbafy(expression=self._ocp._q_funcs, parameters=self._ocp._x_vars, constants=self._ocp._aux_data._consts_vals, num_N=self._mesh._N)
	# 	self._g_numba = integrands

	# 	# Generate objective function and lambdas
	# 	def objective(x):
	# 		return J_lambda(*self._reshape_x_for_numba(x))

	# 	J_lambda = numbafy(expression=self._ocp._J, parameters=self._ocp._x_vars, constants=self._ocp._aux_data._consts_vals, return_non_array=True)
	# 	self._J_numba = objective

	# 	for i in range(100):
	# 		x_data = np.random.rand(self._num_x, 1)
	# 		x_data = self._reshape_x_for_numba(x_data)
	# 		result = self._J_numba(x_data)

	# 	print('\n\n\n\n')
	# 	raise NotImplementedError

	# 	# Generate objective function first derivative and lambdas
	# 	def gradient(x):
	# 		x = self._reshape_x_for_numba(x)
	# 		g = np.empty(self._num_x)
	# 		g[self._yu_slice] = g_yu_lambda(*x[self._ocp._yu_slice])
	# 		g[self._qts_slice] = g_qts_lambda(*x[self._ocp._qts_slice])
	# 		return g

	# 	g_yu_lambda = numbafy(expression=self._ocp._dJ_dx[self._ocp._yu_slice], parameters=self._ocp._x_vars[self._ocp._yu_slice], constants=self._ocp._aux_data._consts_vals, num_N=self._mesh._N, return_flat=True)
	# 	g_qts_lambda = numbafy(expression=self._ocp._dJ_dx[self._ocp._qts_slice], parameters=self._ocp._x_vars[self._ocp._qts_slice], constants=self._ocp._aux_data._consts_vals)
	# 	self._g_numba = gradient

	# 	for i in range(100):
	# 		x_data = np.random.rand(self._num_x, 1)
	# 		x_data = self._reshape_x_for_numba(x_data)
	# 		for item in x_data[self._ocp._yu_slice]:
	# 			print(item, type(item), item.dtype, '\n')
	# 		print(*x_data[self._ocp._yu_slice])
	# 		print(*x_data[self._ocp._qts_slice])
	# 		# print('\n\n')
	# 		result = self._g_numba(x_data)
	# 		# print(x_data[-1])
	# 		# print(result)
	# 		# print('')

	# 	print('\n\n\n\n')
	# 	raise NotImplementedError

	# 	# self._g, self._gradient_lambda = self._generate_objective_function_gradient()

	# 	# Generate_constraints lambdas
	# 	def constraint(x):
	# 		x = self._reshape_x_for_numba(x)
	# 		c = np.empty(self._num_c)
	# 		c[self._c_defect_slice] = c_defect_lambda(*x)
	# 		c[self._c_path_slice] = c_path_lambda(*x)
	# 		c[self._c_integral_slice] = c_integral_lambda(*x)
	# 		c[self._c_boundary_slice] = c_boundary_lambda(*x)
	# 		return c

	# 	c_defect_lambda, c_path_lambda, c_integral_lambda, c_boundary_lambda = self._generate_constraints()
	# 	self._c_numba = constraints

	# 	# self._c, self._num_c, self._constraints_lambda = self._generate_constraints(x_subs)

	# 	print('\n\n\n')
	# 	raise NotImplementedError

	# 	# Generate constraints first derivative, sparsity and lambdas
	# 	self._G, self._G_row, self._G_col, self._num_G_nonzeros, self._jacobian_lambda, self._jacobian_structure_lambda = self._generate_constraints_jacobian()

		

	# # def _generate_objective_function(self):

	# # 	def objective(x):
	# # 		return J_lambda(*x)

	# # 	exec(numbafy(expression=self._ocp._J_subbed, parameters=self._x, new_function_name='J_lambda', use_cse=True))

	# # 	return self._ocp._J_subbed, objective

	# # def _generate_objective_function_gradient(self):

	# # 	def gradient(x):
	# # 		return g_lambda(*x)

	# # 	for deriv in self._ocp._dJ_dx[self._ocp._yu_slice]:
	# # 		if deriv != 0:
	# # 			msg = 'Objective function derivative with respect to state on control is non-zero.'
	# # 			raise ValueError(msg)

	# # 	dJ_dy = sym.Matrix.zeros(self._num_y, 1)
	# # 	dJ_du = sym.Matrix.zeros(self._num_u, 1)
	# # 	g = sym.Matrix([dJ_dy, dJ_du, self._ocp._dJ_dx[self._ocp._qts_slice, :]])
		
	# # 	exec(numbafy(expression=g, parameters=self._x, new_function_name='g_lambda', use_cse=True))

	# # 	return g, gradient

	# def _generate_constraints(self):

	# 		# Generate defect constraints
	# 	c_defect_lambda = self._generate_defect_constraints()

	# 	# Generate path constraints
	# 	c_path_lambda = self._generate_path_constraints()
		
	# 	# Genereate integral constraints
	# 	c_integral_lambda = self._generate_integral_constraints()

	# 	# Genereate boundary constraints
	# 	c_boundary = self._generate_boundary_constraints(x_subs)

	# 	# Constraints vector
	# 	# c = sym.Matrix([c_defect, c_path, c_integral, c_boundary])
	# 	self._num_c = self._num_c_defect + self._num_c_path + self._num_c_integral + self._num_c_boundary

	# 	return c_defect_lambda, c_path_lambda, c_integral_lambda, c_boundary_lambda

	# # @profile
	# # def _generate_constraints(self, x_subs):

	# # 	def constraints(x):
	# # 		return c_lambda(*x)

	# # 	# Generate defect constraints
	# # 	c_defect = self._generate_defect_constraints()
	# # 	num_c_defect = c_defect.shape[0]
	# # 	self._c_defect_slice = slice(0, num_c_defect)

	# # 	# Generate path constraints
	# # 	c_path = self._generate_path_constraints()
	# # 	num_c_path = c_path.shape[0]
	# # 	self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + num_c_path)

	# # 	# Genereate integral constraints
	# # 	c_integral = self._generate_integral_constraints()
	# # 	num_c_integral = self._ocp._num_q_vars
	# # 	self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + num_c_integral)

	# # 	# Genereate boundary constraints
	# # 	c_boundary = self._generate_boundary_constraints(x_subs)
	# # 	num_c_boundary = self._ocp._num_b_cons
	# # 	self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + num_c_boundary)

	# # 	# Constraints vector
	# # 	c = sym.Matrix([c_defect, c_path, c_integral, c_boundary])
	# # 	num_c = num_c_defect + num_c_path + num_c_integral + num_c_boundary

	# # 	exec(numbafy(expression=c, parameters=self._x, new_function_name='c_lambda', use_cse=True))

	# # 	return c, num_c, constraints

	# # @profile
	# def _generate_defect_constraints(self):

	# 	@numba.jit
	# 	def c_defect_lambda(y, u, q, t, s):
	# 		dy = self._dy_numba(y, u, q, t, s)
	# 		return self._mesh._D_matrix*y - self._mesh._A_matrix*dy

	# 	# row_starts = self._mesh._mesh_index_boundaries[:-1]
	# 	# num_rows = self._mesh._mesh_index_boundaries[-1]
	# 	# col_starts = self._mesh._mesh_index_boundaries_offset[:-1]
	# 	# num_cols = self._mesh._mesh_index_boundaries_offset[-1] + 1
	# 	# matrix_dims = (num_rows, num_cols)

	# 	# D_matrix = np.zeros(matrix_dims)
	# 	# A_matrix = np.zeros(matrix_dims)
		
	# 	# # D_matrix = sym.Matrix.zeros(num_rows, num_cols)
	# 	# # A_matrix = sym.Matrix.zeros(num_rows, num_cols)

	# 	# for block_size, hK, row_start, col_start in zip(self._mesh._mesh_col_points, self._mesh._hK, row_starts, col_starts):
	# 	# 	col_slice = slice(col_start, col_start+block_size)
	# 	# 	row_slice = slice(row_start, row_start+block_size)
	# 	# 	A_matrix[row_slice, col_slice] = (hK/2)*np.eye(block_size)
	# 	# 	D_matrix[row_slice, col_slice] = D_matricies[block_size]
	# 	# 	# A_matrix[row_slice, col_slice] = (hK/2)*sym.Matrix.eye(block_size)
	# 	# 	# D_matrix[row_slice, col_slice] = D_matricies[block_size]

	# 	# # y = self._y.transpose()
	# 	# # dy = self._dy.transpose().reshape(*y.shape)

	# 	# # print(D_matrix)
	# 	# # print(A_matrix)

	# 	# # delta_matrix = D_matrix*y - A_matrix*dy
	# 	# # delta_c = delta_matrix.vec().subs(self._dy_subs_dict)

	# 	# self._defect_constraints_D_matrix = D_matrix
	# 	# self._defect_constraints_A_matrix = A_matrix

	# 	self._num_c_defect = self._mesh._A_matrix.shape[0]
	# 	self._c_defect_slice = slice(0, self._num_c_defect)
		
	# 	return c_defect_lambda

	# def _generate_path_constraints(self):

	# 	def c_path_lambda(y, u, q, t, s):
	# 		return np.empty(0)

	# 	self._num_c_path = 0
	# 	self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_path)

	# 	return c_path_lambda

	# 	# return sym.Matrix.zeros(0, 1)

	# def _generate_integral_constraints(self):

	# 	def c_integral_lambda(y, u, q, t, s):
	# 		g = self._g_numba(y, u, q, t, s)
	# 		return q - np.sum(np.dot(self._mesh._W_matrix, g), 0)

	# 	self._num_c_integral = self._ocp._num_q_vars
	# 	self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_c_integral)

	# 	return c_integral_lambda

	# 	# def integral_constraints(x):
	# 	# 	g = self._integrands_lambda(x)
	# 	# 	q_vars = x[self._q_index]
	# 	# 	rho_vector = q_vars - np.sum(np.dot(self._integral_constraints_W_matrix, g), 0)
	# 	# 	return rho_vector

	# 	# row_starts = self._mesh._mesh_index_boundaries[:-1]
	# 	# num_rows = self._mesh._mesh_index_boundaries[-1]
	# 	# col_starts = self._mesh._mesh_index_boundaries_offset[:-1]
	# 	# num_cols = self._mesh._mesh_index_boundaries_offset[-1] + 1
		
	# 	# W_matrix = np.zeros((num_rows, num_cols))

	# 	# for block_size, hK, row_start, col_start in zip(self._mesh._mesh_col_points, self._mesh._hK, row_starts, col_starts):

	# 	# 	col_slice = slice(col_start, col_start+block_size)
	# 	# 	row_slice = slice(row_start, row_start+block_size)

	# 	# 	W_matrix[row_slice, col_slice] = (hK/2)*np.diagflat(quadrature_weights[block_size])

	# 	# # MUST ADD IN (( t_0-t_f/2 )) TERM
	# 	# rho_c = (self._x[self._q_slice] - np.sum(np.dot(W_matrix, self._g_integrands), 0))
	# 	# v_func = np.vectorize(lambda g: g.subs(self._g_subs_dict))
	# 	# rho_c = v_func(rho_c)

	# 	# self._integral_constraints_W_matrix = W_matrix
	# 	# self._integral_constraints_lambda = integral_constraints

	# 	# print('\n\n\n')
	# 	# for item in delta_c:
	# 	# 	print(item, '\n')
	# 	# print('\n\n\n')
	# 	# raise NotImplementedError

	# 	# return sym.Matrix.zeros(self._ocp._num_q_vars, 1)#rho_c

	# def _generate_boundary_constraints(self, x_subs):

	# 	def c_boundary_lambda(y, u, q, t, s):
	# 		return 

	# 	self._num_c_boundary = self._ocp._num_b_cons
	# 	self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + num_c_boundary)

	# 	return _

	# 	# def boundary_constraints(x):
	# 	# 	b = b_lambda(*x)
	# 	# 	return np.array(b)

	# 	# x0_dict = dict(zip(self._ocp._y_t0, sym.Matrix(self._ocp._y_vars).subs(x_subs[0])))
	# 	# xF_dict = dict(zip(self._ocp._y_tF, sym.Matrix(self._ocp._y_vars).subs(x_subs[-1])))
	# 	# x_boundary_dict = {**x0_dict, **xF_dict}

	# 	# b_c = np.array([c.subs(x_boundary_dict) for c in self._ocp._b_cons])#[np.newaxis].transpose()

	# 	# b_lambda = sym.lambdify(self._x, np.array([c.subs(x_boundary_dict) for c in self._ocp._b_cons]), 'numpy')
	# 	# self._boundary_constraints_lambda = boundary_constraints

	# 	# return sym.Matrix.zeros(self._ocp._num_b_cons, 1)#b_c

	# def _generate_constraints_jacobian(self):

	# 	def jacobian(x):
	# 		G = G_lambda(*x)
	# 		return G

	# 	def jacobian_structure():
	# 		return G_structure_tuple

	# 	# Genereate defect constraints Jacobian
	# 	G_defect, G_defect_rows, G_defect_cols = self._generate_defect_constraints_jacobian()

	# 	# Generate path constraints Jacobian
	# 	G_path, G_path_rows, G_path_cols = self._generate_path_constraints_jacobian()

	# 	# Generate integral constraints Jacobian
	# 	G_integral, G_integral_rows, G_integral_cols = self._generate_integral_constraints_jacobian()

	# 	# Generate boundary constraints Jacobian
	# 	G_boundary, G_boundary_rows, G_boundary_cols = self._generate_boundary_constraints_jacobian()

	# 	# Combine Jacobian
	# 	G = np.hstack([G_defect, G_path, G_integral, G_boundary])
	# 	G_rows = np.hstack([G_defect_rows, G_path_rows, G_integral_rows, G_boundary_rows])
	# 	G_cols = np.hstack([G_defect_cols, G_path_cols, G_integral_cols, G_boundary_cols])
	# 	num_G_nonzeros = len(G)
	# 	G_lambda = sym.lambdify(self._x, G, 'numpy')

	# 	G_structure_tuple = (G_rows.astype(np.int8).tolist(), G_cols.astype(np.int8).tolist())

	# 	return G, G_rows, G_cols, num_G_nonzeros, jacobian, jacobian_structure

	# # @profile
	# def _generate_defect_constraints_jacobian(self):

	# 	c_defect_per_block = np.sum(self._mesh._mesh_col_points)
	# 	subs_dicts_list = []
	# 	for y_node, u_node in zip(self._y, self._u):
	# 		y_subs_dict = dict(zip(self._ocp._y_vars, y_node))
	# 		u_subs_dict = dict(zip(self._ocp._u_vars, u_node))
	# 		subs_dict = {**y_subs_dict, **u_subs_dict}
	# 		subs_dicts_list.append(subs_dict)

	# 	G = np.array([])
	# 	G_row = np.array([], dtype=np.int8)
	# 	G_col = np.array([], dtype=np.int8)

	# 	for i_c, c in enumerate(self._c[self._c_defect_slice]):
	# 		jacobian_row = np.array(c.diff(self._x))
	# 		nonzero_cols = np.flatnonzero(jacobian_row)
	# 		G = np.hstack([G, jacobian_row[nonzero_cols]])
	# 		G_row = np.hstack([G_row, i_c*np.ones(nonzero_cols.shape, dtype=np.int8)])
	# 		G_col = np.hstack([G_col, nonzero_cols])
	# 		# print(G)
	# 		# print(G_row)
	# 		# print(G_col)
	# 		# input()

	# 	return G, G_row, G_col

	# 	"""
	# 	for defect_index, deriv_row in enumerate(ddelta_dx):

	# 		# State variables
	# 		for state_index, (y_var, deriv) in enumerate(zip(self._ocp._y_vars, deriv_row[self._ocp._y_slice])):
	# 			deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])

	# 			print(self._defect_constraints_A_matrix.shape)
	# 			print(deriv_at_nodes.reshape(1, -1).shape)
	# 			input()

	# 			block = -deriv_at_nodes.reshape(1, -1).dot(self._defect_constraints_A_matrix)
	# 			if defect_index == state_index:
	# 				block += self._defect_constraints_D_matrix
				
	# 			print(block)
	# 			nonzero_elements = np.flatnonzero(block)
	# 			block_eqn = block.flatten()[nonzero_elements]
	# 			G = np.hstack([G, block_eqn])

	# 			block_row, block_col = np.nonzero(block)
	# 			block_row += defect_index*c_defect_per_block
	# 			G_row = np.hstack([G_row, block_row])
	# 			block_col += state_index*self._mesh._N
	# 			G_col = np.hstack([G_col, block_col])

	# 		# Control variables
	# 		for control_index, (u_var, deriv) in enumerate(zip(self._ocp._u_vars, deriv_row[self._ocp._u_slice])):
	# 			deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])
	# 			block = self._defect_constraints_A_matrix * deriv_at_nodes
				
	# 			block_eqn = block.flatten()[np.flatnonzero(block)]
	# 			G = np.hstack([G, block_eqn])

	# 			block_row, block_col = np.nonzero(block)
	# 			block_row += defect_index*c_defect_per_block
	# 			G_row = np.hstack([G_row, block_row])
	# 			block_col += self._u_slice.start + control_index*self._mesh._N
	# 			G_col = np.hstack([G_col, block_col])

	# 		# Integral variables
	# 		# All G entries are zero

	# 		# Time variables
	# 		# No current support for time variables, but will be dense

	# 		# Parameter variables
	# 		for parameter_index, (s_var, deriv) in enumerate(zip(self._ocp._s_vars, deriv_row[self._ocp._s_slice])):
	# 			deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])
	# 			block = -self._defect_constraints_A_matrix * deriv_at_nodes
	# 			block = np.reshape(block[np.nonzero(block)], (-1, 1))

	# 			block_eqn = block.flatten()[np.flatnonzero(block)]
	# 			G = np.hstack([G, block_eqn])

	# 			block_row, block_col = np.nonzero(block)
	# 			block_row += defect_index*c_defect_per_block
	# 			G_row = np.hstack([G_row, block_row])
	# 			block_col += self._s_slice.start + parameter_index
	# 			G_col = np.hstack([G_col, block_col])

	# 		ind = np.lexsort((G_row, G_col))
	# 		G = G[ind]
	# 		G_row = G_row[ind]
	# 		G_col = G_col[ind]

	# 	return G, G_row, G_col
	# 	"""

	# def _generate_path_constraints_jacobian(self):
	# 	return np.array([]), np.array([]), np.array([])

	# # @profile
	# def _generate_integral_constraints_jacobian(self):

	# 	G = np.array([])
	# 	G_row = np.array([], dtype=np.int8)
	# 	G_col = np.array([], dtype=np.int8)

	# 	# print(self._c[self._c_integral_slice].squeeze())
	# 	# raise NotImplementedError

	# 	for i_c, c in enumerate(self._c[self._c_integral_slice], self._q_slice.start):
	# 		jacobian_row = np.array(c.diff(self._x))
	# 		nonzero_cols = np.flatnonzero(jacobian_row)
	# 		G = np.hstack([G, jacobian_row[nonzero_cols]])
	# 		G_row = np.hstack([G_row, i_c*np.ones(nonzero_cols.shape, dtype=np.int8)])
	# 		G_col = np.hstack([G_col, nonzero_cols])

	# 	return G, G_row, G_col

	# def _generate_boundary_constraints_jacobian(self):

	# 	G = np.array([])
	# 	G_row = np.array([], dtype=np.int8)
	# 	G_col = np.array([], dtype=np.int8)

	# 	sym_subs_into_np_array = np.vectorize(lambda vec, subs_dict: vec.subs(subs_dict))

	# 	t0_subs_dict = dict(zip(self._ocp._y_t0, self._y[1, :]))
	# 	tF_subs_dict = dict(zip(self._ocp._y_tF, self._y[-1, :]))

	# 	subs_dict = {**t0_subs_dict, **tF_subs_dict}

	# 	dbdy0 = sym_subs_into_np_array(self._ocp._G_dbdy0, subs_dict)
	# 	dbdyF = sym_subs_into_np_array(self._ocp._G_dbdyF, subs_dict)

	# 	# y variables
	# 	row_indexes = np.array(range(self._c_boundary_slice.start, self._c_boundary_slice.stop))
	# 	for i_y, (col_0, col_F) in enumerate(zip(dbdy0.transpose(), dbdyF.transpose())):
	# 		nonzeros_0 = np.flatnonzero(col_0)
	# 		nonzeros_F = np.flatnonzero(col_F)

	# 		entries_0 = col_0[nonzeros_0]
	# 		entries_F = col_F[nonzeros_F]
	# 		row_indexes_0 = row_indexes[nonzeros_0]
	# 		row_indexes_F = row_indexes[nonzeros_F]
	# 		col_indexes_0 = i_y*self._mesh._N*np.ones_like(row_indexes_0)
	# 		col_indexes_F = (((i_y+1)*self._mesh._N)-1)*np.ones_like(row_indexes_F)

	# 		G = np.hstack([G, entries_0, entries_F])
	# 		G_row = np.hstack([G_row, row_indexes_0, row_indexes_F])
	# 		G_col = np.hstack([G_col, col_indexes_0, col_indexes_F])

	# 	# u variables
	# 	# Empty block

	# 	# q, t & s variables ()
	# 	qts_c_boundary_block = self._ocp._dc_dx[self._ocp._c_boundary_slice, self._ocp._qts_slice]
	# 	qts_c_boundary_block = sym_subs_into_np_array(qts_c_boundary_block, subs_dict)
	# 	nonzeros_row, nonzeros_col = np.nonzero(qts_c_boundary_block)
	# 	block_row_offset = self._c_boundary_slice.start
	# 	block_col_offset = self._q_slice.start
	# 	G = np.hstack([G, qts_c_boundary_block[nonzeros_row, nonzeros_col]])
	# 	G_row = np.hstack([G_row, nonzeros_row + block_row_offset])
	# 	G_col = np.hstack([G_col, nonzeros_row + block_col_offset])

	# 	ind = np.lexsort((G_col, G_row))
	# 	G = G[ind]
	# 	G_row = G_row[ind]
	# 	G_col = G_col[ind]

	# 	return G, G_row, G_col

	def _generate_x_bounds(self):

		bnd_l = np.empty((self._num_x, ))
		bnd_u = np.empty((self._num_x, ))

		# y bounds
		bnd_l[self._y_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._y_l.reshape(1, -1)).flatten('F').squeeze()
		bnd_u[self._y_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._y_u.reshape(1, -1)).flatten('F').squeeze()

		# u bounds
		bnd_l[self._u_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._u_l.reshape(1, -1)).flatten('F').squeeze()
		bnd_u[self._u_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._u_u.reshape(1, -1)).flatten('F').squeeze()

		# q bounds
		bnd_l[self._q_slice] = self._ocp._bounds._q_l
		bnd_u[self._q_slice] = self._ocp._bounds._q_u

		# t bounds
		if self._ocp._num_t_vars:
			print(self._ocp._t_vars)
			raise NotImplementedError

		# s bounds
		bnd_l[self._s_slice] = self._ocp._bounds._s_l
		bnd_u[self._s_slice] = self._ocp._bounds._s_u

		return bnd_l, bnd_u

	def _generate_c_bounds(self):

		bnd_l = np.zeros((self._num_c, ))
		bnd_u = np.zeros((self._num_c, ))

		# Path constraints bounds
		if self._ocp._num_c_cons:
			print(self._ocp._c_cons)
			raise NotImplementedError

		# Boundary constrants bounds
		bnd_l[self._c_boundary_slice] = self._ocp._bounds._b_l
		bnd_u[self._c_boundary_slice] = self._ocp._bounds._b_u

		return bnd_l, bnd_u

	def _initialise_nlp(self):

		if self._ocp._settings._nlp_solver == 'ipopt':

			self._ipopt_problem = IPOPTProblem(
				self._objective_lambda, 
				self._gradient_lambda, 
				self._constraint_lambda, 
				self._jacobian_lambda, 
				self._jacobian_structure_lambda)

			self._nlp_problem = ipopt.problem(
				n=self._num_x,
				m=self._num_c,
				problem_obj=self._ipopt_problem,
				lb=self._x_bnd_l,
				ub=self._x_bnd_u,
				cl=self._c_bnd_l,
				cu=self._c_bnd_u)

		else:
			raise NotImplementedError

	def _solve(self):
		nlp_solution, nlp_solution_info = self._nlp_problem.solve(self._guess._x)
		self._solution = Solution(self, nlp_solution, nlp_solution_info)
		
		# solution = np.array(solution)
		# y = solution[self._y_slice].reshape(self._ocp._num_y_vars, -1)
		# u = solution[self._u_slice].reshape(self._ocp._num_u_vars, -1)
		# q = solution[self._q_slice]
		# print(y)

		# plt.plot(self._mesh._t, y[0, :], 'b', self._mesh._t, u[0, :], 'r')
		# plt.show()
		# print(info)
		# pass

	def _calculate_discretisation_mesh_error(self):
		pass

	def _refine_new_mesh(self):
		pass


class Solution:

	def __init__(self, iteration, nlp_solution, nlp_solution_info):
		self._it = iteration
		self._ocp = iteration._ocp
		self._nlp_solution = np.array(nlp_solution)
		self._nlp_solution_info = nlp_solution_info
		if self._ocp._settings._nlp_solver == 'ipopt':
			self._process_solution = self._process_ipopt_solution
		self._process_solution()

	@property
	def state(self):
		return self._y

	@property
	def control(self):
		return self._u
	
	@property
	def integral(self):
		return self._q

	@property
	def time(self):
		return self._t
	
	@property
	def parameter(self):
		return self._s

	def _process_ipopt_solution(self):
		self._y = self._nlp_solution[self._it._y_slice].reshape(self._ocp._num_y_vars, -1)
		self._u = self._nlp_solution[self._it._u_slice].reshape(self._ocp._num_u_vars, -1)
		self._q = self._nlp_solution[self._it._q_slice]
		self._t = self._it._mesh._t
		self._s = self._nlp_solution[self._it._s_slice]

	def _interpolate_solution(self):
		for mesh_sec in self._it._mesh._

		print(self.state)

	def _process_snopt_solution(self, solution):
		raise NotImplementedError


class IPOPTProblem:

	def __init__(self, J, g, c, G, G_struct):
		self.objective = J
		self.gradient = g
		self.constraints = c
		self.jacobian = G
		self.jacobianstructure = G_struct





