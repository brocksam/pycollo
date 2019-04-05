import collections

import ipopt
import numpy as np
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import sympy as sym

from opytimise.guess import Guess
from opytimise.utils import (numbafy, quadrature_points, quadrature_weights, D_matricies)

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
		return self._x[self._y_slice].reshape((self._ocp._num_y_vars, -1)).transpose()
		
	@property
	def _u(self):
		return self._x[self._u_slice].reshape((self._ocp._num_u_vars, -1)).transpose()
	
	@property
	def _q(self):
		return self._x[self._q_slice]

	@property
	def _t(self):
		return self._x[self._t_slice]

	@property
	def _s(self):
		return self._x[self._s_slice]

	# @property
	# def a(self):
	# 	return self._a

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

		x = np.empty(self._num_x, dtype=object)
		dy = np.empty(self._num_y, dtype=object)
		g = np.empty(self._num_q * self._mesh._N, dtype=object)
		x_subs = np.empty(self._mesh._N, dtype=object)
		for i_h, _ in enumerate(self._mesh._t):
			subs_dict = {}
			for i_y, y in enumerate(self._ocp._y_vars):
				index = i_y*self._mesh._N + i_h
				symbol = sym.symbols("y_{1}_N{0}".format(i_h, i_y))
				x[index] = symbol
				dy[index] = sym.symbols("dy_{1}_N{0}".format(i_h, i_y))
				subs_dict.update({y: symbol})
			for i_u, u in enumerate(self._ocp._u_vars):
				index = self._u_slice.start + i_u*self._mesh._N + i_h
				symbol = sym.symbols("u_{1}_N{0}".format(i_h, i_u))
				x[index] = symbol
				subs_dict.update({u: symbol})
			for i_q, q in enumerate(self._ocp._q_vars):
				index = i_q*self._mesh._N + i_h
				g[index] = sym.symbols("g_{1}_N{0}".format(i_h, i_q))
			x_subs[i_h] = subs_dict

		x[self._q_slice] = self._ocp._q_vars
		x[self._t_slice] = self._ocp._t_vars
		x[self._s_slice] = self._ocp._s_vars

		self._x = x
		self._dy = dy.reshape((self._ocp._num_y_vars, -1)).transpose()
		self._g_integrands = g.reshape((self._ocp._num_q_vars, -1)).transpose()

		# Generate state derivatives lambdas
		def state_derivatives(x):
			dy = np.array(dy_lambda(*x))
			return np.reshape(dy, (self._mesh._N, -1), order='F')

		self._dy_explicit = np.empty(self._num_y, dtype=object)

		for i_col, dy_eqn in enumerate(self._ocp._y_eqns_subbed):
			for i_row, (y, u) in enumerate(zip(self._y, self._u)):
				x = np.hstack([y, u, self._q, self._t, self._s])
				index = i_col*self._mesh._N + i_row
				subs_dict = dict(zip(self._ocp._x_vars, x))
				self._dy_explicit[index] = dy_eqn.subs(subs_dict)

		self._dy_subs_dict = dict(zip(self._dy.flatten('F'), self._dy_explicit))

		dy_lambda = sym.lambdify(self._x, self._dy_explicit, 'numpy')
		self._state_derivatives_lambda = state_derivatives

		# Generate integrand lambdas
		def integrands(x):
			g = np.array(g_integrand_lambda(*x))
			return np.reshape(g, (self._mesh._N, -1), order='F')

		self._g_explicit = np.empty(self._num_q * self._mesh._N, dtype=object)

		for i_col, q_func in enumerate(self._ocp._q_funcs_subbed):
			for i_row, (y, u) in enumerate(zip(self._y, self._u)):
				x = np.hstack([y, u, self._q, self._t, self._s])
				index = i_col*self._mesh._N + i_row
				subs_dict = dict(zip(self._ocp._x_vars, x))
				self._g_explicit[index] = q_func.subs(subs_dict)

		self._g_subs_dict = dict(zip(self._g_integrands.flatten('F'), self._g_explicit))

		g_integrand_lambda = sym.lambdify(self._x, self._g_explicit)
		self._integrands_lambda = integrands

		# Generate objective function and lambdas
		self._J, self._objective_lambda = self._generate_objective_function()

		# Generate objective function first derivative and lambdas
		self._g, self._gradient_lambda = self._generate_objective_function_gradient()

		# Generate constraints and lambdas
		# def constraints(x):
		# 	c = np.empty(self._num_c)
		# 	c[self._c_defect_slice] = self._defect_constraints_lambda(x)
		# 	c[self._c_path_slice] = self._path_constraints_lambda(x)
		# 	c[self._c_integral_slice] = self._integral_constraints_lambda(x)
		# 	c[self._c_boundary_slice] = self._boundary_constraints_lambda(x)
		# 	return c

		self._c, self._num_c, self._constraints_lambda = self._generate_constraints(x_subs)
		# self._constraints_lambda = constraints

		# Generate constraints first derivative, sparsity and lambdas
		self._G, self._G_row, self._G_col, self._num_G_nonzeros, self._jacobian_lambda, self._jacobian_structure_lambda = self._generate_constraints_jacobian()

		# Generate bounds
		self._x_bnd_l, self._x_bnd_u = self._generate_x_bounds()
		self._c_bnd_l, self._c_bnd_u = self._generate_c_bounds()

		# Initialise the NLP problem
		self._initialise_nlp()

	def _generate_objective_function(self):

		def objective(x):
			J = J_lambda(*x)
			return J

		J = self._ocp._J_subbed
		J_lambda = sym.lambdify(self._x, J, 'numpy')
		return J, objective

	def _generate_objective_function_gradient(self):

		def gradient(x):
			g = g_lambda(*x)
			return g

		dJ_dy = self._y * self._ocp._dJ_dx[self._ocp._y_slice]
		dJ_du = self._u * self._ocp._dJ_dx[self._ocp._u_slice]
		g = np.hstack([dJ_dy.flatten('F'), dJ_du.flatten('F'), self._ocp._dJ_dx[self._ocp._qts_slice]])
		g_lambda = sym.lambdify(self._x, g, 'numpy')
		return g, gradient

	def _generate_constraints(self, x_subs):

		def constraints(x):
			c = c_lambda(*x)
			return c

		# Generate defect constraints
		c_defect, self._num_ind_delta = self._generate_defect_constraints()
		num_c_defect = c_defect.shape[0]
		self._c_defect_slice = slice(0, num_c_defect)

		# Generate path constraints
		c_path = self._generate_path_constraints()
		num_c_path = c_path.shape[0]
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + num_c_path)

		# Genereate integral constraints
		c_integral = self._generate_integral_constraints()
		num_c_integral = self._ocp._num_q_vars
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + num_c_integral)

		# Genereate boundary constraints
		c_boundary = self._generate_boundary_constraints(x_subs)
		num_c_boundary = self._ocp._num_b_cons
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._c_integral_slice.stop + num_c_boundary)

		# Constraints vector
		c = np.hstack([c_defect, c_path, c_integral, c_boundary])
		num_c = num_c_defect + num_c_path + num_c_integral + num_c_boundary

		# print('\n\n\n')
		# print(c)
		# print('\n\n\n')
		# raise NotImplementedError

		c_lambda = sym.lambdify(self._x, c, 'numpy')

		return c, num_c, constraints

	def _generate_defect_constraints(self):

		# def defect_constraints(x):
		# 	y = np.reshape(x[self._y_index], (self._mesh._N, -1), order='F')
		# 	dy = self._state_derivatives_lambda(x)
		# 	zeta_matrix = np.dot(self._defect_constraints_D_matrix, y) - np.dot(self._defect_constraints_A_matrix, dy)
		# 	return zeta_matrix.flatten('F')

		row_starts = self._mesh._mesh_index_boundaries[:-1]
		num_rows = self._mesh._mesh_index_boundaries[-1]
		col_starts = self._mesh._mesh_index_boundaries_offset[:-1]
		num_cols = self._mesh._mesh_index_boundaries_offset[-1] + 1
		
		D_matrix = np.zeros((num_rows, num_cols))
		A_matrix = np.zeros((num_rows, num_cols))

		for block_size, hK, row_start, col_start in zip(self._mesh._mesh_col_points, self._mesh._hK, row_starts, col_starts):
			col_slice = slice(col_start, col_start+block_size)
			row_slice = slice(row_start, row_start+block_size)
			A_matrix[row_slice, col_slice] = (hK/2)*np.eye(block_size)
			D_matrix[row_slice, col_slice] = D_matricies[block_size]

		delta_matrix = np.dot(D_matrix, self._y) - np.dot(A_matrix, self._dy)
		delta_c = delta_matrix.flatten('F')
		v_func = np.vectorize(lambda dy: dy.subs(self._dy_subs_dict))
		delta_c = v_func(delta_c)

		self._defect_constraints_D_matrix = D_matrix
		self._defect_constraints_A_matrix = A_matrix
		# self._defect_constraints_lambda = defect_constraints

		self._num_defect_constraints_per_y = int(delta_matrix.shape[0]/self._ocp._num_y_vars)

		return delta_c, delta_matrix.shape[0]

	def _generate_path_constraints(self):

		# def path_constraints(x):
		# 	return np.empty(0)

		# self._path_constraints_lambda = path_constraints

		return np.array([])

	def _generate_integral_constraints(self):

		def integral_constraints(x):
			g = self._integrands_lambda(x)
			q_vars = x[self._q_index]
			rho_vector = q_vars - np.sum(np.dot(self._integral_constraints_W_matrix, g), 0)
			return rho_vector

		row_starts = self._mesh._mesh_index_boundaries[:-1]
		num_rows = self._mesh._mesh_index_boundaries[-1]
		col_starts = self._mesh._mesh_index_boundaries_offset[:-1]
		num_cols = self._mesh._mesh_index_boundaries_offset[-1] + 1
		
		W_matrix = np.zeros((num_rows, num_cols))

		for block_size, hK, row_start, col_start in zip(self._mesh._mesh_col_points, self._mesh._hK, row_starts, col_starts):

			col_slice = slice(col_start, col_start+block_size)
			row_slice = slice(row_start, row_start+block_size)

			W_matrix[row_slice, col_slice] = (hK/2)*np.diagflat(quadrature_weights[block_size])

		# MUST ADD IN (( t_0-t_f/2 )) TERM
		rho_c = (self._x[self._q_slice] - np.sum(np.dot(W_matrix, self._g_integrands), 0))#[np.newaxis].transpose()
		v_func = np.vectorize(lambda g: g.subs(self._g_subs_dict))
		rho_c = v_func(rho_c)

		self._integral_constraints_W_matrix = W_matrix
		self._integral_constraints_lambda = integral_constraints

		return rho_c

	def _generate_boundary_constraints(self, x_subs):

		def boundary_constraints(x):
			b = b_lambda(*x)
			return np.array(b)

		x0_dict = dict(zip(self._ocp._y_t0, sym.Matrix(self._ocp._y_vars).subs(x_subs[0])))
		xF_dict = dict(zip(self._ocp._y_tF, sym.Matrix(self._ocp._y_vars).subs(x_subs[-1])))
		x_boundary_dict = {**x0_dict, **xF_dict}

		b_c = np.array([c.subs(x_boundary_dict) for c in self._ocp._b_cons])#[np.newaxis].transpose()

		b_lambda = sym.lambdify(self._x, np.array([c.subs(x_boundary_dict) for c in self._ocp._b_cons]), 'numpy')
		self._boundary_constraints_lambda = boundary_constraints

		return b_c

	def _generate_constraints_jacobian(self):

		def jacobian(x):
			G = G_lambda(*x)
			return G

		def jacobian_structure():
			return G_structure_tuple

		# Genereate defect constraints Jacobian
		G_defect, G_defect_rows, G_defect_cols = self._generate_defect_constraints_jacobian()

		# Generate path constraints Jacobian
		G_path, G_path_rows, G_path_cols = self._generate_path_constraints_jacobian()

		# Generate integral constraints Jacobian
		G_integral, G_integral_rows, G_integral_cols = self._generate_integral_constraints_jacobian()

		# Generate boundary constraints Jacobian
		G_boundary, G_boundary_rows, G_boundary_cols = self._generate_boundary_constraints_jacobian()

		# Combine Jacobian
		G = np.hstack([G_defect, G_path, G_integral, G_boundary])
		G_rows = np.hstack([G_defect_rows, G_path_rows, G_integral_rows, G_boundary_rows])
		G_cols = np.hstack([G_defect_cols, G_path_cols, G_integral_cols, G_boundary_cols])
		num_G_nonzeros = len(G)
		G_lambda = sym.lambdify(self._x, G, 'numpy')

		G_structure_tuple = (G_rows.astype(np.int8).tolist(), G_cols.astype(np.int8).tolist())

		return G, G_rows, G_cols, num_G_nonzeros, jacobian, jacobian_structure

	# @profile
	def _generate_defect_constraints_jacobian(self):

		c_defect_per_block = np.sum(self._mesh._mesh_col_points)
		subs_dicts_list = []
		for y_node, u_node in zip(self._y, self._u):
			y_subs_dict = dict(zip(self._ocp._y_vars, y_node))
			u_subs_dict = dict(zip(self._ocp._u_vars, u_node))
			subs_dict = {**y_subs_dict, **u_subs_dict}
			subs_dicts_list.append(subs_dict)

		G = np.array([])
		G_row = np.array([], dtype=np.int8)
		G_col = np.array([], dtype=np.int8)

		for i_c, c in enumerate(self._c[self._c_defect_slice]):
			jacobian_row = np.array(c.diff(self._x))
			nonzero_cols = np.flatnonzero(jacobian_row)
			G = np.hstack([G, jacobian_row[nonzero_cols]])
			G_row = np.hstack([G_row, i_c*np.ones(nonzero_cols.shape, dtype=np.int8)])
			G_col = np.hstack([G_col, nonzero_cols])
			# print(G)
			# print(G_row)
			# print(G_col)
			# input()

		return G, G_row, G_col

		"""
		for defect_index, deriv_row in enumerate(ddelta_dx):

			# State variables
			for state_index, (y_var, deriv) in enumerate(zip(self._ocp._y_vars, deriv_row[self._ocp._y_slice])):
				deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])

				print(self._defect_constraints_A_matrix.shape)
				print(deriv_at_nodes.reshape(1, -1).shape)
				input()

				block = -deriv_at_nodes.reshape(1, -1).dot(self._defect_constraints_A_matrix)
				if defect_index == state_index:
					block += self._defect_constraints_D_matrix
				
				print(block)
				nonzero_elements = np.flatnonzero(block)
				block_eqn = block.flatten()[nonzero_elements]
				G = np.hstack([G, block_eqn])

				block_row, block_col = np.nonzero(block)
				block_row += defect_index*c_defect_per_block
				G_row = np.hstack([G_row, block_row])
				block_col += state_index*self._mesh._N
				G_col = np.hstack([G_col, block_col])

			# Control variables
			for control_index, (u_var, deriv) in enumerate(zip(self._ocp._u_vars, deriv_row[self._ocp._u_slice])):
				deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])
				block = self._defect_constraints_A_matrix * deriv_at_nodes
				
				block_eqn = block.flatten()[np.flatnonzero(block)]
				G = np.hstack([G, block_eqn])

				block_row, block_col = np.nonzero(block)
				block_row += defect_index*c_defect_per_block
				G_row = np.hstack([G_row, block_row])
				block_col += self._u_slice.start + control_index*self._mesh._N
				G_col = np.hstack([G_col, block_col])

			# Integral variables
			# All G entries are zero

			# Time variables
			# No current support for time variables, but will be dense

			# Parameter variables
			for parameter_index, (s_var, deriv) in enumerate(zip(self._ocp._s_vars, deriv_row[self._ocp._s_slice])):
				deriv_at_nodes = np.array([deriv.subs(subs_dict) for subs_dict in subs_dicts_list])
				block = -self._defect_constraints_A_matrix * deriv_at_nodes
				block = np.reshape(block[np.nonzero(block)], (-1, 1))

				block_eqn = block.flatten()[np.flatnonzero(block)]
				G = np.hstack([G, block_eqn])

				block_row, block_col = np.nonzero(block)
				block_row += defect_index*c_defect_per_block
				G_row = np.hstack([G_row, block_row])
				block_col += self._s_slice.start + parameter_index
				G_col = np.hstack([G_col, block_col])

			ind = np.lexsort((G_row, G_col))
			G = G[ind]
			G_row = G_row[ind]
			G_col = G_col[ind]

		return G, G_row, G_col
		"""

	def _generate_path_constraints_jacobian(self):
		return np.array([]), np.array([]), np.array([])

	# @profile
	def _generate_integral_constraints_jacobian(self):

		G = np.array([])
		G_row = np.array([], dtype=np.int8)
		G_col = np.array([], dtype=np.int8)

		# print(self._c[self._c_integral_slice].squeeze())
		# raise NotImplementedError

		for i_c, c in enumerate(self._c[self._c_integral_slice], self._q_slice.start):
			jacobian_row = np.array(c.diff(self._x))
			nonzero_cols = np.flatnonzero(jacobian_row)
			G = np.hstack([G, jacobian_row[nonzero_cols]])
			G_row = np.hstack([G_row, i_c*np.ones(nonzero_cols.shape, dtype=np.int8)])
			G_col = np.hstack([G_col, nonzero_cols])

		return G, G_row, G_col

	def _generate_boundary_constraints_jacobian(self):

		G = np.array([])
		G_row = np.array([], dtype=np.int8)
		G_col = np.array([], dtype=np.int8)

		sym_subs_into_np_array = np.vectorize(lambda vec, subs_dict: vec.subs(subs_dict))

		t0_subs_dict = dict(zip(self._ocp._y_t0, self._y[1, :]))
		tF_subs_dict = dict(zip(self._ocp._y_tF, self._y[-1, :]))

		subs_dict = {**t0_subs_dict, **tF_subs_dict}

		dbdy0 = sym_subs_into_np_array(self._ocp._G_dbdy0, subs_dict)
		dbdyF = sym_subs_into_np_array(self._ocp._G_dbdyF, subs_dict)

		# y variables
		row_indexes = np.array(range(self._c_boundary_slice.start, self._c_boundary_slice.stop))
		for i_y, (col_0, col_F) in enumerate(zip(dbdy0.transpose(), dbdyF.transpose())):
			nonzeros_0 = np.flatnonzero(col_0)
			nonzeros_F = np.flatnonzero(col_F)

			entries_0 = col_0[nonzeros_0]
			entries_F = col_F[nonzeros_F]
			row_indexes_0 = row_indexes[nonzeros_0]
			row_indexes_F = row_indexes[nonzeros_F]
			col_indexes_0 = i_y*self._mesh._N*np.ones_like(row_indexes_0)
			col_indexes_F = (((i_y+1)*self._mesh._N)-1)*np.ones_like(row_indexes_F)

			G = np.hstack([G, entries_0, entries_F])
			G_row = np.hstack([G_row, row_indexes_0, row_indexes_F])
			G_col = np.hstack([G_col, col_indexes_0, col_indexes_F])

		# u variables
		# Empty block

		# q, t & s variables ()
		qts_c_boundary_block = self._ocp._dc_dx[self._ocp._c_boundary_slice, self._ocp._qts_slice]
		qts_c_boundary_block = sym_subs_into_np_array(qts_c_boundary_block, subs_dict)
		nonzeros_row, nonzeros_col = np.nonzero(qts_c_boundary_block)
		block_row_offset = self._c_boundary_slice.start
		block_col_offset = self._q_slice.start
		G = np.hstack([G, qts_c_boundary_block[nonzeros_row, nonzeros_col]])
		G_row = np.hstack([G_row, nonzeros_row + block_row_offset])
		G_col = np.hstack([G_col, nonzeros_row + block_col_offset])

		ind = np.lexsort((G_col, G_row))
		G = G[ind]
		G_row = G_row[ind]
		G_col = G_col[ind]

		return G, G_row, G_col

	def _generate_x_bounds(self):

		bnd_l = np.empty_like(self._x)
		bnd_u = np.empty_like(self._x)

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

		c = self._c.squeeze()

		bnd_l = np.zeros_like(c)
		bnd_u = np.zeros_like(c)

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
				self._constraints_lambda, 
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
		self._nlp_problem.solve(self._guess._x)
		# pass


class IPOPTProblem:

	def __init__(self, J, g, c, G, G_struct):
		self.objective = J
		self.gradient = g
		self.constraints = c
		self.jacobian = G
		self.jacobianstructure = G_struct





