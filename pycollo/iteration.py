import collections

import ipopt
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import sympy as sym

from pycollo.guess import Guess
from pycollo.utils import (numbafy, romberg)

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
		t_0 = prev_guess._t[0]
		t_F = prev_guess._t[-1]
		self._mesh._strech = 2/(t_F + t_0)
		self._mesh._shift = (t_F + t_0)/(t_F - t_0)
		self._mesh._generate_mesh(t_0, t_F)

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
			optimal_control_problem=self._ocp)
		self._guess._t = self._mesh._t
		self._guess._y = new_y
		self._guess._u = new_u
		self._guess._q = prev_guess._q
		# self._guess._t = prev_guess._t
		self._guess._s = prev_guess._s

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
					dzeta_dy_D_nonzero.extend(list(D_ind_array + ind_offset))

		dzeta_dy_slice = slice(0, len(G_nonzero_row))

		for i_c in range(self._ocp._num_y_vars):
			for i_u in range(self._ocp._num_u_vars):
				row_offset = i_c * self._mesh._num_c_boundary_per_y
				col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
				G_nonzero_row.extend(list(A_row_col_array[0] + row_offset))
				G_nonzero_col.extend(list(A_row_col_array[1] + col_offset))
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

		self._reshape_x = reshape_x

		def reshape_x_point(x):
			return self._ocp._x_reshape_lambda_point(x, self._ocp._num_y_vars, self._y_slice.stop, self._q_slice.start)

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

		# Generate bounds
		self._x_bnd_l, self._x_bnd_u = self._generate_x_bounds()
		self._c_bnd_l, self._c_bnd_u = self._generate_c_bounds()

		# Initialise the NLP problem
		self._initialise_nlp()

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
		self._solution._calculate_discretisation_mesh_error()
		# solution = np.array(nlp_solution)
		# y = nlp_solution[self._y_slice].reshape(self._ocp._num_y_vars, -1)
		# u = nlp_solution[self._u_slice].reshape(self._ocp._num_u_vars, -1)
		# q = nlp_solution[self._q_slice]

		# plt.plot(self._mesh._t, y[0, :], 'b', self._mesh._t, u[0, :], 'r')
		# plt.show()
		# print(info)
		# pass

	def _refine_new_mesh(self):
		pass


class Solution:

	def __init__(self, iteration, nlp_solution, nlp_solution_info):
		self._it = iteration
		self._ocp = iteration._ocp
		self._mesh = iteration._mesh
		self._nlp_solution = nlp_solution
		self._nlp_solution_info = nlp_solution_info
		self._x = np.array(nlp_solution)
		if self._ocp._settings._nlp_solver == 'ipopt':
			self._process_solution = self._process_ipopt_solution
		else:
			raise NotImplementedError
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
		self._y = self._x[self._it._y_slice].reshape(self._ocp._num_y_vars, -1)
		self._dy = self._ocp._dy_lambda(*self._it._reshape_x(self._x))
		self._u = self._x[self._it._u_slice].reshape(self._ocp._num_u_vars, -1)
		self._q = self._x[self._it._q_slice]
		self._t = self._x[self._it._t_slice]
		self._s = self._x[self._it._s_slice]
		self._interpolate_solution()

	def _process_snopt_solution(self, solution):
		raise NotImplementedError

	def _interpolate_solution(self):
		self._y_polys = np.empty((self._ocp._num_y_vars, self._mesh._K), dtype=object)
		self._dy_polys = np.empty((self._ocp._num_y_vars, self._mesh._K), dtype=object)
		self._u_polys = np.empty((self._ocp._num_u_vars, self._mesh._K), dtype=object)

		for i_y, state_deriv in enumerate(self._dy):
			for i_k, (i_start, i_stop) in enumerate(zip(self._mesh._mesh_index_boundaries[:-1], self._mesh._mesh_index_boundaries[1:])):
				t_k = self._mesh._t[i_start:i_stop+1]
				dy_k = state_deriv[i_start:i_stop+1]
				dy_poly = np.polynomial.Polynomial.fit(t_k, dy_k, deg=self._mesh._mesh_col_points[i_k]-1, window=[0, 1])
				y_poly = dy_poly.integ(k=self._y[i_y, i_start])
				self._y_polys[i_y, i_k] = y_poly
				self._dy_polys[i_y, i_k] = dy_poly

		for i_u, control in enumerate(self._u):
			for i_k, (i_start, i_stop) in enumerate(zip(self._mesh._mesh_index_boundaries[:-1], self._mesh._mesh_index_boundaries[1:])):
				t_k = self._mesh._t[i_start:i_stop+1]
				u_k = control[i_start:i_stop+1]
				u_poly = np.polynomial.Polynomial.fit(t_k, u_k, deg=self._mesh._mesh_col_points[i_k]-1, window=[0, 1])
				self._u_polys[i_u, i_k] = u_poly

	def _calculate_discretisation_mesh_error(self):

		def eval_poly(t, polys):
			return np.array([poly(t) for poly in polys])

		def error_calc(t, i_y, y_polys, dy_polys, u_polys):
			y_vars = eval_poly(t, y_polys).tolist()
			u_vars = eval_poly(t, u_polys).tolist()
			q_vars = self._q.tolist()
			t_vars = self._t.tolist()
			s_vars = self._s.tolist()
			x = ([np.array([val]) for val in y_vars + u_vars] + q_vars + t_vars + s_vars)
			dy = self._ocp._dy_lambda(*x)
			epsilon = eval_poly(t, dy_polys)[i_y] - dy[i_y]
			return np.abs(epsilon)

		mesh_error = np.empty((self._ocp._num_y_vars, self._mesh._K))
		t_knots = (self._mesh._t[self._mesh._mesh_index_boundaries])
		for i_k, (t_knot_start, t_knot_end) in enumerate(zip(t_knots[:-1], t_knots[1:])):
			for i_y in range(self._ocp._num_y_vars):
				y_polys = self._y_polys[:, i_k]
				dy_polys = self._dy_polys[:, i_k]
				u_polys = self._u_polys[:, i_k] 

				mesh_error[i_y, i_k] = romberg(error_calc, t_knot_start, t_knot_end, args=(i_y, y_polys, dy_polys, u_polys), divmax=25)

		print(mesh_error)

		print(np.amax(mesh_error, axis=0))

	def _patterson_discreisation_mesh_error(self):
		pass



class IPOPTProblem:

	def __init__(self, J, g, c, G, G_struct):
		self.objective = J
		self.gradient = g
		self.constraints = c
		self.jacobian = G
		self.jacobianstructure = G_struct





