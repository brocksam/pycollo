import collections
import csv
import itertools
import json
from timeit import default_timer as timer
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import sympy as sym

from .guess import (PhaseGuess, EndpointGuess, Guess)
from .nlp import initialise_nlp_backend
from .solution import Solution
from .scaling import IterationScaling
from .utils import console_out

class Iteration:

	# def __init__(self, optimal_control_problem=None, iteration_number=None, *, mesh=None, guess=None):
	def __init__(self, backend, index, mesh, guess):

		self.backend = backend
		self.ocp = self.backend.ocp

		self.index = index
		self.number = index + 1
		self._mesh = mesh
		self.prev_guess = guess

		self.initialise()

		# # Optimal control problem
		# self._ocp = optimal_control_problem

		# # Iteration number
		# self._iteration_number = int(iteration_number)

		# # Mesh
		# self._mesh = mesh
		# self._mesh._iteration = self

		# # Guess
		# self._guess = guess
		
		# # Result
		# self._result = None

	@property
	def optimal_control_problem(self):
		return self.ocp

	@property
	def mesh(self):
		return self._mesh
	
	# @property
	# def guess(self):
	# 	return self._guess
	
	@property
	def result(self):
		return self._result

	# @property
	# def x(self):
	# 	return self._x

	# @property
	# def _y(self):
	# 	return self._x[self._y_slice, :].reshape(self._ocp._num_y_vars, self._mesh._N)
		
	# @property
	# def _u(self):
	# 	return self._x[self._u_slice, :].reshape(self._ocp._num_u_vars, self._mesh._N)
	
	# @property
	# def _q(self):
	# 	return self._x[self._q_slice, :]

	# @property
	# def _t(self):
	# 	return self._x[self._t_slice, :]

	# @property
	# def _s(self):
	# 	return self._x[self._s_slice, :]

	@property
	def solution(self):
		return self._solution

	def initialise(self):
		self.console_out_initialising_iteration()
		self.interpolate_guess_to_mesh(self.prev_guess)
		self.create_variable_constraint_numbers_slices()
		self.generate_nlp_lambdas()
		self.generate_bounds()
		self.generate_scaling()
		self.reset_bounds()
		self.initialise_nlp()
		self.check_nlp_functions()

	def console_out_initialising_iteration(self):
		msg = f"Initialising mesh iteration #{self.number}."
		console_out(msg, heading=True)

	def interpolate_guess_to_mesh(self, prev_guess):

		def interpolate_to_new_mesh(prev_tau, tau, num_vars, prev, N):
			new_guess = np.empty((num_vars, N))
			for index, row in enumerate(prev):
				interp_func = interpolate.interp1d(prev_tau, row)
				new_guess[index, :] = interp_func(tau)
			return new_guess

		# Guess
		self.guess_tau = self.mesh.tau
		self.guess_t0 = prev_guess.t0
		self.guess_tF = prev_guess.tF
		self.guess_stretch = [0.5 * (tF - t0) for t0, tF in zip(self.guess_t0, self.guess_tF)]
		self.guess_shift = [0.5 * (t0 + tF) for t0, tF in zip(self.guess_t0, self.guess_tF)]
		self.guess_time = [tau*stretch + shift for tau, stretch, shift in zip(self.guess_tau, self.guess_stretch, self.guess_shift)]
		self.guess_y = [interpolate_to_new_mesh(prev_tau, tau, p.num_y_vars, prev_y, N) for prev_tau, tau, p, prev_y, N in zip(prev_guess.tau, self.guess_tau, self.backend.p, prev_guess.y, self.mesh.N)]
		self.guess_u = [interpolate_to_new_mesh(prev_tau, tau, p.num_u_vars, prev_u, N) for prev_tau, tau, p, prev_u, N in zip(prev_guess.tau, self.guess_tau, self.backend.p, prev_guess.u, self.mesh.N)]
		self.guess_q = prev_guess.q
		self.guess_t = prev_guess.t
		self.guess_s = prev_guess.s

		self.guess_x = []
		for y, u, q, t in zip(self.guess_y, self.guess_u, self.guess_q, self.guess_t):
			self.guess_x.extend(y.tolist())
			self.guess_x.extend(u.tolist())
			self.guess_x.extend([q.tolist()])
			self.guess_x.extend([t.tolist()])
		self.guess_x.extend([self.guess_s.tolist()])
		self.guess_x = np.array(list(itertools.chain.from_iterable(self.guess_x)))

		msg = ("Guess interpolated to iteration mesh.")
		console_out(msg)

	def create_variable_constraint_numbers_slices(self):

		self.num_y_per_phase = []
		self.num_u_per_phase = []
		self.num_q_per_phase = []
		self.num_t_per_phase = []
		self.num_x_per_phase = []
		for p_backend, N in zip(self.backend.p, self.mesh.N):
			num_y = p_backend.num_y_vars * N
			num_u = p_backend.num_u_vars * N
			num_q = p_backend.num_q_vars
			num_t = p_backend.num_t_vars
			num_x = num_y + num_u + num_q + num_t
			self.num_y_per_phase.append(num_y)
			self.num_u_per_phase.append(num_u)
			self.num_q_per_phase.append(num_q)
			self.num_t_per_phase.append(num_t)
			self.num_x_per_phase.append(num_x)
		self.num_y = sum(num_y for num_y in self.num_y_per_phase)
		self.num_u = sum(num_u for num_u in self.num_u_per_phase)
		self.num_q = sum(num_q for num_q in self.num_q_per_phase)
		self.num_t = sum(num_t for num_t in self.num_t_per_phase)
		self.num_s = self.backend.num_s_vars
		self.num_x = self.num_y + self.num_u + self.num_q + self.num_t + self.num_s

		self.y_slices = []
		self.u_slices = []
		self.q_slices = []
		self.t_slices = []
		self.x_slices = []
		total = 0
		for num_y, num_u, num_q, num_t, num_x in zip(self.num_y_per_phase, self.num_u_per_phase, self.num_q_per_phase, self.num_t_per_phase, self.num_x_per_phase):
			y_slice = slice(total, total + num_y)
			u_slice = slice(y_slice.stop, y_slice.stop + num_u)
			q_slice = slice(u_slice.stop, u_slice.stop + num_q)
			t_slice = slice(q_slice.stop, q_slice.stop + num_t)
			x_slice = slice(y_slice.start, t_slice.stop)
			self.y_slices.append(y_slice)
			self.u_slices.append(u_slice)
			self.q_slices.append(q_slice)
			self.t_slices.append(t_slice)
			self.x_slices.append(x_slice)
			total += num_x
		self.s_slice = slice(self.num_x - self.num_s, self.num_x)

		# self._yu_slice = slice(self._y_slice.start, self._u_slice.stop)
		# self._qts_slice = slice(self._q_slice.start, self._s_slice.stop)
		# self._yu_qts_split = self._yu_slice.stop

		# # Constraints
		self.num_c_defect_per_phase = []
		self.num_c_path_per_phase = []
		self.num_c_integral_per_phase = []
		self.num_c_per_phase = []
		for p_backend, N, num_c_defect_per_y in zip(self.backend.p, self.mesh.N, self.mesh.num_c_defect_per_y):
			num_c_defect = p_backend.num_c_defect * num_c_defect_per_y
			num_c_path = p_backend.num_c_path * N
			num_c_integral = p_backend.num_c_integral
			num_c = num_c_defect + num_c_path + num_c_integral
			self.num_c_defect_per_phase.append(num_c_defect)
			self.num_c_path_per_phase.append(num_c_path)
			self.num_c_integral_per_phase.append(num_c_integral)
			self.num_c_per_phase.append(num_c)
		self.num_c_defect = sum(num_c for num_c in self.num_c_defect_per_phase)
		self.num_c_path = sum(num_c for num_c in self.num_c_path_per_phase)
		self.num_c_integral = sum(num_c for num_c in self.num_c_integral_per_phase)
		self.num_c_endpoint = self.backend.num_c_endpoint
		self.num_c = self.num_c_defect + self.num_c_path + self.num_c_integral + self.num_c_endpoint

		self.c_lambda_dy_slices = []
		self.c_lambda_p_slices = []
		self.c_lambda_g_slices = []
		for num_y, num_c_path, num_c_integral, N in zip(self.num_y_per_phase, self.num_c_path_per_phase, self.num_c_integral_per_phase, self.mesh.N):
			dy_slice = slice(0, num_y)
			p_slice = slice(dy_slice.stop, dy_slice.stop + num_c_path)
			g_slice = slice(p_slice.stop, p_slice.stop + num_c_integral * N)
			self.c_lambda_dy_slices.append(dy_slice)
			self.c_lambda_p_slices.append(p_slice)
			self.c_lambda_g_slices.append(g_slice)
		# self._c_lambda_dy_slice = slice(0, self._num_y)
		# self._c_lambda_p_slice = slice(self._c_lambda_dy_slice.stop, self._c_lambda_dy_slice.stop + self._num_c_path*self._mesh._N)
		# self._c_lambda_g_slice = slice(self._c_lambda_p_slice.stop, self._c_lambda_p_slice.stop + self._num_c_integral*self._mesh._N)

		self.c_defect_slices = []
		self.c_path_slices = []
		self.c_integral_slices = []
		self.c_slices = []
		total = 0
		for num_c_defect, num_c_path, num_c_integral, num_c in zip(self.num_c_defect_per_phase, self.num_c_path_per_phase, self.num_c_integral_per_phase, self.num_c_per_phase):
			c_defect_slice = slice(total, total + num_c_defect)
			c_path_slice = slice(c_defect_slice.stop, c_defect_slice.stop + num_c_path)
			c_integral_slice = slice(c_path_slice.stop, c_path_slice.stop + num_c_integral)
			c_slice = slice(c_defect_slice.start, c_integral_slice.stop)
			self.c_defect_slices.append(c_defect_slice)
			self.c_path_slices.append(c_path_slice)
			self.c_integral_slices.append(c_integral_slice)
			self.c_slices.append(c_slice)
			total += num_c
		self.c_endpoint_slice = slice(self.num_c - self.num_c_endpoint, self.num_c)

		# self._c_defect_slice = slice(0, self._num_c_defect)
		# self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_path)
		# self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_c_integral)
		# self._c_boundary_slice = slice(self._c_integral_slice.stop, self._num_c)

	def generate_nlp_lambdas(self):

		self.generate_unscale_variables_lambda()
		self.generate_continuous_variables_reshape_lambda()
		self.generate_endpoint_variables_reshape_lambda()
		self.generate_objective_lambda()
		self.generate_gradient_lambda()
		self.generate_constraints_lambda()
		self.generate_jacobian_lambda()
		if self.ocp.settings.derivative_level == 2:
			self.generate_hessian_lambda()

	def generate_unscale_variables_lambda(self):

		def unscale_x(x):
			# x = self.scaling.x_stretch*(x - self.scaling.x_shift)
			return x

		self._unscale_x = unscale_x
		msg = "Variable unscale lambda generated successfully."
		console_out(msg)

	def generate_continuous_variables_reshape_lambda(self):

		def reshape_x(x):
			x = self._unscale_x(x)
			x_tuple = self.backend.compiled_functions.x_reshape_lambda(
				x, 
				self.y_slices, 
				self.u_slices, 
				self.q_slices, 
				self.t_slices, 
				self.s_slice,
				self.mesh.N,
				)
			return x_tuple

		self._reshape_x = reshape_x
		msg = "Continuous variable reshape lambda generated successfully."
		console_out(msg)

	def generate_endpoint_variables_reshape_lambda(self):

		self._x_endpoint_indices = []
		for p_backend, x_slice, q_slice, t_slice, N in zip(self.backend.p, self.x_slices, self.q_slices, self.t_slices, self.mesh.N):
			for i in range(p_backend.num_y_vars):
				start = x_slice.start
				i_y_t0 = start + i*N
				i_y_tF = start + (i+1)*N - 1
				self._x_endpoint_indices.append(i_y_t0)
				self._x_endpoint_indices.append(i_y_tF)
			self._x_endpoint_indices.extend(list(range(q_slice.start, q_slice.stop)))
			self._x_endpoint_indices.extend(list(range(t_slice.start, t_slice.stop)))
		self._x_endpoint_indices.extend(list(range(self.s_slice.start, self.s_slice.stop)))

		def reshape_x_point(x):
			x = self._unscale_x(x)
			return self.backend.compiled_functions.x_reshape_lambda_point(x, self._x_endpoint_indices)

		self._reshape_x_point = reshape_x_point
		msg = "Endpoint variable reshape lambda generated successfully."
		console_out(msg)

	def generate_objective_lambda(self):

		def objective(x):
			x_tuple_point = self._reshape_x_point(x)
			J = self.backend.compiled_functions.J_lambda(x_tuple_point)
			return J

		self._objective_lambda = objective
		msg = "Objective function lambda generated successfully."
		console_out(msg)

	def generate_gradient_lambda(self):

		def gradient(x):
			x_tuple_point = self._reshape_x_point(x)
			g = self.backend.compiled_functions.g_lambda(x_tuple_point, self.mesh.N)
			return g

		self._gradient_lambda = gradient
		msg = "Objective function gradient lambda generated successfully."
		console_out(msg)

	def generate_constraints_lambda(self):

		def constraint(x):
			x_tuple = self._reshape_x(x)
			x_tuple_point = self._reshape_x_point(x)
			c = self.backend.compiled_functions.c_lambda(
				x_tuple, 
				x_tuple_point, 
				self.mesh.sA_matrix, 
				self.mesh.sD_matrix, 
				self.mesh.W_matrix, 
				self.mesh.N,
				[slice(p_var_slice.start, p_var_slice.start + p.num_y_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				[slice(p_var_slice.start + p.num_y_vars + p.num_u_vars, p_var_slice.start + p.num_y_vars + p.num_u_vars + p.num_q_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				self.c_lambda_dy_slices,
				self.c_lambda_p_slices,
				self.c_lambda_g_slices,
				)#, self._mesh._N, self._ocp._y_slice, self._ocp._q_slice, self._num_c, self._c_lambda_dy_slice, self._c_lambda_p_slice, self._c_lambda_g_slice, self._c_defect_slice, self._c_path_slice, self._c_integral_slice, self._c_boundary_slice, self._mesh._sA_matrix, self._mesh._sD_matrix, self._mesh._W_matrix)
			return c

		self._constraint_lambda = constraint
		msg = "Constraints lambda generated successfully."
		console_out(msg)

	def generate_jacobian_lambda(self):

		def jacobian_data(x):
			G = jacobian(x)
			G_zeros = sparse.coo_matrix((np.full(self._G_nnz, 1e-20), self._jacobian_structure_lambda()), shape=self._G_shape).tocsr()
			return (G + G_zeros).tocoo().data

		def jacobian(x):
			x_tuple = self._reshape_x(x)
			x_tuple_point = self._reshape_x_point(x)
			G = self.backend.compiled_functions.G_lambda(
				self._G_shape,
				x_tuple, 
				x_tuple_point,
				self.mesh.sA_matrix, 
				self.mesh.sD_matrix, 
				self.mesh.W_matrix, 
				self.mesh.N,
				[c_slice.start for c_slice in self.c_slices],
				[x_slice.start for x_slice in self.x_slices],
				[slice(p_var_slice.start, p_var_slice.start + p.num_y_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				[slice(p_var_slice.start + p.num_y_vars, p_var_slice.start + p.num_y_vars + p.num_u_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				[slice(p_var_slice.start + p.num_y_vars + p.num_u_vars, p_var_slice.start + p.num_y_vars + p.num_u_vars + p.num_q_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				[slice(p_var_slice.start + p.num_y_vars + p.num_u_vars + p.num_q_vars, p_var_slice.start + p.num_y_vars + p.num_u_vars + p.num_q_vars + p.num_t_vars) for p, p_var_slice in zip(self.backend.p, self.backend.phase_variable_slices)],
				self.c_lambda_dy_slices,
				self.c_lambda_p_slices,
				self.c_lambda_g_slices,
				[slice(p_c_slice.start, p_c_slice.start + p.num_c_defect) for p, p_c_slice in zip(self.backend.p, self.backend.phase_constraint_slices)],
				[slice(p_c_slice.start + p.num_c_defect, p_c_slice.start + p.num_c_defect + p.num_c_path) for p, p_c_slice in zip(self.backend.p, self.backend.phase_constraint_slices)],
				[slice(p_c_slice.start + p.num_c_defect + p.num_c_path, p_c_slice.start + p.num_c_defect + p.num_c_path + p.num_c_integral) for p, p_c_slice in zip(self.backend.p, self.backend.phase_constraint_slices)],
				self.num_y_per_phase,
				self.num_u_per_phase,
				self.num_q_per_phase,
				self.num_t_per_phase,
				self.num_x_per_phase,
				self.backend.num_s_vars,
				self.num_c_defect_per_phase,
				self.num_c_path_per_phase,
				self.num_c_integral_per_phase,
				self.num_c_per_phase,
				self.num_c_endpoint,
				[p.y_slice for p in self.backend.p],
				[p.u_slice for p in self.backend.p],
				[p.q_slice for p in self.backend.p],
				[p.t_slice for p in self.backend.p],
				[p.c_defect_slice for p in self.backend.p],
				[p.c_path_slice for p in self.backend.p],
				[p.c_integral_slice for p in self.backend.p],
				)
			return G

		self._G_shape = (self.num_c, self.num_x)
		x_sparsity_detect = np.full(self.num_x, np.nan)
		G_sparsity_detect = jacobian(x_sparsity_detect).tocoo()
		self._G_nonzero_row = G_sparsity_detect.row
		self._G_nonzero_col = G_sparsity_detect.col
		self._G_nnz = G_sparsity_detect.nnz

		self._jacobian_lambda = jacobian_data

		def jacobian_structure():
			return (self._G_nonzero_row, self._G_nonzero_col)

		self._jacobian_structure_lambda = jacobian_structure

		msg = "Jacobian of the constraints lambda generated successfully."
		console_out(msg)

	def generate_hessian_lambda(self):

		def detect_hessian_sparsity():

			def build_endpoint_blocks():
				blocks = []
				return blocks

			sH_endpoint_blocks = []
			sH_continuous_blocks = []

			sH_J = detect_endpoint_hessian_sparsity(self.backend.compiled_functions.ddL_J_dxbdxb_nonzero_indices)
			self._sH_J_nz_inds_ = (sH_J.row, sH_J.col)

			sH = sH_J
			self._H_nonzero_row = sH.row
			self._H_nonzero_col = sH.col
			self._H_nnz = sH.nnz

		def detect_endpoint_hessian_sparsity(nz_inds):
			nnz = len(nz_inds[0])
			data = np.full(nnz, np.nan)
			H_ocp = sparse.coo_matrix((data, nz_inds), shape=self._H_shape)

			H = sparse.tril(H)
			return H

		def hessian_data(x):
			G = hessian(x)
			G_zeros = sparse.coo_matrix((np.full(self._G_nnz, 1e-20), self._hessian_structure_lambda()), shape=self._G_shape).tocsr()
			return (G + G_zeros).tocoo().data

		def hessian(x):
			x_tuple = self._reshape_x(x)
			x_tuple_point = self._reshape_x_point(x)
			H = self.backend.compiled_functions.H_lambda(
				self._H_shape,
				x_tuple, 
				x_tuple_point,
				)
			return G

		self._H_shape = (self.num_x, self.num_x)
		detect_hessian_sparsity()
		self._hessian_lambda = hessian_data

		def hessian_structure():
			return (self._H_nonzero_row, self._H_nonzero_col)

		self._hessian_structure_lambda = hessian_structure

		msg = "Hessian of the Lagrangian lambda generated successfully."
		console_out(msg)

	def generate_bounds(self):
		self.generate_variable_bounds()
		self.generate_constraint_bounds()
		msg = "Mesh-specific bounds generated."
		console_out(msg)

	def generate_variable_bounds(self):
		bnds = []
		for p, N in zip(self.backend.p, self.mesh.N):
			p_bnds = p.ocp_phase.bounds
			y_bnds = p_bnds._y_bnds[p_bnds._y_needed]
			y_t0_bnds = p_bnds._y_t0_bnds[p_bnds._y_needed]
			y_tF_bnds = p_bnds._y_tF_bnds[p_bnds._y_needed]
			u_bnds = p_bnds._u_bnds[p_bnds._u_needed]
			q_bnds = p_bnds._q_bnds[p_bnds._q_needed]
			t_bnds = p_bnds._t_bnds[p_bnds._t_needed]
			for y_bnd, y_t0_bnd, y_tF_bnd in zip(y_bnds, y_t0_bnds, y_tF_bnds):
				bnds.extend([y_t0_bnd] + [y_bnd]*(N-2) + [y_tF_bnd])
			for u_bnd in u_bnds:
				bnds.extend([u_bnd]*N)
			bnds.extend(q_bnds)
			bnds.extend(t_bnds)
		bnds.extend(self.backend.ocp.bounds._s_bnds[self.backend.ocp.bounds._s_needed])
		bnds = np.array(bnds)
		self._x_bnd_l = bnds[:, 0]
		self._x_bnd_u = bnds[:, 1]

	def generate_constraint_bounds(self):
		bnds = []
		for p, N, num_c_defect, num_c_integral in zip(self.backend.p, self.mesh.N, self.num_c_defect_per_phase, self.num_c_integral_per_phase):
			bnds.extend([np.array((0, 0))]*num_c_defect)
			for c_bnd in p.ocp_phase.bounds._c_path_bnds:
				bnds.extend([c_bnd]*N)
			bnds.extend([np.array((0, 0))]*num_c_integral)
		bnds.extend(self.backend.ocp.bounds._c_endpoint_bnds)
		bnds = np.array(bnds)
		self._c_bnd_l = bnds[:, 0]
		self._c_bnd_u = bnds[:, 1]

	def generate_scaling(self):
		self.scaling = IterationScaling(self)
		self.scaling._generate()
		msg = "Scaling generated."
		console_out(msg)

	def reset_bounds(self):
		self._x_bnd_l = (self._x_bnd_l - self.scaling.x_shift) * self.scaling.x_scaling
		self._x_bnd_u = (self._x_bnd_u - self.scaling.x_shift) * self.scaling.x_scaling

	def initialise_nlp(self):
		self._nlp_problem = initialise_nlp_backend(self)
		msg = "NLP initialised successfully."
		console_out(msg)

	def check_nlp_functions(self):
		if self.backend.ocp.settings.check_nlp_functions:
			print('\n\n\n')
			x_data = np.array(range(1, self.num_x+1), dtype=float)
			# lagrange = np.array(range(self._num_c), dtype=float)
			# obj_factor = 2.0
			print(f"x Variables:\n{self.backend.x_vars}\n")
			print(f"x Data:\n{x_data}\n")
			# print(f"Lagrange Multipliers:\n{lagrange}\n")

			J = self._objective_lambda(x_data)
			print(f"J:\n{J}\n")

			g = self._gradient_lambda(x_data)
			print(f"g:\n{g}\n")

			c = self._constraint_lambda(x_data)
			print(f"c:\n{c}\n")

			G_struct = self._jacobian_structure_lambda()
			print(f"G Structure:\n{G_struct[0]}\n{G_struct[1]}\n")

			G_nnz = len(G_struct[0])
			print(f"G Nonzeros:\n{G_nnz}\n")

			G = self._jacobian_lambda(x_data)
			print(f"G:\n{G}\n")

			if self.optimal_control_problem.settings.dump_nlp_check_json:
				file_extension = ".json"
				filename_full = str(self.optimal_control_problem.settings.dump_nlp_check_json) + file_extension

				sG = sparse.coo_matrix((G, G_struct), shape=self._G_shape)

				data = {
					"x": x_data.tolist(),
					"J": float(J),
					"g": np.array(g).tolist(),
					"c": np.array(c).tolist(), 
					"G_data": sG.data.tolist(),
					"G_row": sG.row.tolist(),
					"G_col": sG.col.tolist(),
					"G_nnz": int(sG.nnz),
					"num_x": int(self.num_x),
					"num_c": int(self.num_c),
					}

				with open(filename_full, "w", encoding="utf-8") as file:
					json.dump(data, file, ensure_ascii=False, indent=4)

			# if self._ocp.settings.derivative_level == 2:

			# 	H_struct = self._hessian_structure_lambda()
			# 	print(f"H Structure:\n{H_struct}\n")

			# 	H = self._hessian_lambda(x_data, lagrange, obj_factor)
			# 	print(f"H:\n{H}\n")

			# 	sH = sparse.coo_matrix((H, (H_struct[0], H_struct[1])), shape=self._H_shape)
			# 	sH = sparse.coo_matrix(np.tril(sH.toarray()))
			# 	print(sH)
			
			print('\n\n\n')
			raise NotImplementedError

	def solve(self):

		time_start = time.time()
		nlp_time_start = timer()

		nlp_solution, nlp_solution_info = self._nlp_problem.solve(self.guess_x)

		nlp_time_stop = timer()
		time_stop = time.time()
		self._nlp_time = nlp_time_stop - nlp_time_start

		process_results_time_start = timer()

		self._solution = Solution(self, nlp_solution, nlp_solution_info)
		next_iter_mesh = self._refine_new_mesh()
		next_iter_guess = self.generate_guess_for_next_mesh_iteration()

		process_results_time_stop = timer()

		mesh_tolerance_met = self.check_if_mesh_tolerance_met(next_iter_mesh)
		self._display_mesh_iteration_info(mesh_tolerance_met, next_iter_mesh)

		self._process_results_time = process_results_time_stop - process_results_time_start

		return mesh_tolerance_met, next_iter_mesh, next_iter_guess

	def generate_guess_for_next_mesh_iteration(self):
		phase_guesses = self.collect_next_mesh_iteration_phase_guesses()
		endpoint_guess = self.collect_next_mesh_iteration_endpoint_guess()
		next_guess = Guess(self.backend, phase_guesses, endpoint_guess)
		return next_guess

	def collect_next_mesh_iteration_phase_guesses(self):
		phase_guesses = [self.collect_single_next_mesh_iteration_phase_guess(p)
			for p in self.backend.p]
		return phase_guesses

	def collect_single_next_mesh_iteration_phase_guess(self, phase_backend):
		phase_guess = PhaseGuess(phase_backend)
		phase_guess.time = self.solution._time[phase_backend.i]
		phase_guess.state_variables = self.solution._y[phase_backend.i]
		phase_guess.control_variables = self.solution._u[phase_backend.i]
		phase_guess.integral_variables = self.solution._q[phase_backend.i]
		return phase_guess

	def collect_next_mesh_iteration_endpoint_guess(self):
		endpoint_guess = EndpointGuess(self.optimal_control_problem)
		endpoint_guess.parameter_variables = self.solution._s
		return endpoint_guess

	def check_if_mesh_tolerance_met(self, next_iter_mesh):
		mesh_tol = self.optimal_control_problem.settings.mesh_tolerance
		error_over_max_mesh_tol = [np.max(max_rel_mesh_errors) > mesh_tol
			for max_rel_mesh_errors in self.solution._maximum_relative_mesh_errors]
		mesh_tol_met = not any(error_over_max_mesh_tol)
		return mesh_tol_met


	def _display_mesh_iteration_info(self, mesh_tol_met, next_iter_mesh):
		msg = (f"Pycollo Analysis of Mesh Iteration {self.number}")
		console_out(msg, subheading=True, suffix=":")

		max_rel_mesh_error = np.max(np.array([np.max(element) 
			for element in self._solution._maximum_relative_mesh_errors]))

		print(f'Objective Evaluation:       {self.solution.objective}')
		print(f'Max Relative Mesh Error:    {max_rel_mesh_error}\n')
		if mesh_tol_met:
			print(f'Adjusting Collocation Mesh: {next_iter_mesh.K} mesh sections\n')

		settings = self.optimal_control_problem.settings
		if settings.display_mesh_result_info:
			raise NotImplementedError

		if settings.display_mesh_result_graph:
			self.solution._plot_interpolated_solution(plot_y=True, plot_dy=True, plot_u=True)

	def _refine_new_mesh(self):
		self.solution._patterson_rao_discretisation_mesh_error()
		next_iter_mesh = self.solution._patterson_rao_next_iteration_mesh()
		return next_iter_mesh
























		# def _initialise_old(self):

		# def hessian_objective_sparsity():
		# 	H_objective_nonzero_row = []
		# 	H_objective_nonzero_col = []

		# 	ddL_J_dxbdxb_nonzero = self._ocp._expression_graph.ddL_J_dxbdxb

		# 	for i_row in range(self._ocp._num_point_vars):
		# 		row = ddL_J_dxbdxb_nonzero[i_row, :i_row+1]
		# 		if i_row < 2*self._ocp._num_y_vars:
		# 			if i_row % 2:
		# 				row_offset = int((i_row+1)/2) * self._mesh._N - 1
		# 			else:
		# 				row_offset = int(i_row/2) * self._mesh._N
		# 		else:
		# 			row_offset = self._ocp._yu_qts_split * self._mesh._N + (i_row - 2*self._ocp._num_y_vars)
		# 		for i_col, entry in enumerate(row):
		# 			entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			if entry != 0:
		# 				if i_col < 2*self._ocp._num_y_vars:
		# 					if i_col % 2:
		# 						col_offset = int((i_col+1)/2) * self._mesh._N - 1
		# 					else:
		# 						col_offset = int(i_col/2) * self._mesh._N
		# 				else:
		# 					col_offset = self._ocp._yu_qts_split * self._mesh._N + (i_col - 2*self._ocp._num_y_vars)
		# 				H_objective_nonzero_row.append(row_offset)
		# 				H_objective_nonzero_col.append(col_offset)

		# 	num_H_objective_nonzero = len(H_objective_nonzero_row)
		# 	sH_objective_matrix = sparse.coo_matrix(([1]*num_H_objective_nonzero, (H_objective_nonzero_row, H_objective_nonzero_col)), shape=self._H_shape)
		# 	sH_objective_indices = list(zip(sH_objective_matrix.row, sH_objective_matrix.col))
		# 	return sH_objective_matrix, sH_objective_indices

		# def hessian_defect_sparsity():
		# 	H_defect_nonzero_row = []
		# 	H_defect_nonzero_col = []
		# 	H_defect_sum_flag = []

		# 	ddL_zeta_dxdx_nonzero = self._ocp._expression_graph.ddL_zeta_dxdx[0]
		# 	for matrix in self._ocp._expression_graph.ddL_zeta_dxdx[1:]:
		# 		ddL_zeta_dxdx_nonzero += matrix
			
		# 	for i_row in range(self._ocp._num_vars):
		# 		row = ddL_zeta_dxdx_nonzero[i_row, :i_row+1]
		# 		if i_row < self._ocp._yu_qts_split:
		# 			row_offset = i_row * self._mesh._N
		# 			row_numbers = list(range(row_offset, row_offset + self._mesh._N))
		# 		else:
		# 			row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
		# 			row_numbers = [row_offset]*self._mesh._N
		# 		for i_col, entry in enumerate(row):
		# 			entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			if entry != 0:
		# 				if i_col < self._ocp._yu_qts_split:
		# 					col_offset = i_col * self._mesh._N
		# 					col_numbers = list(range(col_offset, col_offset + self._mesh._N))
		# 					H_defect_sum_flag.append(False)
		# 				else:
		# 					col_offset = self._ocp._yu_qts_split* (self._mesh._N - 1) + i_col
		# 					row_numbers = [row_offset]
		# 					col_numbers = [col_offset]
		# 					H_defect_sum_flag.append(True)
		# 				H_defect_nonzero_row.extend(row_numbers)
		# 				H_defect_nonzero_col.extend(col_numbers)

		# 	num_H_defect_nonzero = len(H_defect_nonzero_row)
		# 	sH_defect_matrix = sparse.coo_matrix(([1]*num_H_defect_nonzero, (H_defect_nonzero_row, H_defect_nonzero_col)), shape=self._H_shape)
		# 	sH_defect_matrix = sH_defect_matrix.tocsr().tocoo()
		# 	sH_defect_indices = list(zip(sH_defect_matrix.row, sH_defect_matrix.col))
		# 	return sH_defect_matrix, sH_defect_indices, H_defect_sum_flag

		# def hessian_path_sparsity():
		# 	H_path_nonzero_row = []
		# 	H_path_nonzero_col = []
		# 	H_path_sum_flag = []

		# 	try:
		# 		ddL_gamma_dxdx_nonzero = self._ocp._expression_graph.ddL_gamma_dxdx[0]
		# 	except IndexError:
		# 		raise NotImplementedError
		# 	else:
		# 		for matrix in self._ocp._expression_graph.ddL_gamma_dxdx[1:]:
		# 			ddL_gamma_dxdx_nonzero += matrix

		# 	for i_row in range(self._ocp._num_vars):
		# 		row = ddL_gamma_dxdx_nonzero[i_row, :i_row+1]
		# 		if i_row < self._ocp._yu_qts_split:
		# 			row_offset = i_row * self._mesh._N
		# 			row_numbers = list(range(row_offset, row_offset + self._mesh._N))
		# 		else:
		# 			row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
		# 			row_numbers = [row_offset]*self._mesh._N
		# 		for i_col, entry in enumerate(row):
		# 			entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			if entry != 0:
		# 				if i_col < self._ocp._yu_qts_split:
		# 					col_offset = i_col * self._mesh._N
		# 					col_numbers = list(range(col_offset, col_offset + self._mesh._N))
		# 				else:
		# 					col_offset = self._ocp._yu_qts_split* (self._mesh._N - 1) + i_col
		# 					row_numbers = [row_offset]
		# 					col_numbers = [col_offset]
		# 				H_path_nonzero_row.extend(row_numbers)
		# 				H_path_nonzero_col.extend(col_numbers)

		# 	num_H_path_nonzero = len(H_path_nonzero_row)
		# 	sH_path_matrix = sparse.coo_matrix(([1]*num_H_path_nonzero, (H_path_nonzero_row, H_path_nonzero_col)), shape=self._H_shape).tocsr().tocoo()
		# 	sH_path_indices = list(zip(sH_path_matrix.row, sH_path_matrix.col))
		# 	return sH_path_matrix, sH_path_indices

		# def hessian_integral_sparsity():
		# 	H_integral_nonzero_row = []
		# 	H_integral_nonzero_col = []
		# 	H_integral_sum_flag = []

		# 	ddL_rho_dxdx_nonzero = self._ocp._expression_graph.ddL_rho_dxdx[0]
		# 	for matrix in self._ocp._expression_graph.ddL_rho_dxdx[1:]:
		# 		ddL_rho_dxdx_nonzero += matrix
			
		# 	for i_row in range(self._ocp._num_vars):
		# 		row = ddL_rho_dxdx_nonzero[i_row, :i_row+1]
		# 		if i_row < self._ocp._yu_qts_split:
		# 			row_offset = i_row * self._mesh._N
		# 			row_numbers = list(range(row_offset, row_offset + self._mesh._N))
		# 		else:
		# 			row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
		# 			row_numbers = [row_offset]*self._mesh._N
		# 		for i_col, entry in enumerate(row):
		# 			entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			if entry != 0:
		# 				if i_col < self._ocp._yu_qts_split:
		# 					col_offset = i_col * self._mesh._N
		# 					col_numbers = list(range(col_offset, col_offset + self._mesh._N))
		# 					H_integral_sum_flag.append(False)
		# 				else:
		# 					col_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_col
		# 					row_numbers = [row_offset]
		# 					col_numbers = [col_offset]
		# 					H_integral_sum_flag.append(True)
		# 				H_integral_nonzero_row.extend(row_numbers)
		# 				H_integral_nonzero_col.extend(col_numbers)

		# 	num_H_integral_nonzero = len(H_integral_nonzero_row)
		# 	sH_integral_matrix = sparse.coo_matrix(([1]*num_H_integral_nonzero, (H_integral_nonzero_row, H_integral_nonzero_col)), shape=self._H_shape).tocsr().tocoo()
		# 	sH_integral_indices = list(zip(sH_integral_matrix.row, sH_integral_matrix.col))
		# 	return sH_integral_matrix, sH_integral_indices, H_integral_sum_flag

		# def hessian_endpoint_sparsity():
		# 	H_endpoint_nonzero_row = []
		# 	H_endpoint_nonzero_col = []

		# 	if len(self._ocp._expression_graph.ddL_b_dxbdxb) > 0:
		# 		ddL_b_dxbdxb_nonzero = sym.zeros(*self._ocp._expression_graph.ddL_b_dxbdxb[0].shape)
		# 		for ddL_b_dxbdxb in self._ocp._expression_graph.ddL_b_dxbdxb:
		# 			ddL_b_dxbdxb = ddL_b_dxbdxb.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			ddL_b_dxbdxb_nonzero += ddL_b_dxbdxb
		# 	else:
		# 		raise NotImplementedError

		# 	for i_row in range(self._ocp._num_point_vars):
		# 		row = ddL_b_dxbdxb_nonzero[i_row, :i_row+1]
		# 		if i_row < 2*self._ocp._num_y_vars:
		# 			if i_row % 2:
		# 				row_offset = int((i_row+1)/2) * self._mesh._N - 1
		# 			else:
		# 				row_offset = int(i_row/2) * self._mesh._N
		# 		else:
		# 			row_offset = self._ocp._yu_qts_split * self._mesh._N + (i_row - 2*self._ocp._num_y_vars)
		# 		for i_col, entry in enumerate(row):
		# 			entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
		# 			if entry != 0:
		# 				if i_col < 2*self._ocp._num_y_vars:
		# 					if i_col % 2:
		# 						col_offset = int((i_col+1)/2) * self._mesh._N - 1
		# 					else:
		# 						col_offset = int(i_col/2) * self._mesh._N
		# 				else:
		# 					col_offset = self._ocp._yu_qts_split * self._mesh._N + (i_col - 2*self._ocp._num_y_vars)
		# 				H_endpoint_nonzero_row.append(row_offset)
		# 				H_endpoint_nonzero_col.append(col_offset)

		# 	num_H_endpoint_nonzero = len(H_endpoint_nonzero_row)
		# 	sH_endpoint_matrix = sparse.coo_matrix(([1]*num_H_endpoint_nonzero, (H_endpoint_nonzero_row, H_endpoint_nonzero_col)), shape=self._H_shape)
		# 	sH_endpoint_indices = list(zip(sH_endpoint_matrix.row, sH_endpoint_matrix.col))
		# 	return sH_endpoint_matrix, sH_endpoint_indices

		# def reorder_full_hessian_sparsity(sH):

		# 	row_old = sH_matrix.row
		# 	col_old = sH_matrix.col

		# 	row_new = []
		# 	col_new = []

		# 	row_temp = []
		# 	col_temp = []

		# 	for r, c in zip(row_old, col_old):
		# 		max_row_temp = max(row_temp) if row_temp else r
		# 		if r < self._yu_qts_split:
		# 			if (r % self._mesh._N) == 0 and r > max_row_temp:
		# 				ind_sorted = np.argsort(col_temp)
		# 				row_sorted = np.array(row_temp)[ind_sorted]
		# 				col_sorted = np.array(col_temp)[ind_sorted]
		# 				row_new.extend(row_sorted.tolist())
		# 				col_new.extend(col_sorted.tolist())
		# 				row_temp = []
		# 				col_temp = []
		# 			row_temp.append(r)
		# 			col_temp.append(c)
		# 		else:
		# 			if row_temp or col_temp:
		# 				ind_sorted = np.argsort(col_temp)
		# 				row_sorted = np.array(row_temp)[ind_sorted]
		# 				col_sorted = np.array(col_temp)[ind_sorted]
		# 				row_new.extend(row_sorted.tolist())
		# 				col_new.extend(col_sorted.tolist())
		# 				row_temp = []
		# 				col_temp = []
		# 			row_new.append(r)
		# 			col_new.append(c)

		# 	return row_new, col_new

		# if self._ocp.settings.derivative_level == 2:
		# 	sH_objective_matrix, sH_objective_indices = hessian_objective_sparsity()
		# 	sH_defect_matrix, sH_defect_indices, H_defect_sum_flag = hessian_defect_sparsity()
		# 	sH_path_matrix, sH_path_indices = hessian_path_sparsity()
		# 	sH_integral_matrix, sH_integral_indices, H_integral_sum_flag = hessian_integral_sparsity()
		# 	sH_endpoint_matrix, sH_endpoint_indices = hessian_endpoint_sparsity()			

		# 	sH_matrix = (sH_objective_matrix + sH_defect_matrix + sH_path_matrix + sH_integral_matrix + sH_endpoint_matrix).tocoo()
		# 	sH_indices = list(zip(sH_matrix.row, sH_matrix.col))
		# 	sH_row_reordered, sH_col_reordered = reorder_full_hessian_sparsity(sH_matrix)
		# 	sH_indices_reordered = list(zip(sH_row_reordered, sH_col_reordered))

		# 	swap_indices = [sH_indices.index(ind) 
		# 		for ind in sH_indices_reordered]

		# 	H_objective_indices = []
		# 	H_defect_indices = []
		# 	H_path_indices = []
		# 	H_integral_indices = []
		# 	H_endpoint_indices = []
		# 	for i, pair in zip(swap_indices, sH_indices):
		# 		if pair in sH_objective_indices:
		# 			H_objective_indices.append(i)
		# 		if pair in sH_defect_indices:
		# 			H_defect_indices.append(i)
		# 		if pair in sH_path_indices:
		# 			H_path_indices.append(i)
		# 		if pair in sH_integral_indices:
		# 			H_integral_indices.append(i)
		# 		if pair in sH_endpoint_indices:
		# 			H_endpoint_indices.append(i)

		# 	self._H_nonzero_row = tuple(sH_matrix.row)
		# 	self._H_nonzero_col = tuple(sH_matrix.col)
		# 	self._num_H_nonzero = len(self._H_nonzero_row)

		# 	print('Full Hessian sparsity computed.')

		# def reshape_lagrange(lagrange):
		# 	lagrange = np.array(lagrange)
		# 	zeta_lagrange = lagrange[self._c_defect_slice].reshape((self._ocp._num_y_vars, self._mesh._num_c_boundary_per_y))
		# 	gamma_lagrange = lagrange[self._c_path_slice].reshape((self._ocp.number_path_constraints, self._mesh._N))
		# 	rho_lagrange = lagrange[self._c_integral_slice].reshape(-1, )
		# 	beta_lagrange = lagrange[self._c_boundary_slice].reshape(-1, )
		# 	return tuple([*zeta_lagrange]), tuple([*gamma_lagrange]), tuple([*rho_lagrange]), tuple([*beta_lagrange])

		# def hessian(x, lagrange, obj_factor):
		# 	x_tuple = reshape_x(x)
		# 	x_tuple_point = reshape_x_point(x)
		# 	zeta_lagrange, gamma_lagrange, rho_lagrange, beta_lagrange = reshape_lagrange(lagrange)
		# 	H = self._ocp._H_lambda(x_tuple, x_tuple_point, obj_factor, zeta_lagrange, gamma_lagrange, rho_lagrange, beta_lagrange, self._mesh._N, self._num_H_nonzero, H_objective_indices, H_defect_indices, H_path_indices, H_integral_indices, H_endpoint_indices, self._mesh._sA_matrix, self._mesh._W_matrix, H_defect_sum_flag, H_integral_sum_flag)
		# 	return H

		# def hessian_structure():
		# 	return (self._H_nonzero_row, self._H_nonzero_col)

		# if self._ocp.settings.derivative_level == 2:
		# 	self._hessian_lambda = hessian
		# 	self._hessian_structure_lambda = hessian_structure
		# print('IPOPT functions compiled.')

	

		# # ========================================================
		# # PROFILE
		# # ========================================================
		# if False:
		# 	print('\n\n\n')
		# 	num_loops = 100
		# 	for i in range(num_loops):
		# 		x_data = np.random.rand(self._num_x)
		# 		lagrange = np.random.rand(self._num_c)
		# 		obj_factor = np.random.rand()
		# 		J = self._objective_lambda(x_data)
		# 		g = self._gradient_lambda(x_data)
		# 		c = self._constraint_lambda(x_data)
		# 		G = self._jacobian_lambda(x_data)
		# 		G_struct = self._jacobian_structure_lambda()

		# 		if self._ocp.settings.derivative_level == 2:
		# 			H = self._hessian_lambda(x_data, lagrange, obj_factor)
		# 			H_struct = self._hessian_structure_lambda()

		# 	print('\n\n\n')
		# 	raise ValueError

		# # ========================================================

		# # Initialise the NLP problem
		# self._initialise_nlp()

		# initialisation_time_stop = timer()

		# self._initialisation_time = initialisation_time_stop - initialisation_time_start






