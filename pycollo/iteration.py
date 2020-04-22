import collections
import csv
import itertools
from timeit import default_timer as timer
import time

import ipopt
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import sympy as sym

from pycollo.guess import Guess
from pycollo.mesh import Mesh
from pycollo.scaling import IterationScaling
from .utils import console_out

class Iteration:

	# def __init__(self, optimal_control_problem=None, iteration_number=None, *, mesh=None, guess=None):
	def __init__(self, backend, index, mesh, guess):

		self.backend = backend
		self._ocp = self.backend.ocp

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
		return self._ocp

	@property
	def mesh(self):
		return self._mesh
	
	# @property
	# def guess(self):
	# 	return self._guess
	
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

		if self.backend.ocp.settings.derivative_level == 1:
			self._hessian_lambda = None
			self._hessian_structure_lambda = None

		if self.backend.ocp.settings.nlp_solver == "ipopt":

			self._ipopt_problem = IPOPTProblem(
				self._objective_lambda, 
				self._gradient_lambda, 
				self._constraint_lambda, 
				self._jacobian_lambda, 
				self._jacobian_structure_lambda,
				self._hessian_lambda,
				self._hessian_structure_lambda,
				)

			self._nlp_problem = ipopt.problem(
				n=self.num_x,
				m=self.num_c,
				problem_obj=self._ipopt_problem,
				lb=self._x_bnd_l,
				ub=self._x_bnd_u,
				cl=self._c_bnd_l,
				cu=self._c_bnd_u)

			self._nlp_problem.addOption('mu_strategy', 'adaptive')
			self._nlp_problem.addOption('tol', self._ocp._settings.nlp_tolerance)
			self._nlp_problem.addOption('max_iter', self._ocp._settings.max_nlp_iterations)
			self._nlp_problem.addOption('print_level', 5)
			# self._nlp_problem.addOption('nlp_scaling_method', 'user-scaling')

			# self._nlp_problem.setProblemScaling(
			# 	self.scaling.obj_scaling, 
			# 	self.scaling.x_scaling,
			# 	self.scaling.c_scaling,
			# 	)

		else:
			raise NotImplementedError

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

			G = self._jacobian_lambda(x_data)
			print(f"G:\n{G}\n")

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
		next_iter_guess = Guess(optimal_control_problem=self._ocp, time=self._solution.time, state=self._solution.state, control=self._solution.control, integral=self._solution.integral, parameter=self._solution.parameter)
		_ = next_iter_guess._mesh_refinement_bypass_init()

		process_results_time_stop = timer()

		_ = self._display_mesh_iteration_info(next_iter_mesh)

		self._process_results_time = process_results_time_stop - process_results_time_start

		return next_iter_mesh, next_iter_guess





















	def _initialise_iteration(self, prev_guess):

		# _ = self._display_mesh_iteration()

		# initialisation_time_start = timer()

		# def interpolate_to_new_mesh(num_vars, prev):
		# 	new_guess = np.empty((num_vars, self._mesh._N))
		# 	for index, row in enumerate(prev):
		# 		interp_func = interpolate.interp1d(prev_guess._tau, row)
		# 		new_guess[index, :] = interp_func(self._guess._tau)
		# 	return new_guess

		# # Mesh
		# self._mesh._generate_mesh()

		# # Guess
		# self._guess = Guess(
		# 	optimal_control_problem=self._ocp)
		# self._guess._iteration = self
		# self._guess._mesh = self._mesh
		# self._mesh._guess = self._guess
		# self._guess._tau = self._mesh._tau
		# self._guess._t0 = prev_guess._t0
		# self._guess._tF = prev_guess._tF
		# self._guess._stretch = 0.5 * (self._guess._tF - self._guess._t0)
		# self._guess._shift = 0.5 * (self._guess._t0 + self._guess._tF)
		# self._guess._time = (self._mesh._tau * self._guess._stretch) + self._guess._shift
		# self._guess._y = interpolate_to_new_mesh(self._ocp._num_y_vars, prev_guess._y) if self._ocp._num_y_vars else np.array([])
		# self._guess._u = interpolate_to_new_mesh(self._ocp._num_u_vars, prev_guess._u) if self._ocp._num_u_vars else np.array([])
		# self._guess._q = prev_guess._q
		# self._guess._t = prev_guess._t
		# self._guess._s = prev_guess._s
		# print('Guess interpolated to iteration mesh.')

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
		self._yu_qts_split = self._yu_slice.stop

		# Constraints
		self._num_c_defect = self._ocp.number_state_equations * self._mesh._num_c_boundary_per_y
		self._num_c_path = self._ocp.number_path_constraints * self._mesh._N
		self._num_c_integral = self._ocp.number_integrand_functions
		self._num_c_boundary = self._ocp.number_state_endpoint_constraints + self._ocp.number_endpoint_constraints
		self._num_c = self._num_c_defect + self._num_c_path + self._num_c_integral + self._num_c_boundary

		self._c_lambda_dy_slice = slice(0, self._num_y)
		self._c_lambda_p_slice = slice(self._c_lambda_dy_slice.stop, self._c_lambda_dy_slice.stop + self._num_c_path*self._mesh._N)
		self._c_lambda_g_slice = slice(self._c_lambda_p_slice.stop, self._c_lambda_p_slice.stop + self._num_c_integral*self._mesh._N)

		self._c_defect_slice = slice(0, self._num_c_defect)
		self._c_path_slice = slice(self._c_defect_slice.stop, self._c_defect_slice.stop + self._num_c_path)
		self._c_integral_slice = slice(self._c_path_slice.stop, self._c_path_slice.stop + self._num_c_integral)
		self._c_boundary_slice = slice(self._c_integral_slice.stop, self._num_c)

		self._G_shape = (self._num_c, self._num_x)
		self._H_shape = (self._num_x, self._num_x)

		# # Jacobian
		# G_nonzero_row = []
		# G_nonzero_col = []
		# dzeta_dy_D_nonzero = []

		# A_row_col_array = np.vstack(self._mesh._sA_matrix.nonzero())
		# A_ind_array = self._mesh._A_index_array
		# D_ind_array = self._mesh._D_index_array

		# # Defect constraints by state variables
		# for i_c in range(self._ocp._num_y_vars):
		# 	for i_y in range(self._ocp._num_y_vars):
		# 		row_offset = i_c * self._mesh._num_c_boundary_per_y
		# 		col_offset = i_y * self._mesh._N
		# 		ind_offset = len(G_nonzero_row)
		# 		G_nonzero_row.extend(list(A_row_col_array[0] + row_offset))
		# 		G_nonzero_col.extend(list(A_row_col_array[1] + col_offset))
		# 		if i_c == i_y:
		# 			dzeta_dy_D_nonzero.extend(list(D_ind_array + ind_offset))
		# dzeta_dy_slice = slice(0, len(G_nonzero_row))

		# # Defect constraints by control variables
		# for i_c in range(self._ocp._num_y_vars):
		# 	for i_u in range(self._ocp._num_u_vars):
		# 		row_offset = i_c * self._mesh._num_c_boundary_per_y
		# 		col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
		# 		G_nonzero_row.extend(list(A_row_col_array[0] + row_offset))
		# 		G_nonzero_col.extend(list(A_row_col_array[1] + col_offset))
		# dzeta_du_slice = slice(dzeta_dy_slice.stop, len(G_nonzero_row))

		# # Defect constraints by time variables
		# num_rows = self._ocp._num_y_vars * self._mesh._num_c_boundary_per_y
		# num_cols = self._ocp._num_t_vars
		# col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars
		# row_indices = list(range(num_rows))
		# col_indices = list(range(col_offset, col_offset+num_cols))
		# G_nonzero_row.extend(np.repeat(row_indices, num_cols))
		# G_nonzero_col.extend(np.tile(col_indices, num_rows))
		# dzeta_dt_slice = slice(dzeta_du_slice.stop, len(G_nonzero_row))

		# # Defect constraint by parameter variables
		# num_rows = self._ocp._num_y_vars * self._mesh._num_c_boundary_per_y
		# num_cols = self._ocp._num_s_vars
		# col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars + self._ocp._num_t_vars
		# row_indices = list(range(num_rows))
		# col_indices = list(range(col_offset, col_offset+num_cols))
		# G_nonzero_row.extend(np.repeat(row_indices, num_cols))
		# G_nonzero_col.extend(np.tile(col_indices, num_rows))
		# dzeta_ds_slice = slice(dzeta_dt_slice.stop, len(G_nonzero_row))

		# # Path constraints by state variables
		# for i_c in range(self._ocp.number_path_constraints):
		# 	for i_y in range(self._ocp._num_y_vars):
		# 		row_offset = self._ocp.number_state_equations*self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = i_y * self._mesh._N
		# 		G_nonzero_row.extend(list(range(row_offset, self._mesh._N + row_offset)))
		# 		G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		# dgamma_dy_slice = slice(dzeta_ds_slice.stop, len(G_nonzero_row))

		# # Path constraints by control variables
		# for i_c in range(self._ocp.number_path_constraints):
		# 	for i_u in range(self._ocp._num_u_vars):
		# 		row_offset = self._ocp.number_state_equations*self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
		# 		G_nonzero_row.extend(list(range(row_offset, self._mesh._N + row_offset)))
		# 		G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		# dgamma_du_slice = slice(dgamma_dy_slice.stop, len(G_nonzero_row))

		# # Path constraints by time variables
		# num_rows = self._ocp.number_path_constraints * self._mesh._N
		# num_cols = self._ocp._num_t_vars
		# row_offset = self._ocp._num_y_vars * self._mesh._num_c_boundary_per_y
		# col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars
		# row_indices = list(range(row_offset, row_offset+num_rows))
		# col_indices = list(range(col_offset, col_offset+num_cols))
		# G_nonzero_row.extend(np.repeat(row_indices, num_cols))
		# G_nonzero_col.extend(np.tile(col_indices, num_rows))
		# dgamma_dt_slice = slice(dgamma_du_slice.stop, len(G_nonzero_row))

		# # Path constraints by parameter variables
		# num_rows = self._ocp.number_path_constraints * self._mesh._N
		# num_cols = self._ocp._num_s_vars
		# row_offset = self._ocp._num_y_vars * self._mesh._num_c_boundary_per_y
		# col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars + self._ocp._num_t_vars
		# row_indices = list(range(row_offset, row_offset+num_rows))
		# col_indices = list(range(col_offset, col_offset+num_cols))
		# G_nonzero_row.extend(np.repeat(row_indices, num_cols))
		# G_nonzero_col.extend(np.tile(col_indices, num_rows))
		# dgamma_ds_slice = slice(dgamma_dt_slice.stop, len(G_nonzero_row))

		# # Integral constraints by state variables
		# for i_c in range(self._ocp._num_q_vars):
		# 	for i_y in range(self._ocp._num_y_vars):
		# 		row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = i_y * self._mesh._N
		# 		G_nonzero_row.extend(list(row_offset*np.ones(self._mesh._N, dtype=int)))
		# 		G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		# drho_dy_slice = slice(dgamma_ds_slice.stop, len(G_nonzero_row))

		# # Integral constraints by control variables
		# for i_c in range(self._ocp._num_q_vars):
		# 	for i_u in range(self._ocp._num_u_vars):
		# 		row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = (self._ocp._num_y_vars + i_u) * self._mesh._N
		# 		G_nonzero_row.extend(list(row_offset*np.ones(self._mesh._N, dtype=int)))
		# 		G_nonzero_col.extend(list(range(col_offset, self._mesh._N + col_offset)))
		# drho_du_slice = slice(drho_dy_slice.stop, len(G_nonzero_row))

		# # Integral constraints by integral variables
		# for i_c in range(self._ocp._num_q_vars):
		# 	row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + i_c
		# 	col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + i_c
		# 	G_nonzero_row.extend(list(row_offset*np.ones(self._ocp._num_q_vars, dtype=int)))
		# 	G_nonzero_col.extend(list(range(col_offset, self._ocp._num_q_vars + col_offset)))
		# drho_dq_slice = slice(drho_du_slice.stop, len(G_nonzero_row))

		# # Integral constraints by time variables
		# for i_c in range(self._ocp._num_q_vars):
		# 	for i_t in range(self._ocp._num_t_vars):
		# 		row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars + i_t
		# 		G_nonzero_row.append(row_offset)
		# 		G_nonzero_col.append(col_offset)
		# drho_dt_slice = slice(drho_dq_slice.stop, len(G_nonzero_row))

		# # Integral constraints by parameter variables
		# for i_c in range(self._ocp._num_q_vars):
		# 	for i_s in range(self._ocp._num_s_vars):
		# 		row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + i_c
		# 		col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + self._ocp._num_q_vars + self._ocp._num_t_vars + i_s
		# 		G_nonzero_row.append(row_offset)
		# 		G_nonzero_col.append(col_offset)
		# drho_ds_slice = slice(drho_dt_slice.stop, len(G_nonzero_row))

		# # Boundary constraints
		# for i_c in range(self._ocp.number_state_endpoint_constraints + self._ocp.number_endpoint_constraints):
		# 	for i_y in range(self._ocp._num_y_vars):
		# 		row_offset = [self._ocp.number_state_equations*self._mesh._num_c_boundary_per_y + self._ocp.number_path_constraints*self._mesh._N + self._ocp._num_q_vars + i_c]*2
		# 		col_offset = [i_y * self._mesh._N, (i_y+1) * self._mesh._N - 1]
		# 		G_nonzero_row.extend(row_offset)
		# 		G_nonzero_col.extend(col_offset)
		# 	for i_qts in range(self._ocp._num_q_vars + self._ocp._num_t_vars + self._ocp._num_s_vars):
		# 		row_offset = (self._ocp.number_state_equations + self._ocp.number_path_constraints) * self._mesh._num_c_boundary_per_y + self._ocp.number_integrand_functions + i_c
		# 		col_offset = (self._ocp._num_y_vars + self._ocp._num_u_vars) * self._mesh._N + i_qts
		# 		G_nonzero_row.append(row_offset)
		# 		G_nonzero_col.append(col_offset)
		# dbeta_dxb_slice = slice(drho_ds_slice.stop, len(G_nonzero_row))

		# self._G_nonzero_row = tuple(G_nonzero_row)
		# self._G_nonzero_col = tuple(G_nonzero_col)
		# self._num_G_nonzero = len(self._G_nonzero_row)
		# print('Full Jacobian sparsity computed.')

		def hessian_objective_sparsity():
			H_objective_nonzero_row = []
			H_objective_nonzero_col = []

			ddL_J_dxbdxb_nonzero = self._ocp._expression_graph.ddL_J_dxbdxb

			for i_row in range(self._ocp._num_point_vars):
				row = ddL_J_dxbdxb_nonzero[i_row, :i_row+1]
				if i_row < 2*self._ocp._num_y_vars:
					if i_row % 2:
						row_offset = int((i_row+1)/2) * self._mesh._N - 1
					else:
						row_offset = int(i_row/2) * self._mesh._N
				else:
					row_offset = self._ocp._yu_qts_split * self._mesh._N + (i_row - 2*self._ocp._num_y_vars)
				for i_col, entry in enumerate(row):
					entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					if entry != 0:
						if i_col < 2*self._ocp._num_y_vars:
							if i_col % 2:
								col_offset = int((i_col+1)/2) * self._mesh._N - 1
							else:
								col_offset = int(i_col/2) * self._mesh._N
						else:
							col_offset = self._ocp._yu_qts_split * self._mesh._N + (i_col - 2*self._ocp._num_y_vars)
						H_objective_nonzero_row.append(row_offset)
						H_objective_nonzero_col.append(col_offset)

			num_H_objective_nonzero = len(H_objective_nonzero_row)
			sH_objective_matrix = sparse.coo_matrix(([1]*num_H_objective_nonzero, (H_objective_nonzero_row, H_objective_nonzero_col)), shape=self._H_shape)
			sH_objective_indices = list(zip(sH_objective_matrix.row, sH_objective_matrix.col))
			return sH_objective_matrix, sH_objective_indices

		def hessian_defect_sparsity():
			H_defect_nonzero_row = []
			H_defect_nonzero_col = []
			H_defect_sum_flag = []

			ddL_zeta_dxdx_nonzero = self._ocp._expression_graph.ddL_zeta_dxdx[0]
			for matrix in self._ocp._expression_graph.ddL_zeta_dxdx[1:]:
				ddL_zeta_dxdx_nonzero += matrix
			
			for i_row in range(self._ocp._num_vars):
				row = ddL_zeta_dxdx_nonzero[i_row, :i_row+1]
				if i_row < self._ocp._yu_qts_split:
					row_offset = i_row * self._mesh._N
					row_numbers = list(range(row_offset, row_offset + self._mesh._N))
				else:
					row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
					row_numbers = [row_offset]*self._mesh._N
				for i_col, entry in enumerate(row):
					entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					if entry != 0:
						if i_col < self._ocp._yu_qts_split:
							col_offset = i_col * self._mesh._N
							col_numbers = list(range(col_offset, col_offset + self._mesh._N))
							H_defect_sum_flag.append(False)
						else:
							col_offset = self._ocp._yu_qts_split* (self._mesh._N - 1) + i_col
							row_numbers = [row_offset]
							col_numbers = [col_offset]
							H_defect_sum_flag.append(True)
						H_defect_nonzero_row.extend(row_numbers)
						H_defect_nonzero_col.extend(col_numbers)

			num_H_defect_nonzero = len(H_defect_nonzero_row)
			sH_defect_matrix = sparse.coo_matrix(([1]*num_H_defect_nonzero, (H_defect_nonzero_row, H_defect_nonzero_col)), shape=self._H_shape)
			sH_defect_matrix = sH_defect_matrix.tocsr().tocoo()
			sH_defect_indices = list(zip(sH_defect_matrix.row, sH_defect_matrix.col))
			return sH_defect_matrix, sH_defect_indices, H_defect_sum_flag

		def hessian_path_sparsity():
			H_path_nonzero_row = []
			H_path_nonzero_col = []
			H_path_sum_flag = []

			try:
				ddL_gamma_dxdx_nonzero = self._ocp._expression_graph.ddL_gamma_dxdx[0]
			except IndexError:
				raise NotImplementedError
			else:
				for matrix in self._ocp._expression_graph.ddL_gamma_dxdx[1:]:
					ddL_gamma_dxdx_nonzero += matrix

			for i_row in range(self._ocp._num_vars):
				row = ddL_gamma_dxdx_nonzero[i_row, :i_row+1]
				if i_row < self._ocp._yu_qts_split:
					row_offset = i_row * self._mesh._N
					row_numbers = list(range(row_offset, row_offset + self._mesh._N))
				else:
					row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
					row_numbers = [row_offset]*self._mesh._N
				for i_col, entry in enumerate(row):
					entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					if entry != 0:
						if i_col < self._ocp._yu_qts_split:
							col_offset = i_col * self._mesh._N
							col_numbers = list(range(col_offset, col_offset + self._mesh._N))
						else:
							col_offset = self._ocp._yu_qts_split* (self._mesh._N - 1) + i_col
							row_numbers = [row_offset]
							col_numbers = [col_offset]
						H_path_nonzero_row.extend(row_numbers)
						H_path_nonzero_col.extend(col_numbers)

			num_H_path_nonzero = len(H_path_nonzero_row)
			sH_path_matrix = sparse.coo_matrix(([1]*num_H_path_nonzero, (H_path_nonzero_row, H_path_nonzero_col)), shape=self._H_shape).tocsr().tocoo()
			sH_path_indices = list(zip(sH_path_matrix.row, sH_path_matrix.col))
			return sH_path_matrix, sH_path_indices

		def hessian_integral_sparsity():
			H_integral_nonzero_row = []
			H_integral_nonzero_col = []
			H_integral_sum_flag = []

			ddL_rho_dxdx_nonzero = self._ocp._expression_graph.ddL_rho_dxdx[0]
			for matrix in self._ocp._expression_graph.ddL_rho_dxdx[1:]:
				ddL_rho_dxdx_nonzero += matrix
			
			for i_row in range(self._ocp._num_vars):
				row = ddL_rho_dxdx_nonzero[i_row, :i_row+1]
				if i_row < self._ocp._yu_qts_split:
					row_offset = i_row * self._mesh._N
					row_numbers = list(range(row_offset, row_offset + self._mesh._N))
				else:
					row_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_row
					row_numbers = [row_offset]*self._mesh._N
				for i_col, entry in enumerate(row):
					entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					if entry != 0:
						if i_col < self._ocp._yu_qts_split:
							col_offset = i_col * self._mesh._N
							col_numbers = list(range(col_offset, col_offset + self._mesh._N))
							H_integral_sum_flag.append(False)
						else:
							col_offset = self._ocp._yu_qts_split * (self._mesh._N - 1) + i_col
							row_numbers = [row_offset]
							col_numbers = [col_offset]
							H_integral_sum_flag.append(True)
						H_integral_nonzero_row.extend(row_numbers)
						H_integral_nonzero_col.extend(col_numbers)

			num_H_integral_nonzero = len(H_integral_nonzero_row)
			sH_integral_matrix = sparse.coo_matrix(([1]*num_H_integral_nonzero, (H_integral_nonzero_row, H_integral_nonzero_col)), shape=self._H_shape).tocsr().tocoo()
			sH_integral_indices = list(zip(sH_integral_matrix.row, sH_integral_matrix.col))
			return sH_integral_matrix, sH_integral_indices, H_integral_sum_flag

		def hessian_endpoint_sparsity():
			H_endpoint_nonzero_row = []
			H_endpoint_nonzero_col = []

			if len(self._ocp._expression_graph.ddL_b_dxbdxb) > 0:
				ddL_b_dxbdxb_nonzero = sym.zeros(*self._ocp._expression_graph.ddL_b_dxbdxb[0].shape)
				for ddL_b_dxbdxb in self._ocp._expression_graph.ddL_b_dxbdxb:
					ddL_b_dxbdxb = ddL_b_dxbdxb.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					ddL_b_dxbdxb_nonzero += ddL_b_dxbdxb
			else:
				raise NotImplementedError

			for i_row in range(self._ocp._num_point_vars):
				row = ddL_b_dxbdxb_nonzero[i_row, :i_row+1]
				if i_row < 2*self._ocp._num_y_vars:
					if i_row % 2:
						row_offset = int((i_row+1)/2) * self._mesh._N - 1
					else:
						row_offset = int(i_row/2) * self._mesh._N
				else:
					row_offset = self._ocp._yu_qts_split * self._mesh._N + (i_row - 2*self._ocp._num_y_vars)
				for i_col, entry in enumerate(row):
					entry = entry.subs({self._ocp._expression_graph._zero_node.symbol: 0})
					if entry != 0:
						if i_col < 2*self._ocp._num_y_vars:
							if i_col % 2:
								col_offset = int((i_col+1)/2) * self._mesh._N - 1
							else:
								col_offset = int(i_col/2) * self._mesh._N
						else:
							col_offset = self._ocp._yu_qts_split * self._mesh._N + (i_col - 2*self._ocp._num_y_vars)
						H_endpoint_nonzero_row.append(row_offset)
						H_endpoint_nonzero_col.append(col_offset)

			num_H_endpoint_nonzero = len(H_endpoint_nonzero_row)
			sH_endpoint_matrix = sparse.coo_matrix(([1]*num_H_endpoint_nonzero, (H_endpoint_nonzero_row, H_endpoint_nonzero_col)), shape=self._H_shape)
			sH_endpoint_indices = list(zip(sH_endpoint_matrix.row, sH_endpoint_matrix.col))
			return sH_endpoint_matrix, sH_endpoint_indices

		def reorder_full_hessian_sparsity(sH):

			row_old = sH_matrix.row
			col_old = sH_matrix.col

			row_new = []
			col_new = []

			row_temp = []
			col_temp = []

			for r, c in zip(row_old, col_old):
				max_row_temp = max(row_temp) if row_temp else r
				if r < self._yu_qts_split:
					if (r % self._mesh._N) == 0 and r > max_row_temp:
						ind_sorted = np.argsort(col_temp)
						row_sorted = np.array(row_temp)[ind_sorted]
						col_sorted = np.array(col_temp)[ind_sorted]
						row_new.extend(row_sorted.tolist())
						col_new.extend(col_sorted.tolist())
						row_temp = []
						col_temp = []
					row_temp.append(r)
					col_temp.append(c)
				else:
					if row_temp or col_temp:
						ind_sorted = np.argsort(col_temp)
						row_sorted = np.array(row_temp)[ind_sorted]
						col_sorted = np.array(col_temp)[ind_sorted]
						row_new.extend(row_sorted.tolist())
						col_new.extend(col_sorted.tolist())
						row_temp = []
						col_temp = []
					row_new.append(r)
					col_new.append(c)

			return row_new, col_new

		if self._ocp.settings.derivative_level == 2:
			sH_objective_matrix, sH_objective_indices = hessian_objective_sparsity()
			sH_defect_matrix, sH_defect_indices, H_defect_sum_flag = hessian_defect_sparsity()
			sH_path_matrix, sH_path_indices = hessian_path_sparsity()
			sH_integral_matrix, sH_integral_indices, H_integral_sum_flag = hessian_integral_sparsity()
			sH_endpoint_matrix, sH_endpoint_indices = hessian_endpoint_sparsity()			

			sH_matrix = (sH_objective_matrix + sH_defect_matrix + sH_path_matrix + sH_integral_matrix + sH_endpoint_matrix).tocoo()
			sH_indices = list(zip(sH_matrix.row, sH_matrix.col))
			sH_row_reordered, sH_col_reordered = reorder_full_hessian_sparsity(sH_matrix)
			sH_indices_reordered = list(zip(sH_row_reordered, sH_col_reordered))

			swap_indices = [sH_indices.index(ind) 
				for ind in sH_indices_reordered]

			H_objective_indices = []
			H_defect_indices = []
			H_path_indices = []
			H_integral_indices = []
			H_endpoint_indices = []
			for i, pair in zip(swap_indices, sH_indices):
				if pair in sH_objective_indices:
					H_objective_indices.append(i)
				if pair in sH_defect_indices:
					H_defect_indices.append(i)
				if pair in sH_path_indices:
					H_path_indices.append(i)
				if pair in sH_integral_indices:
					H_integral_indices.append(i)
				if pair in sH_endpoint_indices:
					H_endpoint_indices.append(i)

			self._H_nonzero_row = tuple(sH_matrix.row)
			self._H_nonzero_col = tuple(sH_matrix.col)
			self._num_H_nonzero = len(self._H_nonzero_row)

			print('Full Hessian sparsity computed.')

		# Lambda to prepare x from IPOPT for numba funcs
		def unscale_x(x):
			# x = self.scaling.x_stretch*(x - self.scaling.x_shift)
			return x

		def reshape_x(x):
			x = unscale_x(x)
			num_yu = self._ocp._num_y_vars + self._ocp._num_u_vars
			yu_qts_split = self._q_slice.start
			x_tuple = self._ocp._x_reshape_lambda(x, num_yu, yu_qts_split)
			return x_tuple

		self._reshape_x = reshape_x

		self._x_endpoint_indices = []
		N = self._mesh._N
		for i in range(self._ocp._num_y_vars):
			self._x_endpoint_indices.append(i*N)
			self._x_endpoint_indices.append((i+1)*N - 1)
		self._x_endpoint_indices.extend(list(range(self._q_slice.start, self._s_slice.stop)))

		def reshape_x_point(x):
			x = unscale_x(x)
			return self._ocp._x_reshape_lambda_point(x, self._x_endpoint_indices)

		# Generate objective function lambda
		def objective(x):
			x_tuple_point = reshape_x_point(x)
			J = self._ocp._J_lambda(x_tuple_point)
			return J

		self._objective_lambda = objective

		# Generate objective function gradient lambda
		def gradient(x):
			x_tuple_point = reshape_x_point(x)
			g = self._ocp._g_lambda(x_tuple_point, self._mesh._N)
			return g

		self._gradient_lambda = gradient

		# Generate constraint lambdas
		def constraint(x):
			x_tuple = reshape_x(x)
			x_tuple_point = reshape_x_point(x)
			c = self._ocp._c_lambda(x_tuple, x_tuple_point, self._mesh._N, self._ocp._y_slice, self._ocp._q_slice, self._num_c, self._c_lambda_dy_slice, self._c_lambda_p_slice, self._c_lambda_g_slice, self._c_defect_slice, self._c_path_slice, self._c_integral_slice, self._c_boundary_slice, self._mesh._sA_matrix, self._mesh._sD_matrix, self._mesh._W_matrix)
			return c

		self._constraint_lambda = constraint

		OCPNumX = collections.namedtuple('OCPNumX', ['y', 'u', 'q', 't', 's'])
		ocp_num_x = OCPNumX(y=self._ocp._num_y_vars, u=self._ocp._num_u_vars, q=self._ocp._num_q_vars, t=self._ocp._num_t_vars, s=self._ocp._num_s_vars)

		def jacobian(x):
			x_tuple = reshape_x(x)
			x_tuple_point = reshape_x_point(x)
			G = self._ocp._G_lambda(x_tuple, x_tuple_point, self._mesh._N, self._num_G_nonzero, ocp_num_x, self._mesh._sA_matrix, self._mesh._sD_matrix, self._mesh._W_matrix, A_row_col_array, self._c_lambda_dy_slice, self._c_lambda_p_slice, self._c_lambda_g_slice, dzeta_dy_D_nonzero, dzeta_dy_slice, dzeta_du_slice, dzeta_dt_slice, dzeta_ds_slice, dgamma_dy_slice, dgamma_du_slice, dgamma_dt_slice, dgamma_ds_slice, drho_dy_slice, drho_du_slice, drho_dq_slice, drho_dt_slice, drho_ds_slice, dbeta_dxb_slice)
			return G

		self._jacobian_lambda = jacobian

		def jacobian_structure():
			return (self._G_nonzero_row, self._G_nonzero_col)

		self._jacobian_structure_lambda = jacobian_structure

		def reshape_lagrange(lagrange):
			lagrange = np.array(lagrange)
			zeta_lagrange = lagrange[self._c_defect_slice].reshape((self._ocp._num_y_vars, self._mesh._num_c_boundary_per_y))
			gamma_lagrange = lagrange[self._c_path_slice].reshape((self._ocp.number_path_constraints, self._mesh._N))
			rho_lagrange = lagrange[self._c_integral_slice].reshape(-1, )
			beta_lagrange = lagrange[self._c_boundary_slice].reshape(-1, )
			return tuple([*zeta_lagrange]), tuple([*gamma_lagrange]), tuple([*rho_lagrange]), tuple([*beta_lagrange])

		def hessian(x, lagrange, obj_factor):
			x_tuple = reshape_x(x)
			x_tuple_point = reshape_x_point(x)
			zeta_lagrange, gamma_lagrange, rho_lagrange, beta_lagrange = reshape_lagrange(lagrange)
			H = self._ocp._H_lambda(x_tuple, x_tuple_point, obj_factor, zeta_lagrange, gamma_lagrange, rho_lagrange, beta_lagrange, self._mesh._N, self._num_H_nonzero, H_objective_indices, H_defect_indices, H_path_indices, H_integral_indices, H_endpoint_indices, self._mesh._sA_matrix, self._mesh._W_matrix, H_defect_sum_flag, H_integral_sum_flag)
			return H

		def hessian_structure():
			return (self._H_nonzero_row, self._H_nonzero_col)

		if self._ocp.settings.derivative_level == 2:
			self._hessian_lambda = hessian
			self._hessian_structure_lambda = hessian_structure
		print('IPOPT functions compiled.')

		# Generate bounds
		self._x_bnd_l, self._x_bnd_u = self._generate_x_bounds()
		self._c_bnd_l, self._c_bnd_u = self._generate_c_bounds()
		print('Mesh-specific bounds generated.')

		# Scaling
		self.scaling = IterationScaling(self)
		self.scaling._generate()
		print('Scaling generated.\n\n')

		# Reset mesh-specific bounds for variables
		self._x_bnd_l, self._x_bnd_u = self._generate_x_bounds()

		# ========================================================
		# JACOBIAN CHECK
		# ========================================================
		if False:
			print('\n\n\n')
			x_data = np.array(range(1, self._num_x+1), dtype=float)
			lagrange = np.array(range(self._num_c), dtype=float)
			obj_factor = 2.0
			print(f"x Variables:\n{self._ocp._x_vars}\n")
			print(f"x Data:\n{x_data}\n")
			print(f"Lagrange Multipliers:\n{lagrange}\n")

			J = self._objective_lambda(x_data)
			print(f"J:\n{J}\n")

			g = self._gradient_lambda(x_data)
			print(f"g:\n{g}\n")

			c = self._constraint_lambda(x_data)
			print(f"c:\n{c}\n")

			G_struct = self._jacobian_structure_lambda()
			print(f"G Structure:\n{G_struct[0]}\n{G_struct[1]}")

			G = self._jacobian_lambda(x_data)
			print(f"G:\n{G}\n")

			if self._ocp.settings.derivative_level == 2:

				H_struct = self._hessian_structure_lambda()
				print(f"H Structure:\n{H_struct}\n")

				H = self._hessian_lambda(x_data, lagrange, obj_factor)
				print(f"H:\n{H}\n")

				sH = sparse.coo_matrix((H, (H_struct[0], H_struct[1])), shape=self._H_shape)
				sH = sparse.coo_matrix(np.tril(sH.toarray()))
				print(sH)
			
			print('\n\n\n')
			raise NotImplementedError

		# ========================================================
		# PROFILE
		# ========================================================
		if False:
			print('\n\n\n')
			num_loops = 100
			for i in range(num_loops):
				x_data = np.random.rand(self._num_x)
				lagrange = np.random.rand(self._num_c)
				obj_factor = np.random.rand()
				J = self._objective_lambda(x_data)
				g = self._gradient_lambda(x_data)
				c = self._constraint_lambda(x_data)
				G = self._jacobian_lambda(x_data)
				G_struct = self._jacobian_structure_lambda()

				if self._ocp.settings.derivative_level == 2:
					H = self._hessian_lambda(x_data, lagrange, obj_factor)
					H_struct = self._hessian_structure_lambda()

			print('\n\n\n')
			raise ValueError

		# ========================================================

		# Initialise the NLP problem
		self._initialise_nlp()

		initialisation_time_stop = timer()

		self._initialisation_time = initialisation_time_stop - initialisation_time_start

	# def _generate_x_bounds(self):

	# 	bnd_l = np.empty((self._num_x, ))
	# 	bnd_u = np.empty((self._num_x, ))

	# 	# y bounds
	# 	bnd_l[self._y_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._y_l_needed.reshape(1, -1)).flatten('F').squeeze()
	# 	bnd_u[self._y_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._y_u_needed.reshape(1, -1)).flatten('F').squeeze()

	# 	# u bounds
	# 	bnd_l[self._u_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._u_l_needed.reshape(1, -1)).flatten('F').squeeze()
	# 	bnd_u[self._u_slice] = (np.ones((self._mesh._N, 1))*self._ocp._bounds._u_u_needed.reshape(1, -1)).flatten('F').squeeze()

	# 	# q bounds
	# 	bnd_l[self._q_slice] = self._ocp._bounds._q_l_needed
	# 	bnd_u[self._q_slice] = self._ocp._bounds._q_u_needed

	# 	# t bounds
	# 	bnd_l[self._t_slice] = self._ocp._bounds._t_l_needed
	# 	bnd_u[self._t_slice] = self._ocp._bounds._t_u_needed

	# 	# s bounds
	# 	bnd_l[self._s_slice] = self._ocp._bounds._s_l_needed
	# 	bnd_u[self._s_slice] = self._ocp._bounds._s_u_needed

	# 	return bnd_l, bnd_u

	# def _generate_c_bounds(self):

	# 	bnd_l = np.zeros((self._num_c, ))
	# 	bnd_u = np.zeros((self._num_c, ))

	# 	# Path constraints bounds
	# 	bnd_l[self._c_path_slice] = (self._ocp._bounds._c_l * self._mesh._N).flatten()
	# 	bnd_u[self._c_path_slice] = (self._ocp._bounds._c_u * self._mesh._N).flatten()

	# 	# Boundary constrants bounds
	# 	bnd_l[self._c_boundary_slice] = np.concatenate((self._ocp._bounds._y_b_l, self._ocp._bounds._b_l))
	# 	bnd_u[self._c_boundary_slice] = np.concatenate((self._ocp._bounds._y_b_u, self._ocp._bounds._b_u))

	# 	return bnd_l, bnd_u

	# def _initialise_nlp(self):

	# 	if self._ocp._settings.derivative_level == 1:
	# 		self._hessian_lambda = None
	# 		self._hessian_structure_lambda = None

	# 	if self._ocp._settings._nlp_solver == 'ipopt':

	# 		self._ipopt_problem = IPOPTProblem(
	# 			self._objective_lambda, 
	# 			self._gradient_lambda, 
	# 			self._constraint_lambda, 
	# 			self._jacobian_lambda, 
	# 			self._jacobian_structure_lambda,
	# 			self._hessian_lambda,
	# 			self._hessian_structure_lambda,
	# 			)

	# 		self._nlp_problem = ipopt.problem(
	# 			n=self._num_x,
	# 			m=self._num_c,
	# 			problem_obj=self._ipopt_problem,
	# 			lb=self._x_bnd_l,
	# 			ub=self._x_bnd_u,
	# 			cl=self._c_bnd_l,
	# 			cu=self._c_bnd_u)

	# 		self._nlp_problem.addOption('mu_strategy', 'adaptive')
	# 		self._nlp_problem.addOption('tol', self._ocp._settings.nlp_tolerance)
	# 		self._nlp_problem.addOption('max_iter', self._ocp._settings.max_nlp_iterations)
	# 		self._nlp_problem.addOption('print_level', 5)
	# 		# self._nlp_problem.addOption('nlp_scaling_method', 'gradient-based')

	# 		# self._nlp_problem.setProblemScaling(
	# 		# 	self.scaling.obj_scaling, 
	# 		# 	self.scaling.x_scaling,
	# 			# self.scaling.c_scaling,
	# 			# )

	# 	else:
	# 		raise NotImplementedError

	def _solve(self):

		time_start = time.time()
		nlp_time_start = timer()

		nlp_solution, nlp_solution_info = self._nlp_problem.solve(self._guess._x)

		nlp_time_stop = timer()
		time_stop = time.time()
		self._nlp_time = nlp_time_stop - nlp_time_start

		process_results_time_start = timer()

		self._solution = Solution(self, nlp_solution, nlp_solution_info)
		next_iter_mesh = self._refine_new_mesh()
		next_iter_guess = Guess(optimal_control_problem=self._ocp, time=self._solution.time, state=self._solution.state, control=self._solution.control, integral=self._solution.integral, parameter=self._solution.parameter)
		_ = next_iter_guess._mesh_refinement_bypass_init()

		process_results_time_stop = timer()

		_ = self._display_mesh_iteration_info(next_iter_mesh)

		self._process_results_time = process_results_time_stop - process_results_time_start

		return next_iter_mesh, next_iter_guess

	def _display_mesh_iteration_info(self, next_iter_mesh):

		print(f'\n\npycollo Analysis of Mesh Iteration {self.iteration_number}:\n======================================\n')

		print(f'Objective Evaluation:       {self._solution._J}\n')
		print(f'Max Relative Mesh Error:    {np.max(self._solution._maximum_relative_error)}\n')
		if next_iter_mesh is not None:
			print(f'Adjusting Collocation Mesh: {next_iter_mesh._K} mesh sections\n')

		if self._ocp._settings.display_mesh_result_info:
			print('\n')
			print('Solution:\n=========')
			print('\n')
			print('Objective:\n----------')
			print(self._solution._J, '\n')
			print('Temporal Grid:\n--------------')
			print(self._solution._tau, '\n')
			print('State:\n------')
			print(self._solution._y, '\n')
			print('Control:\n--------')
			print(self._solution._u, '\n')
			print('Integral:\n---------')
			print(self._solution._q, '\n')
			print('Time:\n-----')
			print(self._solution._t, '\n')
			print('Parameter:\n----------')
			print(self._solution._s, '\n')

		if self._ocp._settings.display_mesh_result_graph:
			self._solution._plot_interpolated_solution(plot_y=True, plot_dy=True, plot_u=True)

	def _refine_new_mesh(self):
		_ = self._solution._patterson_rao_discretisation_mesh_error()
		next_iter_mesh = self._solution._patterson_rao_next_iteration_mesh()
		return next_iter_mesh
		

phase_solution_data_fields = ("tau", "y", "dy", "u", "q", "t", "t0", "tF", 
	"T", "stretch", "shift", "time")
PhaseSolutionData = collections.namedtuple("PhaseSolutionData", 
	phase_solution_data_fields)


class Solution:

	def __init__(self, iteration, nlp_solution, nlp_solution_info):
		self._it = iteration
		self._ocp = iteration.optimal_control_problem
		self._backend = iteration.optimal_control_problem._backend
		self._tau = iteration.mesh.tau
		
		self._nlp_solution = nlp_solution
		self._nlp_solution_info = nlp_solution_info
		self._J = self._nlp_solution_info['obj_val']
		self._x = np.array(nlp_solution)
		if self._ocp._settings._nlp_solver == 'ipopt':
			self._process_solution = self._process_ipopt_solution
		else:
			raise NotImplementedError
		self._process_solution()

	@property
	def initial_time(self):
		return self._t0

	@property
	def final_time(self):
		return self._tF
	
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
		return self._time
	
	@property
	def parameter(self):
		return self._s

	def _process_ipopt_solution(self):
		self._phase_data = []
		for tau, time_guess, phase, c_continuous_lambda, y_slice, u_slice, q_slice, t_slice, dy_slice, N in zip(self._tau, self._it.guess_time, self._backend.p, self._backend.compiled_functions.c_continuous_lambdas, self._it.y_slices, self._it.u_slices, self._it.q_slices, self._it.t_slices, self._it.c_lambda_dy_slices, self._it.mesh.N):
			y = self._x[y_slice].reshape(phase.num_y_vars, -1) if phase.num_y_vars else np.array([], dtype=float)
			dy = c_continuous_lambda(*self._it._reshape_x(self._x), N)[dy_slice].reshape((-1, N)) if phase.num_y_vars else np.array([], dtype=float)
			u = self._x[u_slice].reshape(phase.num_u_vars, -1) if phase.num_u_vars else np.array([], dtype=float)
			q = self._x[q_slice]
			t = self._x[t_slice]
			
			t0 = t[0] if phase.ocp_phase.bounds._t_needed[0] else time_guess[0]
			tF = t[-1] if phase.ocp_phase.bounds._t_needed[1] else time_guess[-1]
			T = tF - t0
			stretch = T/2
			shift = (t0 + tF) / 2
			time = tau*stretch + shift
			phase_data = PhaseSolutionData(tau=tau, y=y, dy=dy, u=u, q=q, t=t, t0=t0, tF=tF, T=T, stretch=stretch, shift=shift, time=time)
			self._phase_data.append(phase_data)
		self._phase_data = tuple(self._phase_data)
		self._y = tuple(p.y for p in self._phase_data)
		self._dy = tuple(p.dy for p in self._phase_data)
		self._u = tuple(p.u for p in self._phase_data)
		self._q = tuple(p.q for p in self._phase_data)
		self._t = tuple(p.t for p in self._phase_data)
		self._t0 = tuple(p.t0 for p in self._phase_data)
		self._tF = tuple(p.tF for p in self._phase_data)
		self._T = tuple(p.T for p in self._phase_data)
		self._stretch = tuple(p.stretch for p in self._phase_data)
		self._shift = tuple(p.shift for p in self._phase_data)
		self._time = tuple(p.time for p in self._phase_data)
		self._s = self._x[self._it.s_slice]
		self._interpolate_solution()

	def _process_snopt_solution(self, solution):
		raise NotImplementedError

	def _interpolate_solution(self):

		self._phase_y_polys = []
		self._phase_dy_polys = []
		self._phase_u_polys = []
		for p, p_data, K, N_K, mesh_index_boundaries in zip(self._backend.p, self._phase_data, self._it.mesh.K, self._it.mesh.N_K, self._it.mesh.mesh_index_boundaries):

			y_polys = np.empty((p.num_y_vars, K), dtype=object)
			dy_polys = np.empty((p.num_y_vars, K), dtype=object)
			u_polys = np.empty((p.num_u_vars, K), dtype=object)

			for i_y, state_deriv in enumerate(p_data.dy):
				for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
					t_k = p_data.tau[i_start:i_stop+1]
					dy_k = state_deriv[i_start:i_stop+1]
					dy_poly = np.polynomial.Polynomial.fit(t_k, dy_k, deg=N_K[i_k]-1, window=[0, 1])
					scale_factor = self._it.mesh._PERIOD/p_data.T
					y_poly = dy_poly.integ(k=scale_factor*p_data.y[i_y, i_start])
					y_poly = np.polynomial.Polynomial(coef=y_poly.coef/scale_factor, window=y_poly.window, domain=y_poly.domain)
					y_polys[i_y, i_k] = y_poly
					dy_polys[i_y, i_k] = dy_poly

			for i_u, control in enumerate(p_data.u):
				for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
					t_k = p_data.tau[i_start:i_stop+1]
					u_k = control[i_start:i_stop+1]
					u_poly = np.polynomial.Polynomial.fit(t_k, u_k, deg=N_K[i_k]-1, window=[0, 1])
					u_polys[i_u, i_k] = u_poly

			self._phase_y_polys.append(y_polys)
			self._phase_dy_polys.append(dy_polys)
			self._phase_u_polys.append(u_polys)

	def _plot_interpolated_solution(self, plot_y=False, plot_dy=False, plot_u=False):

		t_data_phases = []
		y_datas_phases = []
		dy_datas_phases = []
		u_datas_phases = []

		for y_polys, dy_polys, u_polys, p_data, mesh_index_boundaries in zip(self._phase_y_polys, self._phase_dy_polys, self._phase_u_polys, self._phase_data, self._it.mesh.mesh_index_boundaries):

			t_start_stops = list(zip(p_data.tau[mesh_index_boundaries[:-1]], p_data.tau[mesh_index_boundaries[1:]]))

			t_data = []
			y_datas = []
			dy_datas = []
			u_datas = []

			for i_y, state in enumerate(y_polys):
				t_list = []
				y_list = []
				for t_start_stop, y_poly in zip(t_start_stops, state):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					y_linspace = y_poly(t_linspace)
					t_list.extend(t_linspace)
					y_list.extend(y_linspace)
				t_list.append(p_data.tau[-1])
				y_list.append(p_data.y[i_y, -1])
				t_data.append(t_list)
				y_datas.append(y_list)

			for i_dy, dstate in enumerate(dy_polys):
				t_list = []
				dy_list = []
				for t_start_stop, dy_poly in zip(t_start_stops, dstate):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					dy_linspace = dy_poly(t_linspace)
					t_list.extend(t_linspace)
					dy_list.extend(dy_linspace)
				t_list.append(p_data.tau[-1])
				dy_list.append(p_data.dy[i_dy, -1])
				t_data.append(t_list)
				dy_datas.append(dy_list)

			for i_u, control in enumerate(u_polys):
				t_list = []
				u_list = []
				for t_start_stop, u_poly in zip(t_start_stops, control):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					u_linspace = u_poly(t_linspace)
					t_list.extend(t_linspace)
					u_list.extend(u_linspace)
				t_list.append(p_data.tau[-1])
				u_list.append(p_data.u[i_u, -1])
				t_data.append(t_list)
				u_datas.append(u_list)

			t_data = np.array(t_data[0])*p_data.stretch + p_data.shift
			y_datas = np.array(y_datas)
			dy_datas = np.array(dy_datas)
			u_datas = np.array(u_datas)

			t_data_phases.append(t_data)
			y_datas_phases.append(y_datas)
			dy_datas_phases.append(dy_datas)
			u_datas_phases.append(u_datas)

		if plot_y:
			for p, p_data, t_data, y_datas in zip(self._backend.p, self._phase_data, t_data_phases, y_datas_phases):
				for i_y, y_data in enumerate(y_datas):
					plt.plot(p_data.time, p_data.y[i_y], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, y_data, linewidth=2, label=str(p.y_vars[i_y])[1:])
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('States')
			plt.xlabel('Time / $s$')
			plt.show()

		if plot_dy:
			for p, p_data, t_data, dy_datas in zip(self._backend.p, self._phase_data, t_data_phases, dy_datas_phases):
				for i_y, dy_data in enumerate(dy_datas):
					plt.plot(p_data.time, p_data.dy[i_y], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, dy_data, linewidth=2, label=str(p.y_vars[i_y])[1:])	
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('State Derivatives')
			plt.xlabel('Time / $s$')
			plt.show()

		if plot_u:
			for p, p_data, t_data, u_datas in zip(self._backend.p, self._phase_data, t_data_phases, u_datas_phases):
				for i_u, u_data in enumerate(u_datas):
					plt.plot(p_data.time, p_data.u[i_u], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, u_data, linewidth=2, label=str(p.u_vars[i_u])[1:])
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('Controls')
			plt.xlabel('Time / $s$')
			plt.show()

		# plt.plot(self._y[0], self._y[1], marker='x', markersize=5, linestyle='', color='black')
		# plt.plot(y_datas[0], y_datas[1], linewidth=2)
		# plt.grid(True)
		# plt.gca().set_aspect('equal', adjustable='box')
		# plt.title('Path')
		# plt.xlabel('x / $m$')
		# plt.ylabel('y / $m$')
		# plt.show()

		

		if False:

			with open('result_tradjectory.csv', mode='w') as result_tradjectory_file:

			    result_tradjectory_writer = csv.writer(result_tradjectory_file, delimiter=',')

			    result_tradjectory_writer.writerow(t_data)

			    for i_y, y_data in enumerate(y_datas):
			    	result_tradjectory_writer.writerow(y_data)

			    for i_y, dy_data in enumerate(dy_datas):
			    	result_tradjectory_writer.writerow(dy_data)

			    for i_u, u_data in enumerate(u_datas):
			    	result_tradjectory_writer.writerow(u_data)

		if False:
			fig, ax1 = plt.subplots()
			plt.grid(True, axis='both')

			color = 'tab:blue'
			ax1.set_xlabel('Time / $s$')
			ax1.set_ylabel('Angle / $rad$', color=color)
			ax1.plot(self._time, self._y[0], marker='x', markersize=7, linestyle='', color='black')
			ax1.plot(t_data, y_datas[0], color=color, linewidth=2)
			ax1.tick_params(axis='y', labelcolor=color)
			plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], ['$-\\pi/2$', '$-\\pi/4$', '0', '$\\pi/4$', '$\\pi/2$'])

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

			color = 'tab:orange'
			ax2.set_ylabel('Angular Velocity / $rad \\cdot s^{-1}$', color=color)  # we already handled the x-label with ax1
			ax2.plot(self._time, self._dy[0], marker='x', markersize=7, linestyle='', color='black')
			ax2.plot(t_data, dy_datas[0], color=color, linewidth=2)
			ax2.tick_params(axis='y', labelcolor=color)
			plt.yticks([-10, -5, 0, 5, 10])

			fig.tight_layout()  # otherwise the right y-label is slightly clipped

			plt.show()



			fig, ax1 = plt.subplots()
			plt.grid(True, axis='both')

			color = 'tab:green'
			ax1.set_xlabel('Time / $s$')
			ax1.set_ylabel('Torque / $N\\cdot m$', color=color)
			plt.plot(self._time, self._u[0]*15, marker='x', markersize=7, linestyle='', color='black')
			ax1.plot(t_data, u_datas[0]*15, color=color, linewidth=2)
			ax1.tick_params(axis='y', labelcolor=color)
			plt.yticks([-2, -1, 0, 1, 2])

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			plt.show()


	def _patterson_rao_discretisation_mesh_error(self):

		def eval_polynomials(polys, mesh, vals):
			sec_bnd_inds = mesh._mesh_index_boundaries
			for i_var, poly_row in enumerate(polys):
				for i_k, (poly, i_start, i_stop) in enumerate(zip(poly_row, sec_bnd_inds[:-1], sec_bnd_inds[1:])):
					sec_slice = slice(i_start+1, i_stop)
					vals[i_var, sec_slice] = poly(mesh._tau[sec_slice])
			return vals

		ph_mesh = Mesh(optimal_control_problem=self._ocp,
			mesh_sections=self._mesh._K,
			mesh_section_fractions=self._mesh._mesh_sec_fracs,
			mesh_collocation_points=(self._mesh._mesh_col_points+1))
		ph_mesh._generate_mesh()

		y_tilde = np.zeros((self._ocp._num_y_vars, ph_mesh._N))
		y_tilde[:, ph_mesh._mesh_index_boundaries] = self._y[:, self._mesh._mesh_index_boundaries]
		y_tilde = eval_polynomials(self._y_polys, ph_mesh, y_tilde)

		if self._ocp._num_u_vars:
			u_tilde = np.zeros((self._ocp._num_u_vars, ph_mesh._N))
			u_tilde[:, ph_mesh._mesh_index_boundaries] = self._u[:, self._mesh._mesh_index_boundaries]
			u_tilde = eval_polynomials(self._u_polys, ph_mesh, u_tilde)
		else:
			u_tilde = np.array([])

		dy_tilde = self._ocp._c_continuous_lambda(*y_tilde, *u_tilde, *self._q, *self._t, *self._s, ph_mesh._N)[:self._ocp._num_y_vars*ph_mesh._N].reshape(-1, ph_mesh._N)

		stretch = 0.5 * (self._tF - self._t0)
		A_dy_tilde = stretch*ph_mesh._sA_matrix.dot(dy_tilde.T)

		mesh_error = np.zeros((self._mesh._K, self._ocp._num_y_vars, max(ph_mesh._mesh_col_points)-1))
		rel_error_scale_factor = np.zeros((self._mesh._K, self._ocp._num_y_vars))

		for i_k, (i_start, m_k) in enumerate(zip(ph_mesh._mesh_index_boundaries[:-1], ph_mesh._mesh_col_points-1)):
			y_k = y_tilde[:, i_start]
			Y_tilde_k = (y_k + A_dy_tilde[i_start:i_start+m_k]).T
			Y_k = y_tilde[:, i_start+1:i_start+1+m_k]
			mesh_error_k = Y_tilde_k - Y_k
			mesh_error[i_k, :, :m_k] = mesh_error_k
			rel_error_scale_factor_k = np.max(np.abs(Y_k), axis=1) + 1
			rel_error_scale_factor[i_k, :] = rel_error_scale_factor_k

		if False:
			tau = ph_mesh._tau
			plt.plot(self._tau, self._dy[0, :])
			plt.plot(tau, dy_tilde[0, :])

			plt.plot(self._tau, self._dy[1, :])
			plt.plot(tau, dy_tilde[1, :])

			plt.show()

		self._absolute_mesh_error = np.abs(mesh_error)

		relative_mesh_error = np.zeros_like(self._absolute_mesh_error)
		for i_k in range(ph_mesh._K):
			for i_y in range(self._ocp._num_y_vars):
				for i_m in range(ph_mesh._mesh_col_points[i_k]-1):
					val = self._absolute_mesh_error[i_k, i_y, i_m] / (1 + rel_error_scale_factor[i_k, i_y])
					relative_mesh_error[i_k, i_y, i_m] = val

		self._relative_mesh_error = relative_mesh_error

		max_relative_error = np.zeros(self._mesh._K)
		for i_k in range(ph_mesh._K):
			max_relative_error[i_k] = np.max(self._relative_mesh_error[i_k, :, :])

		self._maximum_relative_error = max_relative_error

		return None

	def _patterson_rao_next_iteration_mesh(self):

		def merge_sections(new_mesh_sec_fracs, new_mesh_col_points, merge_group):
			merge_group = np.array(merge_group)
			P_q = merge_group[:, 0]
			h_q = merge_group[:, 1]
			p_q = merge_group[:, 2]

			N = np.sum(p_q)
			T = np.sum(h_q)

			merge_ratio = p_q / (self._ocp._settings.collocation_points_min - P_q)
			mesh_secs_needed = np.ceil(np.sum(merge_ratio)).astype(np.int)
			if mesh_secs_needed == 1:
				new_mesh_secs = np.array([T])
			else:
				required_reduction = np.divide(h_q, merge_ratio)
				weighting_factor = np.reciprocal(np.sum(required_reduction))
				reduction_factor = weighting_factor * required_reduction
				knot_locations = np.cumsum(h_q) / T
				current_density = np.cumsum(reduction_factor)

				density_function = interpolate.interp1d(knot_locations, current_density, bounds_error=False, fill_value='extrapolate')
				new_density = np.linspace(1/mesh_secs_needed, 1, mesh_secs_needed)
				new_knots = density_function(new_density)

				new_mesh_secs = T * np.diff(np.concatenate([np.array([0]), new_knots]))

			new_mesh_sec_fracs.extend(new_mesh_secs.tolist())
			new_mesh_col_points.extend([self._ocp._settings.collocation_points_min]*mesh_secs_needed)

			return new_mesh_sec_fracs, new_mesh_col_points

		def subdivide_sections(new_mesh_sec_fracs, new_mesh_col_points, subdivide_group, reduction_tolerance):
			subdivide_group = np.array(subdivide_group)
			subdivide_required = subdivide_group[:, 0].astype(np.bool)
			subdivide_factor = subdivide_group[:, 1].astype(np.int)
			P_q = subdivide_group[:, 2]
			h_q = subdivide_group[:, 3]
			p_q = subdivide_group[:, 4]

			is_node_reduction = P_q <= 0

			predicted_nodes = P_q + p_q
			predicted_nodes[is_node_reduction] = np.ceil(P_q[is_node_reduction] * reduction_tolerance) + p_q[is_node_reduction]

			next_mesh_nodes = np.ones_like(predicted_nodes, dtype=np.int) * self._ocp._settings.collocation_points_min
			next_mesh_nodes[np.invert(subdivide_required)] = predicted_nodes[np.invert(subdivide_required)]
			next_mesh_nodes_lower_than_min = next_mesh_nodes < self._ocp._settings.collocation_points_min
			next_mesh_nodes[next_mesh_nodes_lower_than_min] = self._ocp._settings.collocation_points_min

			for h, k, n in zip(h_q, subdivide_factor, next_mesh_nodes):
				new_mesh_sec_fracs.extend([h/k]*k)
				new_mesh_col_points.extend([n]*k)

			return new_mesh_sec_fracs, new_mesh_col_points

		if np.max(self._maximum_relative_error) > self._ocp._settings.mesh_tolerance:

			P_q = np.ceil(np.divide(np.log(self._maximum_relative_error/self._ocp._settings._mesh_tolerance), np.log(self._mesh._mesh_col_points)))
			P_q_zero = P_q == 0
			P_q[P_q_zero] = 1
			predicted_nodes = P_q + self._mesh._mesh_col_points

			log_tolerance = np.log(self._ocp._settings.mesh_tolerance / np.max(self._maximum_relative_error))
			merge_tolerance = 50 / log_tolerance
			merge_required = predicted_nodes < merge_tolerance

			reduction_tolerance = 1 - (-1 / log_tolerance)
			if reduction_tolerance < 0:
				reduction_tolerance = 0

			subdivide_required = predicted_nodes >= self._ocp._settings.collocation_points_max
			subdivide_level = np.ones_like(predicted_nodes)
			subdivide_level[subdivide_required] = np.ceil(predicted_nodes[subdivide_required] / self._ocp._settings.collocation_points_min)

			merge_group = []
			subdivide_group = []
			new_mesh_sec_fracs = []
			new_mesh_col_points = []
			for need_merge, need_subdivide, subdivide_factor, P, h, p in zip(merge_required, subdivide_required, subdivide_level, P_q, self._mesh._h_K, self._mesh._mesh_col_points):
				if need_merge:
					if subdivide_group != []:
						new_mesh_sec_fracs, new_mesh_col_points = subdivide_sections(new_mesh_sec_fracs, new_mesh_col_points, subdivide_group, reduction_tolerance)
						subdivide_group = []
					merge_group.append([P, h, p])
				else:
					if merge_group != []:
						new_mesh_sec_fracs, new_mesh_col_points = merge_sections(new_mesh_sec_fracs, new_mesh_col_points, merge_group)
						merge_group = []
					subdivide_group.append([need_subdivide, subdivide_factor, P, h, p])
			else:
				if merge_group != []:
					new_mesh_sec_fracs, new_mesh_col_points = merge_sections(new_mesh_sec_fracs, new_mesh_col_points, merge_group)
				elif subdivide_group != []:
					new_mesh_sec_fracs, new_mesh_col_points = subdivide_sections(new_mesh_sec_fracs, new_mesh_col_points, subdivide_group, reduction_tolerance)

			new_mesh_secs = len(new_mesh_sec_fracs)

			new_mesh = Mesh(optimal_control_problem=self._ocp, mesh_sections=new_mesh_secs, mesh_section_fractions=new_mesh_sec_fracs, mesh_collocation_points=new_mesh_col_points)

			return new_mesh

		else:
			return None




class IPOPTProblem:

	def __init__(self, J, g, c, G, G_struct, H, H_struct):
		self.objective = J
		self.gradient = g
		self.constraints = c
		self.jacobian = G
		self.jacobianstructure = G_struct
		if H is not None and H_struct is not None:
			self.hessian = H
			self.hessianstructure = H_struct





