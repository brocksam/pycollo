import numpy as np
import scipy.sparse as sparse


class Scaling:

	def __init__(self, optimal_control_problem):

		self._ocp = optimal_control_problem

	@property
	def optimal_control_problem(self):
		return self._ocp
	

	@property
	def bounds(self):
		return self._ocp.bounds

	def _generate(self):
		scaling_method = self.optimal_control_problem.settings.scaling_method
		self._generator_dispatcher()[scaling_method]()

	def _generator_dispatcher(self):
		return {
			None: self._generate_none,
			'automatic': self._generate_automatic,
			'user': self._generate_user,
			}

	def _generate_none(self):
		needed = np.concatenate([self.bounds._y_needed, self.bounds._u_needed, self.bounds._q_needed, self.bounds._t_needed, self.bounds._s_needed])
		zeros = np.zeros_like(needed)
		ones = np.ones_like(needed)
		self.V_matrix_terms = ones
		self.V_inv_matrix_terms = ones
		self.V_inv_matrix_sqrd_terms = ones
		self.r_vector_terms = zeros

	def _generate_automatic(self):

		x_l = np.concatenate([self.bounds._y_l_needed, self.bounds._u_l_needed, self.bounds._q_l_needed, -0.5*np.ones_like(self.bounds._t_l_needed), self.bounds._s_l_needed])
		x_u = np.concatenate([self.bounds._y_u_needed, self.bounds._u_u_needed, self.bounds._q_u_needed, 0.5*np.ones_like(self.bounds._t_u_needed), self.bounds._s_u_needed])
		diff = x_u - x_l
		diff_recip = 1 / diff
		self.V_matrix_terms = diff_recip
		self.V_inv_matrix_terms = diff
		self.V_inv_matrix_sqrd_terms = diff**2
		self.r_vector_terms = 0.5 - x_u*diff_recip
		# self.V_matrix = sparse.dia_matrix(diff_recip)
		# self.V_inv_matrix = sparse.dia_matrix(diff)
		# self.V_inv_matrix_sqrd = sparse.dia_matrix(diff**2)
		# self.r_vector = 0.5 - x_u*diff_recip

	def _generate_user(self):
		raise NotImplementedError




class IterationScaling:

	def __init__(self, iteration):
		self._iteration = iteration
		self._generate_scaling()

	@property
	def optimal_control_problem(self):
		return self._iteration.optimal_control_problem

	@property
	def base_scaling(self):
		return self.optimal_control_problem.scaling

	def _generate_scaling(self):

		def expand_to_mesh(base_scaling):
			yu = base_scaling[self.optimal_control_problem._yu_slice]
			qts = base_scaling[self.optimal_control_problem._qts_slice]
			return np.concatenate([np.repeat(yu, N), qts])

		N = self._iteration._mesh._N

		# Variables shift
		self.r = expand_to_mesh(self.base_scaling.r_vector_terms)

		# Variables scale
		V_vals = expand_to_mesh(self.base_scaling.V_matrix_terms)
		V_inv_vals = expand_to_mesh(self.base_scaling.V_inv_matrix_terms)
		V_inv_sqrd_vals = expand_to_mesh(self.base_scaling.V_inv_matrix_sqrd_terms)
		self.V = sparse.diags(V_vals)
		self.V_inv = sparse.diags(V_inv_vals)
		self.V_inv_sqrd = sparse.diags(V_inv_sqrd_vals)

		# Objective scale
		self.w_J = 1

		# Constraints scale
		W_defect = V_vals[self._iteration._c_defect_slice]
		W_path = np.zeros(self._iteration._num_c_path)
		W_integral = V_vals[self._iteration._c_integral_slice]
		W_point = np.zeros(self._iteration._num_c_boundary)
		W_vals = np.concatenate([W_defect, W_path, W_integral, W_point])
		self.W = sparse.diags(W_vals)

		# print('\n\n\n')
		# raise NotImplementedError




