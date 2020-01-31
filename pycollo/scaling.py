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
		self.r = 0
		self.V = 1
		self.V_inv = 1
		self.w_J = 1
		self.W = 1
		self.V_inv_sqrd = 1




