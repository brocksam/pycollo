import numpy as np
import scipy.interpolate as interpolate

class Quadrature:

	def __init__(self, *, optimal_control_problem=None):

		# Optimal Control Problem
		self._ocp = optimal_control_problem
		self._polynomials = {}
		self._quadrature_points = {}
		self._quadrature_weights = {}
		self._butcher_arrays = {}
		self._D_matrices = {}
		self._A_matrices = {}
		self._W_matrices = {}
		self._D_index_arrays = {}
		self._A_index_arrays = {}

	@property
	def _settings(self):
		return self._ocp._settings

	@property
	def _ocp(self):
		return self._ocp_priv

	@_ocp.setter
	def _ocp(self, ocp):
		self._ocp_priv = ocp
		self._order_range = list(range(self._settings._col_points_min, self._settings._col_points_max))
		if self._settings._quadrature_method == 'lobatto':
			self._quadrature_generator = self._lobatto_generator
		elif self._settings._quadrature_method == 'radau':
			self._quadrature_generator = self._radau_generator
		elif self._settings._quadrature_method == 'gauss':
			self._quadrature_generator = self._gauss_generator

	def _retrive_or_generate_dict_value(self, quad_dict, order):
		try:
			quad_dict[order]
		except KeyError:
			self._quadrature_generator(order)
		return quad_dict[order]

	def polynomials(self, order):
		return self._retrive_or_generate_dict_value(self._polynomials, order)

	def quadrature_point(self, order, *, domain=None):
		points = self._retrive_or_generate_dict_value(self._quadrature_points, order)
		if domain:
			stretch = 0.5*(domain[1] - domain[0])
			scale = 0.5*(domain[0] + domain[1])
			return stretch*points + scale
		else:
			return points

	def quadrature_weight(self, order):
		return self._retrive_or_generate_dict_value(self._quadrature_weights, order)

	def butcher_array(self, order):
		return self._retrive_or_generate_dict_value(self._butcher_arrays, order)

	def D_matrix(self, order):
		return self._retrive_or_generate_dict_value(self._D_matrices, order)

	def A_matrix(self, order):
		return self._retrive_or_generate_dict_value(self._A_matrices, order)

	def W_matrix(self, order):
		return self._retrive_or_generate_dict_value(self._W_matrices, order)

	def D_index_array(self, order):
		return self._retrive_or_generate_dict_value(self._D_index_arrays, order)

	def A_index_array(self, order):
		return self._retrive_or_generate_dict_value(self._A_index_arrays, order)

	def _radau_generator(self, order):
		coefficients = [0]*(order - 1)
		coefficients.extend([1, 1])
		legendre_polynomial = np.polynomial.legendre.Legendre(coefficients)
		self._polynomials.update({order: legendre_polynomial})

		radau_points = legendre_polynomial.roots()
		self._quadrature_points.update({order: radau_points})

		coefficients = [0]*(order - 1)
		coefficients.extend([1])
		legendre_polynomial = np.polynomial.legendre.Legendre(coefficients)
		radau_weights = [2/order**2]
		radau_weights = np.array(
			radau_weights + [(1 - x)/(order**2*(legendre_polynomial(x)**2)) 
			for x in radau_points[1:]])
		self._quadrature_weights.update({order: radau_weights})

		print(f'x: {radau_points}')
		print(f'w: {radau_weights}, {sum(radau_weights)}')

		butcher_points = self.quadrature_point(order, domain=[0, 1])
		butcher_array = np.zeros((order, order))
		butcher_array[-1, :] = radau_weights
		if order > 2:
			A_row = (order + 1) * (order - 2)
			A_col = order * (order - 2)
			A = np.zeros((A_row, A_col))
			b = np.zeros(A_row)
			for k in range(order-2):
				for j in range(order):
					row = j + k*order
					for i in range(order-2):
						col = i + j*(order-2)
						A[row, col] = radau_weights[i+1] * butcher_points[i+1]**k
					b[row] = (radau_weights[j]/(k+1))*(1 - butcher_points[j]**(k+1)) - radau_weights[-1]*radau_weights[j]
			del_row = []
			for i, row in enumerate(A):
				if np.count_nonzero(row) == 0:
					del_row.append(i)
			A = np.delete(A, del_row, axis=0)
			b = np.delete(b, del_row, axis=0)
			a = np.linalg.solve(A, b)
			butcher_array[1:-1, :] = a.reshape(order-2, -1, order='F')
		self._butcher_arrays.update({order: butcher_array})

		print(butcher_array)
		input()

		D_left = np.ones((order - 1, 1), dtype=int)
		D_right = np.diag(-1*np.ones((order - 1, ), dtype=int))
		D_matrix = np.hstack([D_left, D_right])
		self._D_matrices.update({order: D_matrix})

		A_matrix = self.butcher_array(order)[1:, :]
		self._A_matrices.update({order: A_matrix})

		A_index_array = np.array(range(A_matrix.size), dtype=int)
		self._A_index_arrays.update({order: A_index_array})

		D_num_row, D_num_col = D_matrix.shape
		D_rows = np.array(range(D_num_row), dtype=int)
		D_left = D_rows * D_num_col
		D_right = D_rows * (D_num_col + 1) + 1
		D_index_array = np.concatenate((D_left, D_right))
		D_index_array.sort()
		self._D_index_arrays.update({order: D_index_array})
	
	def _lobatto_generator(self, order):
		num_interior_points = order - 1
		coefficients = [0]*(num_interior_points)
		coefficients.append(1)
		legendre_polynomial = np.polynomial.legendre.Legendre(coefficients)
		self._polynomials.update({order: legendre_polynomial})

		lobatto_points = legendre_polynomial.deriv().roots()
		lobatto_points = np.insert(lobatto_points, 0, -1, axis=0)
		lobatto_points = np.append(lobatto_points, 1)
		self._quadrature_points.update({order: lobatto_points})

		lobatto_weights = np.array([1/(order*(order-1)*(legendre_polynomial(x)**2)) for x in lobatto_points])
		self._quadrature_weights.update({order: lobatto_weights})

		print(f'x: {lobatto_points}')
		print(f'w: {lobatto_weights}')

		butcher_points = self.quadrature_point(order, domain=[0, 1])
		print(f'x\': {butcher_points}')
		butcher_array = np.zeros((order, order))
		butcher_array[-1, :] = lobatto_weights
		if order > 2:
			A_row = (order + 1) * (order - 2)
			A_col = order * (order - 2)
			A = np.zeros((A_row, A_col))
			b = np.zeros(A_row)
			for k in range(order-2):
				print(f'k: {k}')
				for j in range(order):
					print(f'j: {j}')
					row = j + k*order
					print(f'row: {row}')
					for i in range(order-2):
						print(f'i: {i}')
						col = i + j*(order-2)
						print(f'col: {col}')
						A[row, col] = lobatto_weights[i+1] * butcher_points[i+1]**k
						print(f'A: {lobatto_weights[i+1] * butcher_points[i+1]**k}')
					b[row] = (lobatto_weights[j]/(k+1))*(1 - butcher_points[j]**(k+1)) - lobatto_weights[-1]*lobatto_weights[j]
					
					print(f'b: {(lobatto_weights[j]/(k+1))*(1 - butcher_points[j]**(k+1)) - lobatto_weights[-1]*lobatto_weights[j]}\n')
			del_row = []
			for i, row in enumerate(A):
				if np.count_nonzero(row) == 0:
					del_row.append(i)
			A = np.delete(A, del_row, axis=0)
			b = np.delete(b, del_row, axis=0)
			a = np.linalg.solve(A, b)
			print(f'A: {A}')
			print(f'b: {b}')
			print(f'a: {a}')
			butcher_array[1:-1, :] = a.reshape(order-2, -1, order='F')
		self._butcher_arrays.update({order: butcher_array})

		D_left = np.ones((num_interior_points, 1), dtype=int)
		D_right = np.diag(-1*np.ones((num_interior_points, ), dtype=int))
		D_matrix = np.hstack([D_left, D_right])
		self._D_matrices.update({order: D_matrix})

		A_matrix = self.butcher_array(order)[1:, :]
		self._A_matrices.update({order: A_matrix})

		A_index_array = np.array(range(A_matrix.size), dtype=int)
		self._A_index_arrays.update({order: A_index_array})

		D_num_row, D_num_col = D_matrix.shape
		D_rows = np.array(range(D_num_row), dtype=int)
		D_left = D_rows * D_num_col
		D_right = D_rows * (D_num_col + 1) + 1
		D_index_array = np.concatenate((D_left, D_right))
		D_index_array.sort()
		self._D_index_arrays.update({order: D_index_array})

		input()

		


