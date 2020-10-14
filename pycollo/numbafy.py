import itertools
from numbers import Number

import numba as nb
import numpy as np
from numpy import sin, cos, tan, exp, log, sqrt, arccos, arcsin, arctan, tanh
import scipy.interpolate as interpolate
import sympy as sym

acos = arccos
asin = arcsin
atan = arctan

def numbafy(expression_graph=None, expression=None, expression_nodes=None, precomputable_nodes=None, dependent_tiers=None, parameters=None, lagrange_parameters=None, return_dims=None, return_flat=False, N_arg=False, endpoint=False, hessian=None, hessian_sym_set=None, ocp_num_vars=None):

	def grouper(iterable, n):
		args = [iter(iterable)] * n
		return itertools.zip_longest(*args)

	def expand_to_array(e, node):
		if e is zero_sym or e == 0:
			e_entry = f'_np_zeros_array_N'
		elif e is one_sym:
			e_entry = f'_np_ones_array_N'
		elif node.is_vector:
			e_entry = f'{e}'
		else:
			e_entry = f'({e})*_np_ones_array_N'
		return e_entry

	def build_objective_lagrange_matrix(expression, expression_nodes):

		expression_rows = []
		for row_num in range(expression.rows):
			expression_list = []
			row = expression.row(row_num)
			for col_num, e, in enumerate(row):
				index = row_num*expression.cols + col_num
				e_entry = f'{e}'
				expression_list.append(e_entry)
			expression_row = ', '.join(f'{e}' for e in expression_list)
			expression_rows.append(f"np.array([{expression_row}])")
		expression_string = ', '.join(f'{e}' for e in expression_rows)
		return_value = f'np.stack(({expression_string}))'

		return return_value

	def build_endpoint_lagrange_matrix(expression, expression_nodes, L_syms):
		if len(expression) > 0:
			matrix = sym.zeros(*expression[0].shape)
			for mat, L_sym in zip(expression, L_syms):
				mat = mat.subs({expression_graph._zero_node.symbol: 0})
				matrix += L_sym*mat
		else:
			raise NotImplementedError
		return_value = build_objective_lagrange_matrix(matrix, expression_nodes)

		return return_value

	def build_lagrange_matrix_entry(terms, term_nodes, L_syms):

		expressions = []
		for term, node, L_sym in zip(terms, term_nodes, L_syms):
			if term == 0:
				pass
			elif term == zero_sym:
				pass
			else:
				e_entry = expand_to_array(term, node)
				expressions.append(f"np.outer({L_sym}, {e_entry})")
		return_value = ' + '.join(outer for outer in expressions)
		return return_value


	if parameters:
		function_arguments = ', '.join(f'{p}' for p in parameters)
		if lagrange_parameters:
			try:
				iter(lagrange_parameters)
			except TypeError:
				function_arguments += f', {lagrange_parameters}'
			else:
				function_arguments += ', '
				function_arguments += ', '.join(f'{L}' 
					for L in lagrange_parameters)
		if N_arg:
			function_arguments += ', _N'
			numpy_default_arrays = ("_np_zeros_array_N = np.zeros(_N)\n"
				"    _np_ones_array_N = np.ones(_N)")
		else:
			numpy_default_arrays = ""
	else:
		raise NotImplementedError

	if precomputable_nodes:
		precomputed_constants = '\n    '.join(f'{node.numbafy_expression}' 
			for node in precomputable_nodes)
	else:
		precomputed_constants = '    '

	substitution_tiers = []
	for tier in list(dependent_tiers.values())[1:]:
		tier_substitutions = '\n    '.join(f'{node.numbafy_expression}' 
			for node in tier)
		substitution_tiers.append(tier_substitutions)

	intermediate_substitutions = '\n    '.join(f'{sub_tier}' 
			for sub_tier in substitution_tiers)

	if ocp_num_vars:
		yu_qts_continuous_split = sum(ocp_num_vars[0:2])
		yu_qts_endpoint_split = 2*ocp_num_vars[0]

	zero_sym = expression_graph._zero_node.symbol
	one_sym = expression_graph._one_node.symbol

	if lagrange_parameters is not None:

		if not isinstance(expression, list):
			return_value = build_objective_lagrange_matrix(expression, expression_nodes)
		elif endpoint:
			L_syms = lagrange_parameters
			return_value = build_endpoint_lagrange_matrix(expression, expression_nodes, L_syms)
		else:
			array_arguments = []
			L_syms = lagrange_parameters
			for i, _ in enumerate(expression[0]):
				terms = [expr_mat[i] for expr_mat in expression if expr_mat[i] != 0]
				term_nodes = [expr_mat_nodes[i] for expr_mat_nodes in expression_nodes]
				if terms:
					mat_entry = build_lagrange_matrix_entry(terms, term_nodes, L_syms)
					if mat_entry:
						array_arguments.append(mat_entry)
			if array_arguments:
				array_argument = ', '.join(array_arguments)
				return_value = f'np.stack(({array_argument}, ))'
			else:
				return_value = f'np.zeros((_N, _N))'

	elif return_dims is None:
		return_value = f'{expression}'

	elif return_dims == 1:
		if endpoint is True:
			i_t0 = slice(0, yu_qts_endpoint_split, 2)
			i_tF = slice(1, yu_qts_endpoint_split, 2)
			y_t0_vars = parameters[i_t0]
			y_tF_vars = parameters[i_tF]
			qts_vars = parameters[yu_qts_endpoint_split:]

			y_tuple = []
			for y_t0, y_tF in grouper(expression[:yu_qts_endpoint_split], 2):
				if y_t0 == zero_sym and y_tF == zero_sym:
					to_append = '_np_zeros_array_N'
				elif y_t0 == zero_sym:
					to_append = (f'np.hstack((np.zeros(_N-1), {y_tF}))')
				elif y_tF == zero_sym:
					to_append = (f'np.hstack(({y_t0}, np.zeros(_N-1)))')
				else:
					to_append = (f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))')
				y_tuple.append(to_append)

			y_str = ', '.join(f'{e}' for e in y_tuple)

			u_str = ', '.join(f'_np_zeros_array_N' for i in range(ocp_num_vars[1]))

			qts_str = ', '.join(f'{e}' for e in expression[yu_qts_endpoint_split:])

			qts_str = f'np.array([{qts_str}])' if qts_str != '' else ''

			str_list = []
			for s in (y_str, u_str, qts_str):
				if s != '':
					str_list.append(s)

			str_str = ', '.join(s for s in str_list)

			return_value = f'np.concatenate(({str_str}))'
		else:
			expression_string = ', '.join(f'{e}' for e in expression)
			return_value = f'np.array([{expression_string}])'

	elif return_dims == 2:
		expression_rows = []
		parameter_set = set(parameters[:yu_qts_continuous_split])
		if expression.shape[0] != 1 and expression.shape[1] == 1:
			expression = expression.T
		for row_num in range(expression.rows):
			expression_list = []
			row = expression.row(row_num)
			for col_num, e in enumerate(row):
				index = row_num*expression.cols + col_num
				node = expression_nodes[index]
				if e.free_symbols.intersection(parameter_set):
					expression_list.append(e)
				else:
					if node.is_vector:
						e_entry = f'{e}'
					else:
						e_entry = f'({e})*_np_ones_array_N'
					if return_flat:
						raise NotImplementedError
						e_entry = f'{e_entry}.flatten()'
					expression_list.append(e_entry)
			expression_row = ', '.join(f'{e}' for e in expression_list)
			expression_rows.append(f"np.hstack(({expression_row}))")

		expression_string = ', '.join(f'{e}' for e in expression_rows)

		if return_flat:
			raise NotImplementedError
			return_value = f'np.concatenate(({expression_string}))'
		else:
			return_value = expression_string#f'np.array([{expression_string}])'

	elif return_dims == 3:

		expression_rows = []
		for row_num in range(expression.rows):
			expression_list = []
			row = expression.row(row_num)
			for col_num, e in enumerate(row):
				index = row_num*expression.cols + col_num
				node = expression_nodes[index]
				e_entry = expand_to_array(e, node)
				expression_list.append(e_entry)
			expression_row = ', '.join(f'{e}' for e in expression_list)
			expression_rows.append(f"np.stack(({expression_row}))")
		expression_string = ', '.join(f'{e}' for e in expression_rows)
		return_value = f'np.stack(({expression_string}))'

	else:
		msg = ("Value for `return_dims` argument must be 0, 1, or 2 only.")
		raise ValueError(msg)

	function_string = (
		# f"@nb.njit(parallel=True)\n"
		f"def numbafied_func({function_arguments}):\n"
		f"    {numpy_default_arrays}\n"
		f"    {precomputed_constants}\n"
		f"    {intermediate_substitutions}\n"
		# f"    print({return_value})\n"
		f"    return {return_value}")

	# cout(function_string)
	# input()
	
	exec(function_string)
	   
	return locals()['numbafied_func']



def kill():
		print('\n\n')
		raise ValueError

def cout(*args):
	print('\n\n')
	for arg in args:
		print(f'{arg}\n')
