import itertools
from numbers import Number

import numba as nb
import numpy as np
from numpy import (sin, cos, tan, exp, log, sqrt, arccos, arcsin, arctan, tanh)
import scipy.sparse as sparse
import sympy as sym

acos = arccos
asin = arcsin
atan = arctan


def numbafy_objective_hessian(*args, **kwargs):

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

	def collect_function_arguments(parameters, sigma):
		parameters_all = parameters + (sigma, )
		function_arguments = ', '.join(f'{p}' for p in parameters_all)
		return function_arguments

	def collect_numpy_default_arrays():
		return ""

	def collect_precomputable_constants(precomputable_nodes):
		if precomputable_nodes:
			precomputed_constants = '\n    '.join(f'{node.numbafy_expression}' 
				for node in precomputable_nodes)
		else:
			precomputed_constants = '    '
		return precomputed_constants

	def collect_intermediate_substitutions(dependent_tiers):
		substitution_tiers = []
		for tier in list(dependent_tiers.values())[1:]:
			tier_substitutions = '\n    '.join(f'{node.numbafy_expression}' 
				for node in tier)
			substitution_tiers.append(tier_substitutions)
		intermediate_substitutions = '\n    '.join(f'{sub_tier}' 
			for sub_tier in substitution_tiers)
		return intermediate_substitutions

	def collect_return_value(expression, expression_nodes, sigma):
		values = []
		nonzero_row_indices = []
		nonzero_col_indices = []
		for i_row, expr_node_row in enumerate(expression_nodes):
			expr_row = expression.row(i_row)
			for i_col, (expr, expr_node) in enumerate(zip(expr_row, expr_node_row)):
				if expr_node != expression_graph._zero_node and not np.isclose(expr, 0):
					values.append(expr)
					nonzero_row_indices.append(i_row)
					nonzero_col_indices.append(i_row)
		nonzero_indices = (nonzero_row_indices, nonzero_col_indices)
		return_value = f"sigma * np.array([{', '.join(values)}])"
		return return_value, nonzero_indices

	cout_function_string = True

	expression_graph = kwargs.get("expression_graph")
	expression = kwargs.get("expression")
	expression_nodes = kwargs.get("expression_nodes")
	expression_nodes = np.array(expression_nodes).reshape(expression.shape)
	precomputable_nodes = kwargs.get("precomputable_nodes")
	dependent_tiers = kwargs.get("dependent_tiers")
	parameters = kwargs.get("parameters")
	sigma = kwargs.get("sigma")

	function_arguments = collect_function_arguments(parameters, sigma)
	numpy_default_arrays = collect_numpy_default_arrays()
	precomputed_constants = collect_precomputable_constants(precomputable_nodes)
	intermediate_substitutions = collect_intermediate_substitutions(dependent_tiers)
	return_value, nonzero_indices = collect_return_value(expression, expression_nodes, sigma)

	function_string = (
		f"def numbafied_func({function_arguments}):\n"
		f"    {numpy_default_arrays}\n"
		f"    {precomputed_constants}\n"
		f"    {intermediate_substitutions}\n"
		f"    return {return_value}")

	if cout_function_string:
		cout(function_string)
		input()
	
	exec(function_string)
	   
	return locals()['numbafied_func'], nonzero_indices


def cout(*args):
	print('\n\n')
	for arg in args:
		print(f'{arg}\n')

