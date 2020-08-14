import numba as nb
import numpy as np
from numpy import (sin, cos, tan, exp, log, sqrt, arccos, arcsin, arctan, tanh)
import scipy.sparse as sparse
import sympy as sym

acos = arccos
asin = arcsin
atan = arctan


def numbafy_endpoint_hessian(*args, **kwargs):

	def collect_function_arguments(parameters, sigma, lagrange_multipliers):
		parameters_all = parameters + (sigma, ) + lagrange_multipliers
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

	def collect_return_value(expression):
		nonzeros = ', '.join([str(v) for v in expression.entries.values()])
		return_value = f"np.array([{nonzeros}])"
		return return_value

	cout_function_string = False

	expression_graph = kwargs.get("expression_graph")
	expression = kwargs.get("expression")
	precomputable_nodes = kwargs.get("precomputable_nodes")
	dependent_tiers = kwargs.get("dependent_tiers")
	parameters = kwargs.get("parameters")
	sigma = kwargs.get("objective_factor")
	lagrange_multipliers = kwargs.get("lagrange_multipliers")

	function_arguments = collect_function_arguments(parameters, sigma, lagrange_multipliers)
	numpy_default_arrays = collect_numpy_default_arrays()
	precomputed_constants = collect_precomputable_constants(precomputable_nodes)
	intermediate_substitutions = collect_intermediate_substitutions(dependent_tiers)
	return_value = collect_return_value(expression)

	function_string = (
		f"def numbafied_func({function_arguments}):\n"
		f"    {precomputed_constants}\n"
		f"    {intermediate_substitutions}\n"
		f"    return {return_value}")

	if cout_function_string:
		cout(function_string)
		input()
	
	exec(function_string)
	   
	return locals()['numbafied_func']




def numbafy_continuous_hessian(*args, **kwargs):

	def collect_function_arguments(parameters, lagrange_multipliers):
		parameters_all = parameters + lagrange_multipliers
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

	def collect_intermediate_substitutions(dependent_tiers, final_nodes):

		def check_if_needs_summing(node):
			if node in final_nodes:
				new_sym = sym.Symbol(f"_sum{node.symbol}")
				new_parent_syms = [sym.Symbol(f"{parent.symbol}") 
					if parent in summing_nodes else parent.symbol
					for parent in node.parent_nodes]
				new_sum_mapping = {parent: f"np.sum({parent})" for parent in new_parent_syms}
				new_expr = node.operation.SYMPY_OP(*[new_parent_sym 
					for new_parent_sym in new_parent_syms])
				sum_expr = f"{new_sym} = {new_expr}"
				for k, v in new_sum_mapping.items():
					start = sum_expr.find(str(k))
					end = start + len(str(k))
					sum_expr = sum_expr[:start] + str(v) + sum_expr[end:]
				sum_expr = f"{new_sym} = {sum_expr}"
				expr = f"{sum_expr}"
				return expr
			return f"{node.numbafy_expression}"

		substitution_tiers = []
		for tier in list(dependent_tiers.values())[1:]:
			tier_substitutions = '\n    '.join(f"{check_if_needs_summing(node)}\n"
				for node in tier)
			substitution_tiers.append(tier_substitutions)
		intermediate_substitutions = '\n    '.join(f'{sub_tier}' 
			for sub_tier in substitution_tiers)

		return intermediate_substitutions

	def collect_return_value(expression, expression_nodes, final_nodes):

		def check_if_sum_sym_needed(symbol, node):
			if node in final_nodes:
				return str(sym.Symbol(f"_sum{symbol}"))
			else:
				return str(symbol)

		return_value = ", ".join([check_if_sum_sym_needed(symbol, node) for symbol, node in zip(expression.entries.values(), expression_nodes.entries.values())])
		return return_value

	cout_function_string = False

	expression_graph = kwargs.get("expression_graph")
	expression = kwargs.get("expression")
	expression_nodes = kwargs.get("expression_nodes")
	precomputable_nodes = kwargs.get("precomputable_nodes")
	dependent_tiers = kwargs.get("dependent_tiers")
	parameters = kwargs.get("parameters")
	sigma = kwargs.get("objective_factor")
	lagrange_multipliers = kwargs.get("lagrange_multipliers")
	summing_nodes = kwargs.get("summing_nodes")

	final_nodes = set(expression_nodes.entries.values()).intersection(summing_nodes)

	function_arguments = collect_function_arguments(parameters, lagrange_multipliers)
	precomputed_constants = collect_precomputable_constants(precomputable_nodes)
	intermediate_substitutions = collect_intermediate_substitutions(dependent_tiers, final_nodes)
	return_value = collect_return_value(expression, expression_nodes, final_nodes)

	function_string = (
		f"def numbafied_func({function_arguments}):\n"
		f"    {precomputed_constants}\n"
		f"    {intermediate_substitutions}\n"
		f"    return {return_value}")

	if cout_function_string:
		cout(function_string)
		input()
	
	exec(function_string)
	   
	return locals()['numbafied_func']


