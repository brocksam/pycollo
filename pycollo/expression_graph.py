import abc
import itertools
import numbers
from timeit import default_timer as timer
import weakref

import numpy as np
import sympy as sym

from .node import Node


class ExpressionGraph:

	def __init__(self, problem_variables_user, problem_variables, auxiliary_information, objective, constraints, lagrangian):

		self._initialise_node_symbol_number_counters()
		self._initialise_node_mappings()

		self._initialise_problem_variable_information(problem_variables, problem_variables_user)

		self._initialise_default_singleton_number_nodes()
		self._initialise_auxiliary_constant_nodes(auxiliary_information)
		self._initialise_auxiliary_intermediate_nodes()

		# for node in self.nodes:
		# 	print(node.symbol)
		# 	print(node.key)
		# 	print(node)
		# 	print('')

		self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers = self._initialise_function(objective)
		self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers = self._initialise_function(constraints)

		dJ_dx = self.hybrid_symbolic_algorithmic_differentiation(self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers)
		self.dJ_dx, self.dJ_dx_nodes, self.dJ_dx_precomputable, self.dJ_dx_dependent_tiers = self._initialise_function(dJ_dx)

		dc_dx = self.hybrid_symbolic_algorithmic_differentiation(self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers)
		self.dc_dx, self.dc_dx_nodes, self.dc_dx_precomputable, self.dc_dx_dependent_tiers = self._initialise_function(dc_dx)

		# Jacobian of the constraints
		# 
		# self.dc_dx = self.hybrid_symbolic_algorithmic_differentiation(self.c, self.x_vars, self.c_dependent_tiers)
		# dc_dx_cached = self.hybrid_symbolic_algorithmic_differentiation_cached(self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers)

		# print('\n\n\n')
		# if self.dc_dx == dc_dx_cached:
		# 	print("PASSED!")
		# else:
		# 	print("FAILED!")

	def _initialise_node_symbol_number_counters(self):
		self._number_node_num_counter = itertools.count()
		self._constant_node_num_counter = itertools.count()
		self._intermediate_node_num_counter = itertools.count()
	
	def _initialise_node_mappings(self):
		self._variable_nodes = {}
		self._number_nodes = {}
		self._constant_nodes = {}
		self._intermediate_nodes = {}
		self._precomputable_nodes = {}

	def _initialise_problem_variable_information(self, x_vars, x_vars_user):
		self._initialise_problem_variable_attributes(x_vars, x_vars_user)
		self._initialise_problem_variable_nodes()

	def _initialise_default_singleton_number_nodes(self):
		self._zero_node = Node(0, self)
		self._one_node = Node(1, self)
		two_node = Node(2, self)
		neg_one_node = Node(-1, self)
		half_node = Node(0.5, self)

	def _initialise_problem_variable_attributes(self, x_vars, x_vars_user):
		self.problem_variables = set(x_vars)
		self.problem_variables_user = set(x_vars_user)
		self._problem_variables_ordered = tuple(x_vars)
		self.user_to_pycollo_problem_variables_mapping_in_order = dict(zip(x_vars_user, x_vars))

	def _initialise_problem_variable_nodes(self):
		for x_var_user in self.user_to_pycollo_problem_variables_mapping_in_order:
			_ = Node(x_var_user, self)

	def _initialise_auxiliary_constant_nodes(self, aux_info):
		self.user_symbol_to_expression_auxiliary_mapping = {}
		self._user_constants = set()
		for key, value in aux_info.items():
			if isinstance(value, numbers.Real):
				self._user_constants.add(key)
				node = Node(key, self, value=value)
			else:
				self.user_symbol_to_expression_auxiliary_mapping[key] = value

	def _initialise_auxiliary_intermediate_nodes(self):
		for node_symbol, node_expr in self.user_symbol_to_expression_auxiliary_mapping.items():
			_ = Node(node_symbol, self, equation=node_expr)

	def _initialise_function(self, expr):

		def substitute_function_for_root_symbols(expr):

			def traverse_root_branch(expr, max_tier):
				root_node = Node(expr, self)
				max_tier = max(max_tier, root_node.tier)
				return root_node.symbol, root_node, max_tier

			if isinstance(expr, sym.Expr):
				root_symbol, root_node, max_tier = traverse_root_branch(expr, 0)
				return root_symbol, [root_node], max_tier
			else:
				expr_subbed = []
				expr_nodes = []
				max_tier = 0
				for entry_expr in expr:
					root_symbol, root_node, max_tier = traverse_root_branch(entry_expr, max_tier)
					expr_subbed.append(root_symbol)
					expr_nodes.append(root_node)
				return_matrix = sym.Matrix(np.array(expr_subbed).reshape(expr.shape))
				return return_matrix, expr_nodes, max_tier

		def separate_precomputable_and_dependent_nodes(expr, nodes):
			precomputable_nodes = set()
			dependent_nodes = set()
			all_nodes = set(nodes)
			for free_symbol in expr.free_symbols:
				node = self.symbols_to_nodes_mapping[free_symbol]
				all_nodes.update(node.dependent_nodes)
			precomputable_nodes = set()
			dependent_nodes = set()
			for node in all_nodes:
				if node.is_precomputable:
					precomputable_nodes.add(node)
				else:
					dependent_nodes.add(node)
			return precomputable_nodes, dependent_nodes

		def sort_dependent_nodes_by_tier(dependent_nodes, max_tier):
			dependent_tiers = {i: [] for i in range(max_tier+1)}
			for node in dependent_nodes:
				dependent_tiers[node.tier].append(node)
			dependent_tiers[0] = list(self._variable_nodes.values())
			return dependent_tiers

		expr_subbed, expr_nodes, max_tier = substitute_function_for_root_symbols(expr)

		precomputable_nodes, dependent_nodes = separate_precomputable_and_dependent_nodes(expr_subbed, expr_nodes)

		dependent_tiers = sort_dependent_nodes_by_tier(dependent_nodes, max_tier)

		return expr_subbed, expr_nodes, precomputable_nodes, dependent_tiers

	@property
	def problem_variables_ordered(self):
		return self._problem_variables_ordered

	@property
	def variable_nodes(self):
		return tuple(self._variable_nodes.values())

	@property
	def constant_nodes(self):
		return tuple(self._constant_nodes.values())

	@property
	def number_nodes(self):
		return tuple(self._number_nodes.values())

	@property
	def intermediate_nodes(self):
		return tuple(self._intermediate_nodes.values())
	
	@property
	def root_nodes(self):
		return self.variable_nodes + self.constant_nodes + self.number_nodes

	@property
	def nodes(self):
		return self.root_nodes + self.intermediate_nodes

	@property
	def precomputable_nodes(self):
		return tuple(self._precomputable_nodes.values())

	@property
	def symbols_to_nodes_mapping(self):
		return {**self._variable_nodes, **self._constant_nodes, **self._number_nodes, **self._intermediate_nodes}

	def hybrid_symbolic_algorithmic_differentiation(self, target_function, function_nodes, precomputable_nodes, dependent_nodes_by_tier):

		def differentiate(function_nodes, wrt):
			derivative_full = [[function_node.derivative_as_symbol(single_wrt) for single_wrt in wrt] for function_node in function_nodes]
			return sym.Matrix(derivative_full)

		def compute_target_function_derivatives_for_each_tier(dependent_nodes_by_tier_collapsed):
			df_de = []
			for node_tier in dependent_nodes_by_tier_collapsed:
				derivative = differentiate(function_nodes, node_tier)
				df_de.append(derivative)
			return df_de

		def compute_delta_matrices_for_each_tier(num_e0, dependent_nodes_by_tier_collapsed):
			delta_matrices = [1]
			for tier_num, dependent_nodes_tier in enumerate(dependent_nodes_by_tier_collapsed[1:], 1):
				num_ei = len(dependent_nodes_tier)
				delta_matrix_i = sym.Matrix.zeros(num_ei, num_e0)
				for by_tier_num in range(tier_num):
					delta_matrix_j = delta_matrices[by_tier_num]
					deriv_matrix = differentiate(dependent_nodes_tier, dependent_nodes_by_tier_collapsed[by_tier_num])
					delta_matrix_i += deriv_matrix*delta_matrix_j
				delta_matrices.append(delta_matrix_i)
			return delta_matrices

		def compute_derivative_from_function_derivatives_and_delta_matrices():
			num_f = len(function_nodes)
			derivative = sym.Matrix.zeros(num_f, num_e0)
			for df_dei, delta_i in zip(df_de, delta_matrices):
				derivative += df_dei*delta_i
			return derivative

		dependent_nodes_by_tier_collapsed = []
		for nodes in dependent_nodes_by_tier.values():
			if nodes:
				dependent_nodes_by_tier_collapsed.append(nodes)

		# print(f"Target Function:\n----------------\n{target_function}\n")
		# print(f"Function Nodes:\n-----------------")
		# for function_node in function_nodes:
		# 	print(function_node)
		# print(f"\nPrecomputable Nodes:\n-------------")
		# for node in precomputable_nodes:
		# 	print(node)
		# print(f"Dependent Nodes:\n---------------")
		# for nodes in dependent_nodes_by_tier_collapsed:
		# 	print(f"{nodes}")
		# print('')

		df_de = compute_target_function_derivatives_for_each_tier(dependent_nodes_by_tier_collapsed)

		num_e0 = len(dependent_nodes_by_tier_collapsed[0])
		delta_matrices = compute_delta_matrices_for_each_tier(num_e0, dependent_nodes_by_tier_collapsed)

		derivative = compute_derivative_from_function_derivatives_and_delta_matrices()

		# print(f"\nDerivative:\n-------------\n{derivative}\n\n\n")

		return derivative

	def __str__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}({self.problem_variables_user})"

	def __repr__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}(problem_variables_user={self.problem_variables_user})"


