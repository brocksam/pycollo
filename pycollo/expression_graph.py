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
		self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers = self._initialise_function(objective)
		self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers = self._initialise_function(constraints)

		for node in self.nodes:
			print('')
			print('-----------------------------------------')
			print(node)
			print('-----------------------------------------')
			print(f'Tier: {node.tier}')
			print(f'Is Precomputable: {node.is_precomputable}')
			print(f'Value: {node.value}')
			# print('Child Nodes:')
			# for child in node.child_nodes:
			# 	print(child)
			# if not node.is_root:
			# 	print('\nParent Nodes:')
			# 	for parent in node.parent_nodes:
			# 		print(parent)
				# print('\nDerivative Nodes:')
				# for parent in node.parent_nodes:
				# 	node_value = node.value
				# 	deriv = node_value.diff(parent.symbol)
				# 	subs_dict = self.symbols_to_nodes_mapping
				# 	deriv_node = subs_dict.get(deriv)
				# 	if deriv_node is None:
				# 		deriv_expr = deriv.subs(subs_dict)
				# 		deriv_node = Node(deriv_expr, self)
				# 	print(f'd({node.value})/d({parent.symbol}): {deriv}')
				# 	print(deriv_node)
			print('')

		# for node in self.nodes:
		# 	print(f'Initialising node {node.symbol}')
		# 	print(node)
		# 	print(node.is_precomputable)
		# 	# print(node._derivatives_wrt)
		# 	print('')

		# Objective function
		# 
		# self.dJ_dx = self.hybrid_symbolic_algorithmic_differentiation(self.J, self.x_vars, self.J_tiers)
		# dJ_dx_cached = self.hybrid_symbolic_algorithmic_differentiation_cached(self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers)

		# Jacobian of the constraints
		# 
		# self.dc_dx = self.hybrid_symbolic_algorithmic_differentiation(self.c, self.x_vars, self.c_dependent_tiers)
		# dc_dx_cached = self.hybrid_symbolic_algorithmic_differentiation_cached(self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers)

		# print('\n\n\n')
		# if self.dc_dx == dc_dx_cached:
		# 	print("PASSED!")
		# else:
		# 	print("FAILED!")
		print('\n\n\n')
		raise ValueError

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
		zero_node = Node(0, self)
		one_node = Node(1, self)
		two_node = Node(2, self)

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
				node = Node(key, self)
				node._value = float(value)
			else:
				self.user_symbol_to_expression_auxiliary_mapping[key] = value

	def _initialise_auxiliary_intermediate_nodes(self):
		for node_symbol, node_expr in self.user_symbol_to_expression_auxiliary_mapping.items():
			symbol_root_node = Node(node_symbol, self)
			expr_root_node = Node(node_expr, self)
			symbol_root_node.new_parent(expr_root_node)
			expr_root_node.new_child(symbol_root_node)

	def _initialise_function(self, expr):

		def substitute_function_for_root_symbols(expr):

			def traverse_root_branch(expr, max_tier):
				root_node = Node(expr, self)
				max_tier = max(max_tier, root_node.tier)
				return root_node.symbol, root_node, max_tier

			if isinstance(expr, sym.Expr):
				return traverse_root_branch(expr, 0)
			else:
				expr_subbed = []
				expr_nodes = []
				max_tier = 0
				for entry_expr in expr:
					root_symbol, root_node, max_tier = traverse_root_branch(entry_expr, max_tier)
					expr_subbed.append(root_symbol)
					expr_nodes.append(root_node)
				return sym.Matrix(expr_subbed), expr_nodes, max_tier

		def separate_precomputable_and_dependent_nodes(expr):
			
			precomputable_nodes = set()
			dependent_nodes = set()
			
			for free_symbol in expr.free_symbols:
				print(free_symbol)
				# node = self.symbols_to_nodes_mapping[free_symbol]
				# precomputable_nodes.update(node.precomputable_nodes)
				# dependent_nodes.update(node.dependent_nodes)

			return precomputable_nodes, dependent_nodes

		def sort_dependent_nodes_by_tier(dependent_nodes, max_tier):

			dependent_tiers = {i: [] for i in range(max_tier+1)}

			for node in dependent_nodes:
				dependent_tiers[node.tier].append(node)

			dependent_tiers[0] = list(self._variable_nodes.values())

			return dependent_tiers

		expr_subbed, expr_nodes, max_tier = substitute_function_for_root_symbols(expr)

		precomputable_nodes, dependent_nodes = separate_precomputable_and_dependent_nodes(expr_subbed)

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

	# @property
	# def symbols_to_user_symbols_mapping(self):
	# 	return {symbol: node.key for symbol, node in self.symbols_to_nodes_mapping.items()}
	

	# def node_from_key(self, key):
	# 	if key in self.graph.user_to_pycollo_problem_variables_mapping_in_order.get():
	# 		node = VariableNode(key, self)
	# 		if node not in self.variable_nodes:
	# 			self._variable_nodes[node.symbol] = node
	# 	elif key.is_Number:
	# 		node = NumberNode(key, self)
	# 		if node not in self.number_nodes:
	# 			self._number_nodes[node.symbol] = node
	# 	elif key in self._aux_syms_user:
	# 		node = ConstantNode(key, self)
	# 		if node not in self._constant_nodes:
	# 			self._constant_nodes[node.symbol] = node
	# 	else:
	# 		key = self._aux_subs_user.get(key, key)
	# 		node = IntermediateNode(key, self)
	# 		if node not in self._intermediate_nodes:
	# 			self._intermediate_nodes[node.symbol] = node

	# 	return node

	# def traverse_branch(self, child_expr):

	# 	root_node = self.node_from_key(child_expr)

	# 	for expr in sym.preorder_traversal(child_expr):
	# 		child_node = self.node_from_key(expr)
	# 		if child_node is not None and child_node.is_root:
	# 			pass
	# 		else:
	# 			if not expr.is_Atom:
	# 				child_node.operation = expr.func
	# 				for arg in expr.args:
	# 					parent_node = self.node_from_key(arg)
	# 					child_node.new_parent(parent_node)
	# 					parent_node.new_child(child_node)
	# 				child_node._assert_precomputability()
	# 	return root_node

	# @profile
	def hybrid_symbolic_algorithmic_differentiation(self, target_func, variable_symbols, tier_symbols_dict):

		def by_differentiation(function, wrt):
			return function.diff(wrt).T

		def by_jacobian(function, wrt):
			return function.jacobian(wrt)

		symbol_tiers = [sym.Matrix([node.symbol for node in node_list]) for node_list in tier_symbols_dict.values()]
		symbol_tiers[0] = sym.Matrix([node.symbol for node in self.variable_nodes])

		substitution_tiers = [sym.Matrix([node.value for node in node_list]) for node_list in list(tier_symbols_dict.values())[1:]]

		num_e0 = self.x_vars.shape[0]
		if isinstance(target_func, sym.Matrix):
			differentiate = by_jacobian
			num_f = target_func.shape[0]
			transpose_before_return = False
		else:
			differentiate = by_differentiation
			num_f = 1
			transpose_before_return = True

		df_de = [differentiate(target_func, symbol_tier) for symbol_tier in symbol_tiers]

		delta_matrices = [1]

		for i, substitution_tier in enumerate(substitution_tiers):
			num_ei = substitution_tier.shape[0]
			delta_matrix_i = sym.Matrix.zeros(num_ei, num_e0)
			for j in range(i + 1):
				delta_matrix_j = delta_matrices[j]
				deriv_matrix = substitution_tier.jacobian(symbol_tiers[j])
				delta_matrix_i += deriv_matrix*delta_matrix_j
			delta_matrices.append(delta_matrix_i)

		derivative = sym.Matrix.zeros(num_f, num_e0)

		for df_dei, delta_i in zip(df_de, delta_matrices):
			derivative += df_dei*delta_i

		if transpose_before_return:
			return sym.Matrix(derivative).T
		else:
			return derivative

	# @profile
	def hybrid_symbolic_algorithmic_differentiation_cached(self, target_function, function_nodes, precomputable_nodes, dependent_nodes_by_tier):

		# @profile
		def differentiate(function_nodes, wrt):
			derivative_full = [[function_node.derivative_as_symbol(single_wrt.symbol) for single_wrt in wrt] for function_node in function_nodes]
			return sym.Matrix(derivative_full)

		def compute_target_function_derivatives_for_each_tier():
			df_de = []
			for node_tier in dependent_nodes_by_tier.values():
				derivative = differentiate(function_nodes, node_tier)
				df_de.append(derivative)
			return df_de

		def compute_delta_matrices_for_each_tier(num_e0):
			delta_matrices = [1]
			for tier_num, dependent_nodes_tier in list(dependent_nodes_by_tier.items())[1:]:
				num_ei = len(dependent_nodes_tier)
				delta_matrix_i = sym.Matrix.zeros(num_ei, num_e0)
				for by_tier_num in range(tier_num):
					delta_matrix_j = delta_matrices[by_tier_num]
					deriv_matrix = differentiate(dependent_nodes_tier, dependent_nodes_by_tier[by_tier_num])
					delta_matrix_i += deriv_matrix*delta_matrix_j
				delta_matrices.append(delta_matrix_i)
			return delta_matrices

		def compute_derivative_from_function_derivatives_and_delta_matrices():
			num_f = len(function_nodes)
			derivative = sym.Matrix.zeros(num_f, num_e0)
			for df_dei, delta_i in zip(df_de, delta_matrices):
				derivative += df_dei*delta_i
			return derivative

		df_de = compute_target_function_derivatives_for_each_tier()

		for tier in df_de:
			print(tier, '\n')
		print('\n\n\n')
		raise ValueError

		num_e0 = len(dependent_nodes_by_tier[0])
		delta_matrices = compute_delta_matrices_for_each_tier(num_e0)

		derivative = compute_derivative_from_function_derivatives_and_delta_matrices()

		return derivative

	def __str__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}({self.problem_variables_user})"

	def __repr__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}(problem_variables_user={self.problem_variables_user})"


