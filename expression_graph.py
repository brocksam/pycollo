import abc
import itertools
import numbers
from timeit import default_timer as timer
import weakref

import numpy as np
import sympy as sym


class ExpressionGraph:

	def __init__(self, problem_variables, auxiliary_information, objective, constraints, lagrangian):

		self.x_vars = problem_variables

		self._intermediate_node_num = itertools.count()
		self._number_node_num = itertools.count()
		self._constant_node_num = itertools.count()
		# self._variable_node_num = itertools.count()
		self._variable_nodes = {}
		self._constant_nodes = {}
		self._number_nodes = {}
		self._intermediate_nodes = {}
		self._aux_subs_user = {}
		self._aux_syms_user = set()
		_ = self._initialise_problem_variable_nodes(problem_variables)
		aux_subs = self._initialise_auxiliary_information_nodes(auxiliary_information)
		_ = self._initialise_auxiliary_intermediate_nodes(aux_subs)

		for node in self.nodes:
			print(f'Initialising node {node.symbol}')
			print(node)
			print(node.is_precomputable)
			# print(node._derivatives_wrt)
			print('')

		# Objective function
		# self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers = self._initialise_function(objective)
		# self.dJ_dx = self.hybrid_symbolic_algorithmic_differentiation(self.J, self.x_vars, self.J_tiers)
		# dJ_dx_cached = self.hybrid_symbolic_algorithmic_differentiation_cached(self.J, self.J_node, self.J_precomputable, self.J_dependent_tiers)

		# Jacobian of the constraints
		# self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers = self._initialise_function(constraints)
		# self.dc_dx = self.hybrid_symbolic_algorithmic_differentiation(self.c, self.x_vars, self.c_dependent_tiers)
		# dc_dx_cached = self.hybrid_symbolic_algorithmic_differentiation_cached(self.c, self.c_nodes, self.c_precomputable, self.c_dependent_tiers)

		# print('\n\n\n')
		# if self.dc_dx == dc_dx_cached:
		# 	print("PASSED!")
		# else:
		# 	print("FAILED!")
		print('\n\n\n')
		raise ValueError

		# Hessian of the Lagrangian
		self.L, self.L_tiers = self._initialise_lagrangian(lagrangian)

	def _initialise_problem_variable_nodes(self, x_vars):
		
		self._prob_vars_user = set(x_vars)

		for x in x_vars:
			node = self.node_from_key(x)
			self._variable_nodes[node.symbol] = node


		print('\n\n\n')
		raise ValueError
		
		return None

	def _initialise_auxiliary_information_nodes(self, aux_info):

		aux_subs = {}

		for key, value in aux_info.items():
			if isinstance(value, numbers.Real):
				v_sym = sym.sympify(value)
				self._aux_syms_user.add(key)
				node = self.node_from_key(key)
				node.value = v_sym
				self._constant_nodes[node.symbol] = node
			else:
				aux_subs[key] = value
				self._aux_subs_user[value] = key

		return aux_subs

	def _initialise_auxiliary_intermediate_nodes(self, aux_subs):

		for node_expr in aux_subs.values():
			_ = self.traverse_branch(node_expr)
		return None

	def _initialise_function(self, expr):

		def substitute_function_for_root_symbols(expr):

			def traverse_root_branch(expr, max_tier):
				root_node = self.traverse_branch(expr)
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
			
			for free_sym in expr.free_symbols:
				node = self.nodes_dict[free_sym]
				precomputable_nodes.update(node.precomputable_nodes)
				dependent_nodes.update(node.dependent_nodes)

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
	def problem_variables(self):
		return tuple(self._variable_nodes.keys())

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
	def nodes_dict(self):
		return {**self._variable_nodes, **self._constant_nodes, **self._number_nodes, **self._intermediate_nodes}

	def node_from_key(self, key):
		if key in self._prob_vars_user:
			node = VariableNode(key, self)
			if node not in self.variable_nodes:
				self._variable_nodes[node.symbol] = node
		elif key.is_Number:
			node = NumberNode(key, self)
			if node not in self.number_nodes:
				self._number_nodes[node.symbol] = node
		elif key in self._aux_syms_user:
			node = ConstantNode(key, self)
			if node not in self._constant_nodes:
				self._constant_nodes[node.symbol] = node
		else:
			key = self._aux_subs_user.get(key, key)
			node = IntermediateNode(key, self)
			if node not in self._intermediate_nodes:
				self._intermediate_nodes[node.symbol] = node

		return node

	def traverse_branch(self, child_expr):

		root_node = self.node_from_key(child_expr)

		for expr in sym.preorder_traversal(child_expr):
			child_node = self.node_from_key(expr)
			if child_node is not None and child_node.is_root:
				pass
			else:
				if not expr.is_Atom:
					child_node.operation = expr.func
					for arg in expr.args:
						parent_node = self.node_from_key(arg)
						child_node.new_parent(parent_node)
						parent_node.new_child(child_node)
					child_node._assert_precomputability()
		return root_node

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


class cachedproperty:

	def __init__(self, func):
		self.func = func

	def __get__(self, instance, cls):
		if instance is None:
			return self
		else:
			value = self.func(instance)
			setattr(instance, self.func.__name__, value)
			return value


class Cached(type):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__cache = weakref.WeakValueDictionary()

	def __call__(self, *args):
		if args in self.__cache:
			return self.__cache[args]
		else:
			obj = super().__call__(*args)
			self.__cache[args] = obj
			return obj


class Node(metaclass=Cached):

	def __init__(self, key, graph):
		self.key = key
		self.graph = graph
		self._set_node_type_stateful_object()
		
		self._child_nodes = set()

	def _set_node_type_stateful_object(self):
		if key in self.graph._prob_vars_user:
			self._type = VariableNode
		elif key.is_Number:
			self._type = NumberNode
		elif key in self._aux_syms_user:
			self._type = ConstantNode
		else:
			self._type = IntermediateNode

	def _create_and_set_new_node_symbol(self):
		self.symbol = self._type._create_or_get_new_node_symbol(self)

	@property
	def child_nodes(self):
		return self._child_nodes

	def new_child(self, child):
		self._child_nodes.add(child)

	@property
	def parent_nodes(self):
		return self._type._parent_nodes(self)

	def new_parent(self, parent):
		self._type.new_parent

	def __str__(self):
		return self._type.__str__(self)

	def __repr__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}(key={key}, graph={graph})"



class ExpressionNodeABC(abc.ABC):

	@staticmethod
	def _get_new_symbol_number(node_instance):
		return _node_number_counter.__next__()

	@staticmethod
	@abc.abstract_method
	def _node_number_counter(node_instance):
		pass

	@staticmethod
	@abc.abstract_method
	def _node_symbol_number(node_instance):
		pass

	@staticmethod
	@abc.abstract_method
	def _create_or_get_new_node_symbol(node_instance):
		new_symbol_number = _get_new_symbol_number(node_instance)
		new_symbol_name = f"_{_node_symbol_letter}_{_get_new_symbol_number}"
		new_symbol = sym.symbols(new_symbol_name)
		return new_symbol

	@staticmethod
	@abc.abstract_method
	def parent_nodes(node_instance):
		pass

	@staticmethod
	@abc.abstract_method
	def new_parent(node_instance):
		pass

	@staticmethod
	def __str__(node_instance):
		return f"{node_instance.key} = {node_instance.value}"



class RootNode(ExpressionNodeABC):

	@staticmethod
	@abc.abstract_method
	def _create_or_get_new_node_symbol(node_instance):
		new_symbol_number = _get_new_symbol_number(node_instance)
		new_symbol_name = f"_{node_symbol_letter}_{new_symbol_number}"
		return new_symbol





# class IntermediateNode(ExpressionNode):
# 	"""Symbol: _w_{}"""

# 	def __init__(self, key, graph):
# 		super().__init__(key, graph)
# 		number = self.graph._intermediate_node_num.__next__()
# 		self.symbol = sym.symbols(f'_w_{number}')
# 		self._parent_nodes = []
# 		# self._derivatives_as_symbol = {self.symbol: 1}
# 		self._arguments = []
# 		self._num_arguments = 0
# 		# self._derivative_symbols = [1]
# 		if self.is_precomputable:
# 			self._derivatives_wrt = {}
# 		else:
# 			self._derivatives_wrt = {self: 1}

# 	@property
# 	def parent_nodes(self):
# 		return self._parent_nodes
	
# 	def new_parent(self, parent):
# 		self._parent_nodes.append(parent)
# 		self._arguments.append(parent.symbol)
# 		self._num_arguments = len(self.arguments)

# 	@property
# 	def arguments(self):
# 		return self._arguments

# 	@cachedproperty
# 	def num_arguments(self):
# 		return self._num_arguments

# 	@property
# 	def is_root(self):
# 		return False

# 	@property
# 	def is_precomputable(self):
# 		return self._is_precomputable

# 	def _assert_precomputability():
# 		self._is_precomputable = all([parent.is_precomputable for parent in self.parent_nodes])

# 	@cachedproperty
# 	def tier(self):
# 		if self.is_precomputable:
# 			return 0
# 		else:
# 			return max([parent.tier for parent in self.parent_nodes]) + 1

# 	@cachedproperty
# 	def dependent_nodes(self):
# 		return set.union(set([self]) if not self.is_precomputable else set(), *[parent.dependent_nodes for parent in self.parent_nodes])

# 	@cachedproperty
# 	def precomputable_nodes(self):
# 		return set.union(set([self]) if self.is_precomputable else set(), *[parent.precomputable_nodes for parent in self.parent_nodes])

# 	@cachedproperty
# 	def value(self):
# 		if self.is_precomputable:
# 			return float(self.operation(*[parent.value for parent in self.parent_nodes]))
# 		else:
# 			return self.operation(*self.arguments)

# 	def derivative(self, wrt):
# 		return self._derivatives.get(wrt, 0)

# 	def derivative_wrt(self, wrt):
# 		return self._derivatives.get(wrt, 0)

# 	def derivative_as_symbol(self, wrt):
# 		return self._derivatives_as_symbol.get(wrt, 0)

# 	# @cachedproperty
# 	# def _derivatives(self):
# 	# 	return {arg: self.value.diff(arg) for arg in self.arguments}

# 	def __str__(self):
# 		return f"{self.symbol} = {self.key} = {self.value}"


# class RootNode(ExpressionNode):
	
# 	@property
# 	def is_root(self):
# 		return True

# 	@property
# 	def is_precomputable(self):
# 		return True

# 	@property
# 	def tier(self):
# 		return 0

# 	@property
# 	def dependent_nodes(self):
# 		return set()

# 	@property
# 	def precomputable_nodes(self):
# 		return set([self])

# 	@property
# 	def _derivatives_wrt(self):
# 		return {}


# class NumberNode(RootNode):
# 	"""Symbol: _n_{}"""
	
# 	def __init__(self, key, graph):
# 		super().__init__(key, graph)
# 		self.symbol = sym.symbols(f'_n_{self.graph._number_node_num.__next__()}')

# 	@property
# 	def value(self):
# 		return float(self.key)


# class ConstantNode(RootNode):
# 	"""Symbol: _a_{}"""
	
# 	def __init__(self, key, graph):
# 		super().__init__(key, graph)
# 		self.symbol = sym.symbols(f'_a_{self.graph._constant_node_num.__next__()}')
# 		self.value = None

# 	def __str__(self):
# 		return f"{self.symbol} = {self.key} = {self.value}"


# class VariableNode(RootNode):
# 	"""Symbol: _x_{}"""
	
# 	def __init__(self, key, graph):
# 		super().__init__(key, graph)
# 		self.symbol = sym.symbols(f'_x_{self.graph._variable_node_num.__next__()}')
# 		self.value = self.symbol

# 	@property
# 	def is_precomputable(self):
# 		return False

# 	@property
# 	def dependent_nodes(self):
# 		return set([self])

# 	@property
# 	def precomputable_nodes(self):
# 		return set()

# 	@property
# 	def _derivatives_wrt(self):
# 		return {self: 1}


def build_expression_graph(problem_variables, auxiliary_information, objective_function, constraints, hessian_lagrangian):

	expression_graph = ExpressionGraph(problem_variables, auxiliary_information, objective_function, constraints, hessian_lagrangian)

	return expression_graph

	

# class PycolloExpr:

# 	@staticmethod
# 	def evaluate(node):
# 		raise NotImplementedError

# 	@staticmethod
# 	def derivatives(node):
# 		raise NotImplementedError


# class PycolloAdd(PycolloExpr):

# 	@staticmethod
# 	@profile
# 	def evaluate(node):
# 		total = node.arguments[0]
# 		for arg in node.arguments[1:]:
# 			total += arg
# 		return total #sym.Add(*node.arguments)

# 	@staticmethod
# 	@profile
# 	def derivatives(node):
# 		return [1] * node.num_arguments


# class PycolloMul(PycolloExpr):

# 	@staticmethod
# 	@profile
# 	def evaluate(node):
# 		return sym.Mul(*node.arguments)

# 	@staticmethod
# 	@profile
# 	def derivatives(node):
# 		return [node.value/arg for arg in node.arguments]


# class PycolloPow(PycolloExpr):

# 	@staticmethod
# 	@profile
# 	def evaluate(node):
# 		return sym.Pow(*node.arguments)

# 	@staticmethod
# 	@profile
# 	def derivatives(node):
# 		a, b = node.arguments
# 		a**b*b/a
# 		a**b*sym.log(a)
# 		return[a**b*b/a, a**b*sym.log(a)]


# class PycolloSin(PycolloExpr):

# 	@staticmethod
# 	@profile
# 	def evaluate(node):
# 		return sym.sin(*node.arguments)

# 	@staticmethod
# 	@profile
# 	def derivatives(node):
# 		return sym.cos(*node.arguments)


# class PycolloCos(PycolloExpr):

# 	@staticmethod
# 	@profile
# 	def evaluate(node):
# 		return sym.cos(*node.arguments)

# 	@staticmethod
# 	@profile
# 	def derivatives(node):
# 		return -sym.sin(*node.arguments)


# sympy_pycollo_mapping = {
# 	sym.Add: PycolloAdd,
# 	sym.Mul: PycolloMul,
# 	sym.Pow: PycolloPow,
# 	sym.cos: PycolloCos,
# 	sym.sin: PycolloSin,
# }



if __name__ == '__main__':

	print('\n')

	no_subs = False

	y0, y1, u0, q0 = sym.symbols('y0 y1 u0 q0')
	g = sym.symbols('g')
	m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
	phi0mu, phi0sigma = sym.symbols('phi0mu phi0sigma')
	dphi0mu, dphi0sigma = sym.symbols('dphi0mu dphi0sigma')
	T0mu, T0sigma = sym.symbols('T0mu T0sigma')
	phi0, dphi0, T0 = sym.symbols('phi0 dphi0 T0')
	c0, s0 = sym.symbols('c0 s0')
	phi0min, phi0max = sym.symbols('phi0min phi0max')
	dphi0min, dphi0max = sym.symbols('dphi0min dphi0max')
	T0min, T0max = sym.symbols('T0min T0max')
	q0min, q0max = sym.symbols('q0min q0max')
	gm0, p0c0, gm0p0c0, gm0p0c0plusT0, T0T0, q0minusT0T0 = sym.symbols('gm0 p0c0 gm0p0c0 gm0p0c0plusT0 T0T0 q0minusT0T0')

	start_total = timer()

	start = timer()

	if no_subs:
		aux_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, phi0mu: (phi0min + phi0max)/2, phi0sigma: phi0max - phi0min, dphi0mu: (dphi0min + dphi0max)/2, dphi0sigma: dphi0max - dphi0min, T0mu: (T0min + T0max)/2, T0sigma: T0max - T0min, phi0: phi0mu + y0*phi0sigma, phi0min: -np.pi, phi0max: np.pi, dphi0min: -10, dphi0max: 10, T0min: -15, T0max: 15, q0min: 0, q0max: 200})
	else:
		aux_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, I0: m0*(k0**2 + p0**2), l0: p0 + d0, phi0: phi0mu + y0*phi0sigma, phi0mu: (phi0min + phi0max)/2, phi0sigma: phi0max - phi0min, dphi0mu: (dphi0min + dphi0max)/2, dphi0sigma: dphi0max - dphi0min, T0mu: (T0min + T0max)/2, T0sigma: T0max - T0min, dphi0: dphi0mu + y1*dphi0sigma, T0: T0mu + u0*T0sigma, c0: sym.cos(phi0), phi0min: -np.pi, phi0max: np.pi, dphi0min: -10, dphi0max: 10, T0min: -15, T0max: 15, q0min: 0, q0max: 200, gm0: g*m0, p0c0: p0*c0, gm0p0c0: gm0*p0c0, gm0p0c0plusT0: gm0p0c0 + T0, T0T0: T0**2, q0minusT0T0: q0 - T0T0})

	stop = timer()
	print(f"Make aux data dict: {stop - start}", '\n')

	x_vars = sym.Matrix([y0, y1, u0, q0])

	J = (2+I0)*q0

	if no_subs:
		c = sym.Matrix([(dphi0mu + y1*dphi0sigma)/phi0sigma, (g*m0*p0*sym.cos(phi0) + (T0mu + u0*T0sigma))/((m0*(k0**2 + p0**2))*dphi0sigma), q0 - (T0mu + u0*T0sigma)**2])
	else:
		c = sym.Matrix([dphi0/phi0sigma, (gm0p0c0plusT0)/(I0*dphi0sigma), q0minusT0T0])

	L = sym.Matrix([])

	start = timer()
	expression_graph = build_expression_graph(x_vars, aux_data, J, c, L)
	stop = timer()
	print(f"Build expression graph: {stop - start}", '\n')

	print(expression_graph.J)

	print(expression_graph.dJ_dx)

	print('')

	print(expression_graph.c)

	print(expression_graph.dc_dx)

	print('')










	# dJ_dx = hybrid_symbolic_algorithmic_differentiation(J, x_vars, e_vars, e_subs, tier_slices)
	# print('Symbolic objective gradient calculated.\n')
	# print(dJ_dx, '\n\n')

	# start = timer()
	# dc_dx = hybrid_symbolic_algorithmic_differentiation(c, x_vars, e_vars, e_subs, tier_slices)
	# stop = timer()
	# stop_total = timer()
	# print(f"Produce Jacobian: {stop - start}", '\n')
	# print(f"Number subs tiers: {len(tier_slices)}", '\n')
	# print(f"Total time: {stop_total - start_total}", '\n')
	# print(dc_dx, '\n')


