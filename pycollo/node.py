import abc
import itertools
import numbers
from timeit import default_timer as timer
import weakref

import numpy as np
import sympy as sym

# from .operations import determine_operation


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

	def __call__(self, *args, **kwargs):
		if args in self.__cache:
			return self.__cache[args]
		else:
			obj = super().__call__(*args, **kwargs)
			self.__cache[args] = obj
			return obj
	

class Node(metaclass=Cached):

	def __init__(self, key, graph, *, value=None, equation=None):
		self.key = sym.sympify(key)
		self.graph = graph
		self._operation = None
		self._set_node_type_stateful_object()
		self._associate_new_node_with_graph()
		self._child_nodes = set()
		self._parent_nodes = []
		self._derivatives = {}
		self.value = value
		self.equation = equation

	def _set_node_type_stateful_object(self):
		if self.graph.user_to_pycollo_problem_variables_mapping_in_order.get(self.key) is not None:
			self._type = VariableNode
		elif self.key in self.graph.lagrange_syms:
			self._type = VariableNode
		elif self.key.is_Number:
			self._type = NumberNode
		elif self.key in self.graph._user_constants:
			self._type = ConstantNode
		else:
			self._type = IntermediateNode

	def _associate_new_node_with_graph(self):
		self.symbol = self._type._create_or_get_new_node_symbol(self)
		self._type._graph_node_group(self)[self.symbol] = self

	@property
	def child_nodes(self):
		return self._child_nodes

	def new_child(self, child):
		self._child_nodes.add(child)

	@property
	def parent_nodes(self):
		return self._type.parent_nodes(self)

	def new_parent(self, parent):
		self._type.new_parent(self, parent)

	@property
	def arguments(self):
		return self._type.arguments(self)

	@property
	def operation(self):
		return self._operation

	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, value):
		self._value = self._type._set_value(self, value)

	@property
	def equation(self):
		return self._equation

	@equation.setter
	def equation(self, equation):
		self._equation = self._type._set_equation_and_inspect_parents(self, equation)

	@property
	def expression(self):
		return self._expression

	def derivative_as_symbol(self, wrt):
		node = self.derivative_as_node(wrt)
		if node is self.graph._zero_node or node is self.graph._one_node:
			return int(node.value)
		else:
			return node.symbol

	def derivative_as_node(self, wrt):
		return self._type._get_derivative_wrt(self, wrt)

	@cachedproperty
	def is_root(self):
		return self._type.is_root()

	@cachedproperty
	def is_precomputable(self):
		is_precomputable = self._type.is_precomputable(self)
		if is_precomputable:
			self.graph._precomputable_nodes.update({self.symbol: self})
		return is_precomputable

	@cachedproperty
	def is_vector(self):
		return self._type.is_vector(self)

	@cachedproperty
	def dependent_nodes(self):
		if self.is_root:
			return set()
		else:
			nodes = set.union(*[set.union(parent.dependent_nodes, set([parent])) for parent in self.parent_nodes])
			return nodes

	@property
	def numbafy_expression(self):
		if self.value is not None:
			return f'{self.symbol} = {self.value}'
		else:
			return f'{self.symbol} = {self.expression}'

	@cachedproperty
	def tier(self):
		return self._type.tier(self)

	def __str__(self):
		return self._type._str(self)

	def __repr__(self):
		cls_name = self.__class__.__name__
		return f"{cls_name}({self.symbol})"



class ExpressionNodeABC(abc.ABC):

	@staticmethod
	@abc.abstractmethod
	def _graph_node_group(node_instance):
		pass

	@staticmethod
	def _get_new_symbol_number(node_instance):
		return node_instance._type._node_number_counter(node_instance).__next__()

	@staticmethod
	@abc.abstractmethod
	def _node_number_counter(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def _create_or_get_new_node_symbol(node_instance):
		new_symbol_number = node_instance._type._get_new_symbol_number(node_instance)
		node_symbol_letter = node_instance._type._node_symbol_letter(node_instance)
		new_symbol_name = f"_{node_symbol_letter}{new_symbol_number}"
		new_symbol = sym.Symbol(new_symbol_name)
		return new_symbol

	@staticmethod
	@abc.abstractmethod
	def _set_value(node_instance, value):
		pass

	@staticmethod
	@abc.abstractmethod
	def _set_equation_and_inspect_parents(node_instance, equation):
		pass

	@staticmethod
	@abc.abstractmethod
	def _get_derivative_wrt(node_instance, wrt):
		pass

	@staticmethod
	@abc.abstractmethod
	def parent_nodes(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def new_parent(node_instance, parent):
		pass

	@staticmethod
	@abc.abstractmethod
	def arguments(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def tier(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def is_root(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def is_precomputable(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def is_vector(node_instance):
		pass

	@staticmethod
	@abc.abstractmethod
	def _str(node_instance):
		pass



class RootNode(ExpressionNodeABC):

	_parent_nodes_not_allowed_error_message = (f"Object of type RootNode do not have parent nodes.")
	_parent_nodes_not_allowed_error = AttributeError(_parent_nodes_not_allowed_error_message)

	@staticmethod
	def _set_value(node_instance, value):
		return None

	@staticmethod
	def _set_equation_and_inspect_parents(node_instance, equation):
		return None

	@staticmethod
	def _get_derivative_wrt(node_instance, wrt):
		raise ValueError

	@staticmethod
	def parent_nodes(node_instance):
		raise _parent_nodes_not_allowed_error

	@staticmethod
	def new_parent(node_instance, parent):
		raise _parent_nodes_not_allowed_error

	@staticmethod
	def arguments(node_instance):
		raise _parent_nodes_not_allowed_error

	@staticmethod
	def tier(node_instance):
		return 0

	@staticmethod
	def is_root():
		return True

	@staticmethod
	def is_precomputable(node_instance):
		return True

	@staticmethod
	def is_vector(node_instance):
		return False


class VariableNode(RootNode):

	@staticmethod
	def _graph_node_group(node_instance):
		return node_instance.graph._variable_nodes

	@staticmethod
	def _node_number_counter(node_instance):
		raise AttributeError

	@staticmethod
	def _node_symbol_number(node_instance):
		raise AttributeError

	@staticmethod
	def _create_or_get_new_node_symbol(node_instance):
		symbol = node_instance.graph.user_to_pycollo_problem_variables_mapping_in_order.get(node_instance.key, node_instance.key)
		return symbol

	@staticmethod
	def _get_derivative_wrt(node_instance, wrt):
		if wrt is node_instance:
			return node_instance.graph._one_node
		else:
			return node_instance.graph._zero_node

	@staticmethod
	def is_precomputable(node_instance):
		return False

	@staticmethod
	def is_vector(node_instance):
		if node_instance in node_instance.graph.time_function_variable_nodes:
			return True
		else:
			return False

	@staticmethod
	def _str(node_instance):
		return f"{node_instance.symbol} = {node_instance.key}"


class ConstantNode(RootNode):

	@staticmethod
	def _graph_node_group(node_instance):
		return node_instance.graph._constant_nodes

	@staticmethod
	def _node_number_counter(node_instance):
		return node_instance.graph._constant_node_num_counter

	@staticmethod
	def _node_symbol_letter(node_instance):
		return 'a'

	@staticmethod
	def _create_or_get_new_node_symbol(node_instance):
		return super(node_instance._type, node_instance._type)._create_or_get_new_node_symbol(node_instance)

	@staticmethod
	def _get_derivative_wrt(node_instance, wrt):
		return node_instance.graph._zero_node

	@staticmethod
	def _set_value(node_instance, value):
		return float(value)

	@staticmethod
	def _str(node_instance):
		return f"{node_instance.symbol} = {node_instance.key} = {node_instance.value}"


class NumberNode(RootNode):

	@staticmethod
	def _graph_node_group(node_instance):
		return node_instance.graph._number_nodes

	@staticmethod
	def _node_number_counter(node_instance):
		return node_instance.graph._number_node_num_counter

	@staticmethod
	def _node_symbol_letter(node_instance):
		return 'n'

	@staticmethod
	def _create_or_get_new_node_symbol(node_instance):
		return super(node_instance._type, node_instance._type)._create_or_get_new_node_symbol(node_instance)

	@staticmethod
	def _get_derivative_wrt(node_instance, wrt):
		return node_instance.graph._zero_node

	@staticmethod
	def _set_value(node_instance, value):
		return float(node_instance.key)

	@staticmethod
	def _str(node_instance):
		return f"{node_instance.symbol} = {node_instance.key} = {node_instance.value}"


class IntermediateNode(ExpressionNodeABC):

	@staticmethod
	def _graph_node_group(node_instance):
		return node_instance.graph._intermediate_nodes

	@staticmethod
	def _node_number_counter(node_instance):
		return node_instance.graph._intermediate_node_num_counter

	@staticmethod
	def _node_symbol_letter(node_instance):
		return 'w'

	@staticmethod
	def _create_or_get_new_node_symbol(node_instance):
		return super(node_instance._type, node_instance._type)._create_or_get_new_node_symbol(node_instance)

	@staticmethod
	def _set_value(node_instance, value):
		return value

	@staticmethod
	def _set_equation_and_inspect_parents(node_instance, equation):
		IntermediateNode._inspect_parents(node_instance, equation)
		return equation

	@staticmethod
	def _inspect_parents(node_instance, equation):

		def add_new_parent_node(arg):
			parent_node = node_instance.graph.symbols_to_nodes_mapping.get(arg)
			if parent_node is None:
				parent_node = Node(arg, node_instance.graph)
			node_instance.new_parent(parent_node)

		if equation is None:
			equation = node_instance.key

		if equation.args:
			for arg in equation.args:
				add_new_parent_node(arg)
		else:
			add_new_parent_node(equation)

		has_parent_nodes = node_instance.parent_nodes
		has_no_operation = node_instance._operation is None

		if has_parent_nodes and has_no_operation:
			if equation.func is sym.Symbol:
				node_instance._operation = sym.Add
			else:
				node_instance._operation = equation.func

			node_instance._expression = node_instance._operation(*[parent.symbol for parent in node_instance.parent_nodes])

	@staticmethod
	def parent_nodes(node_instance):
		return node_instance._parent_nodes

	@staticmethod
	def new_parent(node_instance, parent):
		node_instance.parent_nodes.append(parent)
		parent._child_nodes.add(node_instance)

	@staticmethod
	def arguments(node_instance):
		return tuple(parent.symbol for parent in node_instance.parent_nodes)

	@staticmethod
	def value(node_instance):
		return node_instance._value

	@staticmethod
	def _get_derivative_wrt(node_instance, wrt):
		if node_instance.is_precomputable:
			raise ValueError
		else:
			if wrt in node_instance.parent_nodes:
				deriv_node = node_instance._derivatives.get(wrt)
				if deriv_node is None:
					if isinstance(node_instance._expression, sym.Pow):
						args = node_instance._expression.args
						deriv = args[1] * args[0]**(args[1] - 1)
					else:
						deriv = node_instance._expression.diff(wrt.symbol)
					if deriv.is_Atom:
						if isinstance(deriv, sym.Symbol):
							deriv_node = node_instance.graph.symbols_to_nodes_mapping.get(deriv)
							if deriv_node is None:
								raise ValueError
						elif deriv.is_Number:
							deriv_node = Node(deriv, node_instance.graph)
						else:
							raise TypeError
					else:
						deriv_node = Node(deriv, node_instance.graph)
					return_val = deriv_node
				else:
					return_val = deriv_node
			else:
				return_val = node_instance.graph._zero_node
			# print(f"Taking the derivative of {node_instance.symbol} = {node_instance.expression} with respect to {wrt.symbol} with answer {return_val.symbol}")
			return return_val

	@staticmethod
	def is_root():
		return False

	@staticmethod
	def tier(node_instance):
		tiers = [parent.tier for parent in node_instance.parent_nodes]
		return max(tiers) + 1

	@staticmethod
	def is_precomputable(node_instance):
		is_precomputable = all([parent.is_precomputable 
			for parent in node_instance.parent_nodes])
		if is_precomputable:
			node_instance._value = node_instance.operation(*[parent.value 
				for parent in node_instance.parent_nodes])
		return is_precomputable

	@staticmethod
	def is_vector(node_instance):
		return any([parent.is_vector 
			for parent in node_instance.parent_nodes])

	@staticmethod
	def _str(node_instance):
		return f"{node_instance.symbol} = {node_instance.key} = {node_instance.equation} = {node_instance.expression}"






