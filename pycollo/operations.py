import abc
import collections

import sympy as sym

from .utils import cachedproperty


class PycolloOp(abc.ABC):

	def __init__(self, node):
		self.node = node

	@classmethod
	@abc.abstractmethod
	def SYMPY_OP(cls): pass

	@cachedproperty
	def expression(self): 
		return self.SYMPY_OP(*[parent_node.symbol 
			for parent_node in self.node.parent_nodes])

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {self.node: self.node.graph._one_node}
		for parent_node in self.node.parent_nodes:
			if not parent_node.is_precomputable:
				derivative = self.expression.diff(parent_node.symbol)
				derivative_node = self.node.new_node(derivative, self.node.graph)
				derivatives[parent_node] = derivative_node
		return derivatives


class PycolloUnsetOp(PycolloOp):

	SYMPY_OP = None

	@cachedproperty
	def expression(self): 
		return None

	@cachedproperty
	def derivatives(self):
		return None


class PycolloNullOp(PycolloOp):

	SYMPY_OP = None

	@cachedproperty
	def expression(self):
		return self.node

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		return {self.node: self.node.graph_one_node}


class PycolloNumber(PycolloOp):

	SYMPY_OP = None




class PycolloMul(PycolloOp):

	SYMPY_OP = sym.Mul


# class PycolloBinaryMul(PycolloOp):

# 	NUM_ARGS = 2

# 	def __init__(self, nodes):
# 		super().__init__(nodes)
# 		symbols = [str(node.symbol) for node in nodes]
# 		self._value = '*'.join(symbols)
# 		self._derivatives = {}


# class PycolloProd(PycolloOp):
# 	pass


class PycolloAdd(PycolloOp):

	SYMPY_OP = sym.Add

# 	NUM_ARGS = 2

# 	def __init__(self, nodes):
# 		super().__init__(nodes)
# 		self._value = ' + '.join([str(node.symbol) for node in nodes])
# 		self._derivatives = {node.symbol: '1' for node in nodes}

# 	@classmethod
# 	def _check_num_arguments(cls, arguments):
# 		if len(arguments) < cls.NUM_ARGS:
# 			raise ValueError
# 		return arguments


# class PycolloSum(PycolloOp):
# 	pass


class PycolloPow(PycolloOp):

	SYMPY_OP = sym.Pow

	

class PycolloSquare(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = 2*self.node.parent_nodes[0].symbol
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node: self.node.graph._one_node,
			self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloExp(PycolloOp):

	SYMPY_OP = sym.exp

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		raise NotImplementedError
		derivatives = {self.node: self.node.graph_one_node, 
			self.node.parent_nodes[0]: 2}
		return derivatives



# class PycolloReciprocal(PycolloOp):

# 	NUM_ARGS = 1

# 	def __init__(self, nodes):
# 		super().__init__(nodes[:-1])
# 		symbol = self.nodes[0].symbol
# 		self._value = f'{symbol}**-1'
# 		self._derivatives = {symbol: f'-{symbol}**-2'}



# class PycolloSquare(PycolloOp):

# 	NUM_ARGS = 1

# 	def __init__(self, nodes):
# 		super().__init__(nodes[:-1])
# 		symbol = self.nodes[0].symbol
# 		self._value = f'{symbol}*{symbol}'
# 		self._derivatives = {symbol: f'2*{symbol}'}


class PycolloCos(PycolloOp):

	SYMPY_OP = sym.cos

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		raise NotImplementedError
		derivatives = {self.node: self.node.graph_one_node, 
			self.node.parent_nodes[0]: -sym.sin(self.node.parent_nodes[0].symbol)}
		return derivatives


class PycolloSin(PycolloOp):

	SYMPY_OP = sym.sin

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		raise NotImplementedError
		derivatives = {self.node: self.node.graph_one_node, 
			self.node.parent_nodes[0]: sym.cos(self.node.parent_nodes[0].symbol)}
		return derivatives


def determine_operation(sympy_func, node):
	op_obj = SYMPY_EXPR_TYPES_DISPATCHER[sympy_func](node)
	return op_obj


def is_sympy_sym(node):
	return PycolloNullOp(node)


# def is_sympy_mul(node):
# 	return PYCOLLO_MUL_DISPATCHER.get(node, PycolloMul)(node)


# def is_sympy_add(nodes):
# 	return PycolloAdd(node)


def is_sympy_pow(node):
	exponent_node_key = node.parent_nodes[1].key
	return PYCOLLO_POW_DISPATCHER.get(exponent_node_key, PycolloPow)(node)


SYMPY_EXPR_TYPES_DISPATCHER = {
	sym.Symbol: is_sympy_sym,
	sym.Mul: PycolloMul,
	sym.Add: PycolloAdd,
	sym.Pow: is_sympy_pow,
	sym.cos: PycolloCos,
	sym.sin: PycolloSin,

	# Pycollo.Negate
	# Pycollo.Reciprocal
	# Pycollo.Square
	# Pycollo.Squareroot
	# Pycollo.Cube
	# Pycollo.Cuberoot

	# Pycollo.Tan
	sym.exp: PycolloExp,
	# Pycollo.Log
	# Pycollo.Arcsin
	# Pycollo.Arccos
	# Pycollo.Arctan
	# Pycollo.Sec
	# Pycollo.Cosec
	# Pycollo.Cot
	# Pycollo.Cosh
	# Pycollo.Sinh
	# Pycollo.Tanh
	# Pycollo.Sech
	# Pycollo.Cosech
	# Pycollo.Coth
	
	}


PYCOLLO_POW_DISPATCHER = {
	#-1: PycolloReciprocal,
	2: PycolloSquare,
	# 3: PycolloCube,
	# 1/2: PycolloSquareroot,
	# 1/3: PycolloCuberoot,
	}




PYCOLLO_MUL_DISPATCHER = {
	# 2: PycolloBinaryMul,
	}





