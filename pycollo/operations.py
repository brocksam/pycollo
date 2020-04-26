import abc
import collections

import sympy as sym

from .utils import cachedproperty


SYMPY_ZERO = sym.numbers.Zero()
SYMPY_ONE = sym.numbers.One()
SYMPY_TWO = sym.numbers.Integer(2)
SYMPY_THREE = sym.numbers.Integer(3)
SYMPY_HALF = sym.numbers.Half()
SYMPY_THIRD = sym.numbers.Rational(1/3)
SYMPY_NEG_ONE = sym.numbers.Integer(-1)
SYMPY_NEG_TWO = sym.numbers.Integer(-2)
SYMPY_NEG_THREE = sym.numbers.Integer(-3)
SYMPY_NEG_HALF = sym.numbers.Rational(-1/2)
SYMPY_NEG_THIRD = sym.numbers.Rational(-1/3)


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
		return {}


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


# class PycolloNumber(PycolloOp):

# 	SYMPY_OP = None



class PycolloMul(PycolloOp):

	SYMPY_OP = sym.Mul

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {}
		for parent_node in self.node.parent_nodes:
			if not parent_node.is_precomputable:
				derivative = self.expression / parent_node.symbol
				derivative_node = self.node.new_node(derivative, self.node.graph)
				derivatives[parent_node] = derivative_node
		return derivatives


class PycolloBinaryMul(PycolloOp):

	SYMPY_OP = sym.Mul

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {}
		if not self.node.parent_nodes[0].is_precomputable:
			derivatives[self.node.parent_nodes[0]] = self.node.parent_nodes[1]
		if not self.node.parent_nodes[1].is_precomputable:
			derivatives[self.node.parent_nodes[1]] = self.node.parent_nodes[0]
		return derivatives


class PycolloTernaryMul(PycolloOp):

	SYMPY_OP = sym.Mul

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {}
		if not self.node.parent_nodes[0].is_precomputable:
			derivative = self.node.parent_nodes[1].symbol * self.node.parent_nodes[2].symbol
			derivative_node = self.node.new_node(derivative, self.node.graph)
			derivatives[self.node.parent_nodes[0]] = derivative_node
		if not self.node.parent_nodes[1].is_precomputable:
			derivative = self.node.parent_nodes[0].symbol * self.node.parent_nodes[2].symbol
			derivative_node = self.node.new_node(derivative, self.node.graph)
			derivatives[self.node.parent_nodes[1]] = derivative_node
		if not self.node.parent_nodes[2].is_precomputable:
			derivative = self.node.parent_nodes[0].symbol * self.node.parent_nodes[1].symbol
			derivative_node = self.node.new_node(derivative, self.node.graph)
			derivatives[self.node.parent_nodes[2]] = derivative_node
		return derivatives


# class PycolloProd(PycolloOp):
# 	pass


class PycolloAdd(PycolloOp):

	SYMPY_OP = sym.Add

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {}
		for parent_node in self.node.parent_nodes:
			if not parent_node.is_precomputable:
				derivatives[parent_node] = self.node.graph._one_node
		return derivatives

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

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		intermediate_1 = sym.Add(self.node.parent_nodes[1].symbol, SYMPY_NEG_ONE)
		intermediate_2 = sym.Pow(self.node.parent_nodes[0].symbol, intermediate_1)
		derivative = sym.Mul(self.node.parent_nodes[1].symbol, intermediate_2)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives
	

class PycolloSquare(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = sym.Mul(SYMPY_TWO, self.node.parent_nodes[0].symbol)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSquareroot(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		intermediate_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_HALF)
		derivative = sym.Mul(SYMPY_HALF, intermediate_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloReciprocal(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		intermediate_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_TWO)
		derivative = sym.Mul(SYMPY_NEG_ONE, intermediate_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloExp(PycolloOp):

	SYMPY_OP = sym.exp

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		raise NotImplementedError
		derivatives = {self.node.parent_nodes[0]: 2}
		return derivatives


class PycolloCos(PycolloOp):

	SYMPY_OP = sym.cos

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = -sym.sin(self.node.parent_nodes[0].symbol)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSin(PycolloOp):

	SYMPY_OP = sym.sin

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = sym.cos(self.node.parent_nodes[0].symbol)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


def determine_operation(sympy_func, node):
	op_obj = SYMPY_EXPR_TYPES_DISPATCHER[sympy_func](node)
	return op_obj


def is_sympy_sym(node):
	return PycolloNullOp(node)


def is_sympy_mul(node):
	num_operands = len(node.parent_nodes)
	return PYCOLLO_MUL_DISPATCHER.get(num_operands, PycolloMul)(node)


# def is_sympy_add(nodes):
# 	return PycolloAdd(node)


def is_sympy_pow(node):
	exponent_node_key = node.parent_nodes[1].key
	return PYCOLLO_POW_DISPATCHER.get(exponent_node_key, PycolloPow)(node)


SYMPY_EXPR_TYPES_DISPATCHER = {
	sym.Symbol: is_sympy_sym,
	sym.Mul: is_sympy_mul,
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
	sym.numbers.Integer(-1): PycolloReciprocal,
	sym.numbers.Integer(2): PycolloSquare,
	# 3: PycolloCube,
	sym.numbers.Half(): PycolloSquareroot,
	# 1/3: PycolloCuberoot,
	}




PYCOLLO_MUL_DISPATCHER = {
	2: PycolloBinaryMul,
	3: PycolloTernaryMul,
	}





