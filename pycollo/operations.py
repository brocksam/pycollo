import abc
import collections

import sympy as sym

from .utils import cachedproperty


SYMPY_ZERO = sym.core.numbers.Zero()
SYMPY_ONE = sym.core.numbers.One()
SYMPY_TWO = sym.core.numbers.Integer(2)
SYMPY_THREE = sym.core.numbers.Integer(3)
SYMPY_HALF = sym.core.numbers.Half()
SYMPY_THIRD = sym.core.numbers.Rational(1/3)
SYMPY_2THIRDS = sym.core.numbers.Rational(2/3)
SYMPY_NEG_ONE = sym.core.numbers.Integer(-1)
SYMPY_NEG_TWO = sym.core.numbers.Integer(-2)
SYMPY_NEG_THREE = sym.core.numbers.Integer(-3)
SYMPY_NEG_HALF = sym.core.numbers.Rational(-1/2)
SYMPY_NEG_THIRD = sym.core.numbers.Rational(-1/3)
SYMPY_NEG_2THIRDS = sym.core.numbers.Rational(-2/3)


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

	@staticmethod
	def SYMPY_OP(arg):
		return arg

	@cachedproperty
	def expression(self):
		return self.node.parent_nodes[0].symbol

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		return {self.node.parent_nodes[0]: self.node.graph._one_node}


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


class PycolloPow(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Add(self.node.parent_nodes[1].symbol, SYMPY_NEG_ONE)
		sub_2 = sym.Pow(self.node.parent_nodes[0].symbol, sub_1)
		derivative = sym.Mul(self.node.parent_nodes[1].symbol, sub_2)
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


class PycolloCube(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		derivative = sym.Mul(SYMPY_THREE, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSquareroot(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_HALF)
		derivative = sym.Mul(SYMPY_HALF, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCuberoot(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_2THIRDS)
		derivative = sym.Mul(SYMPY_THIRD, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloReciprocal(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_TWO)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSquareReciprocal(PycolloOp):

	SYMPY_OP = sym.Pow

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_THREE)
		derivative = sym.Mul(SYMPY_NEG_TWO, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloExp(PycolloOp):

	SYMPY_OP = sym.exp

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivatives = {self.node.parent_nodes[0]: self.node}
		return derivatives


class PycolloLn(PycolloOp):

	SYMPY_OP = sym.log

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_NEG_ONE)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloLog(PycolloOp):

	SYMPY_OP = sym.log

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = self.node.symbol.diff(self.node.parent_nodes[0].symbol)
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


class PycolloCos(PycolloOp):

	SYMPY_OP = sym.cos

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.sin(self.node.parent_nodes[0].symbol)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloTan(PycolloOp):

	SYMPY_OP = sym.tan

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.sec(self.node.parent_nodes[0].symbol)
		derivative = sym.Pow(sub_1, SYMPY_TWO)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCot(PycolloOp):

	SYMPY_OP = sym.cot

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.csc(self.node.parent_nodes[0].symbol)
		sub_2 = sym.Pow(sub_1, SYMPY_TWO)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_2)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSec(PycolloOp):

	SYMPY_OP = sym.sec

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.tan(self.node.parent_nodes[0].symbol)
		sub_2 = sym.sec(self.node.parent_nodes[0].symbol)
		derivative = sym.Mul(sub_1, sub_2)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCosec(PycolloOp):

	SYMPY_OP = sym.csc

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.cot(self.node.parent_nodes[0].symbol)
		sub_2 = sym.csc(self.node.parent_nodes[0].symbol)
		sub_3 = sym.Mul(sub_1, sub_2)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_3)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArcsin(PycolloOp):

	SYMPY_OP = sym.asin

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Mul(SYMPY_NEG_ONE, sub_1)
		sub_3 = sym.Add(SYMPY_ONE, sub_2)
		derivative = sym.Pow(sub_3, SYMPY_NEG_HALF)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArccos(PycolloOp):

	SYMPY_OP = sym.acos

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Mul(SYMPY_NEG_ONE, sub_1)
		sub_3 = sym.Add(SYMPY_ONE, sub_2)
		sub_4 = sym.Pow(sub_3, SYMPY_NEG_HALF)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_4)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArctan(PycolloOp):

	SYMPY_OP = sym.atan

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Add(SYMPY_ONE, sub_1)
		derivative = sym.Pow(sub_2, SYMPY_NEG_ONE)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArccot(PycolloOp):

	SYMPY_OP = sym.acot

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Add(SYMPY_ONE, sub_1)
		sub_3 = sym.Pow(sub_2, SYMPY_NEG_ONE)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_3)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArcsec(PycolloOp):

	SYMPY_OP = sym.asec

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		msg = (f"Pycollo does not currently support differentiation of "
			f"'arccosec' functions as they are discontinuous.")
		raise NotImplementedError(msg)


class PycolloArccosec(PycolloOp):

	SYMPY_OP = sym.acsc

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		msg = (f"Pycollo does not currently support differentiation of "
			f"'arccosec' functions as they are discontinuous.")
		raise NotImplementedError(msg)


class PycolloSinh(PycolloOp):

	SYMPY_OP = sym.sinh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		derivative = sym.cosh(self.node.parent_nodes[0].symbol)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCosh(PycolloOp):

	SYMPY_OP = sym.cosh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.sinh(self.node.parent_nodes[0].symbol)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloTanh(PycolloOp):

	SYMPY_OP = sym.tanh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.tanh(self.node.parent_nodes[0].symbol)
		sub_2 = sym.Pow(sub_1, SYMPY_TWO)
		sub_3 = sym.Mul(SYMPY_NEG_ONE, sub_2)
		derivative = sym.Add(SYMPY_ONE, sub_3)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCoth(PycolloOp):

	SYMPY_OP = sym.coth

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.coth(self.node.parent_nodes[0].symbol)
		sub_2 = sym.Pow(sub_1, SYMPY_TWO)
		sub_3 = sym.Mul(SYMPY_NEG_ONE, sub_2)
		derivative = sym.Add(SYMPY_ONE, sub_3)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloSech(PycolloOp):

	SYMPY_OP = sym.sech

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.tanh(self.node.parent_nodes[0].symbol)
		derivative = sym.Mul(SYMPY_NEG_ONE, self.node.parent_nodes[0].symbol, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloCosech(PycolloOp):

	SYMPY_OP = sym.csch

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.coth(self.node.parent_nodes[0].symbol)
		derivative = sym.Mul(SYMPY_NEG_ONE, self.node.parent_nodes[0].symbol, sub_1)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArcsinh(PycolloOp):

	SYMPY_OP = sym.asinh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Add(SYMPY_ONE, sub_1)
		derivative = sym.Pow(sub_2, SYMPY_NEG_HALF)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArccosh(PycolloOp):

	SYMPY_OP = sym.acosh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Add(SYMPY_NEG_ONE, sub_1)
		derivative = sym.Pow(sub_2, SYMPY_NEG_HALF)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArctanh(PycolloOp):

	SYMPY_OP = sym.atanh

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		msg = (f"Pycollo does not currently support differentiation of "
			f"'arctanh' functions as they are discontinuous.")
		raise NotImplementedError(msg)


class PycolloArccoth(PycolloOp):

	SYMPY_OP = sym.acoth

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		msg = (f"Pycollo does not currently support differentiation of "
			f"'arccoth' functions as they are discontinuous.")
		raise NotImplementedError(msg)


class PycolloArcsech(PycolloOp):

	SYMPY_OP = sym.asech

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		sub_1 = sym.Pow(self.node.parent_nodes[0].symbol, SYMPY_TWO)
		sub_2 = sym.Mul(SYMPY_NEG_ONE, sub_1)
		sub_3 = sym.Add(SYMPY_ONE, sub_2)
		sub_4 = sym.Pow(sub_3, SYMPY_HALF)
		sub_5 = sym.Mul(self.node.parent_nodes[0].symbol, sub_4)
		sub_6 = sym.Pow(sub_5, SYMPY_NEG_ONE)
		derivative = sym.Mul(SYMPY_NEG_ONE, sub_6)
		derivative_node = self.node.new_node(derivative, self.node.graph)
		derivatives = {self.node.parent_nodes[0]: derivative_node}
		return derivatives


class PycolloArccosech(PycolloOp):

	SYMPY_OP = sym.acsch

	@cachedproperty
	def derivatives(self):
		if self.node.is_precomputable:
			return {}
		msg = (f"Pycollo does not currently support differentiation of "
			f"'arccosech' functions as they are discontinuous.")
		raise NotImplementedError(msg)


def determine_operation(sympy_func, node):
	op_obj = SYMPY_EXPR_TYPES_DISPATCHER[sympy_func](node)
	return op_obj


def is_sympy_sym(node):
	return PycolloNullOp(node)


def is_sympy_mul(node):
	num_operands = len(node.parent_nodes)
	return PYCOLLO_MUL_DISPATCHER.get(num_operands, PycolloMul)(node)


def is_sympy_pow(node):
	exponent_node_key = node.parent_nodes[1].key
	return PYCOLLO_POW_DISPATCHER.get(exponent_node_key, PycolloPow)(node)


def is_sympy_log(node):
	num_operands = len(node.parent_nodes)
	return PYCOLLO_LOG_DISPATCHER.get(num_operands, PycolloLog)(node)


SYMPY_EXPR_TYPES_DISPATCHER = {
	sym.Symbol: is_sympy_sym,
	sym.Mul: is_sympy_mul,
	sym.Add: PycolloAdd,
	sym.Pow: is_sympy_pow,
	sym.exp: PycolloExp,
	sym.log: is_sympy_log,
	sym.sin: PycolloSin,
	sym.cos: PycolloCos,
	sym.tan: PycolloTan,
	sym.cot: PycolloCot,
	sym.sec: PycolloSec,
	sym.csc: PycolloCosec,
	sym.asin: PycolloArcsin,
	sym.acos: PycolloArccos,
	sym.atan: PycolloArctan,
	sym.acot: PycolloArccot,
	sym.asec: PycolloArcsec,
	sym.acsc: PycolloArccosec,
	sym.sinh: PycolloSinh,
	sym.cosh: PycolloCosh,
	sym.tanh: PycolloTanh,
	sym.coth: PycolloCoth,
	sym.sech: PycolloSech,
	sym.csch: PycolloCosech,
	sym.asinh: PycolloArcsinh,
	sym.acosh: PycolloArccosh,
	sym.atanh: PycolloArctanh,
	sym.acoth: PycolloArccoth,
	sym.asech: PycolloArcsech,
	sym.acsch: PycolloArccosech,
	}


try:
	import symfit
except ModuleNotFoundError:
	pass
else:
	SYMPY_EXPR_TYPES_DISPATCHER[symfit.Parameter] = is_sympy_sym


PYCOLLO_MUL_DISPATCHER = {
	2: PycolloBinaryMul,
	3: PycolloTernaryMul,
	}


PYCOLLO_POW_DISPATCHER = {
	SYMPY_NEG_ONE: PycolloReciprocal,
	SYMPY_NEG_TWO: PycolloSquareReciprocal,
	SYMPY_TWO: PycolloSquare,
	SYMPY_THREE: PycolloCube,
	SYMPY_HALF: PycolloSquareroot,
	SYMPY_THIRD: PycolloCuberoot,
	}


PYCOLLO_LOG_DISPATCHER = {
	1: PycolloLn,
	}





