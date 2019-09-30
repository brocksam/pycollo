import collections

from ordered_set import OrderedSet

import sympy as sym


class PycolloOp:

	NUM_ARGS = None

	def __init__(self, nodes):
		self.nodes = self._check_num_arguments(nodes)
		self.graph = self.nodes[0].graph
	
	@property
	def value(self):
		return self._value

	def derivative_wrt(self, wrt):
		deriv = self._derivatives.get(wrt)
		if deriv is None:
			deriv = self._take_derivative_wrt(wrt)
		return deriv

	def _take_derivative_wrt(self, wrt):
		if wrt not in self.arguments:
			msg = (f'Trying to take a derivative with respect to a value that {self.value} is not a function of')
			raise ValueError(msg)
		else:
			return 'lalalalala'

	@classmethod
	def _check_num_arguments(cls, nodes):
		if len(nodes) != cls.NUM_ARGS:
			raise ValueError
		return nodes


class PycolloNullOp(PycolloOp):

	NUM_ARGS = 1

	def value(self):
		return self.nodes[0]


class PycolloNumber(PycolloOp):

	NUM_ARGS = 1


class PycolloMul(PycolloOp):

	NUM_ARGS = 2

	def __init__(self, nodes):
		super().__init__(nodes)
		symbols = [str(node.symbol) for node in nodes]
		self._value = '*'.join(symbols)
		self._derivatives = {}
		for node in nodes:
			deriv_symbols = symbols.copy()
			deriv_symbols.remove(str(node.symbol))
			expr = '*'.join(deriv_symbols)
			self._derivatives.update({node.symbol: expr})

	@classmethod
	def _check_num_arguments(cls, arguments):
		if len(arguments) < cls.NUM_ARGS:
			raise ValueError
		return arguments


class PycolloBinaryMul(PycolloOp):

	NUM_ARGS = 2

	def __init__(self, nodes):
		super().__init__(nodes)
		symbols = [str(node.symbol) for node in nodes]
		self._value = '*'.join(symbols)
		self._derivatives = {}


class PycolloProd(PycolloOp):
	pass


class PycolloAdd(PycolloOp):

	NUM_ARGS = 2

	def __init__(self, nodes):
		super().__init__(nodes)
		self._value = ' + '.join([str(node.symbol) for node in nodes])
		self._derivatives = {node.symbol: '1' for node in nodes}

	@classmethod
	def _check_num_arguments(cls, arguments):
		if len(arguments) < cls.NUM_ARGS:
			raise ValueError
		return arguments


class PycolloSum(PycolloOp):
	pass


class PycolloPow(PycolloOp):

	def __init__(self, arguments):
		print('Pycollo Power')
		raise NotImplementedError



class PycolloReciprocal(PycolloOp):

	NUM_ARGS = 1

	def __init__(self, nodes):
		super().__init__(nodes[:-1])
		symbol = self.nodes[0].symbol
		self._value = f'{symbol}**-1'
		self._derivatives = {symbol: f'-{symbol}**-2'}



class PycolloSquare(PycolloOp):

	NUM_ARGS = 1

	def __init__(self, nodes):
		super().__init__(nodes[:-1])
		symbol = self.nodes[0].symbol
		self._value = f'{symbol}*{symbol}'
		self._derivatives = {symbol: f'2*{symbol}'}


class PycolloCos(PycolloOp):

	NUM_ARGS = 1

	def __init__(self, nodes):
		super().__init__(nodes)
		symbol = self.nodes[0].symbol
		self._value = f'cos({symbol})'
		self._derivatives = {symbol: f'-sin({symbol})'}



# class PycolloAdd(PycolloOp):

# 	def __init__(self):
# 		raise NotImplementedError



# class PycolloAdd(PycolloOp):

# 	def __init__(self):
# 		raise NotImplementedError



# class PycolloAdd(PycolloOp):

# 	def __init__(self):
# 		raise NotImplementedError

def determine_operation(sympy_func, nodes):
	op_obj = SYMPY_EXPR_TYPES_DISPATCHER[sympy_func](nodes)
	return op_obj


def is_sympy_sym(nodes):
	return PycolloNullOp(nodes)


def is_sympy_mul(nodes):
	num_args = len(nodes)
	return PYCOLLO_MUL_DISPATCHER.get(num_args, PycolloMul)(nodes)


def is_sympy_add(nodes):
	return PycolloAdd(nodes)


def is_sympy_pow(nodes):
	exponent_node_key = nodes[1].key
	return PYCOLLO_POW_DISPATCHER.get(exponent_node_key, PycolloPow)(nodes)


def is_sympy_cos(nodes):
	return PycolloCos(nodes)



SYMPY_EXPR_TYPES_DISPATCHER = {
	sym.Symbol: is_sympy_sym,
	sym.Mul: is_sympy_mul,
	sym.Add: is_sympy_add,
	sym.Pow: is_sympy_pow,
	sym.cos: is_sympy_cos,
	# sym.sin: PycolloSin,

	# Pycollo.Negate
	# Pycollo.Reciprocal
	# Pycollo.Square
	# Pycollo.Squareroot
	# Pycollo.Cube
	# Pycollo.Cuberoot

	# Pycollo.Tan
	# Pycollo.Exp
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
	-1: PycolloReciprocal,
	2: PycolloSquare,
	# 3: PycolloCube,
	# 1/2: PycolloSquareroot,
	# 1/3: PycolloCuberoot,
	}




PYCOLLO_MUL_DISPATCHER = {
	2: PycolloBinaryMul,
	}





