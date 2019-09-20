import sympy as sym




class PycolloOp:

	NUM_ARGS = None

	def __init__(self, arguments):
		self.arguments = self._check_num_arguments(arguments)
	
	@property
	def value(self):
		return self._value

	def derivative_wrt(self, wrt):
		deriv = self.derivatives.get()
		if deriv is None:
			deriv = self._take_derivative_wrt(wrt)
		return deriv

	def _take_derivative_wrt(self, wrt):
		if wrt not in self.arguments:
			msg = (f'Trying to take a derivative with respect to a value that {self.value} is not a function of')
			raise ValueError(msg)
		else:
			return 

	@classmethod
	def _check_num_arguments(cls, arguments):
		if len(arguments) != cls.NUM_ARGS:
			raise ValueError
		return arguments


class PycolloNumber(PycolloOp):

	NUM_ARGS = 1


class PycolloMul(PycolloOp):

	NUM_ARGS = 2

	@classmethod
	def _check_num_arguments(cls, arguments):
		if len(arguments) < cls.NUM_ARGS:
			raise ValueError
		return arguments

	def value(self):
		pass





sympy_to_pycollo_symbols = {
	sym.Mul: PycolloMul,
	# sym.Add: PycolloAdd,
	# sym.Pow: PycolloPow,
	# sym.cos: PycolloCos,
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