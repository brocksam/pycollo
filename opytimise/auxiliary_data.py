class AuxiliaryData():
	"""
	TODO:
		Make this a named tuple?
	"""
	def __init__(self, *, optimal_control_problem=None):

		# Optimal control problem
		self._optimal_control_problem = optimal_control_problem

		# Mapping
		self._y_map = {}
		self._u_map = {}
		self._q_map = {}
		self._t_map = {}
		self._s_map = {}
		self._x_map = {}

		# Subs dict
		self.constants_values = None
		self.differential_equations = None

	@property
	def optimal_control_problem(self):
		return self._optimal_control_problem