import ipopt

class IPOPTProblem(ipopt.problem):

	def __init__(self, iteration, J, g, c, G, G_struct):
		self.iteration = iteration
		self.objective = J
		self.gradient = g
		self.constraints = c
		self.jacobian = G
		self.jacobianstructure = G_struct





















		