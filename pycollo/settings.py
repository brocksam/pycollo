from typing import Optional

from .backend import BackendABC
from .bounds import BoundsABC
from .mesh import PhaseMesh
from .nlp import NLPSettings
from .scaling import ScalingABC
from .utils import (format_for_output, format_multiple_items_for_output)


FIRST_ORDER = 1
SECOND_ORDER = 2
DIFFERENTIAL_FORM = "differential"
INTEGRAL_FORM = "integral"
GUASS = "gauss"
LOBATTO = "lobatto"
RADAU = "radau"


class Settings():

	_COLLOCATION_MATRIX_FORMS = {DIFFERENTIAL_FORM, INTEGRAL_FORM}
	_QUADRATURE_METHODS = {GUASS, LOBATTO, RADAU}
	_DERIVATIVE_LEVELS = {FIRST_ORDER, SECOND_ORDER}

	def __init__(self, *, 
			optimal_control_problem=None, 
			collocation_matrix_form=INTEGRAL_FORM, 
			nlp_solver=NLPSettings.NLP_SOLVER_DEFAULT, 
			linear_solver=NLPSettings.LINEAR_SOLVER_DEFAULT, 
			nlp_tolerance=1e-8, 
			max_nlp_iterations=2000, 
			quadrature_method=LOBATTO, 
			derivative_level=SECOND_ORDER, 
			max_mesh_iterations=10, 
			mesh_tolerance=1e-7, 
			collocation_points_min=4, 
			collocation_points_max=10, 
			display_mesh_refinement_info=True, 
			display_mesh_result_info=False, 
			display_mesh_result_graph=False, 
			scaling_method=ScalingABC._METHOD_DEFAULT, 
			update_scaling=ScalingABC._UPDATE_DEFAULT, 
			number_scaling_samples=ScalingABC._NUMBER_SAMPLES_DEFAULT, 
			scaling_update_weight=ScalingABC._UPDATE_WEIGHT_DEFAULT,
			inf_value=BoundsABC._NUMERICAL_INF_DEFAULT,
			assume_inf_bounds=BoundsABC._ASSUME_INF_BOUNDS_DEFAULT,
			remove_constant_variables=BoundsABC._REMOVE_CONST_VARS_DEFAULT,
			override_endpoint_bounds=BoundsABC._OVERRIDE_ENDPOINTS_DEFAULT,
			bound_clash_tolerance=BoundsABC._BOUND_CLASH_TOLERANCE_DEFAULT,
			maximise_objective=True,
			backend=BackendABC._DEFAULT_BACKEND,

			):

		# Optimal Control Problem
		self.ocp = optimal_control_problem

		# NLP solver
		self.nlp_solver = nlp_solver
		self.linear_solver = linear_solver
		self.nlp_tolerance = nlp_tolerance
		self.max_nlp_iterations = max_nlp_iterations

		# Collocation and quadrature
		self.quadrature_method = quadrature_method
		self.derivative_level = derivative_level

		# Mesh refinement
		self.collocation_points_min = collocation_points_min
		self.collocation_points_max = collocation_points_max
		self.mesh_tolerance = mesh_tolerance
		self.max_mesh_iterations = max_mesh_iterations
		# TODO: Can these be refactored to default values for kwargs?
		self.default_number_mesh_sections = PhaseMesh._DEFAULT_NUMBER_MESH_SECTIONS
		self.default_mesh_section_fractions = PhaseMesh._DEFAULT_MESH_SECTION_SIZES
		self.default_number_mesh_section_nodes = PhaseMesh._DEFAULT_NUMBER_MESH_SECTION_NODES

		# Scaling
		self.scaling_method = scaling_method
		self.update_scaling = update_scaling
		self.number_scaling_samples = number_scaling_samples
		self.scaling_update_weight = scaling_update_weight

		# Output information
		self.display_mesh_refinement_info = display_mesh_refinement_info
		self.display_mesh_result_info = display_mesh_result_info
		self.display_mesh_result_graph = display_mesh_result_graph

		# Bounds
		self.inf_value = inf_value
		self.assume_inf_bounds = assume_inf_bounds
		self.remove_constant_variables = remove_constant_variables
		self.override_endpoint_bounds = override_endpoint_bounds
		self.bound_clash_tolerance = bound_clash_tolerance

		# Backend
		self.backend = backend
		self.check_nlp_functions = False
		self.dump_nlp_check_json = False

		# Other
		self.maximise_objective = maximise_objective
		self.console_out_progress = True

	@property
	def scaling_method(self) -> Optional[str]:
		return self._scaling_method

	@scaling_method.setter
	def scaling_method(self, method: Optional[str]):
		if method is not None:
			method = method.casefold()
		if method not in ScalingABC._METHOD_OPTIONS:
			msg = (f"{method} is not a valid scaling option.")
			raise ValueError(msg)
		elif method == "default":
			method = ScalingABC._METHOD_DEFAULT
		elif method == "none":
			method = None
		self._scaling_method = method

	@property
	def update_scaling(self) -> bool:
		return self._update_scaling
	
	@update_scaling.setter
	def update_scaling(self, do_update: bool):
		do_update = bool(do_update)
		self._update_scaling = do_update

	@property
	def scaling_update_weight(self) -> float:
		return self._scaling_update_weight
	
	@scaling_update_weight.setter
	def scaling_update_weight(self, weight: float):
		weight = float(weight)
		if weight < 0 or weight > 1:
			msg = (f"Scaling update weight must be a number between 0 and 1. "
				f"{weight} is invalid.")
			raise ValueError(msg)
		self._scaling_update_weight = weight

	@property
	def number_scaling_samples(self) -> int:
		return self._number_scaling_samples
	
	@number_scaling_samples.setter
	def number_scaling_samples(self, num: int):
		num = int(num)
		if num < 0:
			msg = (f'Number of scaling samples must be an integer greater than '
				f'or equal to 0. {num} is invalid.')
		self._number_scaling_samples = num

	@property
	def collocation_matrix_form(self):
		return self._col_mat_form
	
	@collocation_matrix_form.setter
	def collocation_matrix_form(self, form):
		form = form.casefold()
		if form not in self._COLLOCATION_MATRIX_FORMS:
			msg = (f"{form} is not a valid collocation matrix form.")
			raise ValueError(msg)
		elif form == 'derivative':
			msg = ("Derivative matrix form is not currently supported. Please "
				"use integral matrix form.")
			raise ValueError(msg)
		self._col_mat_form = form

	@property
	def nlp_solver(self):
		return self._nlp_solver

	@nlp_solver.setter
	def nlp_solver(self, nlp_solver):
		nlp_solver = nlp_solver.casefold()
		if nlp_solver not in NLPSettings.NLP_SOLVERS.keys():
			nlp_solver = format_for_output(nlp_solver, case="upper")
			supported_nlp_solvers = format_multiple_items_for_output(
				NLPSettings.NLP_SOLVERS.keys(), case="upper")
			plural_needed = "" if len(supported_nlp_solvers) == 1 else "from "
			msg = (f"{nlp_solver} is not a valid NLP solver. Please choose "
				f"{plural_needed}{supported_nlp_solvers}.")
			raise ValueError(msg)
		if not NLPSettings.NLP_SOLVERS.get(nlp_solver):
			nlp_solver = format_for_output(nlp_solver, case="upper")
			supported_solvers = tuple(solver
				for solver, is_supported in NLPSettings.NLP_SOLVERS.items()
				if is_supported)
			supported_solvers_formatted = format_multiple_items_for_output(
				supported_solvers, case="upper")
			plural_needed = "" if len(supported_solvers) == 1 else "from "
			msg = (f"{nlp_solver} is not currently supported. Please choose "
				f"{plural_needed}{supported_solvers_formatted}.")
			raise ValueError(msg)
		self._nlp_solver = nlp_solver

	@property
	def linear_solver(self):
		return self._linear_solver
	
	@linear_solver.setter
	def linear_solver(self, linear_solver):
		linear_solver = linear_solver.casefold()
		supported_linear_solvers = NLPSettings.LINEAR_SOLVERS.get(self.nlp_solver)
		if linear_solver not in supported_linear_solvers:
			supported_linear_solvers = format_multiple_items_for_output(
				supported_linear_solvers)
			msg = (f"'{linear_solver}' is not a valid linear solver. Please "
				f"choose from {supported_linear_solvers}")
			raise ValueError(msg)
		self._linear_solver = linear_solver

	@property
	def nlp_tolerance(self):
		return self._nlp_tolerance
	
	@nlp_tolerance.setter
	def nlp_tolerance(self, tolerance):
		if tolerance <= 0:
			msg = (f"Tolerance for the NLP must be a postive real number. "
				f"{tolerance} is invalid.")
			raise ValueError(msg)
		self._nlp_tolerance = tolerance

	@property
	def max_nlp_iterations(self):
		return self._max_nlp_iterations
	
	@max_nlp_iterations.setter
	def max_nlp_iterations(self, max_nlp_iterations):
		if max_nlp_iterations <= 0:
			msg = (f"Maximum number of NLP iterations must be a positive integer. {max_nlp_iterations} is invalid.")
			raise ValueError(msg)
		self._max_nlp_iterations = int(max_nlp_iterations)

	@property
	def quadrature_method(self):
		return self._quadrature_method
	
	@quadrature_method.setter
	def quadrature_method(self, method):
		method = method.casefold()
		if method not in self._QUADRATURE_METHODS:
			msg = (f"The quadrature method of '{method}' is not a valid "
				f"argument string.")
			raise ValueError(msg)
		self._quadrature_method = method

	@property
	def derivative_level(self):
		return self._derivative_level
	
	@derivative_level.setter
	def derivative_level(self, deriv_level):
		deriv_level = int(deriv_level)
		if deriv_level not in self._DERIVATIVE_LEVELS:
			msg = (f"Derivative level must be set to either 1 (which uses the gradient vector of the objective function and the Jacobian matrix of the constraints vector) or 2 (which also uses the Hessian matrix of the Lagrangian of the constraints).")
			raise ValueError(msg)
		self._derivative_level = deriv_level

	@property
	def mesh_tolerance(self):
		return self._mesh_tolerance
	
	@mesh_tolerance.setter
	def mesh_tolerance(self, tolerance):
		if tolerance <= 0:
			msg = ("Tolerance for the mesh must be a postive real number. {} is invalid.")
			raise ValueError(msg.format(tolerance))
		self._mesh_tolerance = tolerance

	@property
	def max_mesh_iterations(self):
		return self._max_mesh_iterations
	
	@max_mesh_iterations.setter
	def max_mesh_iterations(self, max_mesh_iterations):
		if max_mesh_iterations <= 0:
			msg = (f"Maximum number of mesh iterations must be a positive integer. {max_mesh_iterations} is invalid.")
			raise ValueError(msg)
		self._max_mesh_iterations = int(max_mesh_iterations)

	@property
	def collocation_points_min(self):
		return self._col_points_min
	
	@collocation_points_min.setter
	def collocation_points_min(self, points_min):
		points_min = int(points_min)
		self._col_points_min = points_min
		if points_min < 2:
			msg = ("The minimum number of collocation points must be great than 2.")
			raise ValueError(msg)
		if points_min > 4:
			msg = ("It is recommended that a minimum of 2, 3 or 4 collocation points is used per mesh section to allow for efficient computation.")
			raise ValueError(msg)

	@property
	def collocation_points_max(self):
		return self._col_points_max
	
	@collocation_points_max.setter
	def collocation_points_max(self, points_max):
		points_max = int(points_max)
		if points_max < self._col_points_min:
			msg = ("The maximum number of collocation points must be greater than or equal to {}, the minimum number of collocation points.")
			raise ValueError(msg.format(self._col_points_min))
		if points_max > 10:
			msg = ("The maximum number of collocation points recommended in a single mesh sections is 10 due to the numerical instabilty of Lagrange polynomial interpolation above this threshold.")
			raise ValueError(msg)
		self._col_points_max = points_max

	@property
	def display_mesh_refinement_info(self):
		return self._display_mesh_refinement_info
	
	@display_mesh_refinement_info.setter
	def display_mesh_refinement_info(self, val):
		self._display_mesh_refinement_info = bool(val)

	@property
	def display_mesh_result_info(self):
		return self._display_mesh_result_info
	
	@display_mesh_result_info.setter
	def display_mesh_result_info(self, val):
		self._display_mesh_result_info = bool(val)

	@property
	def display_mesh_result_graph(self):
		return self._display_mesh_result_graph
	
	@display_mesh_result_graph.setter
	def display_mesh_result_graph(self, val):
		self._display_mesh_result_graph = bool(val)

	@property
	def backend(self):
		return self._backend
	
	@backend.setter
	def backend(self, backend):
		backend = backend.casefold()

		if backend not in BackendABC._BACKENDS:
			valid_options = format_multiple_items_for_output(BackendABC._BACKENDS)
			msg = (f"{backend} is not a valid backend option. Please choose "
				f"from {valid_options}.")
			raise ValueError(msg)
		self._backend = backend


