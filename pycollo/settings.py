"""Settings for the Pycollo package.

This module contains all settings for the Pycollo package including settings
for Pycollo :class:`OptimalControlProblem`s as well as settings for dependent
NLP solvers etc.

"""


__all__ = ["Settings"]


from pyproprop import processed_property

from .backend import BACKENDS
from .bounds import DEFAULT_ASSUME_INF_BOUNDS
from .bounds import DEFAULT_BOUND_CLASH_ABSOLUTE_TOLERANCE
from .bounds import DEFAULT_BOUND_CLASH_RELATIVE_TOLERANCE
from .bounds import DEFAULT_NUMERICAL_INF
from .bounds import DEFAULT_OVERRIDE_ENDPOINTS
from .bounds import DEFAULT_REMOVE_CONSTANT_VARIABLES
from .compiled import COLLOCATION_MATRIX_FORMS
from .mesh_refinement import MESH_REFINEMENT_ALGORITHMS
from .mesh_refinement import DEFAULT_MESH_TOLERANCE
from .mesh_refinement import DEFAULT_MAX_MESH_ITERATIONS
from .quadrature import DEFAULT_COLLOCATION_POINTS_MIN
from .quadrature import DEFAULT_COLLOCATION_POINTS_MAX
from .quadrature import QUADRATURES
from .scaling import DEFAULT_NUMBER_SCALING_SAMPLES
from .scaling import DEFAULT_SCALING_WEIGHT
from .scaling import DEFAULT_UPDATE_SCALING
from .scaling import SCALING_METHODS


# Solver constants
NLP_SOLVER_IPOPT_KEYWORD = "ipopt"
NLP_SOLVER_SNOPT_KEYWORD = "snopt"
NLP_SOLVER_BONMIN_KEYWORD = "bonmin"
NLP_SOLVER_COUENNE_KEYWORD = "couenne"
NLP_SOLVER_WORHP_KEYWORD = "worhp"
NLP_SOLVER_KNITRO_KEYWORD = "knitro"
LINEAR_SOLVER_MUMPS_KEYWORD = "mumps"
LINEAR_SOLVER_MA57_KEYWORD = "ma57"
DEFAULT_NLP_SOLVER = NLP_SOLVER_IPOPT_KEYWORD
UNSUPPORTED_NLP_SOLVER = (NLP_SOLVER_SNOPT_KEYWORD,
                          NLP_SOLVER_BONMIN_KEYWORD,
                          NLP_SOLVER_COUENNE_KEYWORD,
                          NLP_SOLVER_WORHP_KEYWORD,
                          NLP_SOLVER_KNITRO_KEYWORD,
                          )
DEFAULT_LINEAR_SOLVER = LINEAR_SOLVER_MUMPS_KEYWORD
UNSUPPORTED_LINEAR_SOLVER = (LINEAR_SOLVER_MA57_KEYWORD, )
DEFAULT_NLP_TOLERANCE = 1e-10
DEFAULT_MAX_NLP_ITERATIONS = 2000
DEFAULT_WARM_START = False

# Derivative level constants
DERIVATIVE_LEVEL_FIRST = 1
DERIVATIVE_LEVEL_SECOND = 2
DEFAULT_DERIVATIVE_LEVEL = DERIVATIVE_LEVEL_SECOND

# Display constants
DEFAULT_CONSOLE_OUT_PROGRESS = True
DEFAULT_DISPLAY_MESH_REFINE_INFO = True
DEFAULT_DISPLAY_MESH_RESULT_INFO = False
DEFAULT_DISPLAY_MESH_RESULT_GRAPH = False

# Debug
DEFAULT_CHECK_NLP_FUNCTIONS = False


class Settings():

    """Settings class for all Pycollo OCP and NLP settings.

    Attributes
    ----------
    backend : :py:class:`Backend <pycollo>`
        The Pycollo backend used for OCP processing, particularly derivative
        generation and NLP construction.
    collocation_matrix_form : str
        Whether the integral or derivative form of the collocation matrix
        should be used. Specifically relates to the construction of the
        dynamics defect constraints.
    collocation_points_max : int
        Minimum allowable number of mesh points per mesh section.
    collocation_points_min : int
        Maximum allowable number of mesh points per mesh section.
    derivative_level : int (1, 2)
        Whether to use exact Hessian in the NLP solving. If value is 1 then
        Pycollo does not produce an exact Hessian, if the value is 2 it
        does.
    display_mesh_refinement_info : bool
        Should information about the mesh refinement step (change in number of
        mesh sections and mesh nodes per mesh section) be outputted to the
        console at the end of the mesh iteration.
    display_mesh_result_graph : bool
        Should the state, state derivatives and contol be plotted at the end of
        each mesh iteration.
    display_mesh_result_info : bool
        Should information about the result to the NLP be dispalyed on the
        console at the end of each mesh iteration.
    linear_solver : str
        Which linear solver should be used by the chosen NLP solver. Note:
        different NLP solvers will support different linear solvers.
    max_mesh_iterations : int
        How many mesh iterations should be conducted by Pycollo (provided that
        the mesh tolerance hasn't been met) before the attempt to solve the OCP
        is terminated.
    max_nlp_iterations : int
        How many NLP iterations should be conducted by the NLP solver (provided
        the NLP tolerance hasn't been met) before the attempt to solve the NLP
        is terminated. Note: this will likely behave slightly different for
        different NLP solvers.
    mesh_tolerance : float
        The minimum acceptable maximum relative mesh error for the OCP to be
        considered solved.
    nlp_solver : str
        Which NLP solver the Pycollo backend should use.
    nlp_tolerance : float
        The minimum acceptable maximum error in the NLP that the NLP solver
        must meet before it can exit successfully.
    number_scaling_samples : int
        How many randomly generated samples should be used when the sampleing
        scaling method is used.
    ocp : :obj:`pycollo.OptimalControlProblem`
        The optimal control problem object with which these settings should be
        associated.
    quadrature_method : str
        Which K-stage Runge-Kutta orthogonal collocation/quadrature method
        should be used to transcribe the OCP to NLP.
    scaling_method : (str, None)
        Which scaling method should be used to scale the NLP.
    scaling_weight : float
        When updating scaling is used, what weighting factor should be used in
        the moving average. The minimum value is 0, the maximum is 1. A larger
        value means that more recent mesh iteration scalings are weighted more
        heavily.
    update_scaling : bool
        Whether the scaling should be automatically updated between mesh
        iterations. If True then the scaling is updated, if False then the
        scaling at every mesh iteration is the scaling used to solve the OCP
        on the first mesh iteration.

    """

    _NLP_SOLVERS = (
        NLP_SOLVER_IPOPT_KEYWORD,
        NLP_SOLVER_SNOPT_KEYWORD,
        NLP_SOLVER_WORHP_KEYWORD,
        NLP_SOLVER_BONMIN_KEYWORD,
        NLP_SOLVER_COUENNE_KEYWORD,
        NLP_SOLVER_KNITRO_KEYWORD,
    )
    _LINEAR_SOLVERS = (
        LINEAR_SOLVER_MUMPS_KEYWORD,
        LINEAR_SOLVER_MA57_KEYWORD,
    )
    _DERIVATIVE_LEVELS = (
        DERIVATIVE_LEVEL_FIRST,
        DERIVATIVE_LEVEL_SECOND,
    )

    backend = processed_property(
        "backend",
        description="Pycollo backend",
        type=str,
        cast=True,
        options=BACKENDS,
    )
    derivative_level = processed_property(
        "derivative_level",
        description="derivative level",
        type=int,
        cast=True,
        options=_DERIVATIVE_LEVELS,
    )
    collocation_matrix_form = processed_property(
        "collocation_matrix_form",
        description="form of the collocation matrices",
        type=str,
        cast=True,
        options=COLLOCATION_MATRIX_FORMS,
    )
    quadrature_method = processed_property(
        "quadrature_method",
        description="quadrature method",
        type=str,
        cast=True,
        options=QUADRATURES,
    )
    nlp_solver = processed_property(
        "nlp_solver",
        description="NLP solver",
        type=str,
        options=_NLP_SOLVERS,
        unsupported_options=UNSUPPORTED_NLP_SOLVER,
    )
    linear_solver = processed_property(
        "linear_solver",
        description="linear solver",
        type=str,
        options=_LINEAR_SOLVERS,
        unsupported_options=UNSUPPORTED_LINEAR_SOLVER,
    )
    nlp_tolerance = processed_property(
        "nlp_tolerance",
        description="NLP tolerance",
        type=float,
        max=1.0,
        min=0.0,
        exclusive=True,
    )
    max_nlp_iterations = processed_property(
        "max_nlp_iterations",
        description="maximum number of NLP iterations",
        type=int,
        cast=True,
        min=1,
    )
    warm_start = processed_property(
        "warm_start",
        description="warm-start the mesh iteration using previous solution",
        type=bool,
        cast=True,
    )
    collocation_points_min = processed_property(
        "collocation_points_min",
        description="minimum number of collocation points per mesh section",
        type=int,
        cast=True,
        min=2,
        max=20,
        at_most="collocation_points_max",
    )
    collocation_points_max = processed_property(
        "collocation_points_max",
        description="maximum number of collocation points per mesh section",
        type=int,
        cast=True,
        min=2,
        max=20,
        at_least="collocation_points_min"
    )
    mesh_tolerance = processed_property(
        "mesh_tolerance",
        description="mesh tolerance",
        type=float,
        max=1.0,
        min=0.0,
        exclusive=True,
    )
    max_mesh_iterations = processed_property(
        "max_mesh_iterations",
        description="maximum number of mesh iterations",
        type=int,
        cast=True,
        min=1,
    )
    scaling_method = processed_property(
        "scaling_method",
        description="scaling method",
        options=SCALING_METHODS,
    )
    update_scaling = processed_property(
        "update_scaling",
        description="update scaling factors after each mesh iteration",
        type=bool,
        cast=True,
    )
    number_scaling_samples = processed_property(
        "number_scaling_samples",
        description="number of samples taken when computing scaling factors",
        type=int,
        cast=True,
        min=0,
    )
    scaling_weight = processed_property(
        "scaling_weight",
        description="inter-mesh scaling adjustment weighting factor",
        type=float,
        max=1.0,
        min=0.0,
        exclusive=True,
    )
    console_out_progress = processed_property(
        "console_out_progress",
        description="output Pycollo progress to the console",
        type=bool,
        cast=True,
    )
    display_mesh_refinement_info = processed_property(
        "display_mesh_refinement_info",
        description="display the mesh refinement information",
        type=bool,
        cast=True,
    )
    display_mesh_result_info = processed_property(
        "display_mesh_result_info",
        description="display the mesh result information",
        type=bool,
        cast=True,
    )
    display_mesh_result_graph = processed_property(
        "display_mesh_result_graph",
        description="display a graph of result after each mesh iteration",
        type=bool,
        cast=True,
    )
    assume_inf_bounds = processed_property(
        "assume_inf_bounds",
        description="assume missing bounds are (numerically) infinite",
        type=bool,
        cast=True,
    )
    bound_clash_absolute_tolerance = processed_property(
        "bound_clash_absolute_tolerance",
        description=("absolute tolerance to wish clashing bounds are assumed "
                     "equal"),
        type=float,
        cast=True,
        min=0,
        max=1,
    )
    bound_clash_relative_tolerance = processed_property(
        "bound_clash_relative_tolerance",
        description=("relative tolerance to wish clashing bounds are assumed "
                     "equal"),
        type=float,
        cast=True,
        min=0,
        max=1,
    )
    numerical_inf = processed_property(
        "numerical_inf",
        description="numerical approximation to infinity for calculations",
        type=float,
        cast=True,
        min=0,
    )
    override_endpoint_bounds = processed_property(
        "override_endpoint_bounds",
        description="override bounds at endpoint if value available",
        type=bool,
        cast=True,
    )
    remove_constant_variables = processed_property(
        "remove_constant_variables",
        description="treat variables with equal bounds as constants",
        type=bool,
        cast=True,
    )
    check_nlp_functions = processed_property(
        "check_nlp_functions",
        description="dump evaluated NLP functions to .json file",
        type=bool,
        cast=True,
    )
    mesh_refinement_algorithm = processed_property(
        "mesh_refinement_algorithm",
        description="algorithm used to refine the mesh for the next iteration",
        type=str,
        cast=True,
        options=MESH_REFINEMENT_ALGORITHMS,
    )

    def __init__(self,
                 *,
                 optimal_control_problem=None,
                 backend=BACKENDS.default,
                 collocation_matrix_form=COLLOCATION_MATRIX_FORMS.default,
                 nlp_solver=DEFAULT_NLP_SOLVER,
                 linear_solver=DEFAULT_LINEAR_SOLVER,
                 nlp_tolerance=DEFAULT_NLP_TOLERANCE,
                 max_nlp_iterations=DEFAULT_MAX_NLP_ITERATIONS,
                 warm_start=DEFAULT_WARM_START,
                 quadrature_method=QUADRATURES.default,
                 derivative_level=DEFAULT_DERIVATIVE_LEVEL,
                 max_mesh_iterations=DEFAULT_MAX_MESH_ITERATIONS,
                 mesh_tolerance=DEFAULT_MESH_TOLERANCE,
                 collocation_points_min=DEFAULT_COLLOCATION_POINTS_MIN,
                 collocation_points_max=DEFAULT_COLLOCATION_POINTS_MAX,
                 console_out_progress=DEFAULT_CONSOLE_OUT_PROGRESS,
                 display_mesh_refinement_info=DEFAULT_DISPLAY_MESH_REFINE_INFO,
                 display_mesh_result_info=DEFAULT_DISPLAY_MESH_RESULT_INFO,
                 display_mesh_result_graph=DEFAULT_DISPLAY_MESH_RESULT_GRAPH,
                 scaling_method=SCALING_METHODS.default,
                 update_scaling=DEFAULT_UPDATE_SCALING,
                 number_scaling_samples=DEFAULT_NUMBER_SCALING_SAMPLES,
                 scaling_weight=DEFAULT_SCALING_WEIGHT,
                 assume_inf_bounds=DEFAULT_ASSUME_INF_BOUNDS,
                 bound_clash_absolute_tolerance=DEFAULT_BOUND_CLASH_ABSOLUTE_TOLERANCE,
                 bound_clash_relative_tolerance=DEFAULT_BOUND_CLASH_RELATIVE_TOLERANCE,
                 numerical_inf=DEFAULT_NUMERICAL_INF,
                 override_endpoint_bounds=DEFAULT_OVERRIDE_ENDPOINTS,
                 remove_constant_variables=DEFAULT_REMOVE_CONSTANT_VARIABLES,
                 mesh_refinement_algorithm=MESH_REFINEMENT_ALGORITHMS.default,
                 check_nlp_functions=DEFAULT_CHECK_NLP_FUNCTIONS,
                 ):

        # Optimal Control Problem
        self.ocp = optimal_control_problem

        # Backend
        self.backend = backend

        # NLP solver
        self.nlp_solver = nlp_solver
        self.linear_solver = linear_solver
        self.nlp_tolerance = nlp_tolerance
        self.max_nlp_iterations = max_nlp_iterations
        self.warm_start = warm_start

        # Collocation and quadrature
        self.collocation_matrix_form = collocation_matrix_form
        self.quadrature_method = quadrature_method
        self.derivative_level = derivative_level

        # Mesh refinement
        self.mesh_refinement_algorithm = mesh_refinement_algorithm
        self.collocation_points_min = collocation_points_min
        self.collocation_points_max = collocation_points_max
        self.mesh_tolerance = mesh_tolerance
        self.max_mesh_iterations = max_mesh_iterations

        # Scaling
        self.scaling_method = scaling_method
        self.update_scaling = update_scaling
        self.number_scaling_samples = number_scaling_samples
        self.scaling_weight = scaling_weight

        # Output information
        self.console_out_progress = console_out_progress
        self.display_mesh_refinement_info = display_mesh_refinement_info
        self.display_mesh_result_info = display_mesh_result_info
        self.display_mesh_result_graph = display_mesh_result_graph

        # Bounds
        self.assume_inf_bounds = assume_inf_bounds
        self.bound_clash_absolute_tolerance = bound_clash_absolute_tolerance
        self.bound_clash_relative_tolerance = bound_clash_relative_tolerance
        self.numerical_inf = numerical_inf
        self.override_endpoint_bounds = override_endpoint_bounds
        self.remove_constant_variables = remove_constant_variables

        # Debugging
        self.check_nlp_functions = check_nlp_functions

    @property
    def optimal_control_problem(self):
        """User-friendly alias to the optimal control problem attribute.

        Returns
        -------
        :obj:`OptimalControlProblem`
            The Pycollo optimal control problem object to which these settings
            are to be associated with.
        """
        return self.ocp
