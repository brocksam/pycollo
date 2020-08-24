"""Settings for the Pycollo package.

This module contains all settings for the Pycollo package including settings
for Pycollo :class:`OptimalControlProblem`s as well as settings for dependent
NLP solvers etc.

Attributes
----------
BACKEND_CASADI_KEYWORD : str
    Description
BACKEND_PYCOLLO_KEYWORD : str
    Description
DEFAULT_BACKEND : TYPE
    Description
DEFAULT_COLLOCATION_POINTS_MAX : int
    Description
DEFAULT_COLLOCATION_POINTS_MIN : int
    Description
DEFAULT_DERIVATIVE_LEVEL : TYPE
    Description
DEFAULT_DISPLAY_MESH_REFINE_INFO : bool
    Description
DEFAULT_DISPLAY_MESH_RESULT_GRAPH : bool
    Description
DEFAULT_DISPLAY_MESH_RESULT_INFO : bool
    Description
DEFAULT_LINEAR_SOLVER : TYPE
    Description
DEFAULT_MATRIX_FORM : TYPE
    Description
DEFAULT_MAX_MESH_ITERATIONS : int
    Description
DEFAULT_MAX_NLP_ITERATIONS : int
    Description
DEFAULT_MESH_TOLERANCE : float
    Description
DEFAULT_NLP_SOLVER : TYPE
    Description
DEFAULT_NLP_TOLERANCE : float
    Description
DEFAULT_NUMBER_SCALING_SAMPLES : int
    Description
DEFAULT_QUADRATURE_METHOD : TYPE
    Description
DEFAULT_SCALING_METHOD : TYPE
    Description
DEFAULT_SCALING_WEIGHT : float
    Description
DEFAULT_UPDATE_SCALING : bool
    Description
DERIVATIVE_LEVEL_FIRST : int
    Description
DERIVATIVE_LEVEL_SECOND : int
    Description
LINEAR_SOLVER_MA57_KEYWORD : str
    Description
LINEAR_SOLVER_MUMPS_KEYWORD : str
    Description
MATRIX_FORM_DIFFERENTIAL_KEYWORD : str
    Description
MATRIX_FORM_INTEGRAL_KEYWORD : str
    Description
NLP_SOLVER_BONMIN_KEYWORD : str
    Description
NLP_SOLVER_IPOPT_KEYWORD : str
    Description
NLP_SOLVER_WORHP_KEYWORD : str
    Description
QUADRATURE_GAUSS_KEYWORD : str
    Description
QUADRATURE_LOBATTO_KEYWORD : str
    Description
QUADRATURE_RADAU_KEYWORD : str
    Description
SCALING_BOUNDS_KEYWORD : str
    Description
SCALING_GUESS_KEYWORD : str
    Description
SCALING_NONE_KEYWORD : TYPE
    Description
SCALING_USER_KEYWORD : str
    Description

"""

from typing import Optional

from pyproprop import processed_property


# Backend constants
BACKEND_PYCOLLO_KEYWORD = "pycollo"
BACKEND_CASADI_KEYWORD = "casadi"
BACKEND_SYMPY_KEYWORD = "sympy"
DEFAULT_BACKEND = BACKEND_CASADI_KEYWORD
UNSUPPORTED_BACKEND = (BACKEND_SYMPY_KEYWORD, )

# Matrix form constants
MATRIX_FORM_DIFFERENTIAL_KEYWORD = "differential"
MATRIX_FORM_INTEGRAL_KEYWORD = "integral"
DEFAULT_MATRIX_FORM = MATRIX_FORM_INTEGRAL_KEYWORD

# Scaling method constants
SCALING_NONE_KEYWORD = None
SCALING_USER_KEYWORD = "user"
SCALING_GUESS_KEYWORD = "guess"
SCALING_BOUNDS_KEYWORD = "bounds"
DEFAULT_SCALING_METHOD = SCALING_BOUNDS_KEYWORD
DEFAULT_UPDATE_SCALING = True
DEFAULT_NUMBER_SCALING_SAMPLES = 0
DEFAULT_SCALING_WEIGHT = 0.8

# Solver constants
NLP_SOLVER_IPOPT_KEYWORD = "ipopt"
NLP_SOLVER_SNOPT_KEYWORD = "snopt"
NLP_SOLVER_BONMIN_KEYWORD = "bonmin"
NLP_SOLVER_WORHP_KEYWORD = "worhp"
LINEAR_SOLVER_MUMPS_KEYWORD = "mumps"
LINEAR_SOLVER_MA57_KEYWORD = "ma57"
DEFAULT_NLP_SOLVER = NLP_SOLVER_IPOPT_KEYWORD
UNSUPPORTED_NLP_SOLVER = (NLP_SOLVER_SNOPT_KEYWORD,
                          NLP_SOLVER_BONMIN_KEYWORD,
                          NLP_SOLVER_WORHP_KEYWORD,
                          )
DEFAULT_LINEAR_SOLVER = LINEAR_SOLVER_MUMPS_KEYWORD
UNSUPPORTED_LINEAR_SOLVER = ("ma57", )
DEFAULT_NLP_TOLERANCE = 1e-10
DEFAULT_MAX_NLP_ITERATIONS = 2000

# Quadrature constants
QUADRATURE_GAUSS_KEYWORD = "gauss"
QUADRATURE_LOBATTO_KEYWORD = "lobatto"
QUADRATURE_RADAU_KEYWORD = "radau"
DEFAULT_QUADRATURE_METHOD = QUADRATURE_LOBATTO_KEYWORD
DEFAULT_COLLOCATION_POINTS_MIN = 4
DEFAULT_COLLOCATION_POINTS_MAX = 10

# Derivative level constants
DERIVATIVE_LEVEL_FIRST = 1
DERIVATIVE_LEVEL_SECOND = 2
DEFAULT_DERIVATIVE_LEVEL = DERIVATIVE_LEVEL_SECOND

# Mesh refinement constants
DEFAULT_MESH_TOLERANCE = 1e-8
DEFAULT_MAX_MESH_ITERATIONS = 10

# Display constants
DEFAULT_DISPLAY_MESH_REFINE_INFO = True
DEFAULT_DISPLAY_MESH_RESULT_INFO = False
DEFAULT_DISPLAY_MESH_RESULT_GRAPH = False


class Settings():

    """Settings class for all Pycollo OCP and NLP settings.

    Attributes
    ----------
    collocation_points_max : int
        Minimum allowable number of mesh points per mesh section.
    collocation_points_min : int
        Maximum allowable number of mesh points per mesh section.
    derivative_level : int (1, 2)
        Whether to use exact Hessian in the NLP solving. If value is 1 then
        Pycollo does not produce an exact Hessian, if the value is 2 it
        does.
    display_mesh_refinement_info : bool
        Description
    display_mesh_result_graph : bool
        Description
    display_mesh_result_info : bool
        Description
    linear_solver : str
        Description
    max_mesh_iterations : int
        Description
    max_nlp_iterations : int
        Description
    mesh_tolerance : float
        Description
    nlp_solver : str
        Description
    nlp_tolerance : float
        Description
    number_scaling_samples : int
        Description
    ocp : :obj:`pycollo.OptimalControlProblem`
        The optimal control problem object with which these settings should be
        associated.
    quadrature_method : str
        Description
    scaling_method : (str, None)
        Description
    scaling_weight : float
        Description
    update_scaling : bool
        Description

    """

    _BACKEND_OPTIONS = (
        BACKEND_PYCOLLO_KEYWORD,
        BACKEND_CASADI_KEYWORD,
    )
    _COLLOCATION_MATRIX_FORMS = (
        MATRIX_FORM_DIFFERENTIAL_KEYWORD,
        MATRIX_FORM_INTEGRAL_KEYWORD,
    )
    _SCALING_OPTIONS = (
        SCALING_NONE_KEYWORD,
        SCALING_USER_KEYWORD,
        SCALING_GUESS_KEYWORD,
        SCALING_BOUNDS_KEYWORD,
    )
    _NLP_SOLVERS = (
        NLP_SOLVER_IPOPT_KEYWORD,
        NLP_SOLVER_SNOPT_KEYWORD,
        NLP_SOLVER_WORHP_KEYWORD,
        NLP_SOLVER_BONMIN_KEYWORD,
    )
    _LINEAR_SOLVERS = (
        LINEAR_SOLVER_MUMPS_KEYWORD,
        LINEAR_SOLVER_MA57_KEYWORD,
    )
    _QUADRATURE_METHODS = (
        QUADRATURE_GAUSS_KEYWORD,
        QUADRATURE_LOBATTO_KEYWORD,
        QUADRATURE_RADAU_KEYWORD,
    )
    _DERIVATIVE_LEVELS = (
        DERIVATIVE_LEVEL_FIRST,
        DERIVATIVE_LEVEL_SECOND,
    )

    backend = processed_property(
        "backend",
        description="Pycollo backend",
        type=str,
        options=_BACKEND_OPTIONS,
        unsupported_options=UNSUPPORTED_BACKEND,
    )
    derivative_level = processed_property(
        "derivative_level",
        description="derivative level",
        type=int,
        cast=True,
        options=_DERIVATIVE_LEVELS,
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

    def __init__(self, *,
                 optimal_control_problem=None,
                 backend=DEFAULT_BACKEND,
                 collocation_matrix_form=DEFAULT_MATRIX_FORM,
                 nlp_solver=DEFAULT_NLP_SOLVER,
                 linear_solver=DEFAULT_LINEAR_SOLVER,
                 nlp_tolerance=DEFAULT_NLP_TOLERANCE,
                 max_nlp_iterations=DEFAULT_MAX_NLP_ITERATIONS,
                 quadrature_method=DEFAULT_QUADRATURE_METHOD,
                 derivative_level=DEFAULT_DERIVATIVE_LEVEL,
                 max_mesh_iterations=DEFAULT_MAX_MESH_ITERATIONS,
                 mesh_tolerance=DEFAULT_MESH_TOLERANCE,
                 collocation_points_min=DEFAULT_COLLOCATION_POINTS_MIN,
                 collocation_points_max=DEFAULT_COLLOCATION_POINTS_MAX,
                 display_mesh_refinement_info=DEFAULT_DISPLAY_MESH_REFINE_INFO,
                 display_mesh_result_info=DEFAULT_DISPLAY_MESH_RESULT_INFO,
                 display_mesh_result_graph=DEFAULT_DISPLAY_MESH_RESULT_GRAPH,
                 scaling_method=DEFAULT_SCALING_METHOD,
                 update_scaling=DEFAULT_UPDATE_SCALING,
                 number_scaling_samples=DEFAULT_NUMBER_SCALING_SAMPLES,
                 scaling_weight=DEFAULT_SCALING_WEIGHT,
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

        # # Collocation and quadrature
        # self.collocation_matrix_form = collocation_matrix_form
        # self.quadrature_method = quadrature_method
        self.derivative_level = derivative_level

        # # Mesh refinement
        # self.collocation_points_min = collocation_points_min
        # self.collocation_points_max = collocation_points_max
        # self.mesh_tolerance = mesh_tolerance
        # self.max_mesh_iterations = max_mesh_iterations

        # # Scaling
        # self.scaling_method = scaling_method
        # self.update_scaling = update_scaling
        # self.number_scaling_samples = number_scaling_samples
        # self.scaling_weight = scaling_weight

        # # Output information
        # self.display_mesh_refinement_info = display_mesh_refinement_info
        # self.display_mesh_result_info = display_mesh_result_info
        # self.display_mesh_result_graph = display_mesh_result_graph

    @property
    def optimal_control_problem(self):
        return self.ocp

    @property
    def scaling_method(self) -> Optional[str]:
        return self._scaling_method

    @scaling_method.setter
    def scaling_method(self, method: Optional[str]):
        if method is not None:
            method = method.casefold()
        if method not in self._SCALING_OPTIONS:
            msg = (f"{method} is not a valid scaling option.")
            raise ValueError(msg)
        self._scaling_method = method

    @property
    def update_scaling(self):
        return self._update_scaling

    @update_scaling.setter
    def update_scaling(self, do_update):
        do_update = bool(do_update)
        self._update_scaling = do_update

    @property
    def scaling_weight(self):
        return self._scaling_weight

    @scaling_weight.setter
    def scaling_weight(self, weight):
        weight = float(weight)
        if weight < 0 or weight > 1:
            msg = (f'Scaling weight must be a number between 0 and 1. {weight} '
                    'is invalid.')
            raise ValueError(msg)
        self._scaling_weight = weight

    @property
    def number_scaling_samples(self):
        return self._number_scaling_samples

    @number_scaling_samples.setter
    def number_scaling_samples(self, num):
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
    def quadrature_method(self):
        return self._quadrature_method

    @quadrature_method.setter
    def quadrature_method(self, method):
        method = method.casefold()
        if method not in self._QUADRATURE_METHODS:
            msg = ("The quadrature method of '{}' is not a valid argument string.")
            raise ValueError(msg.format(method))
        self._quadrature_method = method

    # @property
    # def derivative_level(self):
    #     return self._derivative_level

    # @derivative_level.setter
    # def derivative_level(self, deriv_level):
    #     deriv_level = int(deriv_level)
    #     if deriv_level not in self._DERIVATIVE_LEVELS:
    #         msg = (f"Derivative level must be set to either 1 (which uses the gradient vector of the objective function and the Jacobian matrix of the constraints vector) or 2 (which also uses the Hessian matrix of the Lagrangian of the constraints).")
    #         raise ValueError(msg)
    #     self._derivative_level = deriv_level

    @property
    def mesh_tolerance(self):
        return self._mesh_tolerance

    @mesh_tolerance.setter
    def mesh_tolerance(self, tolerance):
        if tolerance <= 0:
            msg = (
                "Tolerance for the mesh must be a postive real number. {} is invalid.")
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
