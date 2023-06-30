"""The main way to define and interact with a Pycollo optimal control problem.

Method is used to define the objective function, endpoint constraints, and parameter variables. Each :class:`~.OptimalControlProblem` instance is associated with one or more :class:`~.Phase` instance.

OCP require bounds and initial guesses to be prescribed before they can numerically be solved, handeled in :mod:`~.bounds` and :mod:`~.guess`.

Terminolgy is loosely defined in accordance with "Betts, JT (2010). Practical Methods for Optimal Control and Estimiation Using Nonlinear Programming (Second Edition)":

Notes:
    * t: independent parameter (time).
    * x = [y, u, q, t0, tf, s]: vector of problem variables.
    * y: vector state variables (which are functions of time).
    * u: vector control variables (which are functions of time).
    * q: vector of integral constraints.
    * t0: the initial time of the (single) phase.
    * tf: the final time of the (single) phase.
    * s: vector of static parameter variables (which are phase-independent).

    * J: objective function.
    * g: gradient of the objective function w.r.t. x.
    * L: Lagrangian of the objective function and constraints.
    * H: Hessian of the Lagrangian.

    * c = [zeta, gamma, rho, beta]: vector of constraints.
    * zeta: vector of defect constraints.
    * gamma: vector of path constraints.
    * rho: vector of integral constraints.
    * beta: vector of endpoint constraints.
    * G: Jacobian of the constaints.

    * n = len(x): number of free variables
    * m = len(c): number of constraints
"""


from typing import AnyStr, Iterable, Tuple, Optional

import numpy as np

# from ordered_set import OrderedSet
import sympy as sym

from .backend import BACKENDS
from .bounds import EndpointBounds
from .guess import EndpointGuess
from .phase import Phase
from .scaling import EndpointScaling
from .settings import Settings
from .typing import OptionalSymsType
from .utils import check_sym_name_clash, console_out, format_as_data_container

__all__ = ["OptimalControlProblem"]


class OptimalControlProblem():
    """The main class for Pycollo optimal control problems"""

    def __init__(self,
                 name: str,
                 parameter_variables=None,
                 *,
                 bounds: Optional[EndpointBounds]=None,
                 guess: Optional[EndpointGuess]=None,
                 scaling: Optional[EndpointScaling]=None,
                 endpoint_constraints: Optional=None,
                 objective_function: Optional[sym.Expr]=None,
                 settings: Optional[Settings]=None,
                 auxiliary_data: Optional[dict]=None,
                 ) -> None:
        """Initialise the optimal control problem with user-passed objects.

        Args:
            name: Minimally required to initialise
            parameter_variables: 
            bounds: 
            guess: 
            scaling: 
            endpoint_constraints: 
            objective_function: 
            settings: 
            auxiliary_data: 
        """

        self.name = name
        self.settings = settings
        self._is_initialised = False
        self._forward_dynamics = False
        self._s_var_user = ()
        self._b_con_user = ()
        self._phases = ()
        self.parameter_variables = parameter_variables
        self.endpoint_constraints = endpoint_constraints
        self.objective_function = objective_function
        self.auxiliary_data = dict(auxiliary_data) if auxiliary_data else {}
        self.scaling = scaling
        self.bounds = bounds
        self.guess = guess

    @property
    def name(self) -> str:
        """The name associated with the optimal control problem. For setter
        behaviour, the supplied `name` is cast to a :class:`str`.

        The name is not strictly needed, however it improves the usefulness of
        Pycollo console output. This is particularly useful in cases where the
        user may wish to instantiate multiple :obj:`OptimalControlProblem`
        objects within a single script, or instantiates other Pycollo objects

        without providing a valid :mod:`~.optimal_control_problem` argument for them to be linked to at initialisation.
        """
        return self._name

    @name.setter
    def name(self, name: AnyStr):
        self._name = str(name)

    @property
    def phases(self) -> Tuple[Phase, ...]:
        """A :class:`tuple` of all phases associated with the optimal control problem.

        :meth:`~.phase_number` are integers beginning at 1 and are
        ordered corresponding to the order that they were added to the optimal
        control problem. As Python uses zero-based indexing the phase numbers
        do not directly map to the indexes of phases within :attr:`~.phases`.
        Phases are however ordered sequentially corresponding to the
        chronological order they were added to the optimal control problem.
        """
        return self._phases

    def add_phase(self, phase: Iterable[Phase]) -> Phase:
        """Add an already instantiated :class:`~.Phase` to this optimal control problem.

        This method is needed as :attr:`~.phases` is read only ("private") and
        therefore users cannot manually add :class:`~.Phase` objects to an optimal
        control problem. :attr:`~.phases` is required to be read only as it is an
        iterable of :class:`~.Phase` objects and must be protected from accidental errors
        introduced by user interacting with it incorrectly.

        Args:
            phase (:class:`~.Phase`): The phase to be added to the optimal control problem

        Returns:
            the phase that has been added. It is the same
        """
        phase.optimal_control_problem = self
        return self.phases[-1]

    def add_phases(self, phases: Iterable[Phase]) -> Tuple[Phase, ...]:
        """Associate multiple already instantiated :class:`~.Phase` objects.

        This is a convinience method to allow the user to add multiple :class:`~.Phase`
        objects to the optimal control problem in one go.
        """
        return tuple(self.add_phase(phase) for phase in phases)

    def new_phase(self,
                  name: str,
                  state_variables: OptionalSymsType = None,
                  control_variables: OptionalSymsType = None) -> Phase:
        """Create a new :obj:`~.Phase` and add to this optimal control problem.

        Provides the same behaviour as manually creating a :obj:`~.Phase` called
        `phase` and calling :meth:`~.add_phase`.
        """
        new_phase = Phase(name, optimal_control_problem=self,
                          state_variables=state_variables,
                          control_variables=control_variables)
        return new_phase

    def new_phase_like(self, phase_for_copying: Phase, name: str, **kwargs):
        """Creates new phase while copying all implemented values from `phase_for_copying`
        """
        return phase_for_copying.create_new_copy(name, **kwargs)

    def new_phases_like(self,
                        phase_for_copying: Phase,
                        number: int,
                        names: Iterable[str],
                        **kwargs) -> Tuple[Phase, ...]:
        """Creates multiple new phases like an already instantiated phase.

        For a list of key word arguments and default values see the docstring
        for the :func:`new_phase_like` method.

        Returns:
            The newly instantiated and associated phases.

        Raises:
            ValueError: If the same number of names are not supplied as the
                number of specified new phases.
        """
        if len(names) != int(number):
            msg = ("Must supply a name for each new phase.")
            raise ValueError(msg)
        new_phases = (self.new_phase_like(phase_for_copying, name, **kwargs)
                      for name in names)
        return new_phases

    @property
    def number_phases(self) -> int:
        """Returns number of phases associated with this optimal control problem."""
        return len(self.phases)

    @property
    def time_symbol(self):
        """Dynamic time symbol, not yet supported
        
        Raises:
            NotImplementedError: Whenever called to inform the user that these
                types of problem are not currently supported.
        """
        msg = ("Pycollo do not currently support dynamic, path or integral "
               "constraints that are explicit functions of continuous time.")
        raise NotImplementedError(msg)

    @property
    def parameter_variables(self):
        """Static parameter variables which are optimized within given bounds.

        When the instance OCP is created you can supply the parameter variables by OCP_instance.parameter_variables = :class:`tuple` [:class:`Symbol<sympy.core.symbol.Symbol>`,...] | :class:`list` [:class:`Symbol<sympy.core.symbol.Symbol>`,..] | :obj:`numpy.array` [:class:`Symbol<sympy.core.symbol.Symbol>`,...]

        As described in Betts, JT (2010). Bounds and guesses need to be implemented. See :obj:`~.EndPointBounds` and :class:`~.EndPointGuess` how to implement them. 
        """
        return self._s_var_user

    @parameter_variables.setter
    def parameter_variables(self, s_vars):
        self._s_var_user = format_as_data_container("ParameterVariables", s_vars)
        _ = check_sym_name_clash(self._s_var_user)

    @property
    def number_parameter_variables(self):
        len(self._s_var_user)

    @property
    def endpoint_constraints(self):
        """Inequality constraints consisting of endpoint variables.

        These constraints are the glue in between the phases. For example, when variables need to be continious throughout multiple phases you can set the y_F of phase A equal to y_0 of phase B. More complex formulations are allowed for example for handeling discontinuities. They are formulated as inequality constraint, but by clever formulating can be handeled as equality constraints (A-B > [0,0])
        
        Bounds should be implemented. See :obj:`~.EndPointBounds` how to implement them.
        """
        return self._b_con_user

    @endpoint_constraints.setter
    def endpoint_constraints(self, b_cons):
        self._b_con_user = format_as_data_container(
            "EndpointConstraints",
            b_cons,
            use_named=False,
        )

    @property
    def number_endpoint_constraints(self) -> int:
        """Returns number of endpoint constraints"""
        return len(self._b_con_user)

    @property
    def objective_function(self):
        """Formula to be minimised during the optimization.
        
        It is a `Bolza` objective function which may be a function of endpoint variables or parameter variables (:attr:`~.integral_variables`, :attr:`~.initial_time_variable`, :attr:`~.final_time_variable`, :attr:`~.initial_state_variable, and :attr:`~.final_state_variable or :attr:`~.parameter_variables` ).
        """
        return self._J_user

    @objective_function.setter
    def objective_function(self, J):
        self._J_user = sym.sympify(J)
        # self._forward_dynamics = True if self._J_user == 1 else False

    @property
    def auxiliary_data(self):
        """:class:`dict` containing extra provided data.
         
        When a symbolic variable (:class:`Symbol<sympy.core.symbol.Symbol>`) is indetermined, make sure to assign it to a numerical value or symbolic formulation with this function """
        return self._aux_data_user

    @auxiliary_data.setter
    def auxiliary_data(self, aux_data):
        self._aux_data_user = dict(aux_data)

    @property
    def bounds(self) -> EndpointBounds:
        """Attribute to provide bounds to :attr:`~.parameter_variables`, and :attr:`~.endpoint_constraints`"""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is None:
            self._bounds = EndpointBounds(optimal_control_problem=self)
        else:
            self._bounds = bounds
            self._bounds._ocp = self

    @property
    def guess(self) -> EndpointGuess:
        """Attribute to provide initial guess to :attr:`~.parameter_variables`, and :attr:`~.endpoint_constraints`"""
        return self._guess

    @guess.setter
    def guess(self, guess):
        if guess is None:
            self._guess = EndpointGuess(optimal_control_problem=self)
        else:
            self._guess = guess
            self._guess.optimal_control_problem = self

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        if scaling is None:
            self._scaling = EndpointScaling(optimal_control_problem=self)
        else:
            self._scaling = scaling
            self._scaling._ocp = self

    @property
    def mesh_iterations(self):
        return self._mesh_iterations

    @property
    def num_mesh_iterations(self) -> int:
        """Returns number of mesh iterations"""
        return len(self._backend.mesh_iterations)

    @property
    def settings(self) -> Settings:
        """Attribute to overwrite default settings.
        
        See :mod:`~.settings` for all settings options"""
        return self._settings

    @settings.setter
    def settings(self, settings):
        if settings is None:
            self._settings = Settings(optimal_control_problem=self)
        else:
            self._settings = settings
            self._settings._ocp = self

    @property
    def solution(self):
        """ Attribute where all solutions are found after solving the OCP (:func:`~.solve`).

        Solutions are provided by indexing. OCP_instance.solution.attribute[x][y] refers to the y`th attribute in phase x
         
         Attributes:
            objective: Numerical solution of :attr:`~.objective_function`
            initial_time: Initial times of phases of solved OCP
            final_time: Final times of phases of solved OCP
            state: Numerical solution of state_variables`
            state_derivative: Numerical solution of :attr:`~.state_equations`
            control: Numerical solution of :attr:`~.control`
            integral: Numerical solution of :attr:`~.integrand_function`
            time: Numerical solution of time
            parameter: Numerical solution of :attr:`~.parameter_variables`
           """
        return self._backend.mesh_iterations[-1].solution

    def initialise(self):
        """Initialise the optimal control problem before solving.

        The initialisation of the optimal control problem involves the
        following stages:

            * 1. Check that for each phase there are the same number of state variables as there are state equations.
            * 2. Check that for each phase the user-supplied bounds are permissible, and check point bounds on optimal control problem.
            * 3. Process bounds that need processing.

        """
        self._console_out_initialisation_message()
        self._check_variables_and_equations()
        self._initialise_backend()
        self._check_problem_and_phase_bounds()
        self._initialise_scaling()
        self._initialise_quadrature()
        self._postprocess_backend()
        self._initialise_initial_mesh()
        self._check_initial_guess()
        self._initialise_first_mesh_iteration()
        self._is_initialised = True

    def _console_out_initialisation_message(self):
        msg = "Initialising optimal control problem."
        console_out(msg, heading=True)

    def _check_variables_and_equations(self):
        for phase in self.phases:
            phase._check_variables_and_equations()
        msg = "Phase variables and equations checked."
        console_out(msg)

    def _initialise_backend(self):
        self._backend = BACKENDS.dispatcher[self.settings.backend](self)
        msg = "Backend initialised."
        console_out(msg)

    def _check_problem_and_phase_bounds(self):
        self._backend.create_bounds()
        msg = "Bounds checked."
        console_out(msg)

    def _initialise_scaling(self):
        self._backend.create_scaling()
        msg = "Problem scaling initialised."
        console_out(msg)

    def _initialise_quadrature(self):
        self._backend.create_quadrature()
        msg = "Quadrature scheme initialised."
        console_out(msg)

    def _check_initial_guess(self):
        self._backend.create_guess()
        msg = "Initial guess checked."
        console_out(msg)

    def _postprocess_backend(self):
        self._backend.postprocess_problem_backend()
        msg = "Backend postprocessing complete."
        console_out(msg)

    def _initialise_initial_mesh(self):
        self._backend.create_initial_mesh()
        msg = "Initial mesh created."
        console_out(msg)

    def _initialise_first_mesh_iteration(self):
        self._backend.create_mesh_iterations()

    def solve(self, display_progress=False):
        """Solve the optimal control problem.

        If the initialisation flag is not set to True then the initialisation
        method is called to initialise the optimal control problem.

        Parameters:
            display_progress : bool
                Option for whether progress updates should be outputted to the
                console during solving. Defaults to False.
        """
        self._check_if_initialisation_required_before_solve()
        self.mesh_tolerance_met = False
        self._set_solve_options(display_progress)
        tolerances_met = False
        while not tolerances_met:
            tolerances_met = self._solve_iteration()
        self._final_output()

    def _check_if_initialisation_required_before_solve(self):
        """Initialise the optimal control problem before solve if required."""
        if not self._is_initialised:
            self.initialise()

    def _solve_iteration(self):
        """Solve a single mesh iteration.

        Return:
        bool
            True is mesh tolerance is met or if maximum number of mesh
            iterations has been reached.

        """

        def tolerances_met(mesh_tolerance_met, mesh_iterations_met):
            return (mesh_iterations_met or mesh_tolerance_met)

        if self._backend.mesh_iterations[-1].solved:
            _ = self._backend.new_mesh_iteration(self._next_iteration_mesh,
                                                 self._next_iteration_guess)
        result = self._backend.mesh_iterations[-1].solve()
        self._next_iteration_mesh = result.next_iteration_mesh
        self._next_iteration_guess = result.next_iteration_guess
        if result.mesh_tolerance_met:
            self.mesh_tolerance_met = True
            msg = (f"Mesh tolerance met in mesh iteration "
                   f"{self.num_mesh_iterations}.\n")
            print(msg)
        if self.num_mesh_iterations >= self.settings.max_mesh_iterations:
            mesh_iterations_met = True
            if not self.mesh_tolerance_met:
                msg = ("Maximum number of mesh iterations reached. Pycollo "
                       "exiting before mesh tolerance met.\n")
                print(msg)
        else:
            mesh_iterations_met = False
        return tolerances_met(result.mesh_tolerance_met, mesh_iterations_met)

    def _set_solve_options(self, display_progress):
        self._display_progress = display_progress

    # def _initialise(self):
    #   self._check_user_supplied_bounds()
    #   self._generate_scaling()
    #   self._generate_expression_graph()
    #   self._generate_quadrature()
    #   self._compile_numba_functions()
    #   self._check_user_supplied_initial_guess()

    #   # Initialise the initial mesh iterations
    #   self._mesh_iterations[0]._initialise_iteration(self.initial_guess)

    #   ocp_initialisation_time_stop = timer()
    #   self._ocp_initialisation_time = (ocp_initialisation_time_stop
    #       - ocp_initialisation_time_start)
    #   self._is_initialised = True

    # def solve_old(self, display_progress=False):
    #   """Solve the optimal control problem.

    #   If the initialisation flag is not set to True then the initialisation
    #   method is called to initialise the optimal control problem.

    #   Parameters:
    #   -----------
    #   display_progress : bool
    #       Option for whether progress updates should be outputted to the
    #       console during solving. Defaults to False.
    #   """

    #   self._set_solve_options(display_progress)
    #   self._check_if_initialisation_required_before_solve()

    #   # Solve the transcribed NLP on the initial mesh
    #   new_iteration_mesh, new_iteration_guess = self._mesh_iterations[0]._solve()

    #   mesh_iterations_met = self._settings.max_mesh_iterations == 1
    #   if new_iteration_mesh is None:
    #           mesh_tolerance_met = True
    #   else:
    #       mesh_tolerance_met = False

    #   while not mesh_iterations_met and not mesh_tolerance_met:
    #       new_iteration = Iteration(optimal_control_problem=self, iteration_number=self.num_mesh_iterations+1, mesh=new_iteration_mesh)
    #       self._mesh_iterations.append(new_iteration)
    #       self._mesh_iterations[-1]._initialise_iteration(new_iteration_guess)
    #       new_iteration_mesh, new_iteration_guess = self._mesh_iterations[-1]._solve()
    #       if new_iteration_mesh is None:
    #           mesh_tolerance_met = True
    #           print(f'Mesh tolerance met in mesh iteration {len(self._mesh_iterations)}.\n')
    #       elif self.num_mesh_iterations >= self._settings.max_mesh_iterations:
    #           mesh_iterations_met = True
    #           print(f'Maximum number of mesh iterations reached. pycollo exiting before mesh tolerance met.\n')

    #   _ = self._final_output()

    def _set_solve_options(self, display_progress):
        self._display_progress = display_progress

    def _check_if_initialisation_required_before_solve(self):
        if not self._is_initialised:
            self.initialise()

    def _final_output(self):

        def solution_results():
            J_msg = (f'Final Objective Function Evaluation: {self._backend.mesh_iterations[-1].solution.objective:.4f}\n')
            print(J_msg)

        def mesh_results():
            section_msg = (f'Final Number of Mesh Sections:       {self.mesh_iterations[-1]._mesh._K}')
            node_msg = (f'Final Number of Collocation Nodes:   {self.mesh_iterations[-1]._mesh._N}\n')
            print(section_msg)
            print(node_msg)

        def time_results():
            ocp_init_time_msg = (f'Total OCP Initialisation Time:       {self._ocp_initialisation_time:.4f} s')
            print(ocp_init_time_msg)

            self._iteration_initialisation_time = np.sum(np.array(
                [iteration._initialisation_time for iteration in self._mesh_iterations]))

            iter_init_time_msg = (f'Total Iteration Initialisation Time: {self._iteration_initialisation_time:.4f} s')
            print(iter_init_time_msg)

            self._nlp_time = np.sum(
                np.array([iteration._nlp_time for iteration in self._mesh_iterations]))

            nlp_time_msg = (f'Total NLP Solver Time:               {self._nlp_time:.4f} s')
            print(nlp_time_msg)

            self._process_results_time = np.sum(np.array(
                [iteration._process_results_time for iteration in self._mesh_iterations]))

            process_results_time_msg = (f'Total Mesh Refinement Time:          {self._process_results_time:.4f} s')
            print(process_results_time_msg)

            total_time_msg = (f'\nTotal Time:                          {self._ocp_initialisation_time + self._iteration_initialisation_time + self._nlp_time + self._process_results_time:.4f} s')
            print(total_time_msg)
            print('\n\n')

        solved_msg = ('Optimal control problem sucessfully solved.')
        console_out(solved_msg, heading=True)

        solution_results()
        if False:
            mesh_results()
            time_results()

    def __str__(self):
        """Returns name of OCP"""
        return self.name

    def __repr__(self):
        return f"OptimalControlProblem('{self.name}')"


def kill():
    print('\n\n')
    raise ValueError


def cout(*args):
    print('\n\n')
    for arg in args:
        print(f'{arg}\n')
