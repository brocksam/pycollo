import abc

import numpy as np
import scipy.sparse as sparse
from pyproprop import Options, processed_property


BOUNDS = "bounds"
GUESS = "guess"
NONE = "none"
USER = "user"


SCALING_METHODS = Options((BOUNDS, GUESS, USER, NONE, None),
                          default=BOUNDS, unsupported=(GUESS, USER))
DEFAULT_NUMBER_SCALING_SAMPLES = 0
DEFAULT_SCALING_WEIGHT = 0.8
DEFAULT_UPDATE_SCALING = False


class ScalingABC(abc.ABC):

    _NONE_SCALING_DEFAULT = 1
    _SCALE_DEFAULT = 1
    _SHIFT_DEFAULT = 0

    optimal_control_problem = processed_property("optimal_control_problem",
                                                 read_only=True)


class EndpointScaling(ScalingABC):

    def __init__(self, optimal_control_problem):

        self.optimal_control_problem = optimal_control_problem

        self.parameter_variables = self._NONE_SCALING_DEFAULT
        self.endpoint_constraints = self._NONE_SCALING_DEFAULT

    def __repr__(self):
        cls_name = self.__class__.__name__
        string = f"{cls_name}(optimal_control_problem={self._ocp}, )"
        return string


class PhaseScaling(ScalingABC):

    phase = processed_property("phase", read_only=True)

    def __init__(self, phase):
        self.phase = phase
        self.optimal_control_problem = phase.optimal_control_problem
        self.time = self._NONE_SCALING_DEFAULT
        self.state_variables = self._NONE_SCALING_DEFAULT
        self.control_variables = self._NONE_SCALING_DEFAULT
        self.integral_variables = self._NONE_SCALING_DEFAULT
        self.path_constraints = self._NONE_SCALING_DEFAULT

    def __repr__(self):
        cls_name = self.__class__.__name__
        string = (f"{cls_name}(phase={self.phase}, )")
        return string


class Scaling(ScalingABC):

    def __init__(self, backend):
        self.backend = backend
        self.ocp = backend.ocp
        self._GENERATE_DISPATCHER = {
            None: self._generate_none,
            NONE: self._generate_none,
            USER: self._generate_user,
            BOUNDS: self._generate_bounds,
            GUESS: self._generate_guess,
        }
        self._generate()

    @property
    def optimal_control_problem(self):
        return self.ocp

    def _generate(self):
        method = self.ocp.settings.scaling_method
        self.x_scales, self.x_shifts = self._GENERATE_DISPATCHER[method]()
        self.c_scales = self._generate_constraint_base()

    def _generate_bounds(self):
        x_l = self.backend.bounds.x_bnd_lower
        x_u = self.backend.bounds.x_bnd_upper
        scales = (x_u - x_l)
        shifts = x_u - (x_u - x_l) / 2
        return scales, shifts

    def _generate_guess(self):
        raise NotImplementedError

    def _generate_none(self):
        num_needed = self.backend.num_var
        scales = self._SCALE_DEFAULT * np.ones(num_needed)
        shifts = self._SHIFT_DEFAULT * np.ones(num_needed)
        return scales, shifts

    def _generate_user(self):
        raise NotImplementedError

    def _generate_constraint_base(self):
        scales = np.ones(self.backend.num_c)
        slices = zip(self.backend.phase_y_var_slices,
                     self.backend.phase_q_var_slices,
                     self.backend.phase_y_eqn_slices,
                     self.backend.phase_q_fnc_slices)
        for y_slice, q_slice, y_eqn_slice, q_fnc_slice in slices:
            scales[y_eqn_slice] = self.x_scales[y_slice]
            scales[q_fnc_slice] = self.x_scales[q_slice]
        return scales


def np_print(to_print):
    for val in to_print:
        prefix = "-" if val < 0 else "+"
        print(f"{prefix}{np.abs(val):.15e},")


class IterationScaling:
    """Variable and constraint scaling, specific to a mesh iteration.

    Attributes
    ----------
    iteration : Iteration
        The associated mesh iteration.
    backend : Union[CasadiBackend, HsadBackend, PycolloBackend, SympyBackend]
        The Pycollo backend for the optimal control problem.
    _GENERATE_DISPATCHER : dict
        Dispatcher for the scaling generation method depending on the option
        settings.
    _V : np.ndarray
        Variable stretch values.
    _r : np.ndarray
        Variable shift values.
    _V_inv : np.ndarray
        Variable unstretch values. Reciprocal of :attr:`_V`.

    """

    def __init__(self, iteration):
        self.iteration = iteration
        self.backend = self.iteration.backend
        self._GENERATE_DISPATCHER = {
            True: self._generate_from_previous,
            False: self._generate_from_base,
        }
        self._initialise_variable_scaling()

    @property
    def optimal_control_problem(self):
        """Convenience property for accessing OCP object."""
        return self.iteration.optimal_control_problem

    @property
    def base_scaling(self):
        """Convenience property for accessing OCP basis scaling."""
        return self.optimal_control_problem._backend.scaling

    def _initialise_variable_scaling(self):
        """Expand basis shift/stretch scaling to initial mesh."""
        self.V_ocp = self.base_scaling.x_scales.copy()
        self.r_ocp = self.base_scaling.x_shifts.copy()
        self.V = self._expand_x_to_mesh(self.V_ocp)
        self.r = self._expand_x_to_mesh(self.r_ocp)
        self.V_inv = np.reciprocal(self.V)

    def scale_x(self, x):
        x_tilde = np.multiply(self.V_inv, (x - self.r))
        return x_tilde

    def unscale_x(self, x_tilde):
        x = np.multiply(self.V, x_tilde) + self.r
        return x

    def scale_sigma(self, sigma_tilde):
        raise NotImplementedError

    def scale_lagrange(self, lagrange_tilde):
        raise NotImplementedError

    def unscale_J(self, J_tilde):
        """Convert J for the scaled NLP to the user basis."""
        J = (1 / self.w) * J_tilde
        return J

    def scale_g(self, g):
        raise NotImplementedError

    def scale_c(self, c):
        c_tilde = np.multiply(self.W, c)
        return c_tilde

    def scale_G(self, sG):
        raise NotImplementedError

    def scale_H(self, sH):
        raise NotImplementedError

    def generate_J_c_scaling(self):
        """Generate scaling factors for the objective and constraints."""
        if self.iteration.number == 1:
            self._generate_first_iteration()
        else:
            use_update = self.optimal_control_problem.settings.update_scaling
            self._GENERATE_DISPATCHER[use_update]()

    def _expand_x_to_mesh(self, base_scaling):
        """Expand basis scaling (OCP variables) to iteration variables."""
        scaling = np.empty(self.iteration.num_x)
        zip_args = zip(self.backend.phase_y_var_slices,
                       self.backend.phase_u_var_slices,
                       self.backend.phase_q_var_slices,
                       self.backend.phase_t_var_slices,
                       self.iteration.y_slices,
                       self.iteration.u_slices,
                       self.iteration.q_slices,
                       self.iteration.t_slices,
                       self.iteration.mesh.N)
        for values in zip_args:
            ocp_y_slice = values[0]
            ocp_u_slice = values[1]
            ocp_q_slice = values[2]
            ocp_t_slice = values[3]
            y_slice = values[4]
            u_slice = values[5]
            q_slice = values[6]
            t_slice = values[7]
            N = values[8]
            scaling[y_slice] = np.repeat(base_scaling[ocp_y_slice], N)
            scaling[u_slice] = np.repeat(base_scaling[ocp_u_slice], N)
            scaling[q_slice] = base_scaling[ocp_q_slice]
            scaling[t_slice] = base_scaling[ocp_t_slice]
        ocp_s_slice = self.backend.s_var_slice
        s_slice = self.iteration.s_slice
        scaling[s_slice] = base_scaling[ocp_s_slice]
        return scaling

    def _expand_c_to_mesh(self, base_scaling):
        """Expand basis scaling (OCP constraints) to iteration constraints."""
        scaling = np.empty(self.iteration.num_c)
        zip_args = zip(self.backend.phase_y_eqn_slices,
                       self.backend.phase_p_con_slices,
                       self.backend.phase_q_fnc_slices,
                       self.iteration.c_defect_slices,
                       self.iteration.c_path_slices,
                       self.iteration.c_integral_slices,
                       self.iteration.mesh.num_c_defect_per_y,
                       self.iteration.mesh.N)
        for values in zip_args:
            ocp_d_slice = values[0]
            ocp_p_slice = values[1]
            ocp_q_slice = values[2]
            d_slice = values[3]
            p_slice = values[4]
            i_slice = values[5]
            num_d = values[6]
            N = values[7]
            scaling[d_slice] = np.repeat(base_scaling[ocp_d_slice], num_d)
            scaling[p_slice] = np.repeat(base_scaling[ocp_p_slice], N)
            scaling[i_slice] = base_scaling[ocp_q_slice]
        ocp_e_slice = self.backend.c_endpoint_slice
        e_slice = self.iteration.c_endpoint_slice
        scaling[e_slice] = base_scaling[ocp_e_slice]
        return scaling

    def _generate_first_iteration(self):
        """Generate objective/constraint scaling for first mesh iteration."""
        self.w = 1.0
        self.W_ocp = self._calculate_constraint_scaling(self.iteration.guess_x)
        self.W = self._expand_c_to_mesh(self.W_ocp)

    def _generate_from_base(self):
        """Generate object/constraint scaling from basis scaling."""
        self.w = self._calculate_objective_scaling(self.iteration.guess_x)
        self.W_ocp = self._calculate_constraint_scaling(self.iteration.guess_x)
        self.W = self._expand_c_to_mesh(self.W_ocp)

    def _generate_from_previous(self):
        """Generate objective/constraint scaling from previous iteration."""
        w = self._calculate_objective_scaling(self.iteration.guess_x)
        prev_w = [mesh_iter.scaling.w
                  for mesh_iter in self.backend.mesh_iterations]
        w_all = np.concatenate([prev_w, np.array([w])])
        alpha = self.optimal_control_problem.settings.scaling_weight
        weights = np.array([alpha * (1 - alpha)**i
                            for i, _ in enumerate(w_all)])
        weights = np.flip(weights)
        weights[0] /= alpha
        self.w = np.average(w_all, weights=weights)

        def set_scales_shifts(ocp_var_slice, var_slice, N=None):
            var = self.iteration.guess_x[var_slice]
            if len(var) == 0:
                pass
            elif N:
                var = var.reshape(N, -1)
                var_min = np.min(var, axis=0)
                var_max = np.max(var, axis=0)
                var_amp = var_max - var_min
                self.V_ocp[ocp_var_slice] = var_amp
                self.r_ocp[ocp_var_slice] = var_max - 0.5 * var_amp
            else:
                V_last = self.V_ocp[ocp_var_slice]
                r_last = self.r_ocp[ocp_var_slice]
                V_next = np.abs(var)
                r_next = (V_next / V_last) * r_last
                self.V_ocp[ocp_var_slice] = V_next
                self.r_ocp[ocp_var_slice] = r_next

        zipped = zip(self.backend.p,
                     self.iteration.mesh.N,
                     self.iteration.y_slices,
                     self.iteration.u_slices,
                     self.iteration.q_slices,
                     self.iteration.t_slices)
        for p, N, y_slice, u_slice, q_slice, t_slice in zipped:
            set_scales_shifts(self.backend.phase_y_var_slices[p.i], y_slice, N)
            set_scales_shifts(self.backend.phase_u_var_slices[p.i], u_slice, N)
            set_scales_shifts(self.backend.phase_q_var_slices[p.i], q_slice)
            set_scales_shifts(self.backend.phase_t_var_slices[p.i], t_slice)
        set_scales_shifts(self.backend.s_var_slice, self.iteration.s_slice)
        prev_V = np.array([mesh_iter.scaling.V_ocp
                           for mesh_iter in self.backend.mesh_iterations])
        prev_r = np.array([mesh_iter.scaling.r_ocp
                           for mesh_iter in self.backend.mesh_iterations])
        V_all = np.vstack([prev_V, self.V_ocp.reshape(1, -1)])
        r_all = np.vstack([prev_r, self.r_ocp.reshape(1, -1)])
        self.V_ocp = np.average(V_all, axis=0, weights=weights)
        self.r_ocp = np.average(r_all, axis=0, weights=weights)
        self.V = self._expand_x_to_mesh(self.V_ocp)
        self.r = self._expand_x_to_mesh(self.r_ocp)
        self.V_inv = np.reciprocal(self.V)

        W = self._calculate_constraint_scaling(self.iteration.guess_x)
        prev_W = np.array([mesh_iter.scaling.W_ocp
                           for mesh_iter in self.backend.mesh_iterations])
        W_all = np.vstack([prev_W, W])
        self.W_ocp = np.average(W_all, axis=0, weights=weights)
        self.W = self._expand_c_to_mesh(self.W_ocp)

    def _calculate_objective_scaling(self, x_guess):
        """Calculate objective function scaling value.

        The scalar scaling for the objective function is based on the
        assumption that the Euclidian-norm (2-norm) of the gradient of the
        objective function (`g`) should be equal to 1.0.

        Returns
        -------
        float
            The scaling factor (`w`) for the objective function (`J`).

        """
        if self.backend.ocp.settings.scaling_method is None:
            return 1
        args = np.concatenate([x_guess, np.ones(1)])
        g = self.backend.g_iter_scale_callable(args)
        g_norm = np.sqrt(np.sum(g**2))
        if np.isclose(g_norm, 0.0):
            obj_scaling = 1
        else:
            obj_scaling = 1 / g_norm
        return obj_scaling

    def _calculate_constraint_scaling(self, x_guess):
        """Calculate constraint function scaling values.

        The scalar scaling for the constraints vector different depending on
        which of the four types of constraint (defect, path, integral and
        endpoint) are being considered. Defect and integral are the simplest
        as they are scaled using the inverse of the corresponding variable
        stretch scalings (`V_inv`), i.e. state variables for defect constraints
        and integral variables for integral constraints. The path and endpoint
        constraints are scaled similarly to the objective function, i.e. such
        that the Euclidian-norm (2-norm) of each row is approximately equal to
        1.0.

        Returns
        -------
        np.ndarray
            The scaling factors (`W`) for the constraints vector (`c`).

        """
        null_scaling = np.ones(self.backend.num_c)
        if self.backend.ocp.settings.scaling_method is None:
            return null_scaling
        args = np.concatenate([x_guess, null_scaling])
        G = self.backend.G_iter_scale_callable(args)
        sG = sparse.coo_matrix(np.array(G))
        G_norm = np.squeeze(np.sqrt(np.array(sG.power(2).sum(axis=1))))
        ocp_c_scales = np.empty(self.backend.num_c)
        zip_args = zip(
            self.backend.phase_y_var_slices,
            self.backend.phase_q_var_slices,
            self.backend.phase_y_eqn_slices,
            self.backend.phase_p_con_slices,
            self.backend.phase_q_fnc_slices,
            self.iteration.c_defect_slices,
            self.iteration.c_path_slices,
            self.iteration.c_integral_slices,
            self.backend.p,
            self.iteration.mesh.num_c_defect_per_y,
            self.iteration.mesh.N)
        for args in zip_args:
            ocp_y_slice = args[0]
            ocp_q_slice = args[1]
            ocp_defect_slice = args[2]
            ocp_path_slice = args[3]
            ocp_integral_slice = args[4]
            defect_slice = args[5]
            path_slice = args[6]
            integral_slice = args[7]
            p = args[8]
            n_defect = args[9]
            N = args[10]
            ocp_c_scales[ocp_defect_slice] = np.reciprocal(
                self.V_ocp[ocp_y_slice])
            ocp_c_scales[ocp_path_slice] = np.reciprocal(
                np.mean(G_norm[path_slice].reshape(p.num_p_con, N), axis=1))
            ocp_c_scales[ocp_integral_slice] = np.reciprocal(
                self.V_ocp[ocp_q_slice])
        ocp_c_scales[self.backend.c_endpoint_slice] = np.reciprocal(
            G_norm[self.iteration.c_endpoint_slice])
        c_scales = ocp_c_scales
        return c_scales

    def _generate_random_sample_variables(self):
        """Generate objective/constraint scaling from random sampling."""
        raise NotImplementedError


class CasadiIterationScaling(IterationScaling):
    """Subclass with CasADi backend-specific scaling overrides."""
    pass


class HsadIterationScaling(IterationScaling):
    """Subclass with hSAD backend-specific scaling overrides."""
    pass


class SympyIterationScaling(IterationScaling):
    """Subclass with Sympy backend-specific scaling overrides."""
    pass


class PycolloIterationScaling(IterationScaling):
    """Subclass with Pycollo backend-specific scaling overrides."""
    pass
