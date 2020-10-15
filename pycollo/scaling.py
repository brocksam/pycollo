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
DEFAULT_UPDATE_SCALING = True


class ScalingABC(abc.ABC):

    _NONE_SCALING_DEFAULT = 1
    _SCALE_DEFAULT = 1
    _SHIFT_DEFAULT = 0
    # _METHOD_OPTIONS = {DEFAULT, BOUNDS, USER, GUESS, NONE, None}
    # _METHOD_DEFAULT = BOUNDS
    # _UPDATE_DEFAULT = False
    # _NUMBER_SAMPLES_DEFAULT = 100
    # _UPDATE_WEIGHT_DEFAULT = 0.8

    optimal_control_problem = processed_property("optimal_control_problem",
                                                 read_only=True)


class EndpointScaling(ScalingABC):

    def __init__(self, optimal_control_problem):

        self.optimal_control_problem = optimal_control_problem

        self.parameter_variables = self._NONE_SCALING_DEFAULT
        self.endpoint_constraints = self._NONE_SCALING_DEFAULT

    def __repr__(self):
        cls_name = self.__class__.__name__
        string = (f"{cls_name}(optimal_control_problem={self._ocp}, )")
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

    # def _generate_bounds(self):
    # 	raise NotImplementedError

    # def _generate_guess(self):
    # 	raise NotImplementedError

    # def _generate_none(self):
    # 	num_needed = self.optimal_control_problem._backend.num_s_vars
    # 	shift = self._SHIFT_DEFAULT * np.ones(num_needed)
    # 	stretch = self._STRETCH_DEFAULT * np.ones(num_needed)
    # 	return shift, stretch

    # def _generate_user(self):
    # 	raise NotImplementedError

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


class IterationScaling:

    def __init__(self, iteration):
        self.iteration = iteration
        self.backend = self.iteration.backend
        self._GENERATE_DISPATCHER = {
            True: self._generate_from_previous,
            False: self._generate_from_base,
        }
        self._initialise_variable_scaling()
        self._w = 1
        self._sW = sparse.diags(np.ones(self.iteration.num_c))

    @property
    def optimal_control_problem(self):
        return self.iteration.optimal_control_problem

    @property
    def base_scaling(self):
        return self.optimal_control_problem._backend.scaling

    def _initialise_variable_scaling(self):
        self.x_scales_ocp = self.base_scaling.x_scales
        self.x_shifts_ocp = self.base_scaling.x_shifts
        self.x_scales = self._expand_x_to_mesh(self.base_scaling.x_scales)
        self.x_shifts = self._expand_x_to_mesh(self.base_scaling.x_shifts)
        self.x_scales_inv = np.reciprocal(self.x_scales)

        self._sV = sparse.diags(self.x_scales)
        self._sV_inv = sparse.diags(self.x_scales_inv)
        self._sV_sqrd_inv = sparse.diags(self.x_scales_inv**2)
        self._r = self.x_shifts

        self._V = np.diag(self.x_scales)
        self._V_inv = np.diag(self.x_scales_inv)
        self._V_sqrd_inv = np.diag(self.x_scales_inv**2)
        self._r = self.x_shifts

        self._V = np.diag(np.ones_like(self.x_scales))
        self._V_inv = np.diag(np.ones_like(self.x_scales_inv))
        self._V_sqrd_inv = np.diag(np.ones_like(self.x_scales_inv**2))
        self._r = self.x_shifts

        self._W = np.eye(self.iteration.num_c)

    def scale_x(self, x):
        x_tilde = np.dot(self._V, (x - self._r))
        # x_tilde = self._sV.dot(x - self._r)
        return x_tilde

    def unscale_x(self, x_tilde):
        x = np.dot(self._V_inv, x_tilde) + self._r
        # x = self._sV_inv.dot(x_tilde) + self._r
        return x

    def scale_sigma(self, sigma_tilde):
        sigma_prime = sigma_tilde
        # sigma_prime = self._w * sigma_tilde
        return sigma_prime

    def scale_lagrange(self, lagrange_tilde):
        lagrange_prime = lagrange_tilde
        # lagrange_prime = self._sW.dot(lagrange_tilde)
        return lagrange_prime

    def scale_J(self, J):
        J_tilde = J
        # J_tilde = self._w * J
        return J_tilde

    def scale_g(self, g):
        g_tilde = np.dot(g, self._V_inv)
        # g_tilde = self._w * self._sV_inv.T.dot(g.T).T
        return g_tilde

    def scale_c(self, c):
        # c_tilde = np.dot(self._W, c)
        c_tilde = c
        # c_tilde = self._sW.dot(c)
        return c_tilde

    def scale_G(self, sG):
        # print(self._W)
        # input()
        # print(sG)
        # G = sG.toarray()
        # G_inter_1 = np.dot(G, self._V_inv)
        # G_inter_2 = np.dot(self._W, G_inter_1)
        # sG_tilde = sparse.csr_matrix(G_inter_2)
        # print(sG_tilde)
        # input()
        sG_tilde = sparse.csr_matrix(np.dot(sG.toarray(), self._V_inv))
        # sG_tilde = self._sW.dot(self._sV_inv.T.dot(sG.T).T).tocoo().tocsr()
        return sG_tilde

    def scale_H(self, sH):
        sH_tilde = sparse.csr_matrix(np.dot(sH.toarray(), self._V_sqrd_inv))
        # H_tilde = self._sV_sqrd_inv.T.dot(H.T).T
        return sH_tilde

    def _generate(self):
        if self.iteration.number == 1:
            self._generate_first_iteration()
        else:
            update_scaling = self.optimal_control_problem.settings.update_scaling
            self._GENERATE_DISPATCHER[update_scaling]()

    def _expand_x_to_mesh(self, base_scaling):
        scaling = np.empty(self.iteration.num_x)
        zip_args = zip(self.backend.phase_y_vars_slices,
                       self.backend.phase_u_vars_slices,
                       self.backend.phase_q_vars_slices,
                       self.backend.phase_t_vars_slices,
                       self.iteration.y_slices,
                       self.iteration.u_slices,
                       self.iteration.q_slices,
                       self.iteration.t_slices,
                       self.iteration.mesh.N)
        for ocp_y_slice, ocp_u_slice, ocp_q_slice, ocp_t_slice, y_slice, u_slice, q_slice, t_slice, N in zip_args:
            scaling[y_slice] = np.repeat(base_scaling[ocp_y_slice], N)
            scaling[u_slice] = np.repeat(base_scaling[ocp_u_slice], N)
            scaling[q_slice] = base_scaling[ocp_q_slice]
            scaling[t_slice] = base_scaling[ocp_t_slice]
        scaling[self.iteration.s_slice] = base_scaling[self.backend.s_vars_slice]
        return scaling

    def _expand_c_to_mesh(self, base_scaling):
        scaling = np.empty(self.iteration.num_c)
        zip_args = zip(self.backend.phase_defect_constraint_slices,
                       self.backend.phase_path_constraint_slices,
                       self.backend.phase_integral_constraint_slices,
                       self.iteration.c_defect_slices,
                       self.iteration.c_path_slices,
                       self.iteration.c_integral_slices,
                       self.iteration.mesh.num_c_defect_per_y,
                       self.iteration.mesh.N)
        for ocp_defect_slice, ocp_path_slice, ocp_integral_slice, defect_slice, path_slice, integral_slice, n_defect, N in zip_args:
            scaling[defect_slice] = np.repeat(
                base_scaling[ocp_defect_slice], n_defect)
            scaling[path_slice] = np.repeat(base_scaling[ocp_path_slice], N)
            scaling[integral_slice] = base_scaling[ocp_integral_slice]
        scaling[self.iteration.c_endpoint_slice] = base_scaling[self.backend.c_endpoint_slice]
        return scaling

    def _generate_first_iteration(self):
        self._generate_from_base()
        # self.J_scale = 1
        # self._w = self.J_scale
        # self.c_scales = self._calculate_constraint_scaling(self.iteration.guess_x)
        # self._sW = sparse.diags(self.c_scales)

    def _generate_from_base(self):
        self.J_scale = self._calculate_objective_scaling(
            self.iteration.guess_x)
        self._w = self.J_scale
        self.c_scales = self._calculate_constraint_scaling(
            self.iteration.guess_x)
        self._sW = sparse.diags(self.c_scales)

        self._W = np.diag(self.c_scales)

        # self._sV = sparse.diags(self.x_scales)
        # self._sV_inv = sparse.diags(self.x_scales_inv)
        # self._sV_sqrd_inv = sparse.diags(self.x_scales_inv**2)

    def _generate_from_previous(self):
        J_scale = self._calculate_objective_scaling(self.iteration.guess_x)
        prev_J_scale = [
            mesh_iter.scaling.J_scale for mesh_iter in self.optimal_control_problem._backend.mesh_iterations[:-1]]
        J_scales = np.array(prev_J_scale + [J_scale])
        alpha = self.optimal_control_problem.settings.scaling_update_weight
        weights = np.array(
            [alpha * (1 - alpha)**i for i, _ in enumerate(J_scales)])
        weights = np.flip(weights)
        weights[0] /= alpha
        self.J_scale = np.average(J_scales, weights=weights)
        self._w = self.J_scale
        self._w = 1

        def set_scales_shifts(var_slice, N=None):
            var = self.iteration.guess_x[var_slice]
            if len(var) == 0:
                pass
            elif N:
                var = var.reshape(N, -1)
                var_min = np.min(var, axis=0)
                var_max = np.max(var, axis=0)
                var_amp = var_max - var_min
                self.x_scales[var_slice] = 1 / var_amp
                self.x_shifts[var_slice] = 0.5 - var_max / var_amp
            else:
                self.x_scales[var_slice] = np.abs(var)

        self.x_scales = self.base_scaling.x_scales
        self.x_shifts = self.base_scaling.x_shifts
        for N, y_slice, u_slice, q_slice, t_slice in zip(self.iteration.mesh.N, self.iteration.y_slices, self.iteration.u_slices, self.iteration.q_slices, self.iteration.t_slices):
            set_scales_shifts(y_slice, N)
            set_scales_shifts(u_slice, N)
            set_scales_shifts(q_slice)
            set_scales_shifts(t_slice)
        set_scales_shifts(self.iteration.s_slice)
        prev_x_scales = np.array([mesh_iter.scaling._x_scales_unexpanded
                                  for mesh_iter in self.optimal_control_problem._backend.mesh_iterations[:-1]])
        prev_x_shifts = np.array([mesh_iter.scaling._x_shifts_unexpanded
                                  for mesh_iter in self.optimal_control_problem._backend.mesh_iterations[:-1]])
        x_scales_all_iters = np.vstack([prev_x_scales, self.x_scales])
        x_shifts_all_iters = np.vstack([prev_x_shifts, self.x_shifts])
        self._x_scales_unexpanded = np.average(
            x_scales_all_iters, axis=0, weights=weights)
        self._x_shifts_unexpanded = np.average(
            x_shifts_all_iters, axis=0, weights=weights)
        self.x_scales = self._expand_x_to_mesh(self._x_scales_unexpanded)
        self.x_shifts = self._expand_x_to_mesh(self._x_shifts_unexpanded)

        self.c_scales = self._calculate_constraint_scaling(
            self.iteration.guess_x)
        self._sW = sparse.diags(self.c_scales)
        self._sW = sparse.diags(np.ones_like(self.c_scales))

        # self._sV = sparse.diags(self.x_scales)
        # self._sV_inv = sparse.diags(self.x_scales_inv)
        # self._sV_sqrd_inv = sparse.diags(self.x_scales_inv**2)

    def _calculate_objective_scaling(self, x_guess):
        g = self.iteration._gradient_lambda(x_guess)
        g_norm = np.sqrt(np.sum(g**2))
        obj_scaling = 1 / g_norm
        # obj_scaling = 1
        return obj_scaling

    def _calculate_constraint_scaling(self, x_guess):
        G = self.iteration._jacobian_lambda(x_guess)
        sG = sparse.coo_matrix(
            (G, self.iteration._jacobian_structure_lambda()), shape=self.iteration._G_shape)
        G_norm = np.squeeze(np.sqrt(np.array(sG.power(2).sum(axis=1))))
        ocp_c_scales = np.empty(self.backend.num_c)
        zip_args = zip(
            self.backend.phase_y_vars_slices,
            self.backend.phase_q_vars_slices,
            self.backend.phase_defect_constraint_slices,
            self.backend.phase_path_constraint_slices,
            self.backend.phase_integral_constraint_slices,
            self.iteration.c_defect_slices,
            self.iteration.c_path_slices,
            self.iteration.c_integral_slices,
            self.backend.p,
            self.iteration.mesh.num_c_defect_per_y,
            self.iteration.mesh.N)
        for ocp_y_slice, ocp_q_slice, ocp_defect_slice, ocp_path_slice, ocp_integral_slice, defect_slice, path_slice, integral_slice, p, n_defect, N in zip_args:
            ocp_c_scales[ocp_defect_slice] = self.x_scales_ocp[ocp_y_slice]
            ocp_c_scales[ocp_path_slice] = np.reciprocal(
                np.mean(G_norm[path_slice].reshape(p.num_c_path, N), axis=1))
            ocp_c_scales[ocp_integral_slice] = self.x_scales_ocp[ocp_q_slice]
        ocp_c_scales[self.backend.c_endpoint_slice] = np.reciprocal(
            G_norm[self.iteration.c_endpoint_slice])
        c_scales = self._expand_c_to_mesh(ocp_c_scales)
        return c_scales

    def _generate_random_sample_variables(self):
        raise NotImplementedError
