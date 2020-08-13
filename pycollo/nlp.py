import abc
import itertools

import numpy as np

try:
    import ipopt
except ModuleNotFoundError:
    pass


IPOPT = "ipopt"
WORHP = "worhp"
SNOPT = "snopt"
KNITRO = "knitro"
MUMPS = "mumps"
MA57 = "ma57"


class NLPSettings:

    IPOPT = IPOPT
    WORHP = WORHP
    SNOPT = SNOPT
    KNITRO = KNITRO
    MUMPS = MUMPS
    MA57 = MA57
    NLP_SOLVERS = {IPOPT: True, WORHP: False, SNOPT: False, KNITRO: False}
    LINEAR_SOLVERS = {IPOPT: {MUMPS, MA57}}
    NLP_SOLVER_DEFAULT = IPOPT
    LINEAR_SOLVER_DEFAULT = MUMPS


class NLPProblem(abc.ABC):
    pass


class IPOPTProblem(NLPProblem):

    def __init__(self, iteration):  # J, g, c, G, G_struct, H, H_struct):
        self.iteration = iteration
        self.set_objective()
        self.set_gradient()
        self.set_constraint()
        self.set_jacobian()
        self.set_hessian()
        self.obj_func_eval_counter = itertools.count()

    def set_objective(self):
        self.objective = self.iteration._objective_lambda

    def set_gradient(self):
        self.gradient = self.iteration._gradient_lambda

    def set_constraint(self):
        self.constraints = self.iteration._constraint_lambda

    def set_jacobian(self):
        self.jacobian = self.iteration._jacobian_lambda
        self.jacobianstructure = self.iteration._jacobian_structure_lambda

    def set_hessian(self):
        if self.iteration.backend.ocp.settings.derivative_level == 2:
            self.hessian = self.iteration._hessian_lambda
            self.hessianstructure = self.iteration._hessian_structure_lambda

    def intermediate(self, alg_mod,
                     iter_count,
                     obj_value,
                     inf_pr,
                     inf_du,
                     mu,
                     d_norm,
                     regularization_size,
                     alpha_du,
                     alpha_pr,
                     ls_trials):
        self.max_iter_count = iter_count


nlp_problem_dispatcher = {
    IPOPT: IPOPTProblem,
}


def initialise_nlp_backend(iteration):
    nlp_solver = iteration.backend.ocp.settings.nlp_solver
    nlp_problem = nlp_problem_dispatcher.get(nlp_solver)(iteration)
    nlp_backend = ipopt.problem(
        n=iteration.num_x,
        m=iteration.num_c,
        problem_obj=nlp_problem,
        lb=iteration._x_bnd_l,
        ub=iteration._x_bnd_u,
        cl=iteration._c_bnd_l,
        cu=iteration._c_bnd_u)

    settings = iteration.optimal_control_problem.settings
    nlp_backend.addOption('mu_strategy', 'adaptive')
    nlp_backend.addOption('tol', settings.nlp_tolerance)
    nlp_backend.addOption('max_iter', settings.max_nlp_iterations)
    nlp_backend.addOption('print_level', 5)
    nlp_backend.addOption('nlp_scaling_method', 'user-scaling')
    # nlp_backend.addOption('nlp_scaling_method', 'none')

    nlp_backend.setProblemScaling(
        iteration.scaling.J_scale,
        iteration.scaling.x_scales,
        iteration.scaling.c_scales,
    )

    # print(iteration.scaling.J_scale)
    # print(iteration.scaling.x_scales)
    # print(iteration.scaling.c_scales)
    # input()

    return nlp_backend, nlp_problem
