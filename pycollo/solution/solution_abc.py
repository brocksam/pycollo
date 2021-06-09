import collections
from abc import ABC, abstractmethod

import numpy as np
from pyproprop import processed_property

from ..mesh_refinement import MESH_REFINEMENT_ALGORITHMS
from ..vis.plot import plot_solution


nlp_result_fields = ("solution", "info", "solve_time")
NlpResult = collections.namedtuple("NlpResult", nlp_result_fields)
phase_solution_data_fields = ("tau", "y", "dy", "u", "q", "t", "t0", "tF",
                              "T", "stretch", "shift", "time")
PhaseSolutionData = collections.namedtuple("PhaseSolutionData",
                                           phase_solution_data_fields)
Polys = collections.namedtuple("Polys", ("y", "dy", "u"))


class SolutionABC(ABC):

    objective = processed_property("objective", read_only=True)
    initial_time = processed_property("initial_time", read_only=True)
    final_time = processed_property("final_time", read_only=True)
    state = processed_property("state", read_only=True)
    state_derivative = processed_property("state_derivative", read_only=True)
    control = processed_property("control", read_only=True)
    integral = processed_property("integral", read_only=True)
    time = processed_property("time", read_only=True)
    parameter = processed_property("parameter", read_only=True)

    def __init__(self, iteration, nlp_result):
        self.it = iteration
        self.ocp = iteration.optimal_control_problem
        self.backend = iteration.backend
        self.tau = iteration.mesh.tau
        self.nlp_result = nlp_result
        self.backend_specific_init()
        self.process_solution()

    @abstractmethod
    def backend_specific_init(self):
        pass

    def process_solution(self):
        self.extract_full_solution()
        self.set_user_attributes()
        if self.ocp.settings.quadrature_method == "lobatto":
            self.interpolate_solution_lobatto()
        elif self.ocp.settings.quadrature_method == "radau":
            self.interpolate_solution_radau()

    @abstractmethod
    def extract_full_solution(self):
        pass

    @abstractmethod
    def set_user_attributes(self):
        pass

    def interpolate_solution_lobatto(self):
        self.phase_polys = []
        zipped = zip(self.backend.p,
                     self.phase_data,
                     self.it.mesh.K,
                     self.it.mesh.N_K,
                     self.it.mesh.mesh_index_boundaries)
        for p, p_data, K, N_K, mesh_index_boundaries in zipped:
            y_polys = np.empty((p.num_y_var, K), dtype=object)
            dy_polys = np.empty((p.num_y_var, K), dtype=object)
            u_polys = np.empty((p.num_u_var, K), dtype=object)
            for i_y, (state, state_deriv) in enumerate(zip(p_data.y, p_data.dy)):
                for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):

                    t_k = p_data.tau[i_start:i_stop + 1]
                    dy_k = state_deriv[i_start:i_stop + 1]
                    dy_poly = np.polynomial.Legendre.fit(t_k,
                                                         dy_k,
                                                         deg=N_K[i_k] - 1,
                                                         window=[0, 1])
                    dy_polys[i_y, i_k] = dy_poly

                    scale_factor = p_data.T / self.it.mesh._PERIOD
                    y_k = state[i_start:i_stop + 1]
                    dy_k = state_deriv[i_start:i_stop + 1] * scale_factor
                    dy_poly = np.polynomial.Legendre.fit(t_k,
                                                         dy_k,
                                                         deg=N_K[i_k] - 1,
                                                         window=[0, 1])
                    y_poly = dy_poly.integ(k=state[i_start])
                    y_polys[i_y, i_k] = y_poly

                    t_data = np.linspace(t_k[0], t_k[-1])

            for i_u, control in enumerate(p_data.u):
                for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
                    t_k = p_data.tau[i_start:i_stop + 1]
                    u_k = control[i_start:i_stop + 1]
                    u_poly = np.polynomial.Polynomial.fit(
                        t_k, u_k, deg=N_K[i_k] - 1, window=[0, 1])
                    u_polys[i_u, i_k] = u_poly
            phase_polys = Polys(y_polys, dy_polys, u_polys)
            self.phase_polys.append(phase_polys)

    def interpolate_solution_radau(self):
        self.phase_polys = []
        zipped = zip(self.backend.p,
                     self.phase_data,
                     self.it.mesh.K,
                     self.it.mesh.N_K,
                     self.it.mesh.mesh_index_boundaries)
        for p, p_data, K, N_K, mesh_index_boundaries in zipped:
            y_polys = np.empty((p.num_y_var, K), dtype=object)
            dy_polys = np.empty((p.num_y_var, K), dtype=object)
            u_polys = np.empty((p.num_u_var, K), dtype=object)
            for i_y, (state, state_deriv) in enumerate(zip(p_data.y, p_data.dy)):
                for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
                    scale_factor = p_data.T / self.it.mesh._PERIOD
                    t_k = p_data.tau[i_start:i_stop + 1]
                    dy_k = state_deriv[i_start:i_stop + 1] * scale_factor
                    dy_poly = np.polynomial.Legendre.fit(t_k[:-1],
                                                         dy_k[:-1],
                                                         deg=N_K[i_k] - 2,
                                                         domain=[t_k[0], t_k[-1]],
                                                         window=[0, 1])
                    dy_polys[i_y, i_k] = dy_poly
                    y_poly = dy_poly.integ(k=state[i_start])
                    y_polys[i_y, i_k] = y_poly
                    dy_poly = np.polynomial.Legendre.fit(t_k[:-1],
                                                         state_deriv[i_start:i_stop + 1][:-1],
                                                         deg=N_K[i_k] - 2,
                                                         domain=[t_k[0], t_k[-1]],
                                                         window=[0, 1])
                    dy_polys[i_y, i_k] = dy_poly
            for i_u, control in enumerate(p_data.u):
                for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
                    t_k = p_data.tau[i_start:i_stop + 1]
                    u_k = control[i_start:i_stop + 1]
                    u_poly = np.polynomial.Polynomial.fit(
                        t_k, u_k, deg=N_K[i_k] - 1, window=[0, 1])
                    u_polys[i_u, i_k] = u_poly
            phase_polys = Polys(y_polys, dy_polys, u_polys)
            self.phase_polys.append(phase_polys)

    def plot(self):
        plot_solution(self)

    def refine_mesh(self):
        dispatcher = MESH_REFINEMENT_ALGORITHMS.dispatcher
        algorithm = self.ocp.settings.mesh_refinement_algorithm
        self.mesh_refinement = dispatcher[algorithm](self)
        return self.mesh_refinement.next_iter_mesh
