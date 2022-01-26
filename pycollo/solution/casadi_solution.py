import numpy as np

from .solution_abc import PhaseSolutionData, SolutionABC

class CasadiSolution(SolutionABC):

    def __init__(self, iteration, nlp_result):
        super().__init__(iteration, nlp_result)

    def backend_specific_init(self):
        self.J = float(self.nlp_result.solution["f"])
        self.x = np.array(self.nlp_result.solution["x"]).squeeze()

    def extract_full_solution(self):
        self.objective = self.it.scaling.unscale_J(self.J)
        x = self.it.scaling.unscale_x(self.x)
        self.phase_data = []
        self.phase_data = tuple(self.extract_full_solution_one_phase(p, x)
                                for p in self.backend.p)
        self._y = tuple(p.y for p in self.phase_data)
        self._dy = tuple(p.dy for p in self.phase_data)
        self._u = tuple(p.u for p in self.phase_data)
        self._q = tuple(p.q for p in self.phase_data)
        self._t = tuple(p.t for p in self.phase_data)
        self._t0 = tuple(p.t0 for p in self.phase_data)
        self._tF = tuple(p.tF for p in self.phase_data)
        self._T = tuple(p.T for p in self.phase_data)
        self._stretch = tuple(p.stretch for p in self.phase_data)
        self._shift = tuple(p.shift for p in self.phase_data)
        self._time_ = tuple(p.time for p in self.phase_data)
        self._s = x[self.it.s_slice]

    def set_user_attributes(self):
        self.state = self._y
        self.control = self._u
        self.integral = self._q
        self.time = self._t
        self.parameter = self._s
        self.initial_time = self._t0
        self.final_time = self._tF

    def extract_full_solution_one_phase(self, p, x):

        def extract_y(p, x):
            if p.num_y_var:
                return x[self.it.y_slices[p.i]].reshape(p.num_y_var, -1)
            return np.array([], dtype=float)

        def extract_dy(p, dy):
            if p.num_y_var:
                dy = np.array(dy[self.it.dy_slices[p.i]])
                return dy.reshape((-1, self.it.mesh.N[p.i]))
            return np.array([], dtype=float)

        def extract_u(p, x):
            if p.num_u_var:
                return x[self.it.u_slices[p.i]].reshape(p.num_u_var, -1)
            return np.array([], dtype=float)

        def extract_t0(p, t):
            if p.ocp_phase.bounds._t_needed[0]:
                return t[0]
            return self.it.guess_time[p.i][0]

        def extract_tF(p, t):
            if p.ocp_phase.bounds._t_needed[1]:
                return t[-1]
            return self.it.guess_time[p.i][-1]

        dy = self.backend.dy_iter_callable(self.x)
        tau = self.tau[p.i]
        y = extract_y(p, x)
        dy = extract_dy(p, dy)
        u = extract_u(p, x)
        q = x[self.it.q_slices[p.i]]
        t = x[self.it.t_slices[p.i]]
        t0 = extract_t0(p, t)
        tF = extract_tF(p, t)
        T = tF - t0
        stretch = T / 2
        shift = (t0 + tF) / 2
        time = tau * stretch + shift

        return PhaseSolutionData(tau, y, dy, u, q, t, t0, tF, T, stretch,
                                 shift, time)
