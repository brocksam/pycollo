import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

from .mesh import (PhaseMesh, Mesh)


phase_solution_data_fields = ("tau", "y", "dy", "u", "q", "t", "t0", "tF", 
	"T", "stretch", "shift", "time")
PhaseSolutionData = collections.namedtuple("PhaseSolutionData", 
	phase_solution_data_fields)


class Solution:

	def __init__(self, iteration, nlp_solution, nlp_solution_info):
		self._it = iteration
		self._ocp = iteration.optimal_control_problem
		self._backend = iteration.optimal_control_problem._backend
		self._tau = iteration.mesh.tau
		
		self._nlp_solution = nlp_solution
		self._nlp_solution_info = nlp_solution_info
		self._J = self._nlp_solution_info['obj_val']
		self._x = np.array(nlp_solution)
		if self._ocp._settings._nlp_solver == 'ipopt':
			self._process_solution = self._process_ipopt_solution
		else:
			raise NotImplementedError
		self._process_solution()

	@property
	def objective(self):
		return self._J
	
	@property
	def initial_time(self):
		return self._t0

	@property
	def final_time(self):
		return self._tF
	
	@property
	def state(self):
		return self._y

	@property
	def control(self):
		return self._u
	
	@property
	def integral(self):
		return self._q

	@property
	def time(self):
		return self._time
	
	@property
	def parameter(self):
		return self._s

	def _process_ipopt_solution(self):
		self._phase_data = []
		for tau, time_guess, phase, c_continuous_lambda, y_slice, u_slice, q_slice, t_slice, dy_slice, N in zip(self._tau, self._it.guess_time, self._backend.p, self._backend.compiled_functions.c_continuous_lambdas, self._it.y_slices, self._it.u_slices, self._it.q_slices, self._it.t_slices, self._it.c_lambda_dy_slices, self._it.mesh.N):
			y = self._x[y_slice].reshape(phase.num_y_vars, -1) if phase.num_y_vars else np.array([], dtype=float)
			dy = c_continuous_lambda(*self._it._reshape_x(self._x), N)[dy_slice].reshape((-1, N)) if phase.num_y_vars else np.array([], dtype=float)
			u = self._x[u_slice].reshape(phase.num_u_vars, -1) if phase.num_u_vars else np.array([], dtype=float)
			q = self._x[q_slice]
			t = self._x[t_slice]
			
			t0 = t[0] if phase.ocp_phase.bounds._t_needed[0] else time_guess[0]
			tF = t[-1] if phase.ocp_phase.bounds._t_needed[1] else time_guess[-1]
			T = tF - t0
			stretch = T/2
			shift = (t0 + tF) / 2
			time = tau*stretch + shift
			phase_data = PhaseSolutionData(tau=tau, y=y, dy=dy, u=u, q=q, t=t, t0=t0, tF=tF, T=T, stretch=stretch, shift=shift, time=time)
			self._phase_data.append(phase_data)
		self._phase_data = tuple(self._phase_data)
		self._y = tuple(p.y for p in self._phase_data)
		self._dy = tuple(p.dy for p in self._phase_data)
		self._u = tuple(p.u for p in self._phase_data)
		self._q = tuple(p.q for p in self._phase_data)
		self._t = tuple(p.t for p in self._phase_data)
		self._t0 = tuple(p.t0 for p in self._phase_data)
		self._tF = tuple(p.tF for p in self._phase_data)
		self._T = tuple(p.T for p in self._phase_data)
		self._stretch = tuple(p.stretch for p in self._phase_data)
		self._shift = tuple(p.shift for p in self._phase_data)
		self._time = tuple(p.time for p in self._phase_data)
		self._s = self._x[self._it.s_slice]
		self._interpolate_solution()

	def _process_snopt_solution(self, solution):
		raise NotImplementedError

	def _interpolate_solution(self):

		self._phase_y_polys = []
		self._phase_dy_polys = []
		self._phase_u_polys = []
		for p, p_data, K, N_K, mesh_index_boundaries in zip(self._backend.p, self._phase_data, self._it.mesh.K, self._it.mesh.N_K, self._it.mesh.mesh_index_boundaries):

			y_polys = np.empty((p.num_y_vars, K), dtype=object)
			dy_polys = np.empty((p.num_y_vars, K), dtype=object)
			u_polys = np.empty((p.num_u_vars, K), dtype=object)

			for i_y, state_deriv in enumerate(p_data.dy):
				for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
					t_k = p_data.tau[i_start:i_stop+1]
					dy_k = state_deriv[i_start:i_stop+1]
					dy_poly = np.polynomial.Polynomial.fit(t_k, dy_k, deg=N_K[i_k]-1, window=[0, 1])
					scale_factor = self._it.mesh._PERIOD/p_data.T
					y_poly = dy_poly.integ(k=scale_factor*p_data.y[i_y, i_start])
					y_poly = np.polynomial.Polynomial(coef=y_poly.coef/scale_factor, window=y_poly.window, domain=y_poly.domain)
					y_polys[i_y, i_k] = y_poly
					dy_polys[i_y, i_k] = dy_poly

			for i_u, control in enumerate(p_data.u):
				for i_k, (i_start, i_stop) in enumerate(zip(mesh_index_boundaries[:-1], mesh_index_boundaries[1:])):
					t_k = p_data.tau[i_start:i_stop+1]
					u_k = control[i_start:i_stop+1]
					u_poly = np.polynomial.Polynomial.fit(t_k, u_k, deg=N_K[i_k]-1, window=[0, 1])
					u_polys[i_u, i_k] = u_poly

			self._phase_y_polys.append(y_polys)
			self._phase_dy_polys.append(dy_polys)
			self._phase_u_polys.append(u_polys)

	def _plot_interpolated_solution(self, plot_y=False, plot_dy=False, plot_u=False):

		t_data_phases = []
		y_datas_phases = []
		dy_datas_phases = []
		u_datas_phases = []

		for y_polys, dy_polys, u_polys, p_data, mesh_index_boundaries in zip(self._phase_y_polys, self._phase_dy_polys, self._phase_u_polys, self._phase_data, self._it.mesh.mesh_index_boundaries):

			t_start_stops = list(zip(p_data.tau[mesh_index_boundaries[:-1]], p_data.tau[mesh_index_boundaries[1:]]))

			t_data = []
			y_datas = []
			dy_datas = []
			u_datas = []

			for i_y, state in enumerate(y_polys):
				t_list = []
				y_list = []
				for t_start_stop, y_poly in zip(t_start_stops, state):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					y_linspace = y_poly(t_linspace)
					t_list.extend(t_linspace)
					y_list.extend(y_linspace)
				t_list.append(p_data.tau[-1])
				y_list.append(p_data.y[i_y, -1])
				t_data.append(t_list)
				y_datas.append(y_list)

			for i_dy, dstate in enumerate(dy_polys):
				t_list = []
				dy_list = []
				for t_start_stop, dy_poly in zip(t_start_stops, dstate):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					dy_linspace = dy_poly(t_linspace)
					t_list.extend(t_linspace)
					dy_list.extend(dy_linspace)
				t_list.append(p_data.tau[-1])
				dy_list.append(p_data.dy[i_dy, -1])
				t_data.append(t_list)
				dy_datas.append(dy_list)

			for i_u, control in enumerate(u_polys):
				t_list = []
				u_list = []
				for t_start_stop, u_poly in zip(t_start_stops, control):
					t_linspace = np.linspace(*t_start_stop)[:-1]
					u_linspace = u_poly(t_linspace)
					t_list.extend(t_linspace)
					u_list.extend(u_linspace)
				t_list.append(p_data.tau[-1])
				u_list.append(p_data.u[i_u, -1])
				t_data.append(t_list)
				u_datas.append(u_list)

			t_data = np.array(t_data[0])*p_data.stretch + p_data.shift
			y_datas = np.array(y_datas)
			dy_datas = np.array(dy_datas)
			u_datas = np.array(u_datas)

			t_data_phases.append(t_data)
			y_datas_phases.append(y_datas)
			dy_datas_phases.append(dy_datas)
			u_datas_phases.append(u_datas)

		if plot_y:
			for p, p_data, t_data, y_datas in zip(self._backend.p, self._phase_data, t_data_phases, y_datas_phases):
				for i_y, y_data in enumerate(y_datas):
					plt.plot(p_data.time, p_data.y[i_y], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, y_data, linewidth=2, label=str(p.y_vars[i_y])[1:])
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('States')
			plt.xlabel('Time / $s$')
			plt.show()

		if plot_dy:
			for p, p_data, t_data, dy_datas in zip(self._backend.p, self._phase_data, t_data_phases, dy_datas_phases):
				for i_y, dy_data in enumerate(dy_datas):
					plt.plot(p_data.time, p_data.dy[i_y], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, dy_data, linewidth=2, label=str(p.y_vars[i_y])[1:])	
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('State Derivatives')
			plt.xlabel('Time / $s$')
			plt.show()

		if plot_u:
			for p, p_data, t_data, u_datas in zip(self._backend.p, self._phase_data, t_data_phases, u_datas_phases):
				for i_u, u_data in enumerate(u_datas):
					plt.plot(p_data.time, p_data.u[i_u], marker='x', markersize=5, linestyle='', color='black')
					plt.plot(t_data, u_data, linewidth=2, label=str(p.u_vars[i_u])[1:])
			plt.legend(loc='upper left')
			plt.grid(True)
			plt.title('Controls')
			plt.xlabel('Time / $s$')
			plt.show()

		# plt.plot(self._y[0], self._y[1], marker='x', markersize=5, linestyle='', color='black')
		# plt.plot(y_datas[0], y_datas[1], linewidth=2)
		# plt.grid(True)
		# plt.gca().set_aspect('equal', adjustable='box')
		# plt.title('Path')
		# plt.xlabel('x / $m$')
		# plt.ylabel('y / $m$')
		# plt.show()

		

		if False:

			with open('result_tradjectory.csv', mode='w') as result_tradjectory_file:

			    result_tradjectory_writer = csv.writer(result_tradjectory_file, delimiter=',')

			    result_tradjectory_writer.writerow(t_data)

			    for i_y, y_data in enumerate(y_datas):
			    	result_tradjectory_writer.writerow(y_data)

			    for i_y, dy_data in enumerate(dy_datas):
			    	result_tradjectory_writer.writerow(dy_data)

			    for i_u, u_data in enumerate(u_datas):
			    	result_tradjectory_writer.writerow(u_data)

		if False:
			fig, ax1 = plt.subplots()
			plt.grid(True, axis='both')

			color = 'tab:blue'
			ax1.set_xlabel('Time / $s$')
			ax1.set_ylabel('Angle / $rad$', color=color)
			ax1.plot(self._time, self._y[0], marker='x', markersize=7, linestyle='', color='black')
			ax1.plot(t_data, y_datas[0], color=color, linewidth=2)
			ax1.tick_params(axis='y', labelcolor=color)
			plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi], ['$-\\pi/2$', '$-\\pi/4$', '0', '$\\pi/4$', '$\\pi/2$'])

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

			color = 'tab:orange'
			ax2.set_ylabel('Angular Velocity / $rad \\cdot s^{-1}$', color=color)  # we already handled the x-label with ax1
			ax2.plot(self._time, self._dy[0], marker='x', markersize=7, linestyle='', color='black')
			ax2.plot(t_data, dy_datas[0], color=color, linewidth=2)
			ax2.tick_params(axis='y', labelcolor=color)
			plt.yticks([-10, -5, 0, 5, 10])

			fig.tight_layout()  # otherwise the right y-label is slightly clipped

			plt.show()



			fig, ax1 = plt.subplots()
			plt.grid(True, axis='both')

			color = 'tab:green'
			ax1.set_xlabel('Time / $s$')
			ax1.set_ylabel('Torque / $N\\cdot m$', color=color)
			plt.plot(self._time, self._u[0]*15, marker='x', markersize=7, linestyle='', color='black')
			ax1.plot(t_data, u_datas[0]*15, color=color, linewidth=2)
			ax1.tick_params(axis='y', labelcolor=color)
			plt.yticks([-2, -1, 0, 1, 2])

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			plt.show()


	def _patterson_rao_discretisation_mesh_error(self):

		phase_ph_meshes = []
		for p in self._backend.p:
			phase_ph_mesh = PhaseMesh(phase=p.ocp_phase,
				number_mesh_sections=self._it.mesh.K[p.i],
				mesh_section_sizes=self._it.mesh.h_K[p.i],
				number_mesh_section_nodes=(self._it.mesh.N_K[p.i]+1))
			phase_ph_meshes.append(phase_ph_mesh)
		self._ph_mesh = Mesh(
			self._backend, phase_ph_meshes)

		y_tildes = []
		u_tildes = []
		x_tilde_full = []
		for p, p_data in zip(self._backend.p, self._phase_data):
			y_tilde, u_tilde = self._get_y_u_tilde(p, p_data)
			y_tildes.append(y_tilde)
			u_tildes.append(u_tilde)
			x_tilde_full = x_tilde_full + [y for y in y_tilde] + [u for u in u_tilde] + [q for q in p_data.q] + [t for t in p_data.t]

		self._absolute_mesh_errors = []
		self._relative_mesh_errors = []
		self._maximum_relative_mesh_errors = []

		for p, p_data, y_tilde, u_tilde in zip(self._backend.p, self._phase_data, y_tildes, u_tildes):
			self._patterson_rao_discretisation_mesh_error_single_phase(p, p_data, y_tilde, u_tilde, x_tilde_full)

	def _get_y_u_tilde(self, p, p_data):
	
		def eval_polynomials(polys, mesh, vals):
			sec_bnd_inds = mesh.mesh_index_boundaries[p.i]
			for i_var, poly_row in enumerate(polys):
				for i_k, (poly, i_start, i_stop) in enumerate(zip(poly_row, sec_bnd_inds[:-1], sec_bnd_inds[1:])):
					sec_slice = slice(i_start+1, i_stop)
					vals[i_var, sec_slice] = poly(mesh.tau[p.i][sec_slice])
			return vals

		y_tilde = np.zeros((p.num_y_vars, self._ph_mesh.N[p.i]))
		y_tilde[:, self._ph_mesh.mesh_index_boundaries[p.i]] = p_data.y[:, self._it.mesh.mesh_index_boundaries[p.i]]
		y_tilde = eval_polynomials(self._phase_y_polys[p.i], self._ph_mesh, y_tilde)

		if p.num_u_vars:
			u_tilde = np.zeros((p.num_u_vars, self._ph_mesh.N[p.i]))
			u_tilde[:, self._ph_mesh.mesh_index_boundaries[p.i]] = p_data.u[:, self._it.mesh.mesh_index_boundaries[p.i]]
			u_tilde = eval_polynomials(self._phase_u_polys[p.i], self._ph_mesh, u_tilde)
		else:
			u_tilde = np.array([])

		return y_tilde, u_tilde

	def _patterson_rao_discretisation_mesh_error_single_phase(self, p, p_data, y_tilde, u_tilde, x_tilde_full):

		dy_tilde = self._backend.compiled_functions.c_continuous_lambdas[p.i](*x_tilde_full, *self._s, self._ph_mesh.N[p.i])[:p.num_y_vars*self._ph_mesh.N[p.i]].reshape(-1, self._ph_mesh.N[p.i])

		A_dy_tilde = p_data.stretch*self._ph_mesh.sA_matrix[p.i].dot(dy_tilde.T)

		mesh_error = np.zeros((self._it.mesh.K[p.i], p.num_y_vars, max(self._ph_mesh.N_K[p.i])-1))
		rel_error_scale_factor = np.zeros((self._it.mesh.K[p.i], p.num_y_vars))

		for i_k, (i_start, m_k) in enumerate(zip(self._ph_mesh.mesh_index_boundaries[p.i][:-1], self._ph_mesh.N_K[p.i]-1)):
			y_k = y_tilde[:, i_start]
			Y_tilde_k = (y_k + A_dy_tilde[i_start:i_start+m_k]).T
			Y_k = y_tilde[:, i_start+1:i_start+1+m_k]
			mesh_error_k = Y_tilde_k - Y_k
			mesh_error[i_k, :, :m_k] = mesh_error_k
			rel_error_scale_factor_k = np.max(np.abs(Y_k), axis=1) + 1
			rel_error_scale_factor[i_k, :] = rel_error_scale_factor_k

		absolute_mesh_error = np.abs(mesh_error)
		self._absolute_mesh_errors.append(absolute_mesh_error)

		relative_mesh_error = np.zeros_like(absolute_mesh_error)
		for i_k in range(self._ph_mesh.K[p.i]):
			for i_y in range(p.num_y_vars):
				for i_m in range(self._ph_mesh.N_K[p.i][i_k]-1):
					val = absolute_mesh_error[i_k, i_y, i_m] / (1 + rel_error_scale_factor[i_k, i_y])
					relative_mesh_error[i_k, i_y, i_m] = val

		self._relative_mesh_errors.append(relative_mesh_error)

		max_relative_error = np.zeros(self._it.mesh.K[p.i])
		for i_k in range(self._ph_mesh.K[p.i]):
			max_relative_error[i_k] = np.max(relative_mesh_error[i_k, :, :])

		self._maximum_relative_mesh_errors.append(max_relative_error)

	def _patterson_rao_next_iteration_mesh(self):

		phase_meshes = []
		for p in self._backend.p:
			phase_mesh = self._patterson_rao_next_iteration_phase_mesh(p)
			phase_meshes.append(phase_mesh)

		new_mesh = Mesh(self._backend, phase_meshes)

		return new_mesh 

	def _patterson_rao_next_iteration_phase_mesh(self, p):

		def merge_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, merge_group):
			merge_group = np.array(merge_group)
			P_q = merge_group[:, 0]
			h_q = merge_group[:, 1]
			p_q = merge_group[:, 2]

			N = np.sum(p_q)
			T = np.sum(h_q)

			merge_ratio = p_q / (self._ocp.settings.collocation_points_min - P_q)
			mesh_secs_needed = np.ceil(np.sum(merge_ratio)).astype(np.int)
			if mesh_secs_needed == 1:
				new_mesh_secs = np.array([T])
			else:
				required_reduction = np.divide(h_q, merge_ratio)
				weighting_factor = np.reciprocal(np.sum(required_reduction))
				reduction_factor = weighting_factor * required_reduction
				knot_locations = np.cumsum(h_q) / T
				current_density = np.cumsum(reduction_factor)

				density_function = interpolate.interp1d(knot_locations, current_density, bounds_error=False, fill_value='extrapolate')
				new_density = np.linspace(1/mesh_secs_needed, 1, mesh_secs_needed)
				new_knots = density_function(new_density)

				new_mesh_secs = T * np.diff(np.concatenate([np.array([0]), new_knots]))

			new_mesh_sec_sizes.extend(new_mesh_secs.tolist())
			new_num_mesh_sec_nodes.extend([self._ocp.settings.collocation_points_min]*mesh_secs_needed)

			return new_mesh_sec_sizes, new_num_mesh_sec_nodes

		def subdivide_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, subdivide_group, reduction_tolerance):
			subdivide_group = np.array(subdivide_group)
			subdivide_required = subdivide_group[:, 0].astype(np.bool)
			subdivide_factor = subdivide_group[:, 1].astype(np.int)
			P_q = subdivide_group[:, 2]
			h_q = subdivide_group[:, 3]
			p_q = subdivide_group[:, 4]

			is_node_reduction = P_q <= 0

			predicted_nodes = P_q + p_q
			predicted_nodes[is_node_reduction] = np.ceil(P_q[is_node_reduction] * reduction_tolerance) + p_q[is_node_reduction]

			next_mesh_nodes = np.ones_like(predicted_nodes, dtype=np.int) * self._ocp.settings.collocation_points_min
			next_mesh_nodes[np.invert(subdivide_required)] = predicted_nodes[np.invert(subdivide_required)]
			next_mesh_nodes_lower_than_min = next_mesh_nodes < self._ocp.settings.collocation_points_min
			next_mesh_nodes[next_mesh_nodes_lower_than_min] = self._ocp.settings.collocation_points_min

			for h, k, n in zip(h_q, subdivide_factor, next_mesh_nodes):
				new_mesh_sec_sizes.extend([h/k]*k)
				new_num_mesh_sec_nodes.extend([n]*k)

			return new_mesh_sec_sizes, new_num_mesh_sec_nodes

		if np.max(self._maximum_relative_mesh_errors[p.i]) > self._ocp.settings.mesh_tolerance:

			P_q = np.ceil(np.divide(np.log(self._maximum_relative_mesh_errors[p.i]/self._ocp.settings.mesh_tolerance), np.log(self._it.mesh.N_K[p.i])))
			P_q_zero = P_q == 0
			P_q[P_q_zero] = 1
			predicted_nodes = P_q + self._it.mesh.N_K[p.i]

			log_tolerance = np.log(self._ocp.settings.mesh_tolerance / np.max(self._maximum_relative_mesh_errors[p.i]))
			merge_tolerance = 50 / log_tolerance
			merge_required = predicted_nodes < merge_tolerance

			reduction_tolerance = 1 - (-1 / log_tolerance)
			if reduction_tolerance < 0:
				reduction_tolerance = 0

			subdivide_required = predicted_nodes >= self._ocp.settings.collocation_points_max
			subdivide_level = np.ones_like(predicted_nodes)
			subdivide_level[subdivide_required] = np.ceil(predicted_nodes[subdivide_required] / self._ocp.settings.collocation_points_min)

			merge_group = []
			subdivide_group = []
			new_mesh_sec_sizes = []
			new_num_mesh_sec_nodes = []
			for need_merge, need_subdivide, subdivide_factor, P, h, N_k in zip(merge_required, subdivide_required, subdivide_level, P_q, self._it.mesh.h_K[p.i], self._it.mesh.N_K[p.i]):
				if need_merge:
					if subdivide_group != []:
						new_mesh_sec_sizes, new_num_mesh_sec_nodes = subdivide_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, subdivide_group, reduction_tolerance)
						subdivide_group = []
					merge_group.append([P, h, p])
				else:
					if merge_group != []:
						new_mesh_sec_sizes, new_num_mesh_sec_nodes = merge_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, merge_group)
						merge_group = []
					subdivide_group.append([need_subdivide, subdivide_factor, P, h, N_k])
			else:
				if merge_group != []:
					new_mesh_sec_sizes, new_num_mesh_sec_nodes = merge_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, merge_group)
				elif subdivide_group != []:
					new_mesh_sec_sizes, new_num_mesh_sec_nodes = subdivide_sections(new_mesh_sec_sizes, new_num_mesh_sec_nodes, subdivide_group, reduction_tolerance)
			new_number_mesh_secs = len(new_mesh_sec_sizes)
			new_mesh = PhaseMesh(phase=p.ocp_phase,
				number_mesh_sections=new_number_mesh_secs,
				mesh_section_sizes=new_mesh_sec_sizes,
				number_mesh_section_nodes=new_num_mesh_sec_nodes)
			return new_mesh
		else:
			return p.ocp_phase.mesh