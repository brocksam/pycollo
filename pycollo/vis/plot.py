import matplotlib.pyplot as plt
import numpy as np


Y = "y"
DY = "dy"
U = "u"
POINT_OPTIONS = {"marker": "x",
                 "markersize": 5,
                 "linestyle": "",
                 "color": "black"}
LINE_OPTIONS = {"linewidth": 2}
MESH_OPTIONS = {"align": "edge",
                "alpha": 0.5}
X_TITLES = {"y": "States", "dy": "State Derivatives", "u": "Controls"}


def plot_solution(solution, *, iterpolated=True, plot_y=True, plot_dy=True,
                  plot_u=True, render=True):
    if plot_y:
        t_data_phases, y_datas_phases = interpolate_x_solution(solution, Y)
        render_x_solution(solution, Y, t_data_phases, y_datas_phases)
    if plot_dy:
        t_data_phases, dy_datas_phases = interpolate_x_solution(solution, DY)
        render_x_solution(solution, DY, t_data_phases, dy_datas_phases)
    if plot_u:
        t_data_phases, u_datas_phases = interpolate_x_solution(solution, U)
        render_x_solution(solution, U, t_data_phases, u_datas_phases)
    plot_buffered_renders(render)


def interpolate_x_solution(solution, x):
    t_data_phases = []
    x_datas_phases = []
    zipped = zip(solution.phase_polys,
                 solution.phase_data,
                 solution.it.mesh.mesh_index_boundaries)
    for phase_polys, p_data, mesh_index_boundaries in zipped:
        p_data_tau_lower = p_data.tau[mesh_index_boundaries[:-1]]
        p_data_tau_upper = p_data.tau[mesh_index_boundaries[1:]]
        t_start_stops = list(zip(p_data_tau_lower, p_data_tau_upper))
        t_data = []
        x_datas = []
        for i_x, x_polys in enumerate(getattr(phase_polys, x)):
            t_list = []
            x_list = []
            for t_start_stop, x_poly in zip(t_start_stops, x_polys):
                t_linspace = np.linspace(*t_start_stop)[:-1]
                x_linspace = x_poly(t_linspace)
                t_list.extend(t_linspace)
                x_list.extend(x_linspace)
            t_list.append(p_data.tau[-1])
            x_list.append(getattr(p_data, x)[i_x, -1])
            t_data.append(t_list)
            x_datas.append(x_list)
        t_data = np.array(t_data[0]) * p_data.stretch + p_data.shift
        x_datas = np.array(x_datas)
        t_data_phases.append(t_data)
        x_datas_phases.append(x_datas)
    return t_data_phases, x_datas_phases


def render_x_solution(solution, x, t_data_phases, x_datas_phases):
    plt.figure()
    zipped = zip(solution.backend.p,
                 solution.phase_data,
                 t_data_phases,
                 x_datas_phases)
    for p, p_data, t_data, x_datas in zipped:
        for i_x, x_data in enumerate(x_datas):
            plt.plot(p_data.time, getattr(p_data, x)[i_x], **POINT_OPTIONS)
            label = str(getattr(p, f"{x[-1]}_var")[i_x])[1:]
            plt.plot(t_data, x_data, **LINE_OPTIONS, label=label)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.title(X_TITLES[x])
    plt.xlabel("Time / $s$")
    plt.ylabel(f"{X_TITLES[x]} Values")


def plot_mesh(mesh, *, render=True):
    try:
        iter(mesh)
    except (TypeError, ValueError):
        mesh = [mesh]
    for i, _ in enumerate(mesh[0].p):
        plot_phase_mesh(mesh, i)
    plot_buffered_renders(render)


def plot_phase_mesh(meshes, i):
    plt.figure()
    for mesh in meshes:
        xs = mesh.tau[i][mesh.mesh_index_boundaries[i][:-1]]
        widths = mesh.h_K[i]
        heights = mesh.N_K[i] / mesh.h_K[i]
        plt.bar(x=xs, height=heights, width=widths, **MESH_OPTIONS)
    plt.grid(True)
    plt.title("Phase Mesh Density")
    plt.xlabel("Tau")
    plt.ylabel("Node Density / $node$")


def plot_buffered_renders(render):
    if render:
        plt.show()
