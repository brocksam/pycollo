# interpolate_solution.py

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate

index_bounds = [0, 2, 5, 9, 12, 14]

nK = [3, 4, 5, 4, 3]

t = np.array([0.0, 0.05, 0.1, 0.15527864, 0.24472136, 0.3, 0.36906927, 0.5, 0.63093073, 0.7, 0.75527864, 0.84472136, 0.9, 0.95, 1.0])

y = np.array([[0.0, 0.02287432, 0.09103844, 0.21737672, 0.51998313, 0.75387517, 1.07687657, 1.71179827, 2.28814291, 2.54677563, 2.72674586, 2.95965109, 3.06148441, 3.12044655, 3.14159265], [0.0, 0.91267852, 1.81159255, 2.74099434, 3.95643187, 4.47147978, 4.82579596, 4.73175852, 3.99844958, 3.48236817, 3.02190462, 2.15559883, 1.51264142, 0.82346329, 0.0]])

dy = np.array([[0.0, 0.91267852, 1.81159255, 2.74099434, 3.95643187, 4.47147978, 4.82579596, 4.73175852, 3.99844958, 3.48236817, 3.02190462, 2.15559883, 1.51264142, 0.82346329, 0.0], [18.22285079, 18.2001078, 17.67227079, 15.79749921, 11.09094167, 7.47875908, 2.86511218, -3.73378637, -6.98847419, -7.93355488, -8.76550412, -10.7687229, -12.57891502, -15.05731211, -17.95032147]])

u = np.array([[-18.22285079, -18.42448526, -18.56412479, -17.91321031, -15.96519223, -14.19340048, -11.5026342, -5.9788561, -0.40387781, 2.43645062, 4.81158636, 8.99370705, 11.79389342, 14.84988429, 17.95032147]])

q = np.array([148.12477813])

t_start_stops = list(zip(t[index_bounds[:-1]], t[index_bounds[1:]]))
y_polys = np.empty((2, 5), dtype=object)
dy_polys = np.empty((2, 5), dtype=object)
u_polys = np.empty((1, 5), dtype=object)

for i_y, state_deriv in enumerate(dy):
	for i_poly, (n, i_start, i_stop) in enumerate(zip(nK, index_bounds[:-1], index_bounds[1:])):
		t_k = t[i_start:i_stop+1]
		dy_k = state_deriv[i_start:i_stop+1]
		dy_poly = np.polynomial.Polynomial.fit(t_k, dy_k, deg=n-1, window=[0, 1])
		y_poly = dy_poly.integ(k=y[i_y, i_start])
		y_polys[i_y, i_poly] = y_poly
		dy_polys[i_y, i_poly] = dy_poly

for i_u, control in enumerate(u):
	for i_poly, (n, i_start, i_stop) in enumerate(zip(nK, index_bounds[:-1], index_bounds[1:])):
		t_k = t[i_start:i_stop+1]
		u_k = control[i_start:i_stop+1]
		u_poly = np.polynomial.Polynomial.fit(t_k, u_k, deg=n-1, window=[0, 1])
		u_polys[i_u, i_poly] = u_poly

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
	t_list.append(t[-1])
	y_list.append(y[i_y, -1])
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
	t_list.append(t[-1])
	dy_list.append(dy[i_dy, -1])
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
	t_list.append(t[-1])
	u_list.append(u[i_u, -1])
	t_data.append(t_list)
	u_datas.append(u_list)

t_data = np.array(t_data[0])
y_datas = np.array(y_datas)
dy_datas = np.array(dy_datas)
u_datas = np.array(u_datas)

print(dy_datas)

if True:
	for i_y, y_data in enumerate(y_datas):
		plt.plot(t_data, y_data)
		plt.plot(t, y[i_y], marker='x', markersize=7, linestyle='')

if False:
	for i_y, dy_data in enumerate(dy_datas):
		plt.plot(t_data, dy_data)
		plt.plot(t, dy[i_y], marker='x', markersize=7, linestyle='')

if False:
	for i_u, u_data in enumerate(u_datas):
		plt.plot(t_data, u_data)
		plt.plot(t, u[i_u], marker='x', markersize=7, linestyle='')



plt.show()
