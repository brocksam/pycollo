# import numba as nb
import numpy as np

# @nb.njit(parallel=True)
# def numbafy(a, b):
# 	return np.array((a, b))

# a = np.array([1, 2])
# b = np.array([3, 4])
# c = numbafy(a, b)
# print(c)

def skew(vec):
	vec = vec.squeeze()
	return np.array([
		[0, -vec[2], vec[1]],
		[vec[2], 0, -vec[0]],
		[vec[1], -vec[0], 0]])

J = np.array([
	[2.80701911616e8, 4.822509936e6, -1.71675094448e8], 
	[4.822509936e6, 9.5144639344e8, 6.02604448e5], 
	[-1.71675094448e8, 6.02604448e5, 7.6594401336e8]])

omega_tF = np.array([
	[-0.009466793093285e-3], 
	[-0.001136333138338e0], 
	[0.005068739149029e-3]])

r_tF = np.array([
	[0.002876564438427e0],
	[0.155161074977957e0],
	[0.003819358953833e0]])

h_tF = np.array([
	[0],
	[0],
	[0]])

omega_orb = 0.06511*np.pi/180
r_skew = skew(r_tF)
omega_skew = skew(omega_tF)
I = np.eye(3)
C = I + 2/(1 + np.dot(r_tF.T, r_tF)) * (np.dot(r_skew, r_skew) - r_skew)
C3 = C[:, 2].reshape(3, 1)
C3_skew = skew(C3)
tau_gg = 3*omega_orb**2 * np.dot(C3_skew, np.dot(J, C3))

omega_dot = np.dot(np.linalg.inv(J), (tau_gg - np.dot(omega_skew, (np.dot(J, omega_tF) + h_tF))))

print(omega_dot)