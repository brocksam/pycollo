import numpy as np
import scipy.sparse as sparse
import sympy as sym

N = 4
K = 2

t0, t1, t2, t3 = sym.symbols('t0 t1 t2 t3')
y0_t0, y0_t1, y0_t2, y0_t3 = sym.symbols('y0_t0 y0_t1 y0_t2 y0_t3')
y1_t0, y1_t1, y1_t2, y1_t3 = sym.symbols('y1_t0 y1_t1 y1_t2 y1_t3')
u0_t0, u0_t1, u0_t2, u0_t3 = sym.symbols('u0_t0 u0_t1 u0_t2 u0_t3')
u1_t0, u1_t1, u1_t2, u1_t3 = sym.symbols('u1_t0 u1_t1 u1_t2 u1_t3')

time = sym.Matrix([t0, t1, t2, t3])

h0 = t1 - t0
h1 = t3 - t0
h = sym.Matrix([h0, h1])

y0 = sym.Matrix([y0_t0, y0_t1, y0_t2, y0_t3])
y1 = sym.Matrix([y1_t0, y1_t1, y1_t2, y1_t3])
u0 = sym.Matrix([u0_t0, u0_t1, u0_t2, u0_t3])
u1 = sym.Matrix([u1_t0, u1_t1, u1_t2, u1_t3])
q0 = sym.Symbol('q0')
q1 = sym.Symbol('q1')
s0 = sym.Symbol('s0')
s1 = sym.Symbol('s1')

y = sym.Matrix([y0, y1])
u = sym.Matrix([u0, u1])
q = sym.Matrix([q0, q1])
t = sym.Matrix([t0, t3])
s = sym.Matrix([s0, s1])

x = sym.Matrix([y, u, q, t, s])

E = sym.Matrix([[1, -1, 0, 0], [0, 1, -1, 0], [0, 1, 0, -1]])
I = sym.Matrix([[1/2, 1/2, 0, 0], [0, 5/24, 1/3, -1/24], [0, 1/6, 2/3, 1/6]])

y0_dot_t0, y0_dot_t1, y0_dot_t2, y0_dot_t3 = sym.symbols('y0_dot_t0 y0_dot_t1 y0_dot_t2 y0_dot_t3')
y1_dot_t0, y1_dot_t1, y1_dot_t2, y1_dot_t3 = sym.symbols('y1_dot_t0 y1_dot_t1 y1_dot_t2 y1_dot_t3')

y0_dot = sym.Matrix([y0_dot_t0, y0_dot_t1, y0_dot_t2, y0_dot_t3])
y1_dot = sym.Matrix([y1_dot_t0, y1_dot_t1, y1_dot_t2, y1_dot_t3])

y_dot = sym.Matrix([y0_dot, y1_dot])

c_defect0 = E*y0 + h0*I*y0_dot
c_defect1 = E*y1 + h1*I*y1_dot

c_defect = sym.Matrix([c_defect0, c_defect1])

print(y_dot.jacobian(x).shape)

c = sym.Matrix([
	y0_t0 - y0_t1 + h0 * (1/2 * y0_dot_t0 + 1/2 * y0_dot_t1),
	y0_t1 - y0_t2 + h0 * (5/24 * y0_dot_t1 + 1/3 * y0_dot_t2 - 1/24 * y0_dot_t3),
	y0_t1 - y0_t3 + h0 * (1/6 * y0_dot_t1 + 2/3 * y0_dot_t2 + 1/6 * y0_dot_t3),
	y1_t0 - y1_t1 + h1 * (1/2 * y1_dot_t0 + 1/2 * y1_dot_t1),
	y1_t1 - y1_t2 + h1 * (5/24 * y1_dot_t1 + 1/3 * y1_dot_t2 - 1/24 * y1_dot_t3),
	y1_t1 - y1_t3 + h1 * (1/6 * y1_dot_t1 + 2/3 * y1_dot_t2 + 1/6 * y1_dot_t3),
	])

y0_dot = (timeF - time0) * s0**2 * s1**2 * y0**2 * y1**2 * u0**2 * u1**2
y1_dot = (timeF - time0) * s0**3 * s1**3 * y0**3 * y1**3 * u0**3 * u1**3

y_dot = sym.Matrix([y0_dot, y1_dot])

dy_dot_dx = y_dot.jacobian(x)

num_y = 2
num_u = 2
num_q = 2
num_t = 2
num_s = 2

sE = sparse.csr_matrix(E)
sI = sparse.csr_matrix(I)

print(dy_dot_dx.shape)