from timeit import default_timer as timer

import numpy as np
import sympy as sym

import pycollo


if __name__ == '__main__':

	print('\n')

	y0, y1, u0, q0 = sym.symbols('y0 y1 u0 q0')
	g = sym.symbols('g')
	m0, p0, d0, l0, k0, I0, J0, m0J0 = sym.symbols('m0 p0 d0 l0 k0 I0 J0 m0J0')
	phi0mu, phi0sigma = sym.symbols('phi0mu phi0sigma')
	dphi0mu, dphi0sigma = sym.symbols('dphi0mu dphi0sigma')
	T0mu, T0sigma = sym.symbols('T0mu T0sigma')
	phi0, dphi0, T0 = sym.symbols('phi0 dphi0 T0')
	c0, s0 = sym.symbols('c0 s0')
	phi0min, phi0max = sym.symbols('phi0min phi0max')
	dphi0min, dphi0max = sym.symbols('dphi0min dphi0max')
	T0min, T0max = sym.symbols('T0min T0max')
	q0min, q0max = sym.symbols('q0min q0max')
	gm0, p0c0, gm0p0c0, gm0p0c0plusT0, T0T0, q0minusT0T0 = sym.symbols('gm0 p0c0 gm0p0c0 gm0p0c0plusT0 T0T0 q0minusT0T0')

	start_total = timer()

	start = timer()

	aux_data = dict({I0: m0*(k0**2 + p0**2), c0: sym.cos(y0), g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12})

	stop = timer()
	print(f"Make aux data dict: {stop - start}", '\n')

	x_vars = sym.Matrix([y0, y1, u0, q0])

	J = (2+I0)*q0

	c = sym.Matrix([y1, (g*m0*p0*c0 + u0)/I0, q0 - u0**2])

	L = sym.Matrix([])

	_y0, _y1, _u0, _q0 = sym.symbols('_y0 _y1 _u0 _q0')
	x_vars_pycollo = sym.Matrix([_y0, _y1, _u0, _q0])

	start = timer()
	expression_graph = pycollo.ExpressionGraph(x_vars, x_vars_pycollo, aux_data, J, c, L)
	stop = timer()
	print(f"Build expression graph: {stop - start}", '\n')

	print(expression_graph.J)

	print(expression_graph.dJ_dx)

	print('')

	print(expression_graph.c)

	print(expression_graph.dc_dx)

	print('')


# if __name__ == '__main__':

# 	print('\n')

# 	no_subs = False

# 	y0, y1, u0, q0 = sym.symbols('y0 y1 u0 q0')
# 	g = sym.symbols('g')
# 	m0, p0, d0, l0, k0, I0, J0, m0J0 = sym.symbols('m0 p0 d0 l0 k0 I0 J0 m0J0')
# 	phi0mu, phi0sigma = sym.symbols('phi0mu phi0sigma')
# 	dphi0mu, dphi0sigma = sym.symbols('dphi0mu dphi0sigma')
# 	T0mu, T0sigma = sym.symbols('T0mu T0sigma')
# 	phi0, dphi0, T0 = sym.symbols('phi0 dphi0 T0')
# 	c0, s0 = sym.symbols('c0 s0')
# 	phi0min, phi0max = sym.symbols('phi0min phi0max')
# 	dphi0min, dphi0max = sym.symbols('dphi0min dphi0max')
# 	T0min, T0max = sym.symbols('T0min T0max')
# 	q0min, q0max = sym.symbols('q0min q0max')
# 	gm0, p0c0, gm0p0c0, gm0p0c0plusT0, T0T0, q0minusT0T0 = sym.symbols('gm0 p0c0 gm0p0c0 gm0p0c0plusT0 T0T0 q0minusT0T0')

# 	start_total = timer()

# 	start = timer()

# 	if no_subs:
# 		aux_data = dict({g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, phi0mu: (phi0min + phi0max)/2, phi0sigma: phi0max - phi0min, dphi0mu: (dphi0min + dphi0max)/2, dphi0sigma: dphi0max - dphi0min, T0mu: (T0min + T0max)/2, T0sigma: T0max - T0min, phi0: phi0mu + y0*phi0sigma, phi0min: -np.pi, phi0max: np.pi, dphi0min: -10, dphi0max: 10, T0min: -15, T0max: 15, q0min: 0, q0max: 200})
# 	else:
# 		aux_data = dict({I0: m0*(k0**2 + l0**2), l0: p0 + d0, c0: sym.cos(phi0), g: -9.81, m0: 1.0, p0: 0.5, d0: 0.5, k0: 1/12, phi0: phi0mu + y0*phi0sigma, phi0mu: (phi0min + phi0max)/2, phi0sigma: phi0max - phi0min, dphi0mu: (dphi0min + dphi0max)/2, dphi0sigma: dphi0max - dphi0min, T0mu: (T0min + T0max)/2, T0sigma: T0max - T0min, dphi0: dphi0mu + y1*dphi0sigma, T0: T0mu + u0*T0sigma, phi0min: -np.pi, phi0max: np.pi, dphi0min: -10, dphi0max: 10, T0min: -15, T0max: 15, q0min: 0, q0max: 200, gm0: g*m0, p0c0: p0*c0, gm0p0c0: gm0*p0c0, gm0p0c0plusT0: gm0p0c0 + T0, T0T0: T0**2, q0minusT0T0: q0 - T0T0})

# 	stop = timer()
# 	print(f"Make aux data dict: {stop - start}", '\n')

# 	x_vars = sym.Matrix([y0, y1, u0, q0])

# 	J = (2+I0)*q0

# 	if no_subs:
# 		c = sym.Matrix([(dphi0mu + y1*dphi0sigma)/phi0sigma, (g*m0*p0*sym.cos(phi0) + (T0mu + u0*T0sigma))/((m0*(k0**2 + p0**2))*dphi0sigma), q0 - (T0mu + u0*T0sigma)**2])
# 	else:
# 		c = sym.Matrix([dphi0/phi0sigma, (gm0p0c0plusT0)/(I0*dphi0sigma), q0minusT0T0])

# 	L = sym.Matrix([])

# 	_y0, _y1, _u0, _q0 = sym.symbols('_y0 _y1 _u0 _q0')
# 	x_vars_pycollo = sym.Matrix([_y0, _y1, _u0, _q0])

# 	start = timer()
# 	expression_graph = pycollo.ExpressionGraph(x_vars, x_vars_pycollo, aux_data, J, c, L)
# 	stop = timer()
# 	print(f"Build expression graph: {stop - start}", '\n')

# 	print(expression_graph.J)

# 	print(expression_graph.dJ_dx)

# 	print('')

# 	print(expression_graph.c)

# 	print(expression_graph.dc_dx)

# 	print('')

	# dJ_dx = hybrid_symbolic_algorithmic_differentiation(J, x_vars, e_vars, e_subs, tier_slices)
	# print('Symbolic objective gradient calculated.\n')
	# print(dJ_dx, '\n\n')

	# start = timer()
	# dc_dx = hybrid_symbolic_algorithmic_differentiation(c, x_vars, e_vars, e_subs, tier_slices)
	# stop = timer()
	# stop_total = timer()
	# print(f"Produce Jacobian: {stop - start}", '\n')
	# print(f"Number subs tiers: {len(tier_slices)}", '\n')
	# print(f"Total time: {stop_total - start_total}", '\n')
	# print(dc_dx, '\n')


