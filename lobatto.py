import numpy as np
import scipy.interpolate as interpolate

order = list(range(2, 5))

quadrature_points = {}
quadrature_weights = {}
D_matricies = {}

for k in order:
	num_interior_points = k - 1
	coefficients = [0]*(num_interior_points)
	coefficients.append(1)
	legendre_polynomial = np.polynomial.legendre.Legendre(coefficients)
	lobatto_points = legendre_polynomial.deriv().roots()
	lobatto_points = np.insert(lobatto_points, 0, -1, axis=0)
	lobatto_points = np.append(lobatto_points, 1)

	lobatto_weights = np.array([2/(k*(k-1)*(legendre_polynomial(x)**2)) for x in lobatto_points])

	basis_polynomials = []
	basis_polynomials_derivs = []
	for i, tau in enumerate(lobatto_points):
		weightings = np.zeros_like(lobatto_points)
		weightings[i] = 1
		basis_polynomial = interpolate.lagrange(lobatto_points, weightings)
		basis_polynomials.append(basis_polynomial)
		basis_polynomials_derivs.append(basis_polynomial.deriv())

	D_matrix = np.empty([k, k], dtype=object)

	for dldtau, _ in enumerate(basis_polynomials_derivs):
		for tau, _ in enumerate(lobatto_points):
			D_matrix[tau, dldtau] = basis_polynomials_derivs[dldtau](lobatto_points[tau])

	quadrature_points.update({k: lobatto_points})
	quadrature_weights.update({k: lobatto_weights})
	D_matricies.update({k: D_matrix})

print(quadrature_points)
print(quadrature_weights)
print(D_matricies)