import numpy as np 
import scipy.interpolate as interpolate
import sympy as sym

def numbafy(expression, parameters=None, constants=None, use_cse=False, new_function_name='numbafy_func'):
    cse = sym.cse(expression)
    
    code_parameters = ''
    code_constants = ''
    code_cse = ''

    if parameters:
        code_parameters = ', '.join(f'{p}' for p in parameters)
    
    if constants:
        code_constants = []
        for k, v in constants.items():
            code_constants.append(f'{k} = {v}')
        code_constants = '\n    '.join(code_constants)
        
    if use_cse:
        expressions = sym.cse(expression)
        code_cse = []
        for e in expressions[0]:
            k, v = e
            code_cse.append(f'{k} = {v}')
        code_cse = '\n    '.join(code_cse)
        code_expression = f'{expressions[1][0]}'
    else:
        code_expression = f'{expression}'
        

    template = f"""@jit
def {new_function_name}({code_parameters}):
    {code_constants}
    {code_cse}
    return {code_expression}"""
       
    return template



order = list(range(2, 7))

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

# print(quadrature_points)
# print(quadrature_weights)
# print(D_matricies)