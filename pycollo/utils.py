import warnings

import numba
import numpy as np
from numpy import sin, cos
import scipy.interpolate as interpolate
import sympy as sym

def numbafy(expression, parameters=None, constants=None, return_dims=None, return_flat=False, ocp_yu_qts_split=None):

    if not parameters:
        raise NotImplementedError

    cse = sym.cse(expression)
    
    code_parameters = ''
    code_constants = ''
    code_N = ''
    code_cse = ''

    if parameters:
        code_parameters = ', '.join(f'{p}' for p in parameters)

    # print('Parameters:')
    # print(code_parameters, '\n')
    
    if constants:
        code_constants = []
        for k, v in constants.items():
            code_constants.append(f'{k} = {v}')
        code_constants = '\n    '.join(code_constants)

    # print('Constants:')
    # print(code_constants, '\n')

    # print('SymPy Expression:')
    # print(expression, '\n')

    def factor_cse(expression):
        # print('Expression:')
        # print(expression, '\n')
        expressions = sym.cse(expression)
        expressions_factored = expressions[1]
        # print('CSE Before:')
        # print(expressions, '\n')
        cse_list = []
        cse_str = ''
        if expressions_factored:
            for e in expressions[0]:
                k, v = e
                cse_list.append(f'{k} = {v}')
            cse_str = '\n    '.join(cse_list)

            expressions_factored_list = []
            for factored_expression in expressions_factored:
                if isinstance(factored_expression, sym.Matrix):
                    for entry in factored_expression:
                        expressions_factored_list.append(entry)
                else:
                    expressions_factored_list.append(factored_expression)
            expressions_factored = expressions_factored_list
        else:
            expressions_factored = expression

        # print('Factored Expressions:')
        # print(expressions_factored, '\n')
        # print('Common Sub-Expressions:')
        # print(cse_str, '\n')

        return expressions_factored, cse_str
        
    if return_dims == 0:
        code_expression = f'{expression}'

    elif return_dims == 1:
        expressions_factored, code_cse = factor_cse(expression)
        expression_string = ', '.join(f'{e}' for e in expressions_factored)
        code_expression = f'np.array([{expression_string}])'

    elif return_dims == 2:
        expressions_factored, code_cse = factor_cse(expression)
        parameter_set = set(parameters[:ocp_yu_qts_split])
        expression_list = []
        for e in expressions_factored[:ocp_yu_qts_split]:
            if e.free_symbols.intersection(parameter_set):
                expression_list.append(e)
            else:
                if not e:
                    e_entry = f'np.zeros(_N)'
                else:
                    e_entry = f'{e}*np.ones(_N)'
                if return_flat:
                    e_entry = f'{e_entry}.flatten()'
                expression_list.append(e_entry)

        code_N = f'_N = {parameters[0]}.shape[0]'

        expression_string = ', '.join(f'{e}' for e in expression_list)

        if return_flat:
            code_expression = f'np.concatenate([{expression_string}])'
        else:
            code_expression = f'np.array([{expression_string}])'

    else:
        msg = ("Value for `return_dims` argument must be 0, 1, or 2 only.")
        raise ValueError(msg)

    # print('Expression List:')
    # print(expression_list, '\n')

    # print('Expression String:')
    # print(expression_string, '\n')

    # print('Code Expression:')
    # print(code_expression, '\n')

    function_string = f"""def numbafied_func({code_parameters}):
    {code_constants}
    {code_N}
    {code_cse}
    return {code_expression}"""

#     function_string = f"""@numba.jit(nopython=True)
# def numbafied_func({code_parameters}):
#     {code_constants}
#     {code_N}
#     {code_cse}
#     return {code_expression}"""

    # print(function_string)
    # print('\n\n\n')
    
    exec(function_string)
       
    return locals()['numbafied_func']





class AccuracyWarning(Warning):
    pass

def vectorize1(func, args=(), vec_func=False):
    """Vectorize the call to a function.
    This is an internal utility function used by `romberg` and
    `quadrature` to create a vectorized version of a function.
    If `vec_func` is True, the function `func` is assumed to take vector
    arguments.
    Parameters
    ----------
    func : callable
        User defined function.
    args : tuple, optional
        Extra arguments for the function.
    vec_func : bool, optional
        True if the function func takes vector arguments.
    Returns
    -------
    vfunc : callable
        A function that will take a vector argument and return the
        result.
    """
    if vec_func:
        def vfunc(x):
            return func(x, *args)
    else:
        def vfunc(x):
            if np.isscalar(x):
                return func(x, *args)
            x = np.asarray(x)
            # call with first point to get output type
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, 'dtype', type(y0))
            output = np.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output
    return vfunc


def _difftrap(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    if numtraps <= 0:
        raise ValueError("numtraps must be > 0 in difftrap().")
    elif numtraps == 1:
        return 0.5*(function(interval[0])+function(interval[1]))
    else:
        numtosum = numtraps/2
        h = float(interval[1]-interval[0])/numtosum
        lox = interval[0] + 0.5 * h
        points = lox + h * np.arange(numtosum)
        s = np.sum(function(points), axis=0)
        return s


def _romberg_diff(b, c, k):
    """
    Compute the differences for the Romberg quadrature corrections.
    See Forman Acton's "Real Computing Made Real," p 143.
    """
    tmp = 4.0**k
    return (tmp * c - b)/(tmp - 1.0)


def _printresmat(function, interval, resmat):
    # Print the Romberg result matrix.
    i = j = 0
    print('Romberg integration of', repr(function), end=' ')
    print('from', interval)
    print('')
    print('%6s %9s %9s' % ('Steps', 'StepSize', 'Results'))
    for i in range(len(resmat)):
        print('%6d %9f' % (2**i, (interval[1]-interval[0])/(2.**i)), end=' ')
        for j in range(i+1):
            print('%9f' % (resmat[i][j]), end=' ')
        print('')
    print('')
    print('The final result is', resmat[i][j], end=' ')
    print('after', 2**(len(resmat)-1)+1, 'function evaluations.')


def romberg(function, a, b, args=(), tol=1.48e-8, rtol=1.48e-8, show=False,
            divmin=3, divmax=10, vec_func=False):
    """
    Romberg integration of a callable function or method.
    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).
    If `show` is 1, the triangular array of the intermediate results
    will be printed.  If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.
    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    Returns
    -------
    results  : float
        Result of the integration.
    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    tol, rtol : float, optional
        The desired absolute and relative tolerances. Defaults are 1.48e-8.
    show : bool, optional
        Whether to print the results. Default is False.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    vec_func : bool, optional
        Whether `func` handles arrays as arguments (i.e whether it is a
        "vector" function). Default is False.
    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simps : Integrators for sampled data.
    cumtrapz : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.
    References
    ----------
    .. [1] 'Romberg's method' https://en.wikipedia.org/wiki/Romberg%27s_method
    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.
    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]
    ::
       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
    The final result is 0.421350396475 after 33 function evaluations.
    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701
    """
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Romberg integration only available "
                         "for finite limits.")
    vfunc = vectorize1(function, args, vec_func=vec_func)
    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap(vfunc, interval, n)
    result = intrange * ordsum
    resmat = [[result]]
    err = np.inf
    last_row = resmat[0]
    for i in range(1, divmax+1):
        n *= 2
        ordsum += _difftrap(vfunc, interval, n)
        row = [intrange * ordsum / n]
        for k in range(i):
            row.append(_romberg_diff(last_row[k], row[k], k+1))
        result = row[i]
        lastresult = last_row[i-1]
        if show:
            resmat.append(row)
        err = abs(result - lastresult)
        if err < tol or err < rtol * abs(result):
            if i <= divmin:
                pass
            else:
                break
        last_row = row
    else:
        warnings.warn(
            "divmax (%d) exceeded. Latest difference = %e" % (divmax, err),
            AccuracyWarning)

    if show:
        _printresmat(vfunc, interval, resmat)
    return result





