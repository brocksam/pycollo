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
        expressions = sym.cse(expression)
        expressions_factored = expressions[1][0]
        cse_list = []
        cse_str = ''
        if expressions_factored:
            for e in expressions[0]:
                k, v = e
                cse_list.append(f'{k} = {v}')
            cse_str = '\n    '.join(cse_list)
            
            try:
                iter(expressions_factored)
            except TypeError:
                expressions_factored = [expressions_factored]
        else:
            expressions_factored = expression
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

