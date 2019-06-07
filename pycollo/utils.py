import itertools

import numba
import numpy as np
from numpy import sin, cos, tan, exp, sqrt
import scipy.interpolate as interpolate
import sympy as sym

def numbafy(expression, parameters=None, constants=None, substitutions=None, return_dims=None, return_flat=False, N_arg=False, endpoint=False, ocp_num_vars=None):

    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args)

    def factor_cse(expression):
        # print('Expression:')
        # print(expression, '\n')
        expressions = sym.cse(expression)
        expressions_factored = expressions[1]
        # print('CSE Before:')
        # print(expressions, '\n')
        cse_list = []
        cse_str = ''
        condition_1 = not expressions_factored
        condition_2 = (isinstance(expressions_factored[0], sym.matrices.MatrixBase) and len(expressions_factored) == 1)
        if condition_1 or condition_2:
            expressions_factored = expression
        else:
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
            

        # print('Factored Expressions:')
        # print(expressions_factored, '\n')
        # print('Common Sub-Expressions:')
        # print(cse_str, '\n')

        return expressions_factored, cse_str

    if not parameters:
        raise NotImplementedError

    # print('Constants:')
    # print(code_constants, '\n')

    # print('Parameters:')
    # print(code_parameters, '\n')

    # print('SymPy Expression:')
    # print(expression, '\n')

    cse = sym.cse(expression)
    
    code_parameters = ''
    code_constants = ''
    code_substitutions = ''
    code_cse = ''

    try:
        expression_free_syms = expression.free_symbols
    except AttributeError:
        expression_free_syms = set()
        for e in expression:
            expression_free_syms.update(e.free_symbols)

    if ocp_num_vars:
        yu_qts_split = 2*ocp_num_vars[0]

    # print('Expression Free Symbols:')
    # print(expression_free_syms, '\n')

    if parameters:
        code_parameters = ', '.join(f'{p}' for p in parameters)
        if N_arg is True:
            code_parameters += ', _N'

    substitution_free_syms = set()
    if substitutions:
        code_substitutions = []
        for k, v in substitutions.items():
            if k in expression_free_syms:
                code_substitutions.append(f'{k} = {v}')
                substitution_free_syms.update(v.free_symbols)
        code_substitutions = '\n    '.join(code_substitutions)

    expr_subs_free_syms = set.union(expression_free_syms, substitution_free_syms)

    # print('Expression & Substituion Free Symbols:')
    # print(expr_subs_free_syms, '\n')
    
    if constants:
        code_constants = []
        for k, v in constants.items():
            if k in expr_subs_free_syms:
                code_constants.append(f'{k} = {v}')
        code_constants = '\n    '.join(code_constants)
        
    if return_dims == 0:
        code_expression = f'{expression}'

    elif return_dims == 1:
        expressions_factored, code_cse = factor_cse(expression)
        if endpoint is True:
            i_t0 = slice(0, yu_qts_split, 2)
            i_tF = slice(1, yu_qts_split, 2)
            y_t0_vars = parameters[i_t0]
            y_tF_vars = parameters[i_tF]
            qts_vars = parameters[yu_qts_split:]
            temp_list = [f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))' for y_t0, y_tF in zip(y_t0_vars, y_tF_vars)]

            y_tuple = (f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))' for y_t0, y_tF in grouper(expressions_factored[:yu_qts_split], 2))
            y_str = ', '.join(f'{e}' for e in y_tuple)

            u_str = ', '.join(f'np.zeros(_N)' for i in range(ocp_num_vars[1]))

            qts_str = ', '.join(f'{e}' for e in expressions_factored[yu_qts_split:])
            qts_str = f'np.array([{qts_str}])'

            code_expression = f'np.concatenate([{y_str}, {u_str}, {qts_str}])'
        else:
            expression_string = ', '.join(f'{e}' for e in expressions_factored)
            code_expression = f'np.array([{expression_string}])'

    elif return_dims == 2:
        expressions_factored, code_cse = factor_cse(expression)
        parameter_set = set(parameters[:yu_qts_split])
        expression_list = []
        for e in expressions_factored:
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
    {code_substitutions}
    {code_cse}
    return {code_expression}"""

    print(function_string)
    print('\n\n\n')
    # input()
    
    exec(function_string)
       
    return locals()['numbafied_func']


