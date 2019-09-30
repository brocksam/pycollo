import itertools
from numbers import Number

import numba
import numpy as np
from numpy import sin, cos, tan, exp, sqrt, arctan, tanh
from ordered_set import OrderedSet
import scipy.interpolate as interpolate
import sympy as sym

atan = arctan

def numbafy(expression, parameters=None, constants=None, substitutions=None, return_dims=None, return_flat=False, N_arg=False, endpoint=False, hessian=None, hessian_sym_set=None, ocp_num_vars=None):

    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args)

    def factor_cse(expression, ignore=[]):
        # print('Expression:')
        # print(expression, '\n')
        if not expression:
            return [], '', {}
        expressions = sym.cse(expression, ignore=ignore)
        expressions_factored = expressions[1]
        # print('CSE Before:')
        # print(expressions, '\n')
        # print('Factored Expressions:')
        # print(expressions_factored, '\n')
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
        # print(dict(expressions[0]), '\n')
        # print('Common Sub-Expressions:')
        # print(cse_str, '\n')

        return expressions_factored, cse_str, dict(expressions[0])

    if not parameters:
        raise NotImplementedError

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
        yu_qts_continuous_split = sum(ocp_num_vars[0:2])
        yu_qts_endpoint_split = 2*ocp_num_vars[0]

    # print('Expression Free Symbols:')
    # print(expression_free_syms, '\n')

    if parameters:
        code_parameters = ', '.join(f'{p}' for p in parameters)
        if N_arg is True:
            code_parameters += ', _N'

    substitution_free_syms = set()
    if substitutions:
        code_substitutions = []
        code_substitutions_dict = {}
        max_substitutions_depth = 100
        for i in range(max_substitutions_depth):
            for k, v in substitutions.items():
                if k in expression_free_syms.union(substitution_free_syms) and k not in code_substitutions_dict.keys():
                    code_substitutions_dict[k] = v
                    substitution_free_syms.update(v.free_symbols)

        # Reorder substitutions
        known_symbols = set(parameters).union(set(constants))
        code_substitutions = []
        for i in range(max_substitutions_depth):
            for k, v in code_substitutions_dict.copy().items():
                if not v.free_symbols.difference(known_symbols):
                    code_substitutions.append(f'{k} = {v}')
                    _ = code_substitutions_dict.pop(k, None)
                    known_symbols.add(k)

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
        expressions_factored, code_cse, _ = factor_cse(expression)
        if endpoint is True:
            i_t0 = slice(0, yu_qts_endpoint_split, 2)
            i_tF = slice(1, yu_qts_endpoint_split, 2)
            y_t0_vars = parameters[i_t0]
            y_tF_vars = parameters[i_tF]
            qts_vars = parameters[yu_qts_endpoint_split:]
            temp_list = [f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))' for y_t0, y_tF in zip(y_t0_vars, y_tF_vars)]

            y_tuple = (f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))' for y_t0, y_tF in grouper(expressions_factored[:yu_qts_endpoint_split], 2))
            y_str = ', '.join(f'{e}' for e in y_tuple)

            u_str = ', '.join(f'np.zeros(_N)' for i in range(ocp_num_vars[1]))

            qts_str = ', '.join(f'{e}' for e in expressions_factored[yu_qts_endpoint_split:])

            qts_str = f'np.array([{qts_str}])' if qts_str is not '' else ''

            str_list = []
            for s in (y_str, u_str, qts_str):
                if s is not '':
                    str_list.append(s)

            str_str = ', '.join(s for s in str_list)

            code_expression = f'np.concatenate([{str_str}])'
        else:
            expression_string = ', '.join(f'{e}' for e in expressions_factored)
            code_expression = f'np.array([{expression_string}])'

    elif return_dims == 2:
        parameter_set = set(parameters[:yu_qts_continuous_split])
        if hessian in ('defect', 'path', 'integral', 'endpoint'):
            expressions_factored, code_cse, factors_dict = factor_cse(expression, ignore=hessian_sym_set)
            continuous_parameter_set = parameter_set.copy()
            for k, v in substitutions.items():
                if v.free_symbols.intersection(continuous_parameter_set):
                    continuous_parameter_set.add(k)
            for k, v in factors_dict.items():
                if v.free_symbols.intersection(continuous_parameter_set):
                    continuous_parameter_set.add(k)
        else:
            expressions_factored, code_cse, _ = factor_cse(expression)
        
        expression_list = []
        for e in expressions_factored:
            if hessian is 'defect':
                lagrange_syms = e.free_symbols.intersection(hessian_sym_set)
                e_lambda_factorised = sym.collect(e.expand(), lagrange_syms)
                lagrange_sym_list = []
                lagrange_arg_list = []
                if len(lagrange_syms) > 1:
                    while lagrange_syms:
                        lagrange_sym = lagrange_syms.pop()
                        for arg in e_lambda_factorised.args:
                            if lagrange_sym in arg.free_symbols:
                                lagrange_sym_list.append(lagrange_sym)
                                lagrange_arg_list.append(arg / lagrange_sym)
                else:
                    lagrange_sym = lagrange_syms.pop()
                    lagrange_sym_list.append(lagrange_sym)
                    lagrange_arg_list.append(e_lambda_factorised / lagrange_sym)
                defect_expression_list = []
                for lagrange_sym, e_no_lagrange in zip(lagrange_sym_list, lagrange_arg_list):
                    if e_no_lagrange.free_symbols.intersection(hessian_sym_set):
                        msg = ("Factorisation failed.")
                        raise ValueError(msg)
                    if not e_no_lagrange.free_symbols.intersection(continuous_parameter_set):

                        e_entry = f'np.outer({lagrange_sym}, ({e_no_lagrange})*np.ones(_N))'
                    else:
                        e_entry = f'np.outer({lagrange_sym}, ({e_no_lagrange}))'
                    defect_expression_list.append(e_entry)
                e_entry = ' + '.join(defect_expression_list)
                expression_list.append(e_entry)
            elif hessian is 'path':
                raise NotImplementedError
            elif hessian is 'integral':
                lagrange_syms = e.free_symbols.intersection(hessian_sym_set)
                if len(lagrange_syms) != 1:
                    raise NotImplementedError
                lagrange_sym = lagrange_syms.pop()
                e_no_lagrange = sym.collect(e, lagrange_sym) / lagrange_sym
                if e_no_lagrange.free_symbols.intersection(hessian_sym_set):
                    msg = ("Factorisation failed.")
                    raise ValueError(msg)
                if not e_no_lagrange.free_symbols.intersection(continuous_parameter_set):

                    e_entry = f'{lagrange_sym}*({e_no_lagrange})*np.ones(_N)'
                else:
                    e_entry = f'{lagrange_sym}*({e_no_lagrange})'
                expression_list.append(e_entry)
            elif hessian is 'endpoint':
                raise NotImplementedError
            elif hessian is 'objective':
                raise NotImplementedError
            elif e.free_symbols.intersection(parameter_set):
                expression_list.append(e)
            else:
                if not e:
                    e_entry = f'np.zeros(_N)'
                else:
                    e_entry = f'({e})*np.ones(_N)'
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

    # print('Constants:')
    # print(code_constants, '\n')

    # print('Parameters:')
    # print(code_parameters, '\n')

    # print('SymPy Expression:')
    # print(expression, '\n')

    # print('Code Expression:')
    # print(code_expression, '\n')

    function_string = f"""def numbafied_func({code_parameters}):
    {code_constants}
    {code_substitutions}
    {code_cse}
    return {code_expression}"""

    # print(function_string)
    # print('\n\n\n')
    # input()
    
    exec(function_string)
       
    return locals()['numbafied_func']


supported_iter_types = (tuple, list, np.ndarray)


def format_as_tuple(iterable):
    if not iterable:
        return ()
    try:
        iter(iterable)
    except TypeError:
        iterable = (iterable, )
    iterable_tuple = tuple(sym.sympify(symbol) for symbol in iterable)
    return iterable_tuple


def check_sym_name_clash(syms):
    for sym in syms:
        if str(sym)[0] == '_':
            msg = (f"The user defined symbol {sym} is invalid as its leading character, '_' is reserved for use by `pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(t0)':
            msg = (f"The user defined symbol {sym} is invalid as it is named with the suffix '_t0' which is reserved for use by `pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(tF)':
            msg = (f"The user defined symbol {sym} is invalid as it is named with the suffix '_tF' which is reserved for use by `pycollo`. Please rename this symbol.")
            raise ValueError(msg)

    if len(set(syms)) != len(syms):
        msg = (f"All user defined symbols must have unique names.")
        raise ValueError(msg)
    return None


def dict_merge(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def parse_arg_type(arg, arg_name_str, arg_type):
    if not isinstance(arg, arg_type):
        msg = (f"`{arg_name_str}` must be a {arg_type}. {arg} of type {type(arg)} is not a valid argument.")
        raise TypeError(msg)
    return arg


def parse_parameter_var(var, var_name_str, var_type):
    try:
        iter(var)
    except TypeError:
        _ = parse_arg_type(var, var_name_str, var_type)
        return var, var
    else:
        var = list(var)
        if len(var) != 2:
            msg = (f"If an iterable of values in being passed for `{var_name_str}` then this must be of length 2. {var} of length {len(var)} is not a valid argument.")
            raise TypeError(msg)
        if not isinstance(var[0], var_type) or not isinstance(var[1], var_type):
            msg = (f"Both items in `{var_name_str}` must be {var_type} objects. {var[0]} at index 0 is of type {type(var[0])} and {var[1]} at index 1 is of type {type(var[1])}.")
            raise TypeError(msg)
        return tuple(var)




