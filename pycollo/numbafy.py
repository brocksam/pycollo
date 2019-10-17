import itertools
from numbers import Number

import numba
import numpy as np
from numpy import sin, cos, tan, exp, sqrt, arctan, tanh
from ordered_set import OrderedSet
import scipy.interpolate as interpolate
import sympy as sym

atan = arctan

def numbafy(expression_graph=None, expression=None, expression_nodes=None, precomputable_nodes=None, dependent_tiers=None, parameters=None, return_dims=None, return_flat=False, N_arg=False, endpoint=False, hessian=None, hessian_sym_set=None, ocp_num_vars=None):

    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args)

    # cout(expression)
    # cout(precomputable_nodes)
    # cout(dependent_tiers)
    # cout(parameters)

    if parameters:
        function_arguments = ', '.join(f'{p}' 
            for p in parameters)
        if N_arg is True:
            function_arguments += ', _N'
    else:
        raise NotImplementedError

    if precomputable_nodes:
        precomputed_constants = '\n    '.join(f'{node.numbafy_expression}' 
            for node in precomputable_nodes)
    else:
        precomputed_constants = '    '

    substitution_tiers = []
    for tier in list(dependent_tiers.values())[1:]:
        tier_substitutions = '\n    '.join(f'{node.numbafy_expression}' 
            for node in tier)
        substitution_tiers.append(tier_substitutions)

    intermediate_substitutions = '\n    '.join(f'{sub_tier}' 
            for sub_tier in substitution_tiers)

    if ocp_num_vars:
        yu_qts_continuous_split = sum(ocp_num_vars[0:2])
        yu_qts_endpoint_split = 2*ocp_num_vars[0]

    zero_sym = expression_graph._zero_node.symbol
    one_sym = expression_graph._one_node.symbol

    if return_dims is None:
        return_value = f'{expression}'

    elif return_dims == 1:
        if endpoint is True:
            i_t0 = slice(0, yu_qts_endpoint_split, 2)
            i_tF = slice(1, yu_qts_endpoint_split, 2)
            y_t0_vars = parameters[i_t0]
            y_tF_vars = parameters[i_tF]
            qts_vars = parameters[yu_qts_endpoint_split:]

            y_tuple = []
            for y_t0, y_tF in grouper(expression[:yu_qts_endpoint_split], 2):
                if y_t0 == zero_sym and y_tF == zero_sym:
                    to_append = 'np.zeros(_N)'
                elif y_t0 == zero_sym:
                    to_append = (f'np.hstack((np.zeros(_N-1), {y_tF}))')
                elif y_tF == zero_sym:
                    to_append = (f'np.hstack(({y_t0}, np.zeros(_N-1)))')
                else:
                    to_append = (f'np.hstack(({y_t0}, np.zeros(_N-2), {y_tF}))')
                y_tuple.append(to_append)

            y_str = ', '.join(f'{e}' for e in y_tuple)

            u_str = ', '.join(f'np.zeros(_N)' for i in range(ocp_num_vars[1]))

            qts_str = ', '.join(f'{e}' for e in expression[yu_qts_endpoint_split:])

            qts_str = f'np.array([{qts_str}])' if qts_str is not '' else ''

            str_list = []
            for s in (y_str, u_str, qts_str):
                if s is not '':
                    str_list.append(s)

            str_str = ', '.join(s for s in str_list)

            return_value = f'np.concatenate([{str_str}])'
        else:
            expression_string = ', '.join(f'{e}' for e in expression)
            return_value = f'np.array([{expression_string}])'

    elif return_dims == 2:
        expression_rows = []
        parameter_set = set(parameters[:yu_qts_continuous_split])
        if expression.shape[0] != 1 and expression.shape[1] == 1:
            expression = expression.T
        for row_num in range(expression.rows):
            expression_list = []
            row = expression.row(row_num)
            for col_num, e in enumerate(row):
                index = row_num*expression.cols + col_num
                node = expression_nodes[index]
                print(node, node.is_vector)
                if e.free_symbols.intersection(parameter_set):
                    expression_list.append(e)
                else:
                    if node.is_vector:
                        e_entry = f'{e}'
                    else:
                        e_entry = f'({e})*np.ones(_N)'
                    if return_flat:
                        raise NotImplementedError
                        e_entry = f'{e_entry}.flatten()'
                    expression_list.append(e_entry)
            expression_row = ', '.join(f'{e}' for e in expression_list)
            expression_rows.append(f"np.hstack([{expression_row}])")

        expression_string = ', '.join(f'{e}' for e in expression_rows)

        if return_flat:
            raise NotImplementedError
            return_value = f'np.concatenate([{expression_string}])'
        else:
            return_value = f'np.array([{expression_string}])'

    elif return_dims == 3:
        expression_rows = []
        for row_num in range(expression.rows):
            expression_list = []
            row = expression.row(row_num)
            for col_num, e in enumerate(row):
                index = row_num*expression.cols + col_num
                node = expression_nodes[index]
                if e is zero_sym:
                    e_entry = f'np.zeros(_N)'
                elif e is one_sym:
                    e_entry = f'np.ones(_N)'
                elif node.is_vector:
                    e_entry = f'{e}'
                else:
                    e_entry = f'({e})*np.ones(_N)'
                expression_list.append(e_entry)
            expression_row = ', '.join(f'{e}' for e in expression_list)
            expression_rows.append(f"np.array([{expression_row}])")
        expression_string = ', '.join(f'{e}' for e in expression_rows)
        return_value = f'np.array([{expression_string}])'

    # elif return_dims == 2:
    #     parameter_set = set(parameters[:yu_qts_continuous_split])
    #     expression_list = []
    #     for i, (e, node) in enumerate(zip(expression, expression_nodes)):
    #         if e.free_symbols.intersection(parameter_set):
    #             expression_list.append(e)
    #         else:
    #             if node.is_vector or i > yu_qts_continuous_split:
    #                 e_entry = f'{e}'
    #             else:
    #                 e_entry = f'({e})*np.ones(_N)'
    #             if return_flat:
    #                 e_entry = f'{e_entry}.flatten()'
    #             expression_list.append(e_entry)

    #     expression_string = ', '.join(f'{e}' for e in expression_list)

    #     if return_flat:
    #         return_value = f'np.concatenate([{expression_string}])'
    #     else:
    #         return_value = f'np.array([{expression_string}])'

    else:
        msg = ("Value for `return_dims` argument must be 0, 1, or 2 only.")
        raise ValueError(msg)

    function_string = (
        f"def numbafied_func({function_arguments}):\n"
        f"    {precomputed_constants}\n"
        f"    {intermediate_substitutions}\n"
        f"    return {return_value}")

    cout(function_string)
    
    exec(function_string)
       
    return locals()['numbafied_func']



def kill():
        print('\n\n')
        raise ValueError

def cout(*args):
    print('\n\n')
    for arg in args:
        print(f'{arg}\n')


def numbafy_old(expression, parameters=None, constants=None, substitutions=None, return_dims=None, return_flat=False, N_arg=False, endpoint=False, hessian=None, hessian_sym_set=None, ocp_num_vars=None):

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
                if node.is_vector:
                    e_entry = f'{e}'
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