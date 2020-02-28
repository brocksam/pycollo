import collections
import itertools
from numbers import Number
from typing import (Tuple)

import numba
import numpy as np
from numpy import sin, cos, tan, exp, sqrt, arctan, tanh
from ordered_set import OrderedSet
import scipy.interpolate as interpolate
import sympy as sym


dcdxInfo = collections.namedtuple('dcdxInfo', [
            'zeta_y', 'zeta_u', 'zeta_s',
            'gamma_y', 'gamma_u', 'gamma_s',
            'rho_y', 'rho_u', 'rho_s',
            ])


supported_iter_types = (tuple, list, np.ndarray)


def format_as_tuple(iterable) -> Tuple:
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
            msg = (f"The user defined symbol {sym} is invalid as its leading character '_' is reserved for use by `Pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(t0)':
            msg = (f"The user defined symbol {sym} is invalid as it is named with the suffix '_t0' which is reserved for use by `Pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(tF)':
            msg = (f"The user defined symbol {sym} is invalid as it is named with the suffix '_tF' which is reserved for use by `Pycollo`. Please rename this symbol.")
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




