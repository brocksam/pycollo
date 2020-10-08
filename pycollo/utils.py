import collections
import itertools
from numbers import Number
from typing import (Iterable, Mapping, NamedTuple, Optional, Tuple)

import numba
import numpy as np
from numpy import sin, cos, tan, exp, sqrt, arctan, tanh
import scipy.interpolate as interpolate
import sympy as sym

from .typing import OptionalSymsType, TupleSymsType


dcdxInfo = collections.namedtuple('dcdxInfo', [
            'zeta_y', 'zeta_u', 'zeta_s',
            'gamma_y', 'gamma_u', 'gamma_s',
            'rho_y', 'rho_u', 'rho_s',
            ])


supported_iter_types = (tuple, list, np.ndarray)


class cachedproperty:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def format_as_named_tuple(
        iterable: OptionalSymsType, 
        use_named: bool = True, 
        named_keys: Optional[NamedTuple] = None,
        sympify: bool = True) -> TupleSymsType:
    """Formats user supplied arguments as a named tuple."""
    
    if not iterable:
        return ()
    try:
        iter(iterable)
    except TypeError:
        iterable = (iterable, )
    else:
        try:
            named_keys = iterable.keys()
            iterable = iterable.values()
        except:
            pass

    if sympify:
        entries = [sym.sympify(entry) for entry in iterable]
    else:
        entries = iterable
    if use_named:
        if named_keys is None:
            named_keys = [str(entry) for entry in entries]
        NamedTuple = collections.namedtuple('NamedTuple', named_keys)
        formatted_entries = NamedTuple(*entries)
    else:
        formatted_entries = tuple(entries)
    
    return formatted_entries


def check_sym_name_clash(syms: TupleSymsType) -> None:
    """Ensures user symbols do not clash with internal Pycollo symbols.

    Pycollo reserves certain naming conventions of sympy symbols for itself. 
    This function enforces those rules to make sure that any symbols Pycollo 
    creates and/or manipulates iternally do not conflict with ones that the 
    user expects ownership of. These naming conventions include all internal 
    Pycollo symbols being named with a leading underscore as well as the 
    suffixes '(t0)' and '(tF)'. Finally all user symbols must be uniquely named 
    for obvious reasons.

    Raises:
        ValueError: If any of the Pycollo naming rules are not obeyed.
    """
    for sym in syms:
        if str(sym)[0] == '_':
            msg = (f"The user defined symbol {sym} is invalid as its leading "
                f"character '_' is reserved for use by `Pycollo`. Please "
                f"rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(t0)':
            msg = (f"The user defined symbol {sym} is invalid as it is named "
                f"with the suffix '(t0)' which is reserved for use by "
                f"`Pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(sym)[-4:] == '(tF)':
            msg = (f"The user defined symbol {sym} is invalid as it is named "
                f"with the suffix '(tF)' which is reserved for use by "
                f"`Pycollo`. Please rename this symbol.")
            raise ValueError(msg)

    if len(set(syms)) != len(syms):
        msg = (f"All user defined symbols must have unique names.")
        raise ValueError(msg)


def dict_merge(*dicts: Iterable[Mapping]) -> dict:
    """Merges multiple dictionaries into a single dictionary."""
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def console_out(msg, heading=False, subheading=False, prefix="", suffix=""):
    msg = f"{prefix}{msg}{suffix}"
    msg_len = len(msg)
    if heading:
        seperator = '=' * msg_len
        output_msg = (f"\n{seperator}\n{msg}\n{seperator}\n")
    elif subheading:
        seperator = '-' * msg_len
        output_msg = (f"\n{msg}\n{seperator}")
    else:
        output_msg = msg
    print(output_msg)


def fast_sympify(arg):
    if not isinstance(arg, sym.Expr):
        return sym.sympify(arg)
    return arg


def format_case(item, case):
    if case is "title":
        return item.title()
    elif case is "upper":
        return item.upper()
    elif case is "lower":
        return item.lower()
    else:
        return item


def format_for_output(items, *args, **kwargs):
    return format_multiple_items_for_output(items, *args, **kwargs)


def format_multiple_items_for_output(items, wrapping_char="'", *, prefix_char="", case=None):
    items = (items, ) if isinstance(items, str) else items
    items = [f"{prefix_char}{format_case(item, case)}" for item in items]
    if len(items) == 1:
        return f"'{items[0]}'"
    else:
        pad = f"{wrapping_char}, {wrapping_char}"
        return (f"{wrapping_char}{pad.join(items[:-1])}{wrapping_char} "
            f"and {wrapping_char}{items[-1]}{wrapping_char}")


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




