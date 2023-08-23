from __future__ import annotations

from keyword import iskeyword
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

import casadi as ca
import numpy as np
import sympy as sm
from sympy.core.backend import AppliedUndef

from pycollo.typing import OptionalSymsType, TupleSymsType

SUPPORTED_ITER_TYPES = (tuple, list, np.ndarray)
SYMPY_TO_CASADI_API_MAPPING = {"ImmutableDenseMatrix": ca.blockcat,
                               "MutableDenseMatrix": ca.blockcat,
                               "Abs": ca.fabs,
                               "sec": lambda x: (1 / ca.cos(x)),
                               "cosec": lambda x: (1 / ca.sin(x)),
                               }


def symbol_name(symbol):
    """Return a symbol primitive's name/identifier.

    This function is required as Pycollo supports multiple different
    types of symbol primitives from different packages which do not have a
    consistent API/set of attributes/methods. This method provides that.

    Args
    ----
    symbol : Union[ca.SX, sm.Symbol]
        Symbol to get name of.

    Raises
    ------
    NotImplementedError
        If a name is requested for an unsupported symbol type.

    """
    if isinstance(symbol, ca.SX):
        return symbol.name()
    elif isinstance(symbol, (sm.Symbol, AppliedUndef)):
        return symbol.name
    msg = f"Cannot get name for symbol of type {type(symbol)}."
    raise NotImplementedError(msg)


def symbol_primitives(eqn):
    """Return primitives associated with equation as set of symbols.

    Raises
    ------
    NotImplementedError
        If an equation of an unsupported type is passed.

    """
    if isinstance(eqn, sm.Expr):
        return eqn.free_symbols
    elif isinstance(eqn, (ca.DM, float, int)):
        return set()
    elif isinstance(eqn, ca.SX):
        return ca.symvar(eqn)
    msg = f"Cannot get primitives for type {type(eqn)}."
    raise NotImplementedError(msg)


def sympy_to_casadi(sympy_expr, sympy_to_casadi_sym_mapping, *, phase=None):
    """Convert a Sympy expression to a CasADi one.

    Recipe adapted from one by Joris Gillis taken from:
    https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8

    Example
    -------
    This example creates some primitive symbols using both Sympy and CasADi.

    >>> x, y = sm.symbols("x, y")
    >>> X = ca.SX.sym("x")
    >>> Y = ca.SX.sym("y")
    >>> xy = sm.Matrix([x, y])

    A matrix consisting of some arbitrary expressions is created to showcase
    the differences in syntax between Sympy's and CasADi's internal
    mathematical functions.

    >>> e = sm.Matrix([x * sm.sqrt(y), sm.sin(x + y), abs(x - y)])
    >>> XY = ca.vertcat(X, Y)
    >>> E = sympy_to_casadi(e, xy, XY)

    Display the Sympy expression.

    >>> print(e)
    Matrix([[x*sqrt(y)], [sin(x + y)], [Abs(x - y)]])

    Display the CasADi expression and note that it is mathematically equivalent
    to the Sympy one.

    >>> print(E)
    [(x*sqrt(y)), sin((x+y)), fabs((x-y))]

    """
    sympy_vars = sm.Matrix(list(sympy_to_casadi_sym_mapping.keys()))
    casadi_vars = ca.vertcat(*sympy_to_casadi_sym_mapping.values())
    if casadi_vars.shape[1] > 1:
        casadi_vars = casadi_vars.T
    vars_not_in_mapping = sympy_expr.free_symbols.difference(set(sympy_vars))
    for sympy_var in vars_not_in_mapping:
        casadi_var_name_suffix = f"_P{phase}" if phase is not None else ""
        casadi_var = ca.SX.sym(f"{str(sympy_var)}{casadi_var_name_suffix}")
        sympy_to_casadi_sym_mapping.update({sympy_var: casadi_var})
        sympy_vars = sm.Matrix.vstack(sympy_vars, sm.Matrix([[sympy_var]]))
        casadi_vars = ca.vertcat(casadi_vars, casadi_var)
    f = sm.lambdify(sympy_vars, sympy_expr,
                     modules=[SYMPY_TO_CASADI_API_MAPPING, ca])
    return f(*ca.vertsplit(casadi_vars)), sympy_to_casadi_sym_mapping


def casadi_substitute(casadi_eqn, casadi_sym_mapping):
    """Substitute a CasADi SX expression with symbols from a mapping.

    Args
    ----
    casadi_eqn : ca.SX

    casadi_sym_mapping : Dict[ca.SX, ca.SX]
        Mapping of CasADi SX symbols to be replaced (keys) to CasADi SX symbols
        to replace with (values).

    Returns
    -------
    ca.SX
        The substituted expression.

    """
    remove_sym = ca.vertcat(*casadi_sym_mapping.keys())
    add_sym = ca.vertcat(*casadi_sym_mapping.values())
    return ca.substitute(casadi_eqn, remove_sym, add_sym)


def needed_to_tuple(var_full, needed):
    """Extract only needed variables to a new tuple."""
    return tuple(var for var, n in zip(var_full, needed) if n)


def create_data_container(type_name, field_names: Sequence[str]) -> type:
    """Create an immutable, indexible data type."""

    type_name = str(type_name)
    if isinstance(field_names, str):
        field_names = field_names.replace(",", " ").split()
    field_names = tuple(map(str, field_names))

    for name in (type_name, ) + field_names:
        if not isinstance(name, str):
            msg = f"Type and field names must be strings: {name!r}."
            raise TypeError(msg)
        if not name.isidentifier():
            msg = f"Type and field names must be valid identifiers: {name!r}."
            raise ValueError(msg)
        if  iskeyword(name):
            msg = f"Type names and field names cannot be a keyword: {name!r}."
            raise ValueError(msg)

    seen = set()
    for name in field_names:
        if name.startswith("_"):
            msg = f"Field names cannot start with an underscore: {name!r}."
            raise ValueError(msg)
        if name in seen:
            msg = f"Encountered duplicate field name: {name!r}."
            raise ValueError(msg)
        seen.add(name)

    def __init__(self, *data: tuple[Any]) -> None:
        if len(data) == 1 and isinstance(data[0], dict):
            mapping = data[0]
            data = ()
        else:
            mapping = {}
        if data:
            self._keys = data
            self._values = data
        elif mapping:
            self._keys = tuple(mapping.keys())
            self._values = tuple(mapping.values())
        else:
            self._keys = ()
            self._values = ()
        for field_name, value in zip(self._field_names, self._values):
            setattr(self, field_name, value)

    def __getitem__(self, key: Any) -> Any:
        if key in self._keys:
            index = self._keys.index(key)
        elif key in self._values:
            return key
        elif not isinstance(key, (int, slice)):
            index = self._field_names.index(key)
        else:
            index = key
        return self._values[index]

    def __len__(self) -> int:
        return len(field_names)

    def __str__(self) -> str:
        if self._keys == self._values:
            return str(self._values)
        return str(self.as_dict())

    def __repr__(self) -> str:
        if self._keys == self._values:
            pairs = (f"{f}={d}" for f, d in zip(self._field_names, self._values))
        else:
            pairs = (
                f"{f}={{{k}: {v}}}"
                for f, k, v in zip(self._field_names, self._keys, self._values)
            )
        kwargs = ", ".join(pairs)
        return f"{self.__class__.__name__}({kwargs})"

    def __iter__(self) -> Any:
        for item in self._values:
            yield item

    def __eq__(self, other) -> bool:
        return self._values == other

    def as_dict(self) -> dict[Any, Any]:
        return dict(zip(self._keys, self._values))

    class_namespace = {
        "__slots__": ("_keys", "_values", *tuple(field_names)),
        "__init__": __init__,
        "__getitem__": __getitem__,
        "__len__": __len__,
        "__str__": __str__,
        "__repr__": __repr__,
        "__iter__": __iter__,
        "__eq__": __eq__,
        "_field_names": tuple(field_names),
        "as_dict": as_dict,
    }

    DataContainer = type(type_name, (), class_namespace)

    return DataContainer


def format_as_data_container(
    type_name: str,
    iterable: OptionalSymsType,
    use_named: bool = True,
    identifiers: Optional[NamedTuple] = None,
    sympify: bool = True
) -> TupleSymsType:
    """Formats user supplied arguments as a custom data container."""

    def remove_prefix(string: str, prefix: str) -> str:
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    def convert_to_valid_identifier(entry: str) -> str:
        identifier = str(entry)
        if not identifier.isidentifier() and identifier[-3:] == "(t)":
            identifier = identifier[:-3]
        elif (
            not identifier.isidentifier()
            and identifier.startswith("Derivative(")
            and identifier.endswith(("(t), t)", "(t), (t, 2))"))
        ):
            identifier = (
                "d" * entry.derivative_count
                + remove_prefix(identifier, "Derivative(").rpartition("(t)")[0]
            )
        return identifier

    symbols = None
    if not iterable:
        return create_data_container(type_name, ())()
    try:
        iter(iterable)
    except TypeError:
        iterable = (iterable, )
    else:
        if isinstance(iterable, dict):
            if not identifiers:
                identifiers = [convert_to_valid_identifier(k) for k in iterable.keys()]
            symbols = tuple(iterable.keys())
            iterable = tuple(iterable.values())

    entries = [sm.sympify(entry) for entry in iterable] if sympify else iterable
    if symbols is None:
        symbols = identifiers if identifiers is not None else entries

    if use_named:
        if identifiers is None:
            identifiers = []
            for entry in entries:
                identifier = convert_to_valid_identifier(entry)
                identifiers.append(identifier)
        else:
            identifiers = [convert_to_valid_identifier(i) for i in identifiers]
        DataContainer = create_data_container(type_name, identifiers)
        if identifiers == entries:
            formatted_entries = DataContainer(*entries)
        else:
            formatted_entries = DataContainer(dict(zip(symbols, entries)))
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
    for symbol in syms:
        if str(symbol)[0] == '_':
            msg = (f"The user defined symbol {symbol} is invalid as its leading "
                   f"character '_' is reserved for use by `Pycollo`. Please "
                   f"rename this symbol.")
            raise ValueError(msg)
        elif str(symbol)[-4:] == '(t0)':
            msg = (f"The user defined symbol {symbol} is invalid as it is named "
                   f"with the suffix '(t0)' which is reserved for use by "
                   f"`Pycollo`. Please rename this symbol.")
            raise ValueError(msg)
        elif str(symbol)[-4:] == '(tF)':
            msg = (f"The user defined symbol {symbol} is invalid as it is named "
                   f"with the suffix '(tF)' which is reserved for use by "
                   f"`Pycollo`. Please rename this symbol.")
            raise ValueError(msg)

    if len(set(syms)) != len(syms):
        msg = ("All user defined symbols must have unique names.")
        raise ValueError(msg)


def dict_merge(*dicts: Iterable[Mapping]) -> dict:
    """Merges multiple dictionaries into a single dictionary."""
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def console_out(msg, heading=False, subheading=False, prefix="", suffix="", *,
                trailing_blank_line=False):
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
    if trailing_blank_line:
        output_msg += "\n"
    print(output_msg)


def fast_sympify(arg):
    if not isinstance(arg, sm.Expr):
        return sm.sympify(arg)
    return arg


def parse_arg_type(arg, arg_name_str, arg_type):
    if not isinstance(arg, arg_type):
        msg = (
            f"`{arg_name_str}` must be a {arg_type}. {arg} of type {type(arg)} "
            f"is not a valid argument."
        )
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
            msg = (
                f"If an iterable of values in being passed for `{var_name_str}` "
                f"then this must be of length 2. {var} of length {len(var)} is "
                f"not a valid argument."
            )
            raise TypeError(msg)
        if not isinstance(var[0], var_type) or not isinstance(var[1], var_type):
            msg = (
                f"Both items in `{var_name_str}` must be {var_type} objects. "
                f"{var[0]} at index 0 is of type {type(var[0])} and {var[1]} at "
                f"index 1 is of type {type(var[1])}."
            )
            raise TypeError(msg)
        return tuple(var)


def format_case(item, case):
    """Allow :obj:`str` case formatting method application from keyword.

    Parameters
    ----------
    item : str
        Item to be case formatted.
    case : str
        Which case format method to use.

    Returns
    -------
    str
        :arg:`item` with case method applied.
    """
    if case == "title":
        return item.title()
    elif case == "upper":
        return item.upper()
    elif case == "lower":
        return item.lower()
    else:
        return item


def format_for_output(items, *args, **kwargs):
    """Utility method for formatting console output.

    Passes directly to :func:`format_multiple_items_for_output` just with a
    shorter function name.

    Parameters
    ----------
    items : iterable
        Items to be formatted for output.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    str
        Formatted string for console output.
    """
    return format_multiple_items_for_output(items, *args, **kwargs)


def format_multiple_items_for_output(items, wrapping_char="'", *,
                                     prefix_char="", case=None,
                                     with_verb=False, with_or=False,
                                     with_preposition=False):
    """Format multiple items for pretty console output.

    Args
    ----
    items : iterable of str
        Items to be formatted.
    wrapping_char : str (default `"'"`)
        Prefix and suffix character for format wrapping.
    prefix_char : str (default `""`)
        Additional prefix.
    case : str (default `None`)
        Keyword for :func:`format_case`.
    with_verb : bool, optional (default `False`)
        Append the correct conjugation of "is"/"are" to end of list.

    Returns
    -------
    str
        Formatted string of multiple items for console output.

    """
    items = (items, ) if isinstance(items, str) else items
    items = [f"{prefix_char}{format_case(item, case)}" for item in items]
    if len(items) == 1:
        formatted_items = f"{wrapping_char}{items[0]}{wrapping_char}"
        if with_preposition and wrapping_char == "":
            first_word, _ = formatted_items.split(maxsplit=1)
            starts_with_vowel = first_word[0] in {"a", "e", "h", "i", "o", "u"}
            is_acronym = first_word.upper() == first_word
            if starts_with_vowel or is_acronym:
                preposition = "an"
            else:
                preposition = "a"
            formatted_items = " ".join([preposition, formatted_items])
    else:
        pad = f"{wrapping_char}, {wrapping_char}"
        joiner = "or" if with_or else "and"
        formatted_items = (f"{wrapping_char}{pad.join(items[:-1])}"
                           f"{wrapping_char} {joiner} {wrapping_char}"
                           f"{items[-1]}{wrapping_char}")
    verb = "is" if len(items) == 1 else "are"
    if with_verb:
        formatted_items = f"{formatted_items} {verb}"

    return formatted_items


def format_time(time):
    """Nicely format a time for console output with correct units.

    Args
    ----
    time : float
        Time (in seconds) for formatting.

    Returns
    -------
    str
        Time formatted with units.

    """

    if time >= 1.0:
        if time < 60:
            return f"{time:.2f}s"
        elif time < 3600:
            return f"{time//60}min {time%60:.2f}s"
        else:
            return f"{time//3600}h {time//60}min {time%60:.2f}s"
    else:
        prefixes = ("m", "u", "n", "p")
        time_formatted = time
        for prefix in prefixes:
            time_formatted = time_formatted * 1000
            if time_formatted > 1:
                return f"{time_formatted:.2f}{prefix}s"
        msg = f"Insufficient time prefixes for {time}s"
        raise ValueError(msg)
