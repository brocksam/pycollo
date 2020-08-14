"""Convenience types for PEP484-style type annotations for use with Pycollo.

This module provides a number of custom type descriptions that can be imported 
by other modules within Pycollo to add PEP484-style type annotations to all 
functions, classes, methods, etc.

The `typing` module is not exposed to the user, i.e. it is not importable as 
part of Pycollo.
"""


from collections import namedtuple
from typing import (Iterable, NamedTuple, Optional, Tuple, Union)

import sympy as sym

SympyType = Union[sym.Symbol, sym.Expr]
"""For Sympy symbols and expressions."""

OptionalBoundsType = Optional[Iterable[Union[None, float, Iterable[float]]]]
"""For user-supplied numerical bounds."""

OptionalExprsType = Union[None, SympyType, Iterable[SympyType]]
"""For user-supplied symbols/equations for functions."""

OptionalSymsType = Union[None, sym.Symbol, Iterable[sym.Symbol]]
"""For user-supplied symbols for variables."""

TupleSymsType = Union[Tuple[sym.Symbol, ...], NamedTuple]
"""For return values of varible properties."""