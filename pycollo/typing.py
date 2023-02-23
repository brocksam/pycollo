"""Convenience types for PEP484-style type annotations for use with Pycollo.

This module provides a number of custom type descriptions that can be imported 
by other modules within Pycollo to add PEP484-style type annotations to all 
functions, classes, methods, etc.

The `typing` module is not exposed to the user, i.e. it is not importable as 
part of Pycollo.
"""


from typing import Iterable, NamedTuple, Optional, Tuple, Union

import sympy as sm
from sympy.core.backend import AppliedUndef

SympyType = Union[sm.Symbol, sm.Expr]
"""For Sympy symbols and expressions."""

OptionalBoundsType = Optional[Iterable[Union[None, float, Iterable[float]]]]
"""For user-supplied numerical bounds."""

OptionalExprsType = Union[None, SympyType, Iterable[SympyType]]
"""For user-supplied symbols/equations for functions."""

OptionalSymsType = Union[
    None,
    sm.Symbol,
    Iterable[sm.Symbol],
    AppliedUndef,  # for sm.physics.mechanics.dynamicsymbols
    Iterable[AppliedUndef],
]
"""For user-supplied symbols for variables."""

TupleSymsType = Union[Tuple[sm.Symbol, ...], NamedTuple]
"""For return values of varible properties."""
