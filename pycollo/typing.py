from typing import ()


"""Convenience types for PEP484-style type annotations for use with Pycollo.

This module provides a number of custom type descriptions that can be imported 
by other modules within Pycollo to add PEP484-style type annotations to all 
functions, classes, methods, etc.

The `typing` module is not exposed to the user, i.e. it is not importable as 
part of Pycollo.
"""


OptionalSymsType = Union[None, sym.Symbol, Iterable[sym.Symbol]]
"""For user-supplied symbols for variables."""

TupleSymsType = Tuple[sym.Symbol, ...]
"""For return values of varible properties."""