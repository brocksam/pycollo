"""Module for Pycollo's native symbol object and associated functions."""


import sympy as sym


__all__ = [
    "Symbol",
    "symbols",
    "Matrix",
    "eye",
    "Add",
    "Mul",
    "Pow",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
]


class Symbol(sym.Symbol):
    """Pycollo-native subclass of Sympy's Symbol class."""
    pass


class Matrix(sym.Matrix):
    """Pycollo-native subclass of Sympy's Matrix class."""
    pass


def symbols(*args, **kwargs):
    """Overload Sympy's multiple symbol instantiation with Pycollo's subclass."""
    return sym.symbols(*args, cls=Symbol, **kwargs)


def eye(*args, **kwargs):
    """Overload of Sympy's identity matrix function with Pycollo's subclass."""
    return Matrix.eye(*args, **kwargs)


class Add(sym.Add):
    """Pycollo-native subclass of Sympy's Add class."""
    pass


class Mul(sym.Mul):
    """Pycollo-native subclass of Sympy's Mul class."""
    pass  


class Pow(sym.Pow):
    """Pycollo-native subclass of Sympy's Pow class."""
    pass


def sqrt(*args, **kwargs):
    """Overload of Sympy's sqrt function with Pycollo-native function."""
    return sym.sqrt(*args, **kwargs)


class exp(sym.exp):
    """Pycollo-native subclass of Sympy's exp class."""
    pass


class log(sym.log):
    """Pycollo-native subclass of Sympy's log class."""
    pass


class sin(sym.sin):
    """Pycollo-native subclass of Sympy's sin class."""
    pass


class cos(sym.cos):
    """Pycollo-native subclass of Sympy's cos class."""
    pass


class tan(sym.tan):
    """Pycollo-native subclass of Sympy's tan class."""
    pass
