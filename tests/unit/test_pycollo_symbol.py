"""Tests for the :class:`Symbol` class."""

import pytest
import sympy as sym

import pycollo


def test_pycollo_symbol_instantiation_single():
    """Pycollo's native symbol class can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    assert isinstance(x, pycollo.Symbol)
    assert isinstance(x, sym.Symbol)


@pytest.mark.parametrize("names, expected_syms",
                         [("x y", ["x", "y"]), ("x y", ["x", "y"])])
def test_pycollo_symbol_instantiation_multiple(names, expected_syms):
    """Multiple Pycollo Symbols can be instantiated by function call."""
    pycollo_syms = pycollo.symbols(names)
    for pycollo_sym, expected_sym in zip(pycollo_syms, expected_syms):
        assert pycollo_sym == pycollo.Symbol(expected_sym)
        assert isinstance(pycollo_sym, pycollo.Symbol)
        assert isinstance(pycollo_sym, sym.Symbol)


def test_pycollo_matrix_instantiation():
    """Pycollo's native matrix class can be instantiated sucessfully."""
    M00, M01, M10, M11 = pycollo.symbols("M00 M01 M10 M11")
    M = pycollo.Matrix([[M00, M01], [M10, M11]])
    assert isinstance(M, pycollo.Matrix)
    assert isinstance(M, sym.Matrix)


def test_pycollo_identity_matrix_instantiation():
    """Instantiating identity matrices as native Pycollo objects."""
    I = pycollo.eye(3)
    expected_structure = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert I == pycollo.Matrix(expected_structure)
    assert I == sym.Matrix(expected_structure)


@pytest.mark.parametrize("expected", [sym.Pow(sym.Symbol("x"), 2)])
def test_pycollo_pow_class(expected):
    """Pycollo-native power symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    pow_x_2 = pycollo.Pow(x, 2)
    assert all(pow_x_2.args) == all(expected.args)
    assert str(pow_x_2) == str(expected)


@pytest.mark.parametrize("expected", [sym.sqrt(sym.Symbol("x"))])
def test_pycollo_sqrt_function(expected):
    """Pycollo-native sqrt symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    sqrt_x = pycollo.sqrt(x)
    assert all(sqrt_x.args) == all(expected.args)
    assert str(sqrt_x) == str(expected)


@pytest.mark.parametrize("expected", [sym.exp(sym.Symbol("x"))])
def test_pycollo_exp_class(expected):
    """Pycollo-native exponential symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    exp_x = pycollo.exp(x)
    assert all(exp_x.args) == all(expected.args)
    assert str(exp_x) == str(expected)


@pytest.mark.parametrize("expected", [sym.log(sym.Symbol("x"))])
def test_pycollo_log_class(expected):
    """Pycollo-native log symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    log_x = pycollo.log(x)
    assert all(log_x.args) == all(expected.args)
    assert str(log_x) == str(expected)


@pytest.mark.parametrize("expected", [sym.sin(sym.Symbol("x"))])
def test_pycollo_sin_class(expected):
    """Pycollo-native sin symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    sin_x = pycollo.sin(x)
    assert all(sin_x.args) == all(expected.args)
    assert str(sin_x) == str(expected)


@pytest.mark.parametrize("expected", [sym.cos(sym.Symbol("x"))])
def test_pycollo_cos_class(expected):
    """Pycollo-native cos symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    cos_x = pycollo.cos(x)
    assert all(cos_x.args) == all(expected.args)
    assert str(cos_x) == str(expected)


@pytest.mark.parametrize("expected", [sym.tan(sym.Symbol("x"))])
def test_pycollo_tan_class(expected):
    """Pycollo-native tan symbols can be instantiated sucessfully."""
    x = pycollo.Symbol("x")
    tan_x = pycollo.tan(x)
    assert all(tan_x.args) == all(expected.args)
    assert str(tan_x) == str(expected)
