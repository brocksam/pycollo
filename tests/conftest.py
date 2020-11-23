"""Generic pytest-related classes/functions used throughout test suite."""

import itertools

import casadi as ca
import numpy as np
import pytest
import sympy as sym


class Utils:
    """Container for utility functions for use within test suite.

    Similar to approach as defined in:
    https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di

    """

    @staticmethod
    def assert_ca_sym_identical(*syms):
        """Assert if a number of CasADi SX symbols appear identical."""
        if len(syms) > 2:
            msg = ("Two or more symbols are required for asserting whether "
                   "they're identical.")
            raise ValueError(msg)
        for sym in syms[1:]:
            assert syms[0].name() == sym.name()
            assert syms[0].size() == sym.size()

    @classmethod
    def assert_ca_syms_identical(cls, *syms_iterables):
        """Check all symbols in iterables using `assert_ca_sym_identical`."""
        for syms in itertools.zip_longest(*syms_iterables):
            cls.assert_ca_sym_identical(*syms)

    @classmethod
    def assert_ca_expr_identical(cls, *exprs, n=10):
        """Assert if a number of CasADi SX expressions appear identical.

        Due to complexities with equating expressions from different systems
        (e.g. comparing CasADi to Sympy, or comparing two CasADi expressions
        that have been build using primitives that appear the same but are not
        the same object), expressions equality is checked by compiling a
        numerical function for both expressions and evaluating these at a
        number of different randomly parametrised points.

        This approach likely falls down in scenarios where the numerical
        computation introduces `NaN` or `Inf`. If a test fails within this
        functionthen this is what should probably be investigated first.

        """
        if len(exprs) > 2:
            msg = ("Two or more expressions are required for asserting "
                   "whether they're identical.")
            raise ValueError(msg)
        base_prims, base_names = cls.get_primitives_and_names(exprs[0])
        num_args = len(base_prims)
        base_func = cls.lambdify_expr(exprs[0], base_prims, base=True)
        for expr in exprs[1:]:
            other_prims, other_names = cls.get_primitives_and_names(expr)
            indices = [base_names.index(name) for name in other_names]
            other_func = cls.lambdify_expr(expr, other_prims)
            for _ in range(n):
                data = np.random.random(num_args)
                base_eval = base_func(*data)
                other_eval = other_func(*data[indices])
                np.testing.assert_almost_equal(base_eval, other_eval)

    @classmethod
    def assert_ca_exprs_identical(cls, *exprs_iterables, n=10):
        """Check all exprs in iterables using `assert_ca_expr_identical`."""
        for exprs in itertools.zip_longest(*exprs_iterables):
            cls.assert_ca_expr_identical(*exprs, n=n)

    @staticmethod
    def get_primitives_and_names(expr):
        """Return correct expr primitives depending on expr type."""
        if isinstance(expr, ca.SX):
            prims = ca.symvar(expr)
            names = [prim.name() for prim in prims]
        elif isinstance(expr, (sym.Expr, sym.Matrix)):
            prims = list(expr.free_symbols)
            names = [str(prim) for prim in prims]
        elif isinstance(expr, (float, int, ca.DM)):
            prims = []
            names = []
        else:
            msg = (f"Unexpected expression type of {type(expr)}. Expecting "
                   f"CasADi or Sympy.")
            raise TypeError(msg)
        return (prims, names)

    @staticmethod
    def lambdify_expr(expr, prims, base=False):
        """Compile a function for numerically computing the expression.

        Supports numerical computation of CasADi and Sympy expressions. Sympy
        expressions are compiled using Numpy as the numerical backend.

        """
        if isinstance(expr, ca.SX):
            func_name = "base_func" if base else "other_func"
            return ca.Function("base_func", prims, [expr])
        elif isinstance(expr, (sym.Expr, sym.Matrix)):
            return sym.lambdify(prims, expr, "numpy")
        elif isinstance(expr, (float, int)):
            return lambda: expr
        elif isinstance(expr, ca.DM):
            return lambda: expr[0]
        msg = (f"Unexpected expression type of {type(expr)}. Expecting CasADi "
               f"or Sympy.")
        raise TypeError(msg)


@pytest.fixture
def utils():
    """Fixture for utils helper.

    Related to Utils class above.

    """
    return Utils


# Store history of failures per test class name and per index in parametrize
# (if parametrize used)
_test_failed_incremental = {}


def pytest_runtest_makereport(item, call):
    """Allow pytest to run tests as incremental.

    Sucessive, related tests are marked as xfailed if a previous test fails.
    Taken from "https://docs.pytest.org/en/latest/example/simple.html"

    """
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed
            # test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )


def pytest_runtest_setup(item):
    """Allow pytest to run tests as incremental.

    Sucessive, related tests are marked as xfailed if a previous test fails.
    Taken from "https://docs.pytest.org/en/latest/example/simple.html"

    """
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in
            # combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this
            # class name and index
            test_class = _test_failed_incremental[cls_name]
            test_name = test_class.get(parametrize_index, None)
            # if name found, test has failed for the combination of class name
            # & test name
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))
