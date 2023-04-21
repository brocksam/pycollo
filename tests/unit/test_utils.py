"""Tests for the utils module."""

import casadi as ca
import pytest
import sympy as sm
import sympy.physics.mechanics as me

from pycollo.utils import create_data_container, sympy_to_casadi


@pytest.mark.parametrize("in_mapping", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_sympy_to_casadi(in_mapping, utils):
    """Test utility function for converting Sympy expressions to CasADi ones.

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
    x_sympy = sm.Symbol("x")
    y_sympy = sm.Symbol("y")
    x_casadi = ca.SX.sym("x")
    y_casadi = ca.SX.sym("y")
    sympy_vars = [x_sympy, y_sympy]
    casadi_vars = [x_casadi, y_casadi]
    sym_mapping = {var_sym: var_ca
                       for var_sym, var_ca, use in zip(sympy_vars, casadi_vars, in_mapping)
                       if use}

    expr_sym = sm.Matrix([x_sympy * sm.sqrt(y_sympy),
                           sm.sin(x_sympy + y_sympy),
                           abs(x_sympy - y_sympy)])
    expr_ca, sym_mapping = sympy_to_casadi(expr_sym, sym_mapping)

    assert str(expr_sym) == "Matrix([[x*sqrt(y)], [sin(x + y)], [Abs(x - y)]])"
    assert str(expr_ca) == "[(x*sqrt(y)), sin((x+y)), fabs((x-y))]"
    utils.assert_ca_expr_identical(expr_ca, expr_sym)


class TestDataContainerSequence:
    """Tests for data containers that contain sequence data."""

    @pytest.fixture(autouse=True)
    def _data_container_fixture(self) -> None:
        """Simple fixture setting up a sequence data container."""
        self.field_names = ["x", "q", "dq", "ddq"]
        self.DataContainer = create_data_container(
            "StateVariables",
            self.field_names,
        )

        self.x = sm.Symbol("x")
        self.q = me.dynamicsymbols("q")
        self.dq = me.dynamicsymbols("q", 1)
        self.ddq = me.dynamicsymbols("q", 2)
        self.data = self.DataContainer(self.x, self.q, self.dq, self.ddq)

    def test_data_container_class_name(self) -> None:
        """Data container's class name is set correctly."""
        assert self.data.__class__.__name__ == "StateVariables"

    def test_data_container_str(self) -> None:
        """Data container implements __str__ method formats correctly."""
        expected_str = "(x, q(t), Derivative(q(t), t), Derivative(q(t), (t, 2)))"
        assert str(self.data) == expected_str

    def test_data_container_repr(self) -> None:
        """Data container implements __repr__ method formats correctly."""
        expected_repr = (
            "StateVariables(x=x, q=q(t), dq=Derivative(q(t), t), "
            "ddq=Derivative(q(t), (t, 2)))"
        )
        assert repr(self.data) == expected_repr

    def test_data_container_len(self) -> None:
        """Data container implements __len__ method correctly."""
        assert len(self.data) == 4

    def test_data_container_init(self) -> None:
        """Data container implements __init__ method correctly."""
        expected = (self.x, self.q, self.dq, self.ddq)
        assert self.data._keys == expected
        assert self.data._values == expected

    def test_data_container_iter(self) -> None:
        """Data container implements __iter__ method correctly."""
        for d, expected in zip(self.data, (self.x, self.q, self.dq, self.ddq)):
            assert d == expected

    def test_data_container_integer_indexing(self) -> None:
        """Square bracket integer indexing is supported."""
        assert self.data[0] == self.x
        assert self.data[1] == self.q
        assert self.data[2] == self.dq
        assert self.data[3] == self.ddq

    def test_data_container_value_indexing(self) -> None:
        """Square bracket value indexing is supported."""
        assert self.data[self.x] == self.x
        assert self.data[self.q] == self.q
        assert self.data[self.dq] == self.dq
        assert self.data[self.ddq] == self.ddq

    def test_data_container_attribute_access(self) -> None:
        """Dot attribute access is supported."""
        assert self.data.x == self.x
        assert self.data.q == self.q
        assert self.data.dq == self.dq
        assert self.data.ddq == self.ddq


class TestDataContainerMapping:
    """Tests for data containers that contain mapping data."""

    @pytest.fixture(autouse=True)
    def _data_container_fixture(self) -> None:
        """Simple fixture setting up a mapping data container."""
        self.field_names = ["x", "q", "dq", "ddq"]
        self.DataContainer = create_data_container(
            "StateEquations",
            self.field_names,
        )

        self.x = sm.Symbol("x")
        self.y = sm.Symbol("y")
        self.z = sm.Symbol("z")
        self.q = me.dynamicsymbols("q")
        self.dq = me.dynamicsymbols("q", 1)
        self.ddq = me.dynamicsymbols("q", 2)
        self.u = me.dynamicsymbols("u")
        self.state_equations = {
            self.x: self.y - self.z,
            self.q: self.dq,
            self.dq: self.ddq,
            self.ddq: self.u + self.y,
        }
        self.data = self.DataContainer(self.state_equations)

    def test_data_container_class_name(self) -> None:
        """Data container's class name is set correctly."""
        assert self.data.__class__.__name__ == "StateEquations"

    def test_data_container_str(self) -> None:
        """Data container implements __str__ method formats correctly."""
        expected_str = (
            "{x: y - z, q(t): Derivative(q(t), t), "
            "Derivative(q(t), t): Derivative(q(t), (t, 2)), "
            "Derivative(q(t), (t, 2)): y + u(t)}"
        )
        assert str(self.data) == expected_str

    def test_data_container_repr(self) -> None:
        """Data container implements __repr__ method formats correctly."""
        expected_repr = (
            "StateEquations(x={x: y - z}, q={q(t): Derivative(q(t), t)}, "
            "dq={Derivative(q(t), t): Derivative(q(t), (t, 2))}, "
            "ddq={Derivative(q(t), (t, 2)): y + u(t)})"
        )
        assert repr(self.data) == expected_repr

    def test_data_container_len(self) -> None:
        """Data container implements __len__ method correctly."""
        assert len(self.data) == 4

    def test_data_container_init(self) -> None:
        """Data container implements __init__ method correctly."""
        assert self.data._keys == tuple(self.state_equations.keys())
        assert self.data._values == tuple(self.state_equations.values())

    def test_data_container_iter(self) -> None:
        """Data container implements __iter__ method correctly."""
        for d, expected in zip(self.data, self.state_equations.values()):
            assert d == expected

    def test_data_container_integer_indexing(self) -> None:
        """Square bracket integer indexing is supported."""
        assert self.data[0] == self.y - self.z
        assert self.data[1] == self.dq
        assert self.data[2] == self.ddq
        assert self.data[3] == self.u + self.y

    def test_data_container_value_indexing(self) -> None:
        """Square bracket value indexing is supported."""
        assert self.data[self.x] == self.y - self.z
        assert self.data[self.q] == self.dq
        assert self.data[self.dq] == self.ddq
        assert self.data[self.ddq] == self.u + self.y

    def test_data_container_attribute_access(self) -> None:
        """Dot attribute access is supported."""
        assert self.data.x == self.y - self.z
        assert self.data.q == self.dq
        assert self.data.dq == self.ddq
        assert self.data.ddq == self.u + self.y
