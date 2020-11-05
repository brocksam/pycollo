"""Tests for Pycollo, OCP and NLP settings."""

import re

from hypothesis import assume, given
import hypothesis.strategies as st
import pytest

import pycollo


@pytest.mark.usefixtures("_ocp_fixture")
class TestSettings:

    @pytest.fixture(autouse=True)
    def _ocp_fixture(self):
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.ocp = pycollo.OptimalControlProblem("Test OCP")
        self.settings = self.ocp.settings

    def test_ocp_settings_attr_type(self):
        """OCP settings should be a :obj:`Settings` object."""
        assert isinstance(self.settings, pycollo.Settings)

    def test_settings_ocp_attr_is_set(self):
        """Creation of :obj:`OptimalControlProblem` should link
        :obj:`Settings`.
        """
        assert self.settings.optimal_control_problem is self.ocp

    def test_default_backend(self):
        """Default backend defaults."""
        assert self.settings.backend == "casadi"
        assert self.settings.derivative_level == 2

    def test_solver_defaults(self):
        """Default NLP and linear solver settings."""
        assert self.settings.nlp_solver == "ipopt"
        assert self.settings.linear_solver == "mumps"
        assert self.settings.nlp_tolerance == 1e-10
        assert self.settings.max_nlp_iterations == 2000

    def test_default_collocation_matrix_form(self):
        """Default collocation matrix form should be integral."""
        assert self.settings.collocation_matrix_form == "integral"

    def test_quadrature_defaults(self):
        """Default quadrature settings."""
        assert self.settings.quadrature_method == "lobatto"
        assert self.settings.collocation_points_min == 4
        assert self.settings.collocation_points_max == 10

    def test_mesh_refinement_defaults(self):
        """Default mesh refinement settings."""
        assert self.settings.max_mesh_iterations == 10
        assert self.settings.mesh_tolerance == 1e-7

    def test_display_defaults(self):
        """Defaults for console output and plotting during/after solve."""
        assert self.settings.display_mesh_refinement_info is True
        assert self.settings.display_mesh_result_info is False
        assert self.settings.display_mesh_result_graph is False

    def test_scaling_defaults(self):
        """Defaults for problem scaling."""
        assert self.settings.scaling_method == "bounds"
        assert self.settings.update_scaling is False
        assert self.settings.number_scaling_samples == 0
        assert self.settings.scaling_weight == 0.8

    @given(st.just("casadi"))
    def test_backend_property_valid(self, test_value):
        """`'casadi'` is the only supported backend."""
        self.settings.backend = test_value
        assert self.settings.backend == test_value

    @given(st.text())
    def test_backend_property_invalid(self, test_value):
        """ValueErrors should be raised for invalid values of backend."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of Pycollo "
            f"backend (`backend`). Choose one of: `'casadi'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.backend = test_value

    @given(st.one_of(st.just("hsad"), st.just("pycollo"), st.just("sympy")))
    def test_backend_property_invalid(self, test_value):
        """ValueErrors should be raised for unsupported values of backend."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a "
            f"Pycollo backend (`backend`). Choose one of: `'casadi'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.backend = test_value

    @given(st.just("ipopt"))
    def test_nlp_solver_property_valid(self, test_value):
        """`'ipopt'` is the only supported NLP solver."""
        self.settings.nlp_solver = test_value
        assert self.settings.nlp_solver == test_value

    @given(st.text())
    def test_nlp_solver_property_invalid(self, test_value):
        """ValueErrors should be raised for invalid values of NLP solver."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of NLP "
            f"solver (`nlp_solver`). Choose one of: `'ipopt'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = test_value

    @given(st.one_of(st.just("snopt"), st.just("worhp"), st.just("bonmin")))
    def test_nlp_solver_property_unsupported(self, test_value):
        """ValueErrors should be raised for unsupported NLP solvers."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a "
            f"NLP solver (`nlp_solver`). Choose one of: `'ipopt'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = test_value

    @given(st.just("mumps"))
    def test_linear_solver_property_valid(self, test_value):
        """`'mumps'` is the only supported linear solver."""
        self.settings.linear_solver = test_value
        assert self.settings.linear_solver == test_value

    @given(st.text())
    def test_linear_solver_property_invalid(self, test_value):
        """ValueErrors should be raised for invalid linear solver."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of linear "
            f"solver (`linear_solver`). Choose one of: `'mumps'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.linear_solver = test_value

    @given(st.just("ma57"))
    def test_linear_solver_property_unsupported(self, test_value):
        """ValueErrors should be raised for unsupported linear solvers."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a "
            f"linear solver (`linear_solver`). Choose one of: `'mumps'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.linear_solver = test_value

    @given(st.just(1e-6))
    def test_nlp_tolerance_property_valid(self, test_value):
        """Valid if <1.0 or >=0.0."""
        self.settings.nlp_tolerance = test_value
        assert self.settings.nlp_tolerance == test_value

    def test_nlp_tolerance_property(self):
        """ValueError if >1.0 or <=0.0."""
        expected_error_msg = re.escape(
            "NLP tolerance (`nlp_tolerance`) must be less than `1.0`. `1.1` "
            "is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_tolerance = 1.1
        expected_error_msg = re.escape(
            "NLP tolerance (`nlp_tolerance`) must be greater than `0.0`. "
            "`0.0` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_tolerance = 0.0

    @given(st.integers())
    def test_max_nlp_iterations_property(self, test_value):
        """ValueError if <1."""
        if test_value > 0:
            self.settings.max_nlp_iterations = test_value
            assert self.settings.max_nlp_iterations == int(test_value)
        else:
            expected_error_msg = re.escape(
                f"Maximum number of NLP iterations (`max_nlp_iterations`) "
                f"must be greater than or equal to `1`. `{test_value}` "
                f"is invalid.")
            with pytest.raises(ValueError, match=expected_error_msg):
                self.settings.max_nlp_iterations = test_value

    @given(st.one_of(st.just(1), st.just(2)))
    def test_derivative_level_property_correct(self, test_value):
        """Only value of 1 or 2 should be accepted without error."""
        self.settings.derivative_level = test_value
        assert self.settings.derivative_level == test_value

    @given(st.one_of(st.integers()))
    def test_derivative_level_property_error(self, test_value):
        """Value other than 1 or 2 should raise ValueError."""
        assume(test_value not in {1, 2})
        with pytest.raises(ValueError):
            self.settings.derivative_level = test_value

    @given(st.just("integral"))
    def test_supported_collocation_matrix_form(self, test_value):
        """Integral only supported collocation matrix forms."""
        self.settings.collocation_matrix_form = test_value
        assert self.settings.collocation_matrix_form == test_value

    @given(st.just("differential"))
    def test_supported_collocation_matrix_form(self, test_value):
        """Differential is unsupported collocation matrix forms."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a form of "
            f"the collocation matrices (`collocation_matrix_form`). Choose "
            f"one of: `'integral'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_matrix_form = test_value

    @given(st.text())
    def test_invalid_collocation_matrix_form(self, test_value):
        """Invalid collocation matrix forms raise ValueError."""
        assume(test_value not in {"integral", "differential"})
        regex_wildcard = r"[\s\S]*"
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of form of the "
            f"collocation matrices (`collocation_matrix_form`). Choose one "
            f"of: `'integral'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_matrix_form = test_value

    @given(st.one_of(st.just("lobatto"), st.just("radau")))
    def test_supported_quadrature_methods(self, test_value):
        """Lobatto and Radau are only valid quadrature methods."""
        self.settings.quadrature_method = test_value
        assert self.settings.quadrature_method == test_value

    @given(st.just("gauss"))
    def test_unsupported_quadrature_methods(self, test_value):
        """Gauss not supported quadrature method."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a quadrature "
            f"method (`quadrature_method`). Choose one of: `'lobatto'` or "
            f"`'radau'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.quadrature_method = test_value

    @given(st.text())
    def test_invalid_quadrature_method(self, test_value):
        """String other than 'radau'/'lobatto'/'gauss' raises ValueError."""
        assume(test_value not in {"radau", "lobatto", "gauss"})
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of quadrature method "
            f"(`quadrature_method`). Choose one of: `'lobatto'` or `'radau'`."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.quadrature_method = test_value

    @given(st.one_of(st.integers(min_value=2, max_value=20),
                     st.floats(min_value=2.0, max_value=20.0)))
    def test_valid_min_max_collocation_points(self, test_value):
        """Integers between 2 and 20 can be set provided min < max."""
        self.settings.collocation_points_min = 2
        self.settings.collocation_points_max = 20
        self.settings.collocation_points_min = test_value
        self.settings.collocation_points_max = test_value
        assert self.settings.collocation_points_min == int(test_value)
        assert self.settings.collocation_points_max == int(test_value)

    @given(st.integers(max_value=1))
    def test_too_small_min_max_collocation_points(self, test_value):
        """Integers <2 raise a ValueError."""
        expected_error_msg = re.escape(
            f"Minimum number of collocation points per mesh section "
            f"(`collocation_points_min`) must be greater than or equal to "
            f"`2`. `{repr(test_value)}` is invalid."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_min = test_value
        expected_error_msg = re.escape(
            f"Maximum number of collocation points per mesh section "
            f"(`collocation_points_max`) must be greater than or equal to "
            f"`2`. `{repr(test_value)}` is invalid."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_max = test_value

    @given(st.integers(min_value=21))
    def test_too_large_min_max_collocation_points(self, test_value):
        """Integers >20 raise a ValueError."""
        expected_error_msg = re.escape(
            f"Minimum number of collocation points per mesh section "
            f"(`collocation_points_min`) must be less than or equal to `20`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_min = test_value
        expected_error_msg = re.escape(
            f"Maximum number of collocation points per mesh section "
            f"(`collocation_points_max`) must be less than or equal to `20`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_max = test_value

    @given(st.integers(min_value=3, max_value=19))
    def test_min_greater_than_max_visa_versa(self, test_value):
        """min > max or max < min raises ValueError."""
        self.settings.collocation_points_min = test_value
        self.settings.collocation_points_max = test_value
        expected_error_msg = re.escape(
            f"Minimum number of collocation points per mesh section "
            f"(`collocation_points_min`) with value `{test_value + 1}` must "
            f"be at most maximum number of collocation points per mesh "
            f"section (`collocation_points_max`) with value `{test_value}`.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_min = test_value + 1
        expected_error_msg = re.escape(
            f"Maximum number of collocation points per mesh section "
            f"(`collocation_points_max`) with value `{test_value - 1}` must "
            f"be at least minimum number of collocation points per mesh "
            f"section (`collocation_points_min`) with value `{test_value}`.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.collocation_points_max = test_value - 1
        self.settings.collocation_points_min = 2
        self.settings.collocation_points_max = 20

    @given(st.floats(min_value=0, max_value=1, exclude_min=True,
                     exclude_max=True))
    def test_valid_mesh_tolerance(self, test_value):
        """Check mesh tolerance value is set correctly."""
        self.settings.mesh_tolerance = test_value
        assert self.settings.mesh_tolerance == test_value

    @given(st.floats(max_value=0))
    def test_too_small_mesh_tolerance(self, test_value):
        """Mesh tolerance <0 raises ValueError."""
        expected_error_msg = re.escape(
            f"Mesh tolerance (`mesh_tolerance`) must be greater than `0.0`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.mesh_tolerance = test_value

    @given(st.floats(min_value=1))
    def test_too_large_mesh_tolerance(self, test_value):
        """Mesh tolerance >1 raises ValueError."""
        expected_error_msg = re.escape(
            f"Mesh tolerance (`mesh_tolerance`) must be less than `1.0`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.mesh_tolerance = test_value

    @given(st.integers())
    def test_max_mesh_iterations_property(self, test_value):
        """ValueError if <1."""
        if test_value > 0:
            self.settings.max_mesh_iterations = test_value
            assert self.settings.max_mesh_iterations == int(test_value)
        else:
            expected_error_msg = re.escape(
                f"Maximum number of mesh iterations (`max_mesh_iterations`) "
                f"must be greater than or equal to `1`. `{repr(test_value)}` "
                f"is invalid.")
            with pytest.raises(ValueError, match=expected_error_msg):
                self.settings.max_mesh_iterations = test_value

    @given(st.one_of(st.just(None), st.just("none"), st.just("bounds")))
    def test_supported_scaling_methods(self, test_value):
        """None and bounds are only valid scaling methods."""
        self.settings.scaling_method = test_value
        assert self.settings.scaling_method == test_value

    @given(st.one_of(st.just("user"), st.just("guess")))
    def test_unsupported_scaling_methods(self, test_value):
        """'user' and 'guess' not supported scaling methods."""
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not currently supported as a scaling "
            f"method (`scaling_method`). Choose one of: `'bounds'`, "
            f"`'none'` or `None`.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.scaling_method = test_value

    @given(st.text())
    def test_invalid_quadrature_method(self, test_value):
        """String other than 'radau'/'lobatto'/'gauss' raises ValueError."""
        assume(test_value not in {"radau", "lobatto", "gauss"})
        expected_error_msg = re.escape(
            f"`{repr(test_value)}` is not a valid option of quadrature method "
            f"(`quadrature_method`). Choose one of: `'lobatto'` or `'radau'`.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.quadrature_method = test_value

    @given(st.booleans())
    def test_valid_update_scaling(self, test_value):
        """Sets correctly to a bool."""
        self.settings.update_scaling = test_value
        assert self.settings.update_scaling == test_value

    @given(st.integers())
    def test_number_scaling_samples_property(self, test_value):
        """ValueError if <0."""
        if test_value >= 0:
            self.settings.number_scaling_samples = test_value
            assert self.settings.number_scaling_samples == int(test_value)
        else:
            expected_error_msg = re.escape(
                f"Number of samples taken when computing scaling factors "
                f"(`number_scaling_samples`) must be greater than or equal to "
                f"`0`. `{repr(test_value)}` is invalid.")
            with pytest.raises(ValueError, match=expected_error_msg):
                self.settings.number_scaling_samples = test_value

    @given(st.floats(min_value=0, max_value=1, exclude_min=True,
                     exclude_max=True))
    def test_valid_scaling_weight(self, test_value):
        """Check mesh tolerance value is set correctly."""
        self.settings.scaling_weight = test_value
        assert self.settings.scaling_weight == test_value

    @given(st.floats(max_value=0))
    def test_too_small_scaling_weight(self, test_value):
        """Scaling weight <0 raises ValueError."""
        expected_error_msg = re.escape(
            f"Inter-mesh scaling adjustment weighting factor "
            f"(`scaling_weight`) must be greater than `0.0`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.scaling_weight = test_value

    @given(st.floats(min_value=1))
    def test_too_large_scaling_weight(self, test_value):
        """Scaling weight >1 raises ValueError."""
        expected_error_msg = re.escape(
            f"Inter-mesh scaling adjustment weighting factor "
            f"(`scaling_weight`) must be less than `1.0`. "
            f"`{repr(test_value)}` is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.scaling_weight = test_value
