"""Tests for Pycollo, OCP and NLP settings."""

import pytest

import pycollo


@pytest.mark.usefixtures("_ocp_fixture")
class TestSettings:

    @pytest.fixture(autouse=True)
    def _ocp_fixture(self):
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.ocp = pycollo.OptimalControlProblem()
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

    @pytest.mark.skip
    def test_default_collocation_matrix_form(self):
        """Default collocation matrix form should be integral."""
        assert self.settings.collocation_matrix_form == "integral"

    @pytest.mark.skip
    def test_quadrature_defaults(self):
        """Default quadrature settings."""
        assert self.settings.quadrature_method == "lobatto"
        assert self.settings.collocation_points_min == 4
        assert self.settings.collocation_points_max == 10

    @pytest.mark.skip
    def test_mesh_refinement_defaults(self):
        """Default mesh refinement settings."""
        assert self.settings.max_mesh_iterations == 10
        assert self.settings.mesh_tolerance == 1e-8

    @pytest.mark.skip
    def test_display_defaults(self):
        """Defaults for console output and plotting during/after solve."""
        assert self.settings.display_mesh_refinement_info is True
        assert self.settings.display_mesh_result_info is False
        assert self.settings.display_mesh_result_graph is False

    @pytest.mark.skip
    def test_scaling_defaults(self):
        """Defaults for problem scaling."""
        assert self.settings.scaling_method == "bounds"
        assert self.settings.update_scaling is True
        assert self.settings.number_scaling_samples == 0
        assert self.settings.scaling_weight == 0.8

    def test_backend_property(self):
        """ValueErrors should be raised for invalid and unsupported values."""
        # Options
        invalid_keyword_str = "sdkfj"
        casadi_keyword_str = "casadi"
        pycollo_keyword_str = "pycollo"
        sympy_keyword_str = "sympy"

        # Check valid keyword
        self.settings.backend = pycollo_keyword_str
        assert self.settings.backend == pycollo_keyword_str

        # Check invalid keyword
        expected_error_msg = (
            f"'{invalid_keyword_str}' is not a valid option of Pycollo "
            f"backend. Choose one of: '{pycollo_keyword_str}' or "
            f"'{casadi_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.backend = invalid_keyword_str

        # Check unsupported keyword
        expected_error_msg = (
            f"'{sympy_keyword_str}' is not currently supported as a Pycollo "
            f"backend. Choose one of: '{pycollo_keyword_str}' or "
            f"'{casadi_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.backend = sympy_keyword_str

    def test_nlp_solver_property(self):
        """ValueError should be raised for invalid and unsupported values."""
        # Options
        invalid_keyword_str = "sdkfj"
        ipopt_keyword_str = "ipopt"
        snopt_keyword_str = "snopt"
        worhp_keyword_str = "worhp"
        bonmin_keyword_str = "bonmin"

        # Check valid keyword
        self.settings.nlp_solver = ipopt_keyword_str
        assert self.settings.nlp_solver == ipopt_keyword_str

        # Check invalid keyword
        expected_error_msg = (
            f"'{invalid_keyword_str}' is not a valid option of NLP solver. "
            f"Choose one of: '{ipopt_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = invalid_keyword_str

        # Check unsupported keyword
        expected_error_msg = (
            f"'{snopt_keyword_str}' is not currently supported as an NLP "
            f"solver. Choose one of: '{ipopt_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = snopt_keyword_str
        expected_error_msg = (
            f"'{worhp_keyword_str}' is not currently supported as an NLP "
            f"solver. Choose one of: '{ipopt_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = worhp_keyword_str
        expected_error_msg = (
            f"'{bonmin_keyword_str}' is not currently supported as an NLP "
            f"solver. Choose one of: '{ipopt_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_solver = bonmin_keyword_str

    def test_linear_solver_property(self):
        """ValueError should be raised for invalid and unsupported values."""
        # Options
        invalid_keyword_str = "sdkfj"
        mumps_keyword_str = "mumps"
        ma57_keyword_str = "ma57"

        # Check valid keyword
        self.settings.linear_solver = mumps_keyword_str
        assert self.settings.linear_solver == mumps_keyword_str

        # Check invalid keyword
        expected_error_msg = (
            f"'{invalid_keyword_str}' is not a valid option of linear solver. "
            f"Choose one of: '{mumps_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.linear_solver = invalid_keyword_str

        # Check unsupported keyword
        expected_error_msg = (
            f"'{ma57_keyword_str}' is not currently supported as a linear "
            f"solver. Choose one of: '{mumps_keyword_str}'."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.linear_solver = ma57_keyword_str

    def test_nlp_tolerance_property(self):
        """ValueError if >1.0 or <=0.0."""
        valid_test_value = 1e-6
        self.settings.nlp_tolerance = valid_test_value
        assert self.settings.nlp_tolerance == valid_test_value

        # Check tolerance greater than 1.0
        expected_error_msg = ("NLP tolerance (.*nlp_tolerance.*) must be less "
                              "than 1.0. 1.1 is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_tolerance = 1.1

        # Check negative tolerance
        expected_error_msg = ("NLP tolerance (.*nlp_tolerance.*) must be "
                              "greater than 0.0. 0.0 is invalid.")
        with pytest.raises(ValueError, match=expected_error_msg):
            self.settings.nlp_tolerance = 0.0

    def test_max_nlp_iterations_property(self):
        """ValueError if <1."""
        for valid_test_value in [1, 100, 10000, "1"]:
            self.settings.max_nlp_iterations = valid_test_value
            assert self.settings.max_nlp_iterations == int(valid_test_value)

        for invalid_test_value in [0, -1]:
            expected_error_msg = (
                f"Maximum number of NLP iterations (.*max_nlp_iterations.*) "
                f"must be greater than or equal to 1. {invalid_test_value} "
                f"is invalid.")
            with pytest.raises(ValueError, match=expected_error_msg):
                self.settings.max_nlp_iterations = invalid_test_value

    def test_derivative_level_property(self):
        """Value other than 1 or 2 should raise ValueError."""
        for valid_test_value in [1, 2, "1", "2"]:
            self.settings.derivative_level = valid_test_value
            assert self.settings.derivative_level == int(valid_test_value)
