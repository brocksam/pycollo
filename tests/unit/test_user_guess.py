"""Tests for sanitising and processing user-supplied guesses."""

import numpy as np
import pytest
import sympy as sym

import pycollo


@pytest.mark.usefixtures("_ocp_without_guess_fixture")
class TestUserGuess:
    """Test class for user-supplied guesses.

    Attributes
    ----------
    t0 : float
        Default numerical value for initial time.
    tF : float
        Default numerical value for final time.

    """

    t0 = 0.0
    tF = 10000.0

    @pytest.fixture(autouse=True)
    def _ocp_without_guess_fixture(self):
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.u, self.v, self.w = sym.symbols("u v w")
        self.x, self.y, self.z = sym.symbols("x y z")
        self.ocp = pycollo.OptimalControlProblem("Test OCP")
        _ = self.ocp.new_phase(name="A")

    def test_time_uninitialised_is_none(self):
        """Time guess initialised to None."""
        assert self.ocp.phases.A.guess.time is None

    @pytest.mark.parametrize("test_input, expected",
                             [([t0, tF], np.array([t0, tF])),
                              ([t0, tF], np.array([t0, tF])),
                              ([t0, 5.0, tF], np.array([t0, 5.0, tF])),
                              ]
                             )
    def test_time_guess(self, test_input, expected):
        """Time guesses are always cast to 1d np.ndarray."""
        self.ocp.phases.A.guess.time = test_input
        assert type(self.ocp.phases.A.guess.time) == type(expected)
        assert np.array_equal(self.ocp.phases.A.guess.time, expected)

    @pytest.mark.parametrize("test_input", [[tF, t0], [t0, tF, 5.0]])
    def test_time_guess_ascending(self, test_input):
        """Time guesses must be increasing."""
        with pytest.raises(ValueError):
            self.ocp.phases.A.guess.time = test_input

    def test_state_variables_guess_uninitialised_is_none(self):
        """State guess initialised to None."""
        assert self.ocp.phases.A.guess.state_variables is None

    def test_control_variables_guess_uninitialised_is_none(self):
        """Control guess initialised to None."""
        assert self.ocp.phases.A.guess.control_variables is None

    def test_integral_variables_guess_uninitialised_is_none(self):
        """Integral guess initialised to None."""
        assert self.ocp.phases.A.guess.integral_variables is None


def test_user_guess_dp_specific(double_pendulum_fixture):
    ocp, user_syms = double_pendulum_fixture
    phase = ocp.phases.A

    pi_by_2 = 0.5 * np.pi
    np.testing.assert_allclose(phase.guess.time, np.array([0, 2]))
    np.testing.assert_allclose(phase.guess.state_variables,
                               np.array([[-pi_by_2, pi_by_2],
                                         [-pi_by_2, pi_by_2],
                                         [0, 0],
                                         [0, 0],
                                         ]))
    np.testing.assert_allclose(phase.guess.control_variables,
                               np.array([[0, 0], [0, 0]]))
    np.testing.assert_allclose(phase.guess.integral_variables,
                               np.array([100]))
    np.testing.assert_allclose(ocp.guess.parameter_variables,
                               np.array([1.0, 1.0]))

    ocp._console_out_initialisation_message()
    ocp._check_variables_and_equations()
    ocp._initialise_backend()
    ocp._check_problem_and_phase_bounds()
    ocp._initialise_scaling()
    ocp._check_initial_guess()

    guess = ocp._backend.initial_guess

    assert isinstance(guess.tau, list)
    assert len(guess.tau) == 1
    np.testing.assert_allclose(guess.tau[0], np.array([-1, 1]))
    assert isinstance(guess.t0, list)
    assert len(guess.t0) == 1
    assert guess.t0[0] == 0.0
    assert isinstance(guess.tF, list)
    assert len(guess.tF) == 1
    assert guess.tF[0] == 2.0
    assert isinstance(guess.y, list)
    assert len(guess.y) == 1
    np.testing.assert_allclose(guess.y[0], np.array([[-pi_by_2, pi_by_2],
                                                     [-pi_by_2, pi_by_2],
                                                     [0, 0],
                                                     [0, 0],
                                                     ]))
    assert isinstance(guess.u, list)
    assert len(guess.u) == 1
    np.testing.assert_allclose(guess.u[0], np.array([[0, 0], [0, 0]]))
    assert isinstance(guess.q, list)
    assert len(guess.q) == 1
    np.testing.assert_allclose(guess.q[0], np.array([100]))
    np.testing.assert_allclose(guess.s, np.array([1.0, 1.0]))
