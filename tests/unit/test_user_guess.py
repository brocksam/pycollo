"""Tests for sanitising and processing user-supplied guesses."""

import numpy as np
import pytest

import pycollo


@pytest.mark.usefixtures("_ocp_without_guess_fixture")
class TestUserGuess:

    t0 = 0.0
    tF = 10000.0

    @pytest.fixture(autouse=True)
    def _ocp_without_guess_fixture(self):
        """Simple fixture setting up an empty :obj:`OptimalControlProblem`."""
        self.ocp = pycollo.OptimalControlProblem("Test OCP")
        _ = self.ocp.new_phase(name="A")

    @pytest.mark.parametrize("test_input, expected",
                             [([t0, tF], np.array([t0, tF])),
                              ([t0, tF], np.array([t0, tF])),
                              ])
    def test_time_guess(self, test_input, expected):
        self.ocp.phases.A.guess.time = test_input
        assert type(self.ocp.phases.A.guess.time) == type(expected)
        assert np.array_equal(self.ocp.phases.A.guess.time, expected)
