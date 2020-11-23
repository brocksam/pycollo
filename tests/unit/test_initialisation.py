"""Test stages of problem initialisation."""


import re

import pytest


def test_console_out_initialisation_message(hypersensitive_problem_fixture):
    ocp = hypersensitive_problem_fixture.ocp
    ocp._console_out_initialisation_message()


def test_check_variables_and_equations(hypersensitive_problem_fixture):
    """This checks variables and equations for each phase in the OCP.

    Todo
    ----
    Refactor in to multiple tests checking for:
        * State variables having one more symbol than state equations.
        * State variables having two or more more symbol than state equations.
        * State equations having one more equation than state variables.
        * State equations having two or more more equation than state
          variables.
        * State equations and state variables having the same number of symbols
          but different symbols.

    """
    ocp = hypersensitive_problem_fixture.ocp
    ocp._check_variables_and_equations()


@pytest.mark.parametrize("backend", ["hSAD", "Sympy"])
def test_unsupported_backends_not_implemented(hypersensitive_problem_fixture,
                                              backend):
    """Attempted instantiation of unsupported backends raises error.

    This should not be possible using the Pycollo API as intended and is only
    achievable by the user directly accessing 'protected' attributes hidden
    within the Pycollo :class:`Settings` class.

    """
    ocp = hypersensitive_problem_fixture.ocp
    ocp.settings._backend = backend.lower()
    expected_error_msg = re.escape(
        f"The {backend} backend for Pycollo is not currently supported or "
        f"implemented.")
    with pytest.raises(NotImplementedError, match=expected_error_msg):
        ocp.initialise()
