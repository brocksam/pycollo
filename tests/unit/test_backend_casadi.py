"""Testing of the CasADi-powered backend."""

import pytest


def test_casadi_backend_init(hypersensitive_problem_fixture):
    ocp, syms = hypersensitive_problem_fixture