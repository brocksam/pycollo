import numpy as np
import pytest
import sympy as sym
import sympy.physics.mechanics as me

try:
	import pycollo
except ModuleNotFoundError:
	from ..optimal_control_problem import OptimalControlProblem

def test_1():
	assert(1 == 1)

def test_2():
	assert(1 != 2)