import numpy as np
import sympy as sym

import pycollo

x1, x2, x3 = sym.symbols('x1 x2 x3')
y1, y2, y3 = sym.symbols('y1 y2 y3')

_x1, _x2, _x3 = sym.symbols('_x1 _x2 _x3')

n = 3

J = (y1 + y2 + y3) / 2
c = sym.Matrix([])
L = sym.Matrix([])

x_vars = sym.Matrix([x1, x2, x3])
x_vars_pycollo = sym.Matrix([_x1, _x2, _x3])

aux_data = {y1: x1**4 - 16*x1**2 + 5*x1, y2: x2**4 - 16*x2**2 + 5*x2, y3: x3**4 - 16*x3**2 + 5*x3}

expression_graph = pycollo.ExpressionGraph(x_vars, x_vars_pycollo, aux_data, J, c, L)