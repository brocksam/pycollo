"""Tumour Anti-Angiogenesis problem.

Example 6.17 from Betts, J. T. (2010). Practical methods for optimal control
and estimation using nonlinear programming - 2nd Edition. Society for
Industrial and Applied Mathematics, p348 - 352.

Attributes
----------
p : `Symbol`
    Primary tumour volume
q : `Symbol`
    Carrying capacity of the vascular
u : `Symbol`
    Angiogenic dose rate
xi : `Symbol`
    Tumour growth parameter (in 1/day)
b : `Symbol`
    "Birth" rate (in 1/day)
d : `Symbol`
    "Death" rate (in 1/mm^2/day)
G : `Symbol`
    Anti-angiogenic killing parameter (in kg/mg/day)
mu : `Symbol`
    Loss of endothelial cells due to natural causes (in 1/day)
a : `Symbol`
    Constant (nondimensional)
A : `Symbol`
    Constant (nondimensional)

"""

import pycollo


# Symbol creation
p = pycollo.Symbol("p")
q = pycollo.Symbol("q")
u = pycollo.Symbol("u")

xi = pycollo.Symbol("xi")
b = pycollo.Symbol("b")
d = pycollo.Symbol("d")
G = pycollo.Symbol("G")
mu = pycollo.Symbol("mu")
a = pycollo.Symbol("a")
A = pycollo.Symbol("A")

p_max = pycollo.Symbol("p_max")
p_min = pycollo.Symbol("p_min")
q_max = pycollo.Symbol("q_max")
q_min = pycollo.Symbol("q_min")
y_max = pycollo.Symbol("y_max")
y_min = pycollo.Symbol("y_min")
u_max = pycollo.Symbol("u_max")
u_min = pycollo.Symbol("u_min")
p_t0 = pycollo.Symbol("p_t0")
q_t0 = pycollo.Symbol("q_t0")
y_t0 = pycollo.Symbol("y_t0")

# Auxiliary information
t0 = 0.0
tF_max = 5.0
tF_min = 0.1

# Optimal control problem definition
problem = pycollo.OptimalControlProblem(name="Tumour Anti-Angiogenesis")
phase = problem.new_phase(name="A",
                          state_variables=[p, q],
                          control_variables=u)

phase.state_equations = {p: -xi * p * pycollo.log(p / q),
                         q: q * (b - (mu + (d * p**(2 / 3)) + (G * u)))}
phase.integrand_functions = [u]

problem.objective_function = phase.final_state_variables.p
problem.auxiliary_data = {xi: 0.084,
                          b: 5.85,
                          d: 0.00873,
                          G: 0.15,
                          mu: 0.02,
                          a: 75,
                          A: 15,
                          p_max: ((b - mu) / d)**(3 / 2),
                          p_min: 0.1,
                          q_max: p_max,
                          q_min: p_min,
                          u_max: a,
                          u_min: 0,
                          p_t0: p_max / 2,
                          q_t0: q_max / 4}

# Problem bounds
phase.bounds.initial_time = t0
phase.bounds.final_time = [tF_min, tF_max]
phase.bounds.state_variables = {p: [p_min, p_max],
                                q: [q_min, q_max]}
phase.bounds.control_variables = {u: [u_min, u_max]}
phase.bounds.integral_variables = [[0, A]]
phase.bounds.initial_state_constraints = {p: p_t0,
                                          q: q_t0}

# Problem guesses
phase.guess.time = [0, 1]
phase.guess.state_variables = [[p_t0, p_max], [q_t0, q_max]]
phase.guess.control_variables = [[u_max, u_max]]
phase.guess.integral_variables = [7.5]

problem.settings.display_mesh_result_graph = True

problem.initialise()
problem.solve()
