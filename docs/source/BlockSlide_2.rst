Block Slide 2
=============

OCP Description
---------------

Let’s introduce a second phase to the previous example where the block
will have to return to it’s original position. Because we know friction
will switch direction when we slide the block back, the equations of
motion change. Since phase A will not change, lets copy everything up
until we start initiating the objective function.

.. code:: ipython3

    # 1D Blockslide
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sym
    import pycollo
    
    # State variables
    x = sym.Symbol("x")  # Position (m) of the point horizontally from the origin (x-axis)
    dx = sym.Symbol("dx")  # Velocity (m/s) of the point horizontally (x-axis)
    # Control variables
    Fx = sym.Symbol("Fx")  # Force (N) applied to the point horizontally (x-axis)
    
    # Static parameter variable
    g = sym.Symbol("g")  # Gravitational acceleration (m/s^2)
    m = sym.Symbol("m")  # Mass (kg) of the point
    mu = sym.Symbol("mu") # Coefficient of friction (-)
    
    # Problem instantiation
    problem = pycollo.OptimalControlProblem(
        name="Simple Block Slide",
        parameter_variables= (m,mu))
        
    problem.bounds.parameter_variables = [[1,2], [0.5,1]]
    problem.guess.parameter_variables = [1.5, 0.75]
    problem.auxiliary_data = {g: 9.81}
    
    phase_A = problem.new_phase(name="A")
    phase_A.bounds.initial_time = 0
    phase_A.bounds.final_time = [0, 10]
    phase_A.guess.time = [0, 1]
    
    phase_A.state_variables = [x, dx]
    phase_A.bounds.state_variables = [[-3,3],[-50,50]]
    phase_A.bounds.state_variables = {
        x: [-3, 3],
        dx: [-50, 50],}
    phase_A.bounds.initial_state_constraints = {
        x: 0,
        dx: 0,}
    phase_A.bounds.final_state_constraints = {
        x: 1,
        dx: 0,}
    phase_A.guess.state_variables = [[0, 0], [0, 0]]
    
    phase_A.control_variables = [Fx]
    phase_A.bounds.control_variables = {
        Fx: [-50, 50],}
    phase_A.guess.control_variables = [
            [0, 0],]
    
    phase_A.state_equations = {
        x: dx,
        dx: Fx / m - m*g*mu,}
    
    phase_A.integrand_functions = [Fx ** 2]
    phase_A.bounds.integral_variables = [[0, 1000]]
    phase_A.guess.integral_variables = [0]

New phase
---------

Now we can copy the previous phase completely to initiate a new phase
completely the same. When doing this make sure you overwrite everything
that changes in this next phase. It is real easy to make mistakes like
this and it is recommended to write out the full phase description as
depicted above.

.. code:: ipython3

    phase_B = problem.new_phase_like(
        phase_for_copying=phase_A,
        name="B",)

Now we can start overwriting everything that will be different from the
previous phase:

-  Time Phase B initial time can start any moment within bounds, final
   time the same
-  Initial and final state constraints The block starts at final state
   of phase A, and will end at 0.
-  State equations Friction changes sign

.. code:: ipython3

    # Time 
    phase_B.bounds.initial_time = [0, 10]
    phase_B.bounds.final_time = [0, 10]
    phase_B.guess.time = [1, 2]
    
    # Initial and final state constraints
    phase_B.bounds.initial_state_constraints = {
        x: 1,
        dx: 0,}
    phase_B.bounds.final_state_constraints = {
        x: 0,
        dx: 0,}
    
    # State equations
    phase_B.state_equations = {
        x: dx,
        dx: Fx / m + m*g*mu,}

Endpoint constraints
--------------------

To make sure all variables are continious, sometimes endpoint
constraints need to be implemented. Endpoint constraints are
constraintes which exist of initial and final variables. When final and
initial states are not bound to a single value, phase A final states
should match phase B initial states to make the states continious. In
this example, the states are constrainted to be continious due to the
initial and final state constraints of both phases. Time variables are
not constrained to be continious (yet) thus we can implement the
following inequality constraint (final time phase A = initial time phase
B -> final time phase A - initial time phase B = 0):

.. code:: ipython3

    problem.endpoint_constraints = [
        phase_A.final_time_variable - phase_B.initial_time_variable,]
        
    problem.bounds.endpoint_constraints = [
        0,]

Objective function - multiphase
-------------------------------

Now the objective to minimize input force Fx should als be updated to
consider both phases, the integrand functions are created with the
copying of the new phase but are yet to be implemented in the objective:

.. code:: ipython3

    problem.objective_function = (
        phase_A.integral_variables[0] + phase_B.integral_variables[0])
    # Bug
    phase_B.guess.integral_variables = 0

Solve
~~~~~

.. code:: ipython3

    problem.settings.display_mesh_result_graph = True
    problem.initialise()
    problem.solve()


.. parsed-literal::

    
    =====================================
    Initialising optimal control problem.
    =====================================
    
    Phase variables and equations checked.
    Pycollo variables and constraints preprocessed.
    Backend initialised.
    Bounds checked.
    Problem scaling initialised.
    Quadrature scheme initialised.
    Backend postprocessing complete.
    Initial mesh created.
    Initial guess checked.
    
    ===============================
    Initialising mesh iteration #1.
    ===============================
    
    Guess interpolated to iteration mesh in 1.27ms.
    Scaling initialised in 136.04us.
    Initial guess scaled in 7.88us.
    Scaling generated in 2.86ms.
    NLP generated in 63.36ms.
    Mesh-specific bounds generated in 257.71us.
    
    Mesh iteration #1 initialised in 67.89ms.
    
    
    ==========================
    Solving mesh iteration #1.
    ==========================
    
    This is Ipopt version 3.14.11, running with linear solver MUMPS 5.2.1.
    
    Number of nonzeros in equality constraint Jacobian...:     1061
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:      312
    
    Total number of variables............................:      185
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:      185
                         variables with only upper bounds:        0
    Total number of equality constraints.................:      123
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  1.9999980e+01 1.67e-01 0.00e+00   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  1.9998000e-01 2.19e-02 8.98e+01  -6.3 1.80e-01    -  7.43e-01 9.90e-01f  1
       2  3.1804676e+00 2.16e-02 9.29e+01  -1.0 2.81e+01    -  2.01e-02 1.51e-02h  1
       3  1.5607163e+01 2.03e-02 1.79e+03   0.5 1.25e+01    -  1.00e+00 5.35e-02f  1
       4  1.8366781e+01 2.57e-02 3.82e+02   1.3 3.26e+00    -  1.00e+00 1.51e-01f  2
       5  1.0348553e+02 1.23e-01 1.48e+04   2.7 5.26e+00    -  2.79e-01 7.22e-02f  2
       6  1.1694462e+02 2.83e-01 6.11e+04   2.1 8.17e-01    -  3.71e-01 6.38e-01h  1
       7  2.7193540e+02 3.51e-02 5.05e+04   2.1 2.83e-01   4.0 4.35e-01 1.00e+00f  1
       8  3.1139688e+02 1.98e-04 2.67e+03   1.1 4.65e-02    -  1.00e+00 1.00e+00f  1
       9  3.0681077e+02 6.89e-06 1.39e+01  -0.6 4.15e-03   3.5 1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.3985372e+02 1.24e-03 1.50e+01  -0.7 4.39e+00    -  1.42e-01 7.96e-02f  1
      11  8.2397205e+01 9.25e-03 1.46e+01  -0.7 6.31e-01    -  9.93e-01 3.20e-01f  1
      12  6.5374506e+01 7.68e-03 1.40e+01  -1.1 2.19e-01    -  1.00e+00 1.88e-01f  1
      13  4.0373560e+01 1.14e-02 1.81e+01  -1.7 1.29e-01    -  1.00e+00 8.94e-01f  1
      14  5.4515584e+01 8.23e-03 4.59e+00  -1.2 5.53e-02    -  1.00e+00 1.00e+00h  1
      15  7.0896703e+01 2.04e-04 1.97e-01  -1.9 2.33e-02    -  1.00e+00 1.00e+00h  1
      16  7.0960220e+01 1.26e-06 1.70e-03  -3.8 1.72e-03    -  1.00e+00 1.00e+00h  1
      17  7.0958246e+01 2.22e-09 2.97e-06  -9.8 1.14e-05    -  9.98e-01 9.98e-01h  1
      18  7.0958243e+01 3.06e-16 5.90e-13 -11.0 1.94e-08    -  1.00e+00 1.00e+00h  1
    
    Number of Iterations....: 18
    
                                       (scaled)                 (unscaled)
    Objective...............:   7.0958242992065896e+00    7.0958242992065891e+01
    Dual infeasibility......:   5.9027021046789531e-13    5.9027021046789531e-12
    Constraint violation....:   3.0553337637684310e-16    3.0553337637684310e-16
    Variable bound violation:   9.9995973723565612e-09    9.9995973723565612e-09
    Complementarity.........:   1.0014486051366179e-11    1.0014486051366178e-10
    Overall NLP error.......:   1.0014486051366179e-11    1.0014486051366178e-10
    
    
    Number of objective function evaluations             = 24
    Number of objective gradient evaluations             = 19
    Number of equality constraint evaluations            = 24
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 19
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 18
    Total seconds in IPOPT                               = 0.024
    
    EXIT: Optimal Solution Found.
          solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
           nlp_f  |  18.00us (750.00ns)  22.67us (944.46ns)        24
           nlp_g  | 340.00us ( 14.17us) 313.75us ( 13.07us)        24
      nlp_grad_f  |  42.00us (  2.00us)  40.96us (  1.95us)        21
      nlp_hess_l  | 537.00us ( 29.83us) 533.92us ( 29.66us)        18
       nlp_jac_g  | 692.00us ( 34.60us) 690.09us ( 34.50us)        20
           total  |  24.59ms ( 24.59ms)  24.60ms ( 24.60ms)         1
    
    ==================================
    Post-processing mesh iteration #1.
    ==================================
    
    Mesh iteration #1 solved in 24.88ms.
    Mesh iteration #1 post-processed in 44.07ms.
    
    
    ============================
    Analysing mesh iteration #1.
    ============================
    
    Objective Evaluation:       70.95824299206589
    Max Relative Mesh Error:    3.5248790057201846e-15
    Collocation Points Used:    62
    
    Adjusting Collocation Mesh: [10, 10] mesh sections
    
    Mesh iteration #1 completed in 136.83ms.
    



.. image:: BlockSlide_2_files/BlockSlide_2_11_1.png



.. image:: BlockSlide_2_files/BlockSlide_2_11_2.png



.. image:: BlockSlide_2_files/BlockSlide_2_11_3.png


.. parsed-literal::

    Mesh tolerance met in mesh iteration 1.
    
    
    ===========================================
    Optimal control problem sucessfully solved.
    ===========================================
    
    Final Objective Function Evaluation: 70.9582
    


Solution
~~~~~~~~

All results can be found in problem.solution, see
[INSERT_LINK_TO_SOLUTION]


