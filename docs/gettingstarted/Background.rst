================
What is Pycollo?
================

Pycollo, a combination of Python and collocation, is a Python package for solving multiphase optimal control problems using direct orthogonal collocation.  
  
In optimization there are shooting methods and simultaneous methods. A shooting method uses a simulation to enforce the system dynamics, while the collocation method enforces the dynamics at given points along the trajectory. Shooting methods are more prone to find a global solution, but are computationally heavy. Collocation methods tend to converge better and faster but are prone to find local minima. Direct methods are generally best for problems where dynamics and control must be computed to a similar accuracy, and the structure of the control trajectory is not known a priori :cite:p:`kelly_introduction_2017`.

Pycollo is a direct orthogonal collocation transcription tool for Python :cite:p:`brockie_predictive_2021`. The dynamics and constraints are enforced over a discretization. The discretization exists of N collocation points. The collocation points are either mesh points or polynomial points, but all collocation points are enforced to the constraints and dynamics. After transcription the problem is passed to the solver IPOPT (Interior Point OPTimizer). When the found IPOPT solution does not meet the error-tolerance set in PyCollo, the discretization is refined and a new itera- tion is solved in IPOPT. Mesh points, polynomial points, mesh sections and mesh refinement are explained in figure 5. The polynomials are of the Legendre-Gauss-Lobatto (LGL) nature. More information on the mesh can be found in appendix A. The integration scheme is an implicit Runge-Kutta Kth method :cite:p:`brockie_predictive_2021`. This high order method will have a high accuracy with the disadvantage that all states and constraints need to be differentiable.

It is highly advised to read :cite:p:`kelly_introduction_2017` if you are new to direct collocation.


Multiphase direct collocation
-----------------------------
Some optimal control problems face discontinuities or changes in dynamics. Boolean operations can work in lower order optimization schemes, but with higher order schemes such as pycollo, every formulation needs to be differentiated. Differentiating of boolean operations is impossible or can lead to numerical instabillities (such as a sigmoid function). To still have the benefits of a higher order scheme and handle discontinuities one can use multiple continious phases. Each phase represents a distinct segment of the problem with its own set of dynamics, constraints, and control inputs. The main idea behind this scheme is to discretize each phase individually and then link the phases together by ensuring continuity of states and control inputs at phase boundaries. Between the phase boundaries discontinuities can be handeled such as impact, friction or change of dynamics. This approach allows you to tackle complex problems with changing dynamics or control strategies over different time intervals. 

Pycollo will help you with the following steps:

1. Discretization within Each Phase:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within each phase, you discretize the continuous-time dynamics using collocation points. Collocation points are time instances within the phase where you will approximate the system dynamics using constraints. These points are often chosen based on established techniques like Gaussian quadrature.

2. State and Control Parameterization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each phase, parameterize the state and control trajectories using LGL polynomial approximations.

3. Dynamics and Constraint Approximation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At each collocation point within a phase, you approximate the system dynamics and constraints using discrete equations. This involves approximating the differential equations that describe the system's behavior and ensuring that path constraints and boundary conditions are met at these points.

4. Endpoint Constraints:
^^^^^^^^^^^^^^^^^^^^^^^^

To ensure smooth transitions between phases, you enforce continuity constraints at the boundaries between phases. This involves linking the state and control variables from the final collocation point of one phase to the initial collocation point of the next phase. 

5. Objective Function Discretization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Discretize the objective function over each phase by approximating integrals as summations over the collocation points. This allows you to express the objective function in terms of the endpoint variables.

6. Formulating the Optimization Problem:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the discretized dynamics, constraints, and objective function in place for each phase, Pycollo formulates the complete optimization problem as a NLP. This program aims to find the optimal values of the discrete state and control variables across all phases, while satisfying the dynamics, constraints, and continuity conditions.

7. Solving the Optimization Problem:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pycollo uses IPOPT to solve the NLP

8. Using a PH method to ensure tolerances are met:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pycollo adjusts the amount of collocation points (P) and size of the mesh sections (H), to ensure all tolerances are met. Whenever a state shows non linear behaviour the amount of collocation points in that section will increase to decrease errors. 

In summary, a multiphase direct collocation scheme breaks down a complex optimal control problem into manageable phases, discretizes the dynamics and constraints within each phase, ensures continuity between phases, and solves the resulting optimization problem to find the optimal control strategy across multiple segments. This approach is particularly useful for problems where the system behavior and control strategies change over different time intervals.

.. bibliography::