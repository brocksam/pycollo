"""Expected data for testing iteration scaling classes.

This is specific to the brachistochrone test problem using Lobatto quadrature
on an evenly spaced temporal grid with four collocation points per mesh
section.

Attributes
----------
EXPECT_V : np.ndarray
    Expected variable stretching values.
EXPECT_R : np.ndarray
    Expected variable shifting values.
EXPECT_V_INV : np.ndarray
    Expected inverse of the variable stretching values.
EXPECT_X : np.ndarray
    Expected unscaled variable values on the temporal mesh.
EXPECT_X_TILDE : np.ndarray
    Expected scaled variable values on the temporal mesh.

"""

import numpy as np

EXPECT_V_BR = np.array([010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        010.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        100.000000000000000,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        003.141592653589793,
                        010.000000000000000,
                        ])
EXPECT_R_BR = np.array([5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        5.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        0.00000000000000000,
                        5.00000000000000000,
                        ])
EXPECT_V_INV_BR = np.array([0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.100000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.010000000000000,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.318309886183791,
                            0.100000000000000,])
EXPECT_X_TILDE_BR = np.array([-5.00000000e-01,
                              -4.99994350e-01,
                              -4.99898613e-01,
                              -4.99732795e-01,
                              -4.99445376e-01,
                              -4.98639586e-01,
                              -4.97880940e-01,
                              -4.96886127e-01,
                              -4.94701341e-01,
                              -4.92951528e-01,
                              -4.90864654e-01,
                              -4.86711766e-01,
                              -4.83629834e-01,
                              -4.80129455e-01,
                              -4.73533307e-01,
                              -4.68855427e-01,
                              -4.63701940e-01,
                              -4.54328828e-01,
                              -4.47883572e-01,
                              -4.40933081e-01,
                              -4.28610079e-01,
                              -4.20328341e-01,
                              -4.11540993e-01,
                              -3.96265963e-01,
                              -3.86184963e-01,
                              -3.75627252e-01,
                              -3.57568968e-01,
                              -3.45830096e-01,
                              -3.33671013e-01,
                              -3.13159398e-01,
                              -3.00000000e-01,
                              -5.00000000e-01,
                              -4.99745466e-01,
                              -4.98259185e-01,
                              -4.96683013e-01,
                              -4.94612510e-01,
                              -4.90239716e-01,
                              -4.86924092e-01,
                              -4.83157496e-01,
                              -4.76151359e-01,
                              -4.71288245e-01,
                              -4.66043629e-01,
                              -4.56809779e-01,
                              -4.50680731e-01,
                              -4.44261742e-01,
                              -4.33334786e-01,
                              -4.26294651e-01,
                              -4.19072930e-01,
                              -4.07085500e-01,
                              -3.99541873e-01,
                              -3.91935535e-01,
                              -3.79581661e-01,
                              -3.71971288e-01,
                              -3.64420717e-01,
                              -3.52415644e-01,
                              -3.45179136e-01,
                              -3.38121487e-01,
                              -3.27160266e-01,
                              -3.20716589e-01,
                              -3.14560484e-01,
                              -3.05277706e-01,
                              -3.00000000e-01,
                              +0.00000000e+00,
                              +2.23471237e-03,
                              +5.84421165e-03,
                              +8.06717369e-03,
                              +1.02811729e-02,
                              +1.38382375e-02,
                              +1.60171568e-02,
                              +1.81782801e-02,
                              +2.16312371e-02,
                              +2.37344609e-02,
                              +2.58113136e-02,
                              +2.91100023e-02,
                              +3.11069778e-02,
                              +3.30693893e-02,
                              +3.61658901e-02,
                              +3.80276078e-02,
                              +3.98470702e-02,
                              +4.26964005e-02,
                              +4.43958158e-02,
                              +4.60458976e-02,
                              +4.86066655e-02,
                              +5.01190915e-02,
                              +5.15758220e-02,
                              +5.38108276e-02,
                              +5.51142935e-02,
                              +5.63565110e-02,
                              +5.82332866e-02,
                              +5.93088571e-02,
                              +6.03185149e-02,
                              +6.18098009e-02,
                              +6.26418391e-02,
                              +1.42545093e-19,
                              +1.06103164e-02,
                              +2.77779883e-02,
                              +3.83883317e-02,
                              +4.89988044e-02,
                              +6.61661787e-02,
                              +7.67766590e-02,
                              +8.73873482e-02,
                              +1.04554322e-01,
                              +1.15164993e-01,
                              +1.25775930e-01,
                              +1.42942430e-01,
                              +1.53553331e-01,
                              +1.64164538e-01,
                              +1.81330518e-01,
                              +1.91941676e-01,
                              +2.02553155e-01,
                              +2.19718601e-01,
                              +2.30330027e-01,
                              +2.40941767e-01,
                              +2.58106693e-01,
                              +2.68718385e-01,
                              +2.79330358e-01,
                              +2.96494810e-01,
                              +3.07106749e-01,
                              +3.17718914e-01,
                              +3.34882964e-01,
                              +3.45495120e-01,
                              +3.56107422e-01,
                              +3.73271168e-01,
                              +3.83888143e-01,
                              -4.17566133e-01,
                              ])
EXPECT_X_BR = np.array([0.00000000e+00,
                        5.65025116e-05,
                        1.01387321e-03,
                        2.67204807e-03,
                        5.54624222e-03,
                        1.36041410e-02,
                        2.11906012e-02,
                        3.11387269e-02,
                        5.29865920e-02,
                        7.04847175e-02,
                        9.13534622e-02,
                        1.32882343e-01,
                        1.63701662e-01,
                        1.98705447e-01,
                        2.64666933e-01,
                        3.11445728e-01,
                        3.62980600e-01,
                        4.56711720e-01,
                        5.21164281e-01,
                        5.90669189e-01,
                        7.13899209e-01,
                        7.96716587e-01,
                        8.84590068e-01,
                        1.03734037e+00,
                        1.13815037e+00,
                        1.24372748e+00,
                        1.42431032e+00,
                        1.54169904e+00,
                        1.66328987e+00,
                        1.86840602e+00,
                        2.00000000e+00,
                        0.00000000e+00,
                        2.54533802e-03,
                        1.74081513e-02,
                        3.31698732e-02,
                        5.38749029e-02,
                        9.76028401e-02,
                        1.30759078e-01,
                        1.68425044e-01,
                        2.38486415e-01,
                        2.87117550e-01,
                        3.39563710e-01,
                        4.31902212e-01,
                        4.93192694e-01,
                        5.57382576e-01,
                        6.66652140e-01,
                        7.37053493e-01,
                        8.09270700e-01,
                        9.29145004e-01,
                        1.00458127e+00,
                        1.08064465e+00,
                        1.20418339e+00,
                        1.28028712e+00,
                        1.35579283e+00,
                        1.47584356e+00,
                        1.54820864e+00,
                        1.61878513e+00,
                        1.72839734e+00,
                        1.79283411e+00,
                        1.85439516e+00,
                        1.94722294e+00,
                        2.00000000e+00,
                        0.00000000e+00,
                        2.23471237e-01,
                        5.84421165e-01,
                        8.06717369e-01,
                        1.02811729e+00,
                        1.38382375e+00,
                        1.60171568e+00,
                        1.81782801e+00,
                        2.16312371e+00,
                        2.37344609e+00,
                        2.58113136e+00,
                        2.91100023e+00,
                        3.11069778e+00,
                        3.30693893e+00,
                        3.61658901e+00,
                        3.80276078e+00,
                        3.98470702e+00,
                        4.26964005e+00,
                        4.43958158e+00,
                        4.60458976e+00,
                        4.86066655e+00,
                        5.01190915e+00,
                        5.15758220e+00,
                        5.38108276e+00,
                        5.51142935e+00,
                        5.63565110e+00,
                        5.82332866e+00,
                        5.93088571e+00,
                        6.03185149e+00,
                        6.18098009e+00,
                        6.26418391e+00,
                        4.47818618e-19,
                        3.33332922e-02,
                        8.72671241e-02,
                        1.20600501e-01,
                        1.53934284e-01,
                        2.07867181e-01,
                        2.41200988e-01,
                        2.74535451e-01,
                        3.28467089e-01,
                        3.61801495e-01,
                        3.95136739e-01,
                        4.49066888e-01,
                        4.82402018e-01,
                        5.15738106e-01,
                        5.69666624e-01,
                        6.03002559e-01,
                        6.36339505e-01,
                        6.90266342e-01,
                        7.23603120e-01,
                        7.56940886e-01,
                        8.10866091e-01,
                        8.44203703e-01,
                        8.77542201e-01,
                        9.31465917e-01,
                        9.64804307e-01,
                        9.98143406e-01,
                        1.05206586e+00,
                        1.08540493e+00,
                        1.11874446e+00,
                        1.17266596e+00,
                        1.20602017e+00,
                        0.82433867e+00,
                        ])
