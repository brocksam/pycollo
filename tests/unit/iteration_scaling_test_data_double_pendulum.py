"""Expected data for testing iteration scaling classes.

This is specific to the double pendulum swing up test problem using Lobatto 
quadrature on an evenly spaced temporal grid with four collocation points per
mesh section.

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

EXPECT_V_DP = np.array(
    [
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +6.283185307179586e00,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +2.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +3.000000000000000e01,
        +1.000000000000000e03,
        +2.000000000000000e00,
        +1.000000000000000e00,
        +1.000000000000000e00,
    ]
)
EXPECT_R_DP = np.array(
    [
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +5.000000000000000e02,
        +2.000000000000000e00,
        +1.000000000000000e00,
        +1.000000000000000e00,
    ]
)
EXPECT_V_INV_DP = np.array(
    [
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +1.591549430918953e-01,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +5.000000000000000e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +3.333333333333333e-02,
        +1.000000000000000e-03,
        +5.000000000000000e-01,
        +1.000000000000000e00,
        +1.000000000000000e00,
    ]
)
EXPECT_X_TILDE_DP = np.array(
    [
        -2.500000000000000e-01,
        -2.361803398874990e-01,
        -2.138196601125011e-01,
        -2.000000000000000e-01,
        -1.861803398874990e-01,
        -1.638196601125010e-01,
        -1.500000000000000e-01,
        -1.361803398874990e-01,
        -1.138196601125011e-01,
        -1.000000000000000e-01,
        -8.618033988749896e-02,
        -6.381966011250108e-02,
        -5.000000000000000e-02,
        -3.618033988749896e-02,
        -1.381966011250108e-02,
        +0.000000000000000e00,
        +1.381966011250101e-02,
        +3.618033988749892e-02,
        +5.000000000000000e-02,
        +6.381966011250108e-02,
        +8.618033988749896e-02,
        +1.000000000000000e-01,
        +1.138196601125011e-01,
        +1.361803398874989e-01,
        +1.500000000000000e-01,
        +1.638196601125010e-01,
        +1.861803398874990e-01,
        +2.000000000000000e-01,
        +2.138196601125011e-01,
        +2.361803398874990e-01,
        +2.500000000000000e-01,
        -2.500000000000000e-01,
        -2.361803398874990e-01,
        -2.138196601125011e-01,
        -2.000000000000000e-01,
        -1.861803398874990e-01,
        -1.638196601125010e-01,
        -1.500000000000000e-01,
        -1.361803398874990e-01,
        -1.138196601125011e-01,
        -1.000000000000000e-01,
        -8.618033988749896e-02,
        -6.381966011250108e-02,
        -5.000000000000000e-02,
        -3.618033988749896e-02,
        -1.381966011250108e-02,
        +0.000000000000000e00,
        +1.381966011250101e-02,
        +3.618033988749892e-02,
        +5.000000000000000e-02,
        +6.381966011250108e-02,
        +8.618033988749896e-02,
        +1.000000000000000e-01,
        +1.138196601125011e-01,
        +1.361803398874989e-01,
        +1.500000000000000e-01,
        +1.638196601125010e-01,
        +1.861803398874990e-01,
        +2.000000000000000e-01,
        +2.138196601125011e-01,
        +2.361803398874990e-01,
        +2.500000000000000e-01,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        -4.000000000000000e-01,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
    ]
)
EXPECT_X_DP = np.array(
    [
        -1.570796326794897e00,
        -1.483964841425814e00,
        -1.343468546805000e00,
        -1.256637061435917e00,
        -1.169805576066835e00,
        -1.029309281446020e00,
        -9.424777960769380e-01,
        -8.556463107078557e-01,
        -7.151500160870412e-01,
        -6.283185307179586e-01,
        -5.414870453488763e-01,
        -4.009907507280619e-01,
        -3.141592653589793e-01,
        -2.273277799898969e-01,
        -8.683148536908258e-02,
        +0.000000000000000e00,
        +8.683148536908214e-02,
        +2.273277799898967e-01,
        +3.141592653589793e-01,
        +4.009907507280619e-01,
        +5.414870453488763e-01,
        +6.283185307179586e-01,
        +7.151500160870410e-01,
        +8.556463107078556e-01,
        +9.424777960769379e-01,
        +1.029309281446020e00,
        +1.169805576066835e00,
        +1.256637061435917e00,
        +1.343468546805000e00,
        +1.483964841425814e00,
        +1.570796326794897e00,
        -1.570796326794897e00,
        -1.483964841425814e00,
        -1.343468546805000e00,
        -1.256637061435917e00,
        -1.169805576066835e00,
        -1.029309281446020e00,
        -9.424777960769380e-01,
        -8.556463107078557e-01,
        -7.151500160870412e-01,
        -6.283185307179586e-01,
        -5.414870453488763e-01,
        -4.009907507280619e-01,
        -3.141592653589793e-01,
        -2.273277799898969e-01,
        -8.683148536908258e-02,
        +0.000000000000000e00,
        +8.683148536908214e-02,
        +2.273277799898967e-01,
        +3.141592653589793e-01,
        +4.009907507280619e-01,
        +5.414870453488763e-01,
        +6.283185307179586e-01,
        +7.151500160870410e-01,
        +8.556463107078556e-01,
        +9.424777960769379e-01,
        +1.029309281446020e00,
        +1.169805576066835e00,
        +1.256637061435917e00,
        +1.343468546805000e00,
        +1.483964841425814e00,
        +1.570796326794897e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +0.000000000000000e00,
        +1.000000000000000e02,
        +2.000000000000000e00,
        +1.000000000000000e00,
        +1.000000000000000e00,
    ]
)
