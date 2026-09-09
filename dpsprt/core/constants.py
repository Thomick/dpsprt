"""
Mathematical constants and parameters used in DP-SPRT algorithms.

This module contains pre-computed values and mathematical constants that are used
across different DP-SPRT implementations.
"""

# Parameter s of the zeta function in the correction function C(n, delta) of the
# paper. Any s > 1 is admissible; the union bound over n in the correctness proof
# needs the paired zeta value to be zeta(s) exactly, since it relies on
# sum_n 1 / (n^s zeta(s)) = 1.
ZETA_S_PARAMETER = 1.1340

# zeta(ZETA_S_PARAMETER), to six decimals. Both this library and the JAX code that
# produced the paper's figures previously carried 5.591 here, which is zeta(1.2000)
# rather than zeta(1.1340), so the union bound spent 1.44 delta instead of delta.
# test_zeta_constant_matches_its_parameter recomputes this from the series and fails
# if the two constants ever drift apart again.
ZETA_S_VALUE = 8.049572

# The paper's error allocation gamma is fixed at max(0.5, 1 - 1/eps) throughout,
# so 1/(1 - gamma) equals max(eps, 2), which is the form the code uses inside the
# correction function.

# Small epsilon value to avoid numerical issues (division by zero, log of zero)
NUMERICAL_EPSILON = 1e-10
