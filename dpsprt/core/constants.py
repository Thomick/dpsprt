"""
Mathematical constants and parameters used in DP-SPRT algorithms.

This module contains pre-computed values and mathematical constants that are used
across different DP-SPRT implementations.
"""

# Pre-computed value of the Riemann zeta function at s=1.1340
# This appears in the privacy-preserving stopping conditions of DP-SPRT
ZETA_S_VALUE = 5.591

# Parameter s for the zeta function used in DP-SPRT privacy analysis
ZETA_S_PARAMETER = 1.1340

# The paper's error allocation gamma is fixed at max(0.5, 1 - 1/eps) throughout,
# so 1/(1 - gamma) equals max(eps, 2), which is the form the code uses inside the
# correction function.

# Small epsilon value to avoid numerical issues (division by zero, log of zero)
NUMERICAL_EPSILON = 1e-10
