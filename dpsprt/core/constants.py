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

# Small epsilon value to avoid numerical issues (division by zero, log of zero)
NUMERICAL_EPSILON = 1e-10

# Default maximum number of samples for any SPRT test
DEFAULT_MAX_SAMPLES = 10000
