"""
User-friendly API for DP-SPRT algorithms.

This module provides clean, object-oriented interfaces to all DP-SPRT variants
with comprehensive parameter validation and documentation. All algorithms use
the incremental interface where data is processed one sample at a time.
"""

# Import core implementations
from .sprt import (
    BaseSPRT,
    ClassicalSPRT,
    DPSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
    run_streaming_sprt_on_batch,
)

# Create clean aliases for public API
ClassicalSPRT = ClassicalSPRT
DPSPRT = DPSPRT
DPSPRTSubsampled = DPSPRTSubsampled
DPSPRTTuned = DPSPRTTuned
DPSPRTGaussian = DPSPRTGaussian
run_sprt_on_batch = run_streaming_sprt_on_batch

from .outside_interval import OutsideInterval, OutsideIntervalParameters

from .parameters import (
    DPSPRTGaussianParameters,
    DPSPRTParameters,
    DPSPRTTunedParameters,
    SPRTParameters,
    validate_bernoulli_parameters,
    validate_error_rates,
    validate_privacy_parameters,
)

__all__ = [
    # Main API classes
    "BaseSPRT",
    "ClassicalSPRT",
    "DPSPRT",
    "DPSPRTSubsampled",
    "DPSPRTTuned",
    "DPSPRTGaussian",
    "OutsideInterval",
    "run_sprt_on_batch",
    # Parameter classes
    "SPRTParameters",
    "DPSPRTParameters",
    "DPSPRTGaussianParameters",
    "DPSPRTTunedParameters",
    "OutsideIntervalParameters",
    "validate_bernoulli_parameters",
    "validate_error_rates",
    "validate_privacy_parameters",
]
