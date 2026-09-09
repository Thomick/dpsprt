"""User-facing API for the DP-SPRT algorithms.

Each class wraps the corresponding implementation in ``dpsprt.core`` with
parameter validation and a uniform incremental interface, where data is
processed one sample at a time.
"""

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
from .sprt import (
    DPSPRT,
    BaseSPRT,
    ClassicalSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
    run_streaming_sprt_on_batch,
)

run_sprt_on_batch = run_streaming_sprt_on_batch

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
