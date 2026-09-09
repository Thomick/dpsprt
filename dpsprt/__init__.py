"""
DP-SPRT: Differentially Private Sequential Probability Ratio Tests

Implementation of Michel, Basu and Kaufmann, AISTATS 2026 (arXiv:2508.06377).
Five sequential tests for two simple Bernoulli hypotheses, one non-private and
four differentially private, plus the OutsideInterval mechanism they are built
on, exposed as a primitive in its own right.

Sampling caveat: noise is drawn with numpy floating-point arithmetic and a
non-cryptographic generator, so the guarantees hold for the idealized
real-valued mechanisms.  Known floating-point attacks on differentially private
samplers are out of scope.

Quick Start:
    >>> import numpy as np
    >>> from dpsprt import DPSPRT
    >>>
    >>> # Create test instance
    >>> test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42)
    >>>
    >>> # Process data points one by one
    >>> for i in range(100):
    ...     x = np.random.binomial(1, 0.6)  # New sample
    ...     can_stop, decision = test.add_sample(x)
    ...     if can_stop:
    ...         print(f"Decision reached after {i+1} samples: {decision}")
    ...         break

Algorithm Variants:
    - ClassicalSPRT: Non-private baseline (Wald's SPRT)
    - DPSPRT: DP-SPRT with Laplace noise (pure ε-DP)
    - DPSPRTSubsampled: DP-SPRT with subsampling amplification
    - DPSPRTTuned: DP-SPRT with tunable optimization parameters
    - DPSPRTGaussian: DP-SPRT with Gaussian noise ((ε,δ)-DP)

Privacy Guarantees:
    All DP variants provide rigorous differential privacy guarantees:
    - Pure ε-DP: Probability ratios bounded by e^ε
    - (ε,δ)-DP: ε-DP holds with probability ≥ (1-δ)

Runnable scripts live in the examples/ directory of the source repository:
    - paper_experiments.py: reproduces the headline AISTATS 2026 figure
    - plot_stopping_times_vs_epsilon.py: stopping-time sweep across epsilon
    - stopping_times_demo.py: side-by-side comparison on a single stream
    - outside_interval_demo.py: OutsideInterval on a band-crossing scenario

Run them from the repository root, for example:
    python examples/stopping_times_demo.py
"""

__version__ = "1.2.0"
__author__ = "Thomas Michel"

# Import main API classes
from .api import (
    DPSPRT,
    ClassicalSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
    OutsideInterval,
    run_sprt_on_batch,
)
from .api.outside_interval import OutsideIntervalParameters

# Import parameter classes for advanced usage
from .api.parameters import (
    DPSPRTGaussianParameters,
    DPSPRTParameters,
    DPSPRTTunedParameters,
    SPRTParameters,
)

# Core algorithms (for advanced users)
from .core import (
    classical_sprt,
    dp_sprt_gaussian,
    dp_sprt_laplace,
    dp_sprt_subsampled,
    dp_sprt_tuned,
)

__all__ = [
    # Main algorithm classes
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
    # Core functions for advanced users
    "classical_sprt",
    "dp_sprt_laplace",
    "dp_sprt_subsampled",
    "dp_sprt_tuned",
    "dp_sprt_gaussian",
]

# Package metadata
__title__ = "dpsprt"
__description__ = "Differentially Private Sequential Probability Ratio Tests"
__url__ = "https://github.com/Thomick/dpsprt"
__license__ = "MIT"


def get_version():
    """Get the version string."""
    return __version__


def list_algorithms():
    """List all available algorithm variants and primitives."""
    algorithms = {
        "ClassicalSPRT": "Non-private Wald SPRT (baseline)",
        "DPSPRT": "DP-SPRT with Laplace noise (pure ε-DP)",
        "DPSPRTSubsampled": "DP-SPRT with subsampling amplification",
        "DPSPRTTuned": "DP-SPRT with tunable parameters c₁, c₂",
        "DPSPRTGaussian": "DP-SPRT with Gaussian noise ((ε,δ)-DP)",
        "OutsideInterval": "ε-DP continual-observation primitive (paper Algorithm 2)",
    }

    print("Available DP-SPRT Algorithm Variants:")
    print("=" * 40)
    for name, description in algorithms.items():
        print(f"{name:<20}: {description}")

    return algorithms


def quick_demo():
    """Run a quick demonstration of the library."""
    import numpy as np

    print("DP-SPRT Library Quick Demo")
    print("=" * 30)

    # Create test instance
    test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=123)
    print("Created DP-SPRT test: H0(mu=0.3) vs H1(mu=0.7), eps=1.0")

    # Generate test data and process one sample at a time
    rng = np.random.default_rng(42)
    print("Processing data samples one at a time...")

    # Process samples until decision or max samples reached
    decision = 0
    samples_processed = 0
    max_samples = 100

    for i in range(max_samples):
        x = int(rng.binomial(1, 0.6))
        can_stop, decision = test.add_sample(x)
        samples_processed += 1

        if can_stop:
            break

    if decision == 1:
        result = "Accept H₁ (μ=0.7)"
    elif decision == -1:
        result = "Accept H₀ (μ=0.3)"
    else:
        result = "No decision"

    print(f"Result: {result} after {samples_processed} samples")
    print(f"Privacy guarantee: ε = {test.params.epsilon}")
    print("\nFor more examples, see examples/ directory in the repository")
