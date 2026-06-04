"""User-facing SPRT and DP-SPRT classes.

Each class is a thin wrapper around the corresponding implementation in
``dpsprt.core.sprt`` that adds parameter validation and a uniform interface.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.sprt import ClassicalSPRT as CoreClassicalSPRT
from ..core.sprt import DPSPRT as CoreDPSPRT
from ..core.sprt import DPSPRTGaussian as CoreDPSPRTGaussian
from ..core.sprt import DPSPRTSubsampled as CoreDPSPRTSubsampled
from ..core.sprt import DPSPRTTuned as CoreDPSPRTTuned
from .parameters import (
    DPSPRTGaussianParameters,
    DPSPRTParameters,
    DPSPRTTunedParameters,
    SPRTParameters,
)


class BaseSPRT:
    """Shared interface over a core SPRT instance."""

    def __init__(self, algorithm, is_dp_algorithm: bool = False):
        self.algorithm = algorithm
        self.is_dp_algorithm = is_dp_algorithm
        # Sample history is recorded for non-DP variants only; recording the
        # raw stream alongside a DP algorithm would defeat its privacy.
        self._sample_history = [] if not is_dp_algorithm else None

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if x not in (0, 1):
            raise ValueError(f"Sample must be 0 or 1, got {x}")
        if not self.is_dp_algorithm:
            self._sample_history.append(x)
        return self.algorithm.add_sample(x)

    def reset(self):
        self.algorithm.reset()
        if not self.is_dp_algorithm:
            self._sample_history = []

    def get_state(self) -> Dict[str, Any]:
        state = self.algorithm.get_state()
        if not self.is_dp_algorithm:
            state["sample_history"] = self._sample_history.copy()
        return state

    def get_sample_count(self) -> int:
        """Number of samples seen so far (non-DP variants only)."""
        if self.is_dp_algorithm:
            raise ValueError(
                "Sample count not available for DP algorithms — exposing it would "
                "leak the stream length."
            )
        return len(self._sample_history)

    def has_stopped(self) -> bool:
        return self.algorithm.stopped

    def get_decision(self) -> int:
        return self.algorithm.decision


class ClassicalSPRT(BaseSPRT):
    """Wald's SPRT for Bernoulli observations."""

    def __init__(self, mu0: float, mu1: float, alpha: float, beta: float):
        self.params = SPRTParameters(mu0, mu1, alpha, beta)
        super().__init__(CoreClassicalSPRT(mu0, mu1, alpha, beta), is_dp_algorithm=False)


class DPSPRT(BaseSPRT):
    """ε-DP SPRT with Laplace noise."""

    def __init__(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        epsilon: float,
        random_seed: Optional[int] = None,
    ):
        self.params = DPSPRTParameters(mu0, mu1, alpha, beta, epsilon)
        super().__init__(
            CoreDPSPRT(mu0, mu1, alpha, beta, epsilon, random_seed),
            is_dp_algorithm=True,
        )


class DPSPRTGaussian(BaseSPRT):
    """(ε,δ)-DP SPRT with Gaussian noise."""

    def __init__(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        epsilon: float,
        delta: float,
        random_seed: Optional[int] = None,
    ):
        self.params = DPSPRTGaussianParameters(mu0, mu1, alpha, beta, epsilon, delta)
        super().__init__(
            CoreDPSPRTGaussian(mu0, mu1, alpha, beta, epsilon, delta, random_seed),
            is_dp_algorithm=True,
        )


class DPSPRTTuned(BaseSPRT):
    """DP-SPRT (Laplace) with tunable constants ``c1`` (threshold) and ``c2`` (privacy term)."""

    def __init__(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        epsilon: float,
        c1: float = 1.0,
        c2: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        self.params = DPSPRTTunedParameters(mu0, mu1, alpha, beta, epsilon, c1, c2)
        super().__init__(
            CoreDPSPRTTuned(mu0, mu1, alpha, beta, epsilon, c1, c2, random_seed),
            is_dp_algorithm=True,
        )


class DPSPRTSubsampled(BaseSPRT):
    """ε-DP SPRT with Bernoulli subsampling."""

    def __init__(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        epsilon: float,
        random_seed: Optional[int] = None,
    ):
        self.params = DPSPRTParameters(mu0, mu1, alpha, beta, epsilon)
        super().__init__(
            CoreDPSPRTSubsampled(mu0, mu1, alpha, beta, epsilon, random_seed),
            is_dp_algorithm=True,
        )

    def get_subsampling_info(self) -> Dict[str, Any]:
        """Always raises — exposing subsampling decisions would leak data."""
        raise ValueError(
            "Subsampling details not available for DP algorithms — exposing per-step "
            "inclusion decisions would leak data."
        )


def run_streaming_sprt_on_batch(
    sprt: BaseSPRT, data: np.ndarray
) -> Tuple[int, int, Dict[str, Any]]:
    """Run an SPRT instance on a full batch and return ``(decision, stopping_time, state)``.

    Calls ``sprt.reset()`` first, so every invocation starts a fresh run with a
    new threshold noise draw.
    """
    sprt.reset()
    for i, x in enumerate(data):
        can_stop, decision = sprt.add_sample(x)
        if can_stop:
            return decision, i + 1, sprt.get_state()
    return 0, len(data), sprt.get_state()
