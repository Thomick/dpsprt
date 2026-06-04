"""Core implementation of the OutsideInterval DP primitive and helpers.

OutsideIntervalCore implements Algorithm 2 from the AISTATS 2026 paper: monitors
a stream of query values and halts when a noisy query falls outside [T̂₀, T̂₁].
Threshold noise is drawn once at construction; per-query noise is fresh each step.

dpsprt_interval_check() is a stateless helper shared by all four DP-SPRT variants.
"""
from typing import Callable, Optional, Tuple

import numpy as np


class OutsideIntervalCore:
    """Stateful implementation of the OutsideInterval algorithm (paper Algorithm 2).

    Threshold noise is drawn once at construction; query noise is fresh per step.
    """

    def __init__(
        self,
        lower_threshold: Callable[[int], float],
        upper_threshold: Callable[[int], float],
        query_noise_scale: float,
        threshold_noise_scale: float,
        rng: np.random.RandomState,
    ):
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold
        self._query_noise_scale = query_noise_scale
        self._threshold_noise_scale = threshold_noise_scale
        self._rng = rng

        self._step = 0
        self._stopped = False
        self._side = 0
        self._threshold_noise_lower = rng.laplace(0, threshold_noise_scale)
        self._threshold_noise_upper = rng.laplace(0, threshold_noise_scale)

    def add_query(self, value: float) -> Tuple[bool, int]:
        """Process next query value. Returns (stopped, side)."""
        if self._stopped:
            return True, self._side

        self._step += 1
        t = self._step

        # Noisy thresholds: T̂₀ = τ₀(t) - noise_lower, T̂₁ = τ₁(t) + noise_upper
        noisy_lower = self._lower_threshold(t) - self._threshold_noise_lower
        noisy_upper = self._upper_threshold(t) + self._threshold_noise_upper

        # Per-query noise
        nu = self._rng.laplace(0, self._query_noise_scale)
        noisy_value = value + nu

        if noisy_value <= noisy_lower:
            self._stopped = True
            self._side = -1
        elif noisy_value >= noisy_upper:
            self._stopped = True
            self._side = 1

        return self._stopped, self._side

    def is_active(self) -> bool:
        return not self._stopped

    @property
    def step(self) -> int:
        return self._step

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def side(self) -> int:
        return self._side


def dpsprt_interval_check(
    noisy_query_h0: float,
    noisy_query_h1: float,
    lower_threshold: float,
    upper_threshold: float,
) -> tuple:
    """DPSPRT stopping-condition check (shared by all four DP-SPRT variants).

    Returns (stopped, side):
      side =  1 → accept H₁ (noisy_query_h1 ≥ upper_threshold)
      side = -1 → accept H₀ (noisy_query_h0 ≤ lower_threshold)
      side =  0 → continue

    H₁ is checked first, matching the original code's priority.
    """
    if noisy_query_h1 >= upper_threshold:
        return True, 1
    elif noisy_query_h0 <= lower_threshold:
        return True, -1
    return False, 0
