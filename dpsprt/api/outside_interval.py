"""User-facing wrapper for the OutsideInterval DP primitive."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from dpsprt.core.outside_interval import OutsideIntervalCore


@dataclass
class OutsideIntervalParameters:
    """Validated noise scales and ε budget for OutsideInterval."""

    query_noise_scale: float
    threshold_noise_scale: float
    epsilon: float

    def __post_init__(self):
        if self.query_noise_scale <= 0:
            raise ValueError(f"query_noise_scale must be > 0, got {self.query_noise_scale}")
        if self.threshold_noise_scale <= 0:
            raise ValueError(
                f"threshold_noise_scale must be > 0, got {self.threshold_noise_scale}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


class OutsideInterval:
    """ε-DP continual-observation interval monitor (paper Algorithm 2).

    Threshold noise is drawn once at construction; per-query noise is fresh at
    every step.  ``reset()`` redraws the threshold noise without re-seeding the
    rng, so reused instances see independent runs.
    """

    def __init__(
        self,
        lower_threshold: Callable[[int], float],
        upper_threshold: Callable[[int], float],
        query_noise_scale: float,
        threshold_noise_scale: float,
        epsilon: float,
        random_seed: Optional[int] = None,
    ):
        self.params = OutsideIntervalParameters(
            query_noise_scale=query_noise_scale,
            threshold_noise_scale=threshold_noise_scale,
            epsilon=epsilon,
        )
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold
        self._rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )
        self._core = self._make_core()

    def _make_core(self) -> OutsideIntervalCore:
        return OutsideIntervalCore(
            lower_threshold=self._lower_threshold,
            upper_threshold=self._upper_threshold,
            query_noise_scale=self.params.query_noise_scale,
            threshold_noise_scale=self.params.threshold_noise_scale,
            rng=self._rng,
        )

    def add_query(self, value: float) -> Tuple[bool, int]:
        """Feed the next query result; return ``(stopped, side)``.

        ``side`` is ``-1`` on lower-threshold crossing, ``+1`` on upper, ``0`` while running.
        """
        return self._core.add_query(value)

    def is_active(self) -> bool:
        return self._core.is_active()

    def reset(self) -> None:
        """Restart with fresh threshold noise (rng is advanced, not re-seeded)."""
        self._core = self._make_core()

    def get_state(self) -> Dict[str, Any]:
        return {
            "stopped": self._core.stopped,
            "side": self._core.side,
            "step": self._core.step,
            "epsilon": self.params.epsilon,
        }
