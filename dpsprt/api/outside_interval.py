"""User-facing wrapper for the OutsideInterval DP primitive."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..core.outside_interval import OutsideIntervalCore


@dataclass
class OutsideIntervalParameters:
    """Validated noise scales, sensitivity and declared eps for OutsideInterval."""

    query_noise_scale: float
    threshold_noise_scale: float
    epsilon: float
    sensitivity: float = 1.0

    def __post_init__(self):
        if self.query_noise_scale <= 0:
            raise ValueError(f"query_noise_scale must be > 0, got {self.query_noise_scale}")
        if self.threshold_noise_scale <= 0:
            raise ValueError(f"threshold_noise_scale must be > 0, got {self.threshold_noise_scale}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {self.sensitivity}")

    def implied_epsilon(self) -> float:
        """Budget the Laplace scales actually implement, eps_Z + eps_Y (Theorem 1(i)).

        Z answers a sensitivity-``sensitivity`` query and Y a sensitivity-``2 * sensitivity``
        one, so eps_Z = sensitivity / threshold_noise_scale and
        eps_Y = 2 * sensitivity / query_noise_scale.
        """
        eps_z = self.sensitivity / self.threshold_noise_scale
        eps_y = 2.0 * self.sensitivity / self.query_noise_scale
        return eps_z + eps_y


class OutsideInterval:
    """eps-DP continual-observation interval monitor (Algorithm 2 of the paper).

    Halts at the first step where the noisy query leaves ``[T0(t) - Z, T1(t) + Z]``.
    A single Z is drawn at construction and shared by both comparisons, which is
    what makes the mechanism (eps_Z + eps_Y)-DP rather than costing two composed
    AboveThreshold instances.  Query noise is fresh at every step.  ``reset()``
    redraws Z without re-seeding the rng, so reused instances give independent runs.

    The noise scales alone set the guarantee.  ``epsilon`` is the budget you
    declare, reported back by ``get_state()`` and never used to compute noise.
    The constructor checks it against ``eps_Z + eps_Y`` implied by the scales and
    raises when they disagree by more than ``epsilon_tol`` in relative terms; pass
    ``epsilon_tol=None`` to skip the check, for instance when supplying
    non-Laplace noise scales.

    The caller is responsible for feeding query values of sensitivity at most
    ``sensitivity``.
    """

    def __init__(
        self,
        lower_threshold: Callable[[int], float],
        upper_threshold: Callable[[int], float],
        query_noise_scale: float,
        threshold_noise_scale: float,
        epsilon: float,
        random_seed: Optional[int] = None,
        sensitivity: float = 1.0,
        epsilon_tol: Optional[float] = 1e-6,
    ):
        self.params = OutsideIntervalParameters(
            query_noise_scale=query_noise_scale,
            threshold_noise_scale=threshold_noise_scale,
            epsilon=epsilon,
            sensitivity=sensitivity,
        )
        if epsilon_tol is not None:
            implied = self.params.implied_epsilon()
            if abs(implied - epsilon) > epsilon_tol * max(implied, epsilon):
                raise ValueError(
                    f"declared epsilon={epsilon} does not match the budget the noise "
                    f"scales implement, eps_Z + eps_Y = {implied:.6g}, for sensitivity "
                    f"{sensitivity}. Fix the scales or the declared epsilon, or pass "
                    "epsilon_tol=None to skip this check."
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
            "sensitivity": self.params.sensitivity,
        }
