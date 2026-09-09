"""Parameter dataclasses for SPRT and DP-SPRT algorithms."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SPRTParameters:
    """Hypothesis-testing parameters: H₀ at μ₀, H₁ at μ₁, target errors α and β."""

    mu0: float
    mu1: float
    alpha: float
    beta: float

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not (0 < self.mu0 < 1):
            raise ValueError(f"mu0 must be in (0, 1), got {self.mu0}")
        if not (0 < self.mu1 < 1):
            raise ValueError(f"mu1 must be in (0, 1), got {self.mu1}")
        if self.mu0 == self.mu1:
            raise ValueError(f"mu0 and mu1 must differ, got {self.mu0}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0 < self.beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")

    def to_dict(self) -> Dict[str, Any]:
        return {"mu0": self.mu0, "mu1": self.mu1, "alpha": self.alpha, "beta": self.beta}

    def effect_size(self) -> float:
        return abs(self.mu1 - self.mu0)

    def power(self) -> float:
        return 1.0 - self.beta


@dataclass
class DPSPRTParameters(SPRTParameters):
    """SPRT parameters plus the ε privacy budget."""

    epsilon: float

    def _validate(self):
        super()._validate()
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["epsilon"] = self.epsilon
        return result


@dataclass
class DPSPRTGaussianParameters(DPSPRTParameters):
    """DP-SPRT parameters with the additional δ for (ε,δ)-DP."""

    delta: float

    def _validate(self):
        super()._validate()
        if not (0 < self.delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["delta"] = self.delta
        return result


@dataclass
class DPSPRTTunedParameters(DPSPRTParameters):
    """DP-SPRT parameters scaling the threshold by ``c1`` and the correction by ``c2``.

    The paper's correction function is ``C(n, delta) = 6 log(n^s zeta(s) / delta) / (n eps)``,
    and ``c2`` scales it, playing the role of the paper's kappa.  Privacy holds for
    every ``c2 > 0``, since the noise is unchanged; only the ``(alpha, beta)``
    guarantee needs ``c2 >= 1``.  The paper's tuned experiment runs at kappa about
    0.5, and that factor has to be re-estimated for each new set of parameters.
    """

    c1: float = 1.0
    c2: float = 1.0

    def _validate(self):
        super()._validate()
        if self.c1 <= 0:
            raise ValueError(f"c1 must be positive, got {self.c1}")
        if self.c2 <= 0:
            raise ValueError(f"c2 must be positive, got {self.c2}")

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({"c1": self.c1, "c2": self.c2})
        return result


def validate_bernoulli_parameters(mu0: float, mu1: float) -> None:
    if not (0 < mu0 < 1):
        raise ValueError(f"mu0 must be in (0, 1), got {mu0}")
    if not (0 < mu1 < 1):
        raise ValueError(f"mu1 must be in (0, 1), got {mu1}")
    if mu0 == mu1:
        raise ValueError("mu0 and mu1 must differ")


def validate_error_rates(alpha: float, beta: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")


def validate_privacy_parameters(epsilon: float, delta: Optional[float] = None) -> None:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta is not None and not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
