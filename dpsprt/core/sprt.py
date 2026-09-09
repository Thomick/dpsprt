"""Incremental SPRT and DP-SPRT classes.

Each algorithm processes Bernoulli observations one at a time via ``add_sample``
and returns ``(can_stop, decision)`` where ``decision`` is ``-1`` for H₀,
``+1`` for H₁, and ``0`` while sampling continues.

For all DP variants, the threshold noise ``Z`` is drawn once at ``reset()`` and
rescaled by ``1/t`` on the mean scale at each step, one Z shared by both
threshold comparisons.  This is the OutsideInterval mechanism of the paper.
``reset()`` advances the rng rather than re-seeding it, so reusing one seeded
instance across trials yields independent Z draws.

Noise is drawn with numpy floating-point arithmetic and a non-cryptographic
generator, so the guarantees are those of the idealized real-valued mechanisms.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import NUMERICAL_EPSILON, ZETA_S_PARAMETER, ZETA_S_VALUE
from .outside_interval import dpsprt_interval_check


def _theta(mu: float) -> float:
    p = np.clip(mu, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    return float(np.log(p / (1 - p)))


def _kl_bernoulli(p: float, q: float) -> float:
    p = np.clip(p, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    q = np.clip(q, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


class ClassicalSPRT:
    """Wald's Sequential Probability Ratio Test for Bernoulli observations."""

    def __init__(self, mu0: float, mu1: float, alpha: float, beta: float):
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta

        self.p0 = np.clip(mu0, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
        self.p1 = np.clip(mu1, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)

        self.upper_threshold = np.log(1 / alpha)
        self.lower_threshold = np.log(beta)

        self.log_ratio_success = np.log(self.p1 / self.p0)
        self.log_ratio_failure = np.log((1 - self.p1) / (1 - self.p0))

        self.reset()

    def reset(self):
        self.t = 0
        self.sum_x = 0
        self.llr = 0.0
        self.decision = 0
        self.stopped = False
        self.llr_trajectory = []

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if self.stopped:
            return True, self.decision

        self.t += 1
        self.sum_x += x
        self.llr += self.log_ratio_success if x == 1 else self.log_ratio_failure
        self.llr_trajectory.append(self.llr)

        if self.llr >= self.upper_threshold:
            self.decision = 1
            self.stopped = True
            return True, self.decision
        if self.llr <= self.lower_threshold:
            self.decision = -1
            self.stopped = True
            return True, self.decision
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "sample_count": self.t,
            "cumulative_sum": self.sum_x,
            "current_llr": self.llr,
            "decision": self.decision,
            "stopped": self.stopped,
            "llr_trajectory": self.llr_trajectory.copy(),
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
        }


class DPSPRT:
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
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.random_seed = random_seed

        self.rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )

        self.kl_h0_h1 = _kl_bernoulli(mu0, mu1)
        self.kl_h1_h0 = _kl_bernoulli(mu1, mu0)

        self.reset()

    def reset(self):
        self.t = 0
        self.sum_x = 0
        self.decision = 0
        self.stopped = False
        self.noisy_means = []
        self.auxiliary_noise = []
        # Threshold noise Z drawn once per run; rng is NOT re-seeded so reused
        # instances see an independent Z on every reset.
        self._aux_z_unit = self.rng.laplace(0.0, 1.0)

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if self.stopped:
            return True, self.decision

        self.t += 1
        self.sum_x += x
        current_mean = self.sum_x / self.t

        lap1 = self.rng.laplace(0, 4.0 / (self.t * self.epsilon))
        lap2 = self._aux_z_unit * (2.0 / (self.t * self.epsilon))
        noisy_mean = current_mean + lap1
        self.noisy_means.append(noisy_mean)
        self.auxiliary_noise.append(lap2)

        theta0, theta1 = _theta(self.mu0), _theta(self.mu1)
        s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE
        n = float(self.t)

        tau0_rhs = (
            (self.kl_h0_h1 - np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.beta)) / n)
            / (theta1 - theta0)
            - lap2
            - 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.beta)
            / (n * self.epsilon)
        )

        tau1_rhs = (
            (-self.kl_h1_h0 + np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.alpha)) / n)
            / (theta1 - theta0)
            + lap2
            + 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.alpha)
            / (n * self.epsilon)
        )

        stopped, side = dpsprt_interval_check(
            current_mean - self.mu0 + lap1,
            current_mean - self.mu1 + lap1,
            tau0_rhs,
            tau1_rhs,
        )
        if stopped:
            self.decision = side
            self.stopped = True
            return True, self.decision
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "decision": self.decision,
            "algorithm_type": "DP-SPRT-Laplace",
            "epsilon": self.epsilon,
        }

    def _get_internal_state(self) -> Dict[str, Any]:
        """Full internal state — NOT privacy-safe; debug/test use only."""
        return {
            "sample_count": self.t,
            "cumulative_sum": self.sum_x,
            "current_mean": self.sum_x / self.t if self.t > 0 else 0.0,
            "decision": self.decision,
            "stopped": self.stopped,
            "noisy_means": self.noisy_means.copy(),
            "auxiliary_noise": self.auxiliary_noise.copy(),
            "epsilon": self.epsilon,
        }


class DPSPRTGaussian:
    """(ε,δ)-DP SPRT with Gaussian noise (RDP analysis)."""

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
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.delta = delta
        self.random_seed = random_seed

        self.rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )

        self.kl_h0_h1 = _kl_bernoulli(mu0, mu1)
        self.kl_h1_h0 = _kl_bernoulli(mu1, mu0)

        self.reset()

    def reset(self):
        self.t = 0
        self.sum_x = 0
        self.decision = 0
        self.stopped = False
        self.noisy_means = []
        self._aux_z_unit = self.rng.normal(0.0, 1.0)

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if self.stopped:
            return True, self.decision

        self.t += 1
        self.sum_x += x
        current_mean = self.sum_x / self.t

        epsilon_prime = self.epsilon / 2
        gaussian_std1 = np.sqrt(8 * np.log(1.25 / self.delta)) / (2 * self.t * epsilon_prime)
        gaussian_std2 = np.sqrt(2 * np.log(1.25 / self.delta)) / (2 * self.t * epsilon_prime)

        gaussian_noise1 = self.rng.normal(0, gaussian_std1)
        aux_noise = self._aux_z_unit * gaussian_std2

        noisy_mean = current_mean + gaussian_noise1
        self.noisy_means.append(noisy_mean)

        theta0, theta1 = _theta(self.mu0), _theta(self.mu1)
        s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE
        n = float(self.t)

        threshold_term_0 = (
            self.kl_h0_h1 - np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon_prime) * self.beta)) / n
        ) / (theta1 - theta0)
        threshold_term_1 = (
            -self.kl_h1_h0 + np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon_prime) * self.alpha)) / n
        ) / (theta1 - theta0)

        privacy_term_0 = np.sqrt(
            2
            * (gaussian_std1**2 + gaussian_std2**2)
            * np.log(np.maximum(epsilon_prime, 2) * (n**s) * zeta_s / self.beta)
        )
        privacy_term_1 = np.sqrt(
            2
            * (gaussian_std1**2 + gaussian_std2**2)
            * np.log(np.maximum(epsilon_prime, 2) * (n**s) * zeta_s / self.alpha)
        )

        stopped, side = dpsprt_interval_check(
            noisy_mean - self.mu0,
            noisy_mean - self.mu1,
            threshold_term_0 - aux_noise - privacy_term_0,
            threshold_term_1 + aux_noise + privacy_term_1,
        )
        if stopped:
            self.decision = side
            self.stopped = True
            return True, self.decision
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "decision": self.decision,
            "algorithm_type": "DP-SPRT-Gaussian",
            "epsilon": self.epsilon,
            "delta": self.delta,
        }

    def _get_internal_state(self) -> Dict[str, Any]:
        """Full internal state — NOT privacy-safe; debug/test use only."""
        return {
            "sample_count": self.t,
            "cumulative_sum": self.sum_x,
            "current_mean": self.sum_x / self.t if self.t > 0 else 0.0,
            "decision": self.decision,
            "stopped": self.stopped,
            "noisy_means": self.noisy_means.copy(),
            "epsilon": self.epsilon,
            "delta": self.delta,
        }


class DPSPRTTuned:
    """DP-SPRT (Laplace) with tunable threshold and privacy-correction constants.

    ``c1=c2=1`` recovers :class:`DPSPRT`.  ``c2 < 1`` has no privacy proof.
    """

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
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.random_seed = random_seed

        self.rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )

        self.kl_h0_h1 = _kl_bernoulli(mu0, mu1)
        self.kl_h1_h0 = _kl_bernoulli(mu1, mu0)

        self.reset()

    def reset(self):
        self.t = 0
        self.sum_x = 0
        self.decision = 0
        self.stopped = False
        self.noisy_means = []
        self.auxiliary_noise = []
        self._aux_z_unit = self.rng.laplace(0.0, 1.0)

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if self.stopped:
            return True, self.decision

        self.t += 1
        self.sum_x += x
        current_mean = self.sum_x / self.t

        mean_noise = self.rng.laplace(0, 4.0 / (self.t * self.epsilon))
        aux_noise = self._aux_z_unit * (2.0 / (self.t * self.epsilon))
        noisy_mean = current_mean + mean_noise
        self.noisy_means.append(noisy_mean)
        self.auxiliary_noise.append(aux_noise)

        theta0, theta1 = _theta(self.mu0), _theta(self.mu1)
        s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE
        n = float(self.t)

        threshold_term_0 = (
            self.kl_h0_h1
            - self.c1 * np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.beta)) / n
        ) / (theta1 - theta0)
        privacy_term_0 = (
            self.c2
            * 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.beta)
            / (n * self.epsilon)
        )

        threshold_term_1 = (
            -self.kl_h1_h0
            + self.c1 * np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.alpha)) / n
        ) / (theta1 - theta0)
        privacy_term_1 = (
            self.c2
            * 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.alpha)
            / (n * self.epsilon)
        )

        stopped, side = dpsprt_interval_check(
            noisy_mean - self.mu0,
            noisy_mean - self.mu1,
            threshold_term_0 - aux_noise - privacy_term_0,
            threshold_term_1 + aux_noise + privacy_term_1,
        )
        if stopped:
            self.decision = side
            self.stopped = True
            return True, self.decision
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "decision": self.decision,
            "algorithm_type": "DP-SPRT-Tuned",
            "epsilon": self.epsilon,
            "c1": self.c1,
            "c2": self.c2,
        }

    def _get_internal_state(self) -> Dict[str, Any]:
        """Full internal state — NOT privacy-safe; debug/test use only."""
        return {
            "sample_count": self.t,
            "cumulative_sum": self.sum_x,
            "current_mean": self.sum_x / self.t if self.t > 0 else 0.0,
            "decision": self.decision,
            "stopped": self.stopped,
            "noisy_means": self.noisy_means.copy(),
            "auxiliary_noise": self.auxiliary_noise.copy(),
            "epsilon": self.epsilon,
            "c1": self.c1,
            "c2": self.c2,
        }


class DPSPRTSubsampled:
    """ε-DP SPRT with Bernoulli subsampling for privacy amplification.

    Subsampling rate ``r = min(1, sqrt(eps/10))``.  Stopping conditions follow the
    subsampling appendix of arXiv:2508.06377, whose denominator is the realised
    subsample count ``M_n``.  The JAX code that produced the paper's figures uses
    the deterministic ``r * n`` instead, so stopping times differ slightly.
    """

    def __init__(
        self,
        mu0: float,
        mu1: float,
        alpha: float,
        beta: float,
        epsilon: float,
        random_seed: Optional[int] = None,
    ):
        self.mu0 = mu0
        self.mu1 = mu1
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.random_seed = random_seed

        self.rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )

        self.kl_h0_h1 = _kl_bernoulli(mu0, mu1)
        self.kl_h1_h0 = _kl_bernoulli(mu1, mu0)

        self.reset()

    def reset(self):
        self.t = 0
        self.t_subsample = 0
        self.sum_x_subsample = 0
        self.decision = 0
        self.stopped = False
        self.subsampling_decisions = []
        self.noisy_means = []
        self._aux_z_unit = self.rng.laplace(0.0, 1.0)

    def add_sample(self, x: int) -> Tuple[bool, int]:
        if self.stopped:
            return True, self.decision

        self.t += 1
        q = min(1.0, np.sqrt(self.epsilon / 10))
        keep_sample = self.rng.random() < q
        self.subsampling_decisions.append(keep_sample)

        if keep_sample:
            self.t_subsample += 1
            self.sum_x_subsample += x

        # Before the first included sample X̄^r_n is undefined; use a
        # data-independent neutral value (midpoint of the two hypotheses).
        # Noise is drawn unconditionally; stopping is suppressed until M_n ≥ 1.
        if self.t_subsample == 0:
            current_mean = 0.5 * (self.mu0 + self.mu1)
        else:
            current_mean = self.sum_x_subsample / self.t_subsample

        laplace_noise1 = self.rng.laplace(0, 4 / self.epsilon)
        laplace_noise2 = (2.0 / self.epsilon) * self._aux_z_unit
        noisy_mean = current_mean + q * laplace_noise1 / self.t
        aux_noise = q * laplace_noise2 / self.t
        self.noisy_means.append(noisy_mean)

        if self.t_subsample == 0:
            return False, 0

        theta0, theta1 = _theta(self.mu0), _theta(self.mu1)
        s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE
        n = float(self.t)
        m_n = float(self.t_subsample)

        threshold_term_0 = (
            self.kl_h0_h1 - np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.beta)) / m_n
        ) / (theta1 - theta0)
        privacy_term_0 = (
            q
            * 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.beta)
            / (n * self.epsilon)
        )

        threshold_term_1 = (
            -self.kl_h1_h0 + np.log(1 / (np.maximum(0.5, 1 - 1 / self.epsilon) * self.alpha)) / m_n
        ) / (theta1 - theta0)
        privacy_term_1 = (
            q
            * 6
            * np.log(np.maximum(self.epsilon, 2) * (n**s) * zeta_s / self.alpha)
            / (n * self.epsilon)
        )

        stopped, side = dpsprt_interval_check(
            noisy_mean - self.mu0,
            noisy_mean - self.mu1,
            threshold_term_0 - aux_noise - privacy_term_0,
            threshold_term_1 + aux_noise + privacy_term_1,
        )
        if stopped:
            self.decision = side
            self.stopped = True
            return True, self.decision
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "stopped": self.stopped,
            "decision": self.decision,
            "algorithm_type": "DP-SPRT-Subsampled",
            "epsilon": self.epsilon,
        }

    def _get_internal_state(self) -> Dict[str, Any]:
        """Full internal state — NOT privacy-safe; debug/test use only."""
        return {
            "total_samples": self.t,
            "subsampled_count": self.t_subsample,
            "subsampled_sum": self.sum_x_subsample,
            "subsampled_mean": (
                self.sum_x_subsample / self.t_subsample if self.t_subsample > 0 else 0.0
            ),
            "decision": self.decision,
            "stopped": self.stopped,
            "subsampling_decisions": self.subsampling_decisions.copy(),
            "noisy_means": self.noisy_means.copy(),
            "epsilon": self.epsilon,
        }
