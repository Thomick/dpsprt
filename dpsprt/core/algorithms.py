"""Vectorized one-shot SPRT and DP-SPRT implementations.

Each function processes an entire observation sequence and returns
``(decision, stopping_time, info)``.  For incremental, streaming use see
:mod:`dpsprt.core.sprt`.

In every DP variant the threshold noise ``Z`` is sampled once (sum scale)
and rescaled by ``1/t`` on the mean scale, matching the OutsideInterval
mechanism from the paper.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import NUMERICAL_EPSILON, ZETA_S_PARAMETER, ZETA_S_VALUE


def _theta(mu: np.ndarray) -> np.ndarray:
    p = np.clip(mu, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    return np.log(p / (1 - p))


def _kl_bernoulli(p, q):
    p_clip = np.clip(p, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    q_clip = np.clip(q, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    return p_clip * np.log(p_clip / q_clip) + (1 - p_clip) * np.log((1 - p_clip) / (1 - q_clip))


def _first_crossing(condition_0: np.ndarray, condition_1: np.ndarray, n: int) -> Tuple[int, int]:
    """Resolve the first of two boolean-trajectory crossings into ``(decision, stopping_time)``."""
    idx0 = np.where(condition_0)[0]
    idx1 = np.where(condition_1)[0]
    if len(idx0) == 0 and len(idx1) == 0:
        return 0, n
    if len(idx0) == 0:
        return 1, int(idx1[0]) + 1
    if len(idx1) == 0:
        return -1, int(idx0[0]) + 1
    if idx0[0] < idx1[0]:
        return -1, int(idx0[0]) + 1
    return 1, int(idx1[0]) + 1


def classical_sprt(
    data: np.ndarray, mu0: float, mu1: float, alpha: float, beta: float
) -> Tuple[int, int, Dict[str, Any]]:
    """Wald's SPRT for Bernoulli observations (non-private baseline)."""
    n = len(data)
    cumsum = np.cumsum(data)
    steps = np.arange(1, n + 1)

    p0 = np.clip(mu0, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    p1 = np.clip(mu1, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    llr = cumsum * np.log(p1 / p0) + (steps - cumsum) * np.log((1 - p1) / (1 - p0))

    upper_threshold = np.log(1 / alpha)
    lower_threshold = np.log(beta)

    upper_crossings = np.where(llr >= upper_threshold)[0]
    lower_crossings = np.where(llr <= lower_threshold)[0]

    if len(upper_crossings) == 0 and len(lower_crossings) == 0:
        decision, stopping_time, first_cross_idx = 0, n, n - 1
    elif len(upper_crossings) == 0:
        first_cross_idx = lower_crossings[0]
        decision, stopping_time = -1, first_cross_idx + 1
    elif len(lower_crossings) == 0:
        first_cross_idx = upper_crossings[0]
        decision, stopping_time = 1, first_cross_idx + 1
    elif upper_crossings[0] < lower_crossings[0]:
        first_cross_idx = upper_crossings[0]
        decision, stopping_time = 1, first_cross_idx + 1
    else:
        first_cross_idx = lower_crossings[0]
        decision, stopping_time = -1, first_cross_idx + 1

    info = {
        "llr_trajectory": llr,
        "cumulative_sum": cumsum,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "upper_crossings": upper_crossings,
        "lower_crossings": lower_crossings,
        "stopping_index": first_cross_idx,
    }
    return decision, stopping_time, info


def dp_sprt_laplace(
    data: np.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    random_seed: Optional[int] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """ε-DP SPRT with Laplace noise."""
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    cumsum = np.cumsum(data)
    steps = np.arange(1, n + 1).astype(float)
    means = cumsum / steps

    laplace_noise1 = np.random.laplace(0, 4 / (steps * epsilon))
    z = np.random.laplace(0.0, 1.0)
    laplace_noise2 = z * (2 / (steps * epsilon))

    noisy_means = means + laplace_noise1
    aux_noise = laplace_noise2

    theta0, theta1 = _theta(mu0), _theta(mu1)
    s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE

    threshold_term_0 = (
        _kl_bernoulli(mu0, mu1)
        - np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * beta)) / steps
    ) / (theta1 - theta0)
    privacy_term_0 = 6 * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / beta) / (steps * epsilon)
    condition_0 = noisy_means - mu0 <= threshold_term_0 - aux_noise - privacy_term_0

    threshold_term_1 = (
        -_kl_bernoulli(mu1, mu0)
        + np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * alpha)) / steps
    ) / (theta1 - theta0)
    privacy_term_1 = 6 * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / alpha) / (steps * epsilon)
    condition_1 = noisy_means - mu1 >= threshold_term_1 + aux_noise + privacy_term_1

    decision, stopping_time = _first_crossing(condition_0, condition_1, n)
    info = {
        "means": means,
        "noisy_means": noisy_means,
        "laplace_noise1": laplace_noise1,
        "laplace_noise2": laplace_noise2,
        "condition_0": condition_0,
        "condition_1": condition_1,
        "threshold_term_0": threshold_term_0,
        "threshold_term_1": threshold_term_1,
        "privacy_term_0": privacy_term_0,
        "privacy_term_1": privacy_term_1,
        "kl_div_01": _kl_bernoulli(mu0, mu1),
        "kl_div_10": _kl_bernoulli(mu1, mu0),
    }
    return decision, stopping_time, info


def dp_sprt_subsampled(
    data: np.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    random_seed: Optional[int] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """ε-DP SPRT with Bernoulli subsampling (rate ``r = min(1, sqrt(ε/10))``)."""
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    q = min(1.0, np.sqrt(epsilon / 10))
    include_mask = np.random.random(n) < q
    data_masked = np.where(include_mask, data, 0.0)
    include_counts = include_mask.astype(float)
    cumsum_included = np.cumsum(data_masked)
    cumcount_included = np.cumsum(include_counts)
    steps = np.arange(1, n + 1).astype(float)

    # Before the first included sample, X̄^r_n is undefined; substitute a
    # data-independent neutral value (midpoint of the two hypotheses) and
    # suppress stopping at those steps below.
    neutral_mean = 0.5 * (mu0 + mu1)
    means = np.where(
        cumcount_included > 0,
        cumsum_included / np.maximum(cumcount_included, 1.0),
        neutral_mean,
    )
    has_sample = cumcount_included > 0

    laplace_noise1 = np.random.laplace(0, q * 4 / (steps * epsilon))
    z = np.random.laplace(0.0, 1.0)
    laplace_noise2 = z * (2 * q / (steps * epsilon))

    noisy_means = means + laplace_noise1
    aux_noise = laplace_noise2

    theta0, theta1 = _theta(mu0), _theta(mu1)
    s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE
    m_n = np.maximum(cumcount_included, 1.0)  # safe denominator; conditions are masked by has_sample

    threshold_term_0 = (
        _kl_bernoulli(mu0, mu1)
        - np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * beta)) / m_n
    ) / (theta1 - theta0)
    privacy_term_0 = (
        6 * q * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / beta) / (steps * epsilon)
    )
    condition_0 = (
        noisy_means - mu0 <= threshold_term_0 - aux_noise - privacy_term_0
    ) & has_sample

    threshold_term_1 = (
        -_kl_bernoulli(mu1, mu0)
        + np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * alpha)) / m_n
    ) / (theta1 - theta0)
    privacy_term_1 = (
        6 * q * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / alpha) / (steps * epsilon)
    )
    condition_1 = (
        noisy_means - mu1 >= threshold_term_1 + aux_noise + privacy_term_1
    ) & has_sample

    decision, stopping_time = _first_crossing(condition_0, condition_1, n)
    info = {
        "subsampling_rate": q,
        "include_mask": include_mask,
        "means": means,
        "noisy_means": noisy_means,
        "samples_included": cumcount_included,
        "laplace_noise1": laplace_noise1,
        "laplace_noise2": laplace_noise2,
        "condition_0": condition_0,
        "condition_1": condition_1,
    }
    return decision, stopping_time, info


def dp_sprt_tuned(
    data: np.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    c1: float = 1.0,
    c2: float = 1.0,
    random_seed: Optional[int] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """DP-SPRT (Laplace) with tunable constants ``c1`` (threshold) and ``c2`` (privacy term).

    ``c1 = c2 = 1`` recovers :func:`dp_sprt_laplace`.  ``c2 < 1`` has no privacy proof.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    cumsum = np.cumsum(data)
    steps = np.arange(1, n + 1).astype(float)
    means = cumsum / steps

    laplace_noise1 = np.random.laplace(0, 4 / (steps * epsilon))
    z = np.random.laplace(0.0, 1.0)
    laplace_noise2 = z * (2 / (steps * epsilon))

    noisy_means = means + laplace_noise1
    aux_noise = laplace_noise2

    theta0, theta1 = _theta(mu0), _theta(mu1)
    s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE

    threshold_term_0 = (
        _kl_bernoulli(mu0, mu1)
        - c1 * np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * beta)) / steps
    ) / (theta1 - theta0)
    privacy_term_0 = (
        c2 * 6 * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / beta) / (steps * epsilon)
    )
    condition_0 = noisy_means - mu0 <= threshold_term_0 - aux_noise - privacy_term_0

    threshold_term_1 = (
        -_kl_bernoulli(mu1, mu0)
        + c1 * np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon) * alpha)) / steps
    ) / (theta1 - theta0)
    privacy_term_1 = (
        c2 * 6 * np.log(np.maximum(epsilon, 2) * (steps**s) * zeta_s / alpha) / (steps * epsilon)
    )
    condition_1 = noisy_means - mu1 >= threshold_term_1 + aux_noise + privacy_term_1

    decision, stopping_time = _first_crossing(condition_0, condition_1, n)
    info = {
        "c1": c1,
        "c2": c2,
        "means": means,
        "noisy_means": noisy_means,
        "laplace_noise1": laplace_noise1,
        "laplace_noise2": laplace_noise2,
        "threshold_term_0": threshold_term_0,
        "threshold_term_1": threshold_term_1,
        "privacy_term_0": privacy_term_0,
        "privacy_term_1": privacy_term_1,
        "condition_0": condition_0,
        "condition_1": condition_1,
    }
    return decision, stopping_time, info


def dp_sprt_gaussian(
    data: np.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,
    random_seed: Optional[int] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """(ε,δ)-DP SPRT with Gaussian noise (RDP analysis)."""
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(data)
    cumsum = np.cumsum(data)
    steps = np.arange(1, n + 1).astype(float)
    means = cumsum / steps

    epsilon_prime = epsilon / 2
    gaussian_std1 = np.sqrt(8 * np.log(1.25 / delta)) / (2 * epsilon_prime * steps)
    gaussian_std2 = np.sqrt(2 * np.log(1.25 / delta)) / (2 * epsilon_prime * steps)

    gaussian_noise1 = np.random.normal(0, gaussian_std1)
    z = np.random.normal(0.0, 1.0)
    gaussian_noise2 = z * gaussian_std2

    noisy_means = means + gaussian_noise1
    aux_noise = gaussian_noise2

    theta0, theta1 = _theta(mu0), _theta(mu1)
    s, zeta_s = ZETA_S_PARAMETER, ZETA_S_VALUE

    threshold_term_0 = (
        _kl_bernoulli(mu0, mu1)
        - np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon_prime) * beta)) / steps
    ) / (theta1 - theta0)
    privacy_term_0 = np.sqrt(
        2 * (gaussian_std1**2 + gaussian_std2**2)
        * np.log(np.maximum(epsilon_prime, 2) * (steps**s) * zeta_s / beta)
    )
    condition_0 = noisy_means - mu0 <= threshold_term_0 - aux_noise - privacy_term_0

    threshold_term_1 = (
        -_kl_bernoulli(mu1, mu0)
        + np.log(1 / (np.maximum(0.5, 1 - 1 / epsilon_prime) * alpha)) / steps
    ) / (theta1 - theta0)
    privacy_term_1 = np.sqrt(
        2 * (gaussian_std1**2 + gaussian_std2**2)
        * np.log(np.maximum(epsilon_prime, 2) * (steps**s) * zeta_s / alpha)
    )
    condition_1 = noisy_means - mu1 >= threshold_term_1 + aux_noise + privacy_term_1

    decision, stopping_time = _first_crossing(condition_0, condition_1, n)
    info = {
        "means": means,
        "noisy_means": noisy_means,
        "gaussian_noise1": gaussian_noise1,
        "gaussian_noise2": gaussian_noise2,
        "gaussian_std1": gaussian_std1,
        "gaussian_std2": gaussian_std2,
        "epsilon_prime": epsilon_prime,
        "delta": delta,
        "threshold_term_0": threshold_term_0,
        "threshold_term_1": threshold_term_1,
        "privacy_term_0": privacy_term_0,
        "privacy_term_1": privacy_term_1,
        "condition_0": condition_0,
        "condition_1": condition_1,
    }
    return decision, stopping_time, info
