"""Empirical DP-guarantee tests for the DP-SPRT variants.

These tests run each algorithm on neighbouring datasets and check
that the empirical decision distribution is consistent with the
claimed (epsilon, delta)-DP. Tests are slow and gated by the
'slow' marker (registered in pyproject.toml).
"""

from collections import Counter

import numpy as np
import pytest

from dpsprt import DPSPRT, DPSPRTGaussian, DPSPRTSubsampled

pytestmark = pytest.mark.slow


def _run_dpsprt_to_decision(data, eps, seed):
    """Run DPSPRT on a fixed dataset and return (decision, stopping-time-bucket)."""
    test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.1, beta=0.1, epsilon=eps, random_seed=seed)
    for i, x in enumerate(data):
        can_stop, decision = test.add_sample(int(x))
        if can_stop:
            return (decision, i // 50)
    return (0, len(data) // 50)


def test_dpsprt_laplace_empirical_dp_neighbours():
    """Empirical sanity check: max log-ratio across outcome buckets ≤ eps + slack.

    Two neighbouring datasets (differ in element 0) are each run 10 000 times
    with different noise seeds.  We bucket outcomes by (decision, stopping-time /
    50) and check the smoothed log-likelihood ratio.  This is NOT a formal proof;
    a failure is a real signal to investigate.
    """
    eps = 1.0
    n_trials = 10_000
    # Neighbouring datasets: differ only at index 0 (1 vs 0)
    base = np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], 200)
    data_a = np.concatenate([np.array([1]), base[1:]])
    data_b = np.concatenate([np.array([0]), base[1:]])
    counts_a = Counter(_run_dpsprt_to_decision(data_a, eps, s) for s in range(n_trials))
    counts_b = Counter(_run_dpsprt_to_decision(data_b, eps, s + n_trials) for s in range(n_trials))
    keys = set(counts_a) | set(counts_b)
    max_log_ratio = 0.0
    for k in keys:
        # Laplace rule-of-succession smoothing avoids log(0)
        pa = (counts_a.get(k, 0) + 1) / (n_trials + len(keys))
        pb = (counts_b.get(k, 0) + 1) / (n_trials + len(keys))
        max_log_ratio = max(max_log_ratio, abs(np.log(pa / pb)))
    assert (
        max_log_ratio <= eps + 0.5
    ), f"max log-ratio {max_log_ratio:.3f} exceeds eps+slack={eps + 0.5}"


def _run_dpsprt_gaussian_to_decision(data, eps, delta, seed):
    """Run DPSPRTGaussian on a fixed dataset and return (decision, stopping-time-bucket)."""
    test = DPSPRTGaussian(
        mu0=0.3,
        mu1=0.7,
        alpha=0.1,
        beta=0.1,
        epsilon=eps,
        delta=delta,
        random_seed=seed,
    )
    for i, x in enumerate(data):
        can_stop, decision = test.add_sample(int(x))
        if can_stop:
            return (decision, i // 50)
    return (0, len(data) // 50)


def test_dpsprt_gaussian_empirical_eps_delta_dp():
    """Empirical (ε,δ)-DP sanity check for DPSPRTGaussian.

    Same setup as the Laplace test but the assertion drops the top-δ fraction
    of outcome ratios before checking the remainder against ε + slack.
    """
    eps, delta = 1.0, 1e-3
    n_trials = 10_000
    base = np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], 200)
    data_a = np.concatenate([np.array([1]), base[1:]])
    data_b = np.concatenate([np.array([0]), base[1:]])
    counts_a = Counter(
        _run_dpsprt_gaussian_to_decision(data_a, eps, delta, s) for s in range(n_trials)
    )
    counts_b = Counter(
        _run_dpsprt_gaussian_to_decision(data_b, eps, delta, s + n_trials) for s in range(n_trials)
    )
    keys = set(counts_a) | set(counts_b)
    ratios = sorted(
        abs(
            np.log(
                (counts_a.get(k, 0) + 1)
                / (n_trials + len(keys))
                / ((counts_b.get(k, 0) + 1) / (n_trials + len(keys)))
            )
        )
        for k in keys
    )
    # Drop the top-δ fraction before checking the high-probability bound
    cutoff = max(0, int(len(ratios) * (1 - delta)))
    high_prob_max = ratios[cutoff - 1] if cutoff > 0 else 0.0
    assert (
        high_prob_max <= eps + 0.5
    ), f"high-prob max log-ratio {high_prob_max:.3f} exceeds eps+slack={eps + 0.5}"


def _run_dpsprt_subsampled_to_decision(data, eps, seed):
    """Run DPSPRTSubsampled on a fixed dataset and return (decision, stopping-time-bucket)."""
    test = DPSPRTSubsampled(mu0=0.3, mu1=0.7, alpha=0.1, beta=0.1, epsilon=eps, random_seed=seed)
    for i, x in enumerate(data):
        can_stop, decision = test.add_sample(int(x))
        if can_stop:
            return (decision, i // 50)
    return (0, len(data) // 50)


def test_dpsprt_subsampled_empirical_dp_neighbours():
    """Empirical DP sanity check for DPSPRTSubsampled (same setup as Laplace test)."""
    eps = 1.0
    n_trials = 10_000
    base = np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], 200)
    data_a = np.concatenate([np.array([1]), base[1:]])
    data_b = np.concatenate([np.array([0]), base[1:]])
    counts_a = Counter(_run_dpsprt_subsampled_to_decision(data_a, eps, s) for s in range(n_trials))
    counts_b = Counter(
        _run_dpsprt_subsampled_to_decision(data_b, eps, s + n_trials) for s in range(n_trials)
    )
    keys = set(counts_a) | set(counts_b)
    max_log_ratio = 0.0
    for k in keys:
        pa = (counts_a.get(k, 0) + 1) / (n_trials + len(keys))
        pb = (counts_b.get(k, 0) + 1) / (n_trials + len(keys))
        max_log_ratio = max(max_log_ratio, abs(np.log(pa / pb)))
    assert (
        max_log_ratio <= eps + 0.5
    ), f"max log-ratio {max_log_ratio:.3f} exceeds eps+slack={eps + 0.5}"
