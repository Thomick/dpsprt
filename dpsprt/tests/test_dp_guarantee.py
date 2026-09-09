"""Empirical DP-guarantee tests for the DP-SPRT variants.

Each test runs an algorithm on two neighbouring datasets and checks that the
empirical distribution over outcomes is consistent with the claimed guarantee.
This is a sanity check and not a proof; a failure is a real signal to investigate.

Outcomes are bucketed by decision and stopping-time band. A bucket seen a handful
of times out of ten thousand trials cannot resolve a log-ratio to within the
tolerance, so the comparison runs only over buckets carrying enough observations,
and each test asserts that those buckets hold nearly all the mass. Without that
second assertion the filter could pass by discarding everything.

Tests are slow and gated by the 'slow' marker (registered in pyproject.toml).
"""

from collections import Counter

import numpy as np
import pytest

from dpsprt import DPSPRT, DPSPRTGaussian, DPSPRTSubsampled

pytestmark = pytest.mark.slow

N_TRIALS = 10_000
BUCKET = 50
# A bucket needs this many observations in its larger arm before its empirical
# log-ratio is worth comparing against epsilon.
MIN_COUNT = 30
# The retained buckets must cover at least this share of all outcomes.
MIN_COVERAGE = 0.95


def _neighbouring_datasets():
    """Two observation sequences differing only in the first element."""
    base = np.tile([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], 200)
    return (
        np.concatenate([np.array([1]), base[1:]]),
        np.concatenate([np.array([0]), base[1:]]),
    )


def _outcome_counts(run_one, data_a, data_b):
    counts_a = Counter(run_one(data_a, seed) for seed in range(N_TRIALS))
    counts_b = Counter(run_one(data_b, seed + N_TRIALS) for seed in range(N_TRIALS))
    return counts_a, counts_b


def _log_ratios(counts_a, counts_b):
    """Per-bucket |log(p_a / p_b)| with add-one smoothing, plus coverage.

    Returns (ratios_over_well_populated_buckets, coverage_of_those_buckets).
    """
    keys = set(counts_a) | set(counts_b)
    ratios, covered, total = [], 0, 2 * N_TRIALS
    for key in keys:
        n_a, n_b = counts_a.get(key, 0), counts_b.get(key, 0)
        p_a = (n_a + 1) / (N_TRIALS + len(keys))
        p_b = (n_b + 1) / (N_TRIALS + len(keys))
        if max(n_a, n_b) >= MIN_COUNT:
            ratios.append(abs(float(np.log(p_a / p_b))))
            covered += n_a + n_b
    return ratios, covered / total


def _assert_consistent_with_epsilon(counts_a, counts_b, eps, slack, label):
    ratios, coverage = _log_ratios(counts_a, counts_b)
    assert coverage >= MIN_COVERAGE, (
        f"{label}: well-populated buckets cover only {coverage:.2%} of outcomes, "
        "so this check would be vacuous"
    )
    assert ratios, f"{label}: no bucket reached {MIN_COUNT} observations"
    worst = max(ratios)
    assert worst <= eps + slack, (
        f"{label}: max log-ratio {worst:.3f} over {len(ratios)} buckets covering "
        f"{coverage:.2%} of outcomes exceeds eps+slack={eps + slack}"
    )


def _to_outcome(test, data):
    """Run to a decision and bucket the outcome as (decision, stopping_time // BUCKET)."""
    for i, x in enumerate(data):
        can_stop, decision = test.add_sample(int(x))
        if can_stop:
            return decision, i // BUCKET
    return 0, len(data) // BUCKET


def test_dpsprt_laplace_empirical_dp_neighbours():
    """Pure eps-DP sanity check for DPSPRT."""
    eps = 1.0
    data_a, data_b = _neighbouring_datasets()

    def run_one(data, seed):
        return _to_outcome(
            DPSPRT(mu0=0.3, mu1=0.7, alpha=0.1, beta=0.1, epsilon=eps, random_seed=seed),
            data,
        )

    counts_a, counts_b = _outcome_counts(run_one, data_a, data_b)
    _assert_consistent_with_epsilon(counts_a, counts_b, eps, 0.5, "DPSPRT")


def test_dpsprt_gaussian_empirical_eps_delta_dp():
    """(eps, delta)-DP sanity check for DPSPRTGaussian.

    The delta allowance is left in the slack rather than modelled, since dropping
    a top-delta fraction of ten buckets cannot resolve delta = 1e-3.
    """
    eps, delta = 1.0, 1e-3
    data_a, data_b = _neighbouring_datasets()

    def run_one(data, seed):
        return _to_outcome(
            DPSPRTGaussian(
                mu0=0.3,
                mu1=0.7,
                alpha=0.1,
                beta=0.1,
                epsilon=eps,
                delta=delta,
                random_seed=seed,
            ),
            data,
        )

    counts_a, counts_b = _outcome_counts(run_one, data_a, data_b)
    _assert_consistent_with_epsilon(counts_a, counts_b, eps, 0.5, "DPSPRTGaussian")


def test_dpsprt_subsampled_empirical_dp_neighbours():
    """Pure eps-DP sanity check for DPSPRTSubsampled."""
    eps = 1.0
    data_a, data_b = _neighbouring_datasets()

    def run_one(data, seed):
        return _to_outcome(
            DPSPRTSubsampled(mu0=0.3, mu1=0.7, alpha=0.1, beta=0.1, epsilon=eps, random_seed=seed),
            data,
        )

    counts_a, counts_b = _outcome_counts(run_one, data_a, data_b)
    _assert_consistent_with_epsilon(counts_a, counts_b, eps, 0.5, "DPSPRTSubsampled")
