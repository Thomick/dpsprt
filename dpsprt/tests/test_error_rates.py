"""Error-rate calibration and agreement between the two implementations.

The tests here check the two properties a user actually relies on. The realised
type I and type II errors stay at or under the nominal alpha and beta, and the
vectorized functions in ``dpsprt.core.algorithms`` behave like the incremental
classes they mirror. Both are slow, so both carry the 'slow' marker.
"""

import numpy as np
import pytest

from dpsprt import DPSPRT, ClassicalSPRT
from dpsprt.core.algorithms import dp_sprt_laplace

pytestmark = pytest.mark.slow

MU0, MU1, ALPHA, BETA = 0.3, 0.7, 0.05, 0.05


def _run_to_decision(test, data):
    """Feed data one sample at a time; return (decision, stopping_time)."""
    for i, x in enumerate(data, 1):
        can_stop, decision = test.add_sample(int(x))
        if can_stop:
            return decision, i
    return 0, len(data)


def _error_rate(make_test, p, wrong_decision, n_runs, max_steps, data_seed):
    rng = np.random.default_rng(data_seed)
    wrong = 0
    for run in range(n_runs):
        data = rng.binomial(1, p, size=max_steps)
        decision, _ = _run_to_decision(make_test(run), data)
        if decision == wrong_decision:
            wrong += 1
    return wrong / n_runs


def test_classical_sprt_controls_both_error_rates():
    """Wald's SPRT must keep type I under alpha and type II under beta.

    Under H0 the data are Bernoulli(mu0) and deciding H1 is the type I error.
    """
    type_i = _error_rate(
        lambda r: ClassicalSPRT(MU0, MU1, ALPHA, BETA), MU0, 1, 1000, 5000, data_seed=1234
    )
    type_ii = _error_rate(
        lambda r: ClassicalSPRT(MU0, MU1, ALPHA, BETA), MU1, -1, 1000, 5000, data_seed=5678
    )
    # 1000 runs put the binomial standard error near 0.007 at the nominal rate,
    # so 0.08 is a wide margin that still catches a broken calibration.
    assert type_i <= 0.08, f"type I error {type_i:.4f} exceeds nominal {ALPHA}"
    assert type_ii <= 0.08, f"type II error {type_ii:.4f} exceeds nominal {BETA}"


def test_dpsprt_controls_both_error_rates():
    """DP-SPRT inherits the guarantee; the correction function makes it conservative."""
    type_i = _error_rate(
        lambda r: DPSPRT(MU0, MU1, ALPHA, BETA, epsilon=2.0, random_seed=r),
        MU0,
        1,
        500,
        20000,
        data_seed=1234,
    )
    type_ii = _error_rate(
        lambda r: DPSPRT(MU0, MU1, ALPHA, BETA, epsilon=2.0, random_seed=r),
        MU1,
        -1,
        500,
        20000,
        data_seed=5678,
    )
    assert type_i <= 0.08, f"type I error {type_i:.4f} exceeds nominal {ALPHA}"
    assert type_ii <= 0.08, f"type II error {type_ii:.4f} exceeds nominal {BETA}"


def test_vectorized_laplace_matches_the_incremental_class():
    """dp_sprt_laplace and DPSPRT implement the same test.

    They consume their generators differently, so a run-by-run match is not
    available. The stopping-time distributions must still agree.
    """
    n_trials, eps = 300, 1.0
    rng = np.random.default_rng(0)
    incremental, vectorized = [], []
    for trial in range(n_trials):
        data = rng.binomial(1, 0.6, size=3000)
        _, t_vec, _ = dp_sprt_laplace(data, MU0, MU1, ALPHA, BETA, eps, random_seed=trial)
        _, t_inc = _run_to_decision(DPSPRT(MU0, MU1, ALPHA, BETA, eps, random_seed=trial), data)
        vectorized.append(t_vec)
        incremental.append(t_inc)

    med_inc, med_vec = np.median(incremental), np.median(vectorized)
    assert med_vec == pytest.approx(
        med_inc, rel=0.15
    ), f"median stopping times disagree: incremental {med_inc}, vectorized {med_vec}"


def test_vectorized_functions_leave_the_global_numpy_stream_alone():
    """Passing random_seed must not reseed the caller's global generator."""
    np.random.seed(12345)
    before = np.random.random()
    np.random.seed(12345)
    dp_sprt_laplace(np.zeros(50, dtype=int), MU0, MU1, ALPHA, BETA, 1.0, random_seed=999)
    after = np.random.random()
    assert before == after, "dp_sprt_laplace reseeded the global numpy generator"
