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


def test_zeta_constant_matches_its_parameter():
    """ZETA_S_VALUE must be zeta(ZETA_S_PARAMETER).

    The correctness proof bounds the tail by sum_n delta / (n^s zeta(s)), which
    telescopes to delta only when the two constants are paired. A mismatched pair
    silently inflates the realised error bound by zeta(s) / ZETA_S_VALUE.
    """
    from dpsprt.core.constants import ZETA_S_PARAMETER, ZETA_S_VALUE

    s = ZETA_S_PARAMETER
    terms = 200_000
    n = np.arange(1, terms + 1, dtype=float)
    # Euler-Maclaurin: partial sum plus the leading tail terms.
    N = float(terms)
    zeta_s = np.sum(n ** (-s)) + N ** (1 - s) / (s - 1) - 0.5 * N ** (-s) + s * N ** (-s - 1) / 12.0
    assert zeta_s == pytest.approx(ZETA_S_VALUE, abs=1e-5), (
        f"ZETA_S_VALUE is {ZETA_S_VALUE} but zeta({s}) = {zeta_s:.6f}. The union "
        f"bound in the correctness proof would spend {zeta_s / ZETA_S_VALUE:.4f} "
        "times the nominal error budget."
    )


def test_tuned_kappa_half_reproduces_the_paper_tradeoff():
    """The paper's tuned experiment: Instance 1, eps=1, alpha=beta=0.1, kappa about 0.5.

    It reports error rates under target and a marked drop in stopping time.
    Both are checked here, along with the paper's caveat that the factor does
    not transfer, by confirming a smaller kappa breaks the target.
    """
    from dpsprt import DPSPRTTuned

    mu0, mu1, alpha, beta, eps = 0.3, 0.7, 0.1, 0.1, 1.0

    def cell(kappa, p):
        wrong_decision = 1 if p == mu0 else -1
        rng = np.random.default_rng(7)
        times, wrong = [], 0
        for run in range(200):
            data = rng.binomial(1, p, size=40000)
            test = DPSPRTTuned(mu0, mu1, alpha, beta, eps, c1=1.0, c2=kappa, random_seed=run)
            decision, stop = _run_to_decision(test, data)
            times.append(stop)
            if decision == wrong_decision:
                wrong += 1
        return float(np.mean(times)), wrong / 200

    mean_untuned, _ = cell(1.0, mu0)
    mean_tuned, err_i = cell(0.5, mu0)
    _, err_ii = cell(0.5, mu1)

    assert err_i <= alpha, f"kappa=0.5 type I error {err_i:.3f} exceeds {alpha}"
    assert err_ii <= beta, f"kappa=0.5 type II error {err_ii:.3f} exceeds {beta}"
    assert mean_tuned < 0.6 * mean_untuned, (
        f"kappa=0.5 should cut the stopping time markedly, got {mean_tuned:.1f} "
        f"against {mean_untuned:.1f}"
    )

    _, err_too_small = cell(0.25, mu0)
    assert err_too_small > alpha, (
        f"kappa=0.25 was expected to break the target but gave {err_too_small:.3f}; "
        "the paper's point is that the factor must be re-estimated per setting"
    )
