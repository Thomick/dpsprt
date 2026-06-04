"""Behavioural conformance tests for all public DP-SPRT classes."""
import itertools
import numpy as np
import pytest
from dpsprt import ClassicalSPRT, DPSPRT, DPSPRTGaussian, DPSPRTTuned, DPSPRTSubsampled


def test_classical_sprt_accepts_h1_on_clear_h1_stream(bernoulli_stream):
    """Wald's SPRT should accept H1 on a stream from p=0.9 when testing
    H0: mu=0.3 vs H1: mu=0.7, across most seeds."""
    n_runs, accepts_h1 = 200, 0
    # Each iteration advances the shared rng — 200 non-overlapping prefixes.
    for _ in range(n_runs):
        test = ClassicalSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05)
        for x in itertools.islice(bernoulli_stream(0.9), 1000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == 1:
                    accepts_h1 += 1
                break
    assert accepts_h1 / n_runs >= 0.95, f"Only {accepts_h1}/{n_runs} runs accepted H1"


def test_classical_sprt_accepts_h0_on_clear_h0_stream(bernoulli_stream):
    """Wald's SPRT should accept H0 on a stream from p=0.1 when testing
    H0: mu=0.3 vs H1: mu=0.7, across most runs."""
    n_runs, accepts_h0 = 200, 0
    for _ in range(n_runs):
        test = ClassicalSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05)
        for x in itertools.islice(bernoulli_stream(0.1), 1000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == -1:
                    accepts_h0 += 1
                break
    assert accepts_h0 / n_runs >= 0.95, f"Only {accepts_h0}/{n_runs} runs accepted H0"



def test_dpsprt_laplace_accepts_correct_hypothesis(bernoulli_stream):
    """DP-SPRT (eps=2) should mostly accept H1 on a stream from p=0.9."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=2.0, random_seed=seed)
        for x in itertools.islice(bernoulli_stream(0.9), 5000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == 1:
                    correct += 1
                break
    assert correct / n_runs >= 0.90, f"Only {correct}/{n_runs} runs accepted H1"


def test_dpsprt_laplace_accepts_h0_on_clear_h0_stream(bernoulli_stream):
    """DP-SPRT (eps=2) should mostly accept H0 on a stream from p=0.1."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=2.0, random_seed=seed)
        for x in itertools.islice(bernoulli_stream(0.1), 5000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == -1:
                    correct += 1
                break
    assert correct / n_runs >= 0.90, f"Only {correct}/{n_runs} runs accepted H0"


def test_dpsprt_stopping_time_decreases_with_epsilon(bernoulli_stream):
    """Median stopping time should decrease as epsilon increases (Corollary 1 trend)."""
    medians = []
    for eps in [0.5, 1.0, 2.0, 4.0]:
        times = []
        for seed in range(100):
            test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=eps, random_seed=seed)
            steps = 0
            for x in itertools.islice(bernoulli_stream(0.9), 20000):
                steps += 1
                can_stop, _ = test.add_sample(x)
                if can_stop:
                    break
            times.append(steps)
        medians.append(np.median(times))
    assert medians == sorted(medians, reverse=True), f"non-monotone median stopping times: {medians}"



def test_dpsprt_gaussian_accepts_correct_hypothesis(bernoulli_stream):
    """DP-SPRT Gaussian (eps=2, delta=1e-5) should mostly accept H1 on a stream from p=0.9."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRTGaussian(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
            epsilon=2.0, delta=1e-5, random_seed=seed,
        )
        for x in itertools.islice(bernoulli_stream(0.9), 5000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == 1:
                    correct += 1
                break
    assert correct / n_runs >= 0.90, f"Only {correct}/{n_runs} runs accepted H1"


def test_dpsprt_gaussian_accepts_h0_on_clear_h0_stream(bernoulli_stream):
    """DP-SPRT Gaussian (eps=2, delta=1e-5) should mostly accept H0 on a stream from p=0.1."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRTGaussian(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
            epsilon=2.0, delta=1e-5, random_seed=seed,
        )
        for x in itertools.islice(bernoulli_stream(0.1), 5000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == -1:
                    correct += 1
                break
    assert correct / n_runs >= 0.90, f"Only {correct}/{n_runs} runs accepted H0"


from dpsprt import DPSPRTTuned


def test_dpsprt_tuned_rejects_unsafe_c2():
    """c2 < 1 has no privacy proof; constructor must reject it."""
    with pytest.raises(ValueError, match="c2"):
        DPSPRTTuned(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
                    epsilon=1.0, c1=1.0, c2=0.5)


def test_dpsprt_tuned_matches_dpsprt_at_unit_constants(bernoulli_stream):
    """DPSPRTTuned(c1=1, c2=1) must be numerically identical to DPSPRT on same seed."""
    data = list(itertools.islice(bernoulli_stream(0.6), 200))
    a = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=7)
    b = DPSPRTTuned(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0,
                    c1=1.0, c2=1.0, random_seed=7)
    out_a = [a.add_sample(x) for x in data]
    out_b = [b.add_sample(x) for x in data]
    assert out_a == out_b, "DPSPRTTuned(c1=1,c2=1) differs from DPSPRT"


from dpsprt import DPSPRTSubsampled


def test_dpsprt_subsampled_accepts_correct_hypothesis(bernoulli_stream):
    """DPSPRTSubsampled (eps=2) should mostly accept H1 on a stream from p=0.9."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRTSubsampled(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=2.0, random_seed=seed,
        )
        for x in itertools.islice(bernoulli_stream(0.9), 10000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == 1:
                    correct += 1
                break
    assert correct / n_runs >= 0.80, f"Only {correct}/{n_runs} runs accepted H1 (subsampled)"


def test_dpsprt_subsampled_accepts_h0_on_clear_h0_stream(bernoulli_stream):
    """DPSPRTSubsampled (eps=2) should mostly accept H0 on a stream from p=0.1."""
    n_runs, correct = 200, 0
    for seed in range(n_runs):
        test = DPSPRTSubsampled(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=2.0, random_seed=seed,
        )
        for x in itertools.islice(bernoulli_stream(0.1), 10000):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                if decision == -1:
                    correct += 1
                break
    assert correct / n_runs >= 0.80, f"Only {correct}/{n_runs} runs accepted H0 (subsampled)"


def test_dpsprt_laplace_refactor_matches_v1_0_0(bernoulli_stream):
    """Hash of decision sequences must match the pre-refactor baseline.

    Run once on un-refactored code to get the hash, paste it here, then
    verify after refactoring.  A mismatch means the refactor changed numerical
    behaviour and must be investigated.
    """
    import hashlib, json

    decisions = []
    for seed in range(50):
        test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=seed)
        for x in itertools.islice(bernoulli_stream(0.6), 1000):
            can_stop, d = test.add_sample(x)
            decisions.append((seed, x, int(d)))
            if can_stop:
                break
    digest = hashlib.sha256(json.dumps(decisions).encode()).hexdigest()
    # Baseline updated after fixing threshold noise Z to be drawn once per run
    # (was: redrawn fresh per step).  Old hash:
    # f725f7a1fb6a574e8261ef774fbe9f9285ca0cd2fda0bfeb6183b4bc6165a783
    assert digest == "b8a082a8978a6e373b845d0e8cf78eb424aefffc6283d95e7a1fd389fe662916"


# ---------------------------------------------------------------------------
# Snapshot hashes for the remaining incremental classes.  Each draws its own
# per-seed Bernoulli stream so the hash does not depend on the bernoulli_stream
# fixture's shared rng state.
# ---------------------------------------------------------------------------
def _per_seed_stream(p, seed):
    rng = np.random.RandomState(seed * 1000 + 1)
    while True:
        yield int(rng.random() < p)


def _decisions_hash(make_test, n_seeds=50, p=0.6, max_steps=1000):
    import hashlib, json
    decisions = []
    for seed in range(n_seeds):
        test = make_test(seed)
        for x in itertools.islice(_per_seed_stream(p, seed), max_steps):
            can_stop, d = test.add_sample(x)
            decisions.append((seed, x, int(d)))
            if can_stop:
                break
    return hashlib.sha256(json.dumps(decisions).encode()).hexdigest()


def test_dpsprt_gaussian_snapshot():
    """Regression anchor for DPSPRTGaussian.  Bump deliberately on intended changes."""
    digest = _decisions_hash(
        lambda s: DPSPRTGaussian(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
            epsilon=1.0, delta=1e-5, random_seed=s,
        )
    )
    assert digest == "e7a781482142e3ce40ed48631fed21c4eb3678f0be2a220c15b1e8881e389526"


def test_dpsprt_tuned_snapshot():
    digest = _decisions_hash(
        lambda s: DPSPRTTuned(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
            epsilon=1.0, c1=1.0, c2=1.0, random_seed=s,
        )
    )
    assert digest == "a405b5b7fba3e3c059fc2f16b17dff8762f9da2f703a95c7b05271978fa88623"


def test_dpsprt_subsampled_snapshot():
    digest = _decisions_hash(
        lambda s: DPSPRTSubsampled(
            mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05,
            epsilon=1.0, random_seed=s,
        )
    )
    # Baseline updated after switching the M_n=0 fallback from a None
    # placeholder (no noise drawn) to a neutral noisy_mean (noise drawn every
    # step).  Old hash:
    # 0c6877d4391cc2559c8ee5c59b7b86b8e53ba516a816d3644ec27af0ffe56660
    assert digest == "9e0444b30f813414ec154c978d90069356cfc44a137dfbc132a525ab4837d2ba"


# ---------------------------------------------------------------------------
# Behavioural tests for the issues fixed after the original code review.
# ---------------------------------------------------------------------------
def test_reset_advances_rng_for_independent_trials():
    """reset() must NOT re-seed self.rng.

    Otherwise reusing one seeded instance across trials gives every trial the
    same threshold noise Z, silently destroying trial independence.  We assert
    that consecutive reset() calls produce DIFFERENT _aux_z_unit draws for each
    of the four core incremental classes.
    """
    from dpsprt.core.sprt import (
        DPSPRT as CoreDPSPRT,
        DPSPRTGaussian as CoreDPSPRTGaussian,
        DPSPRTTuned as CoreDPSPRTTuned,
        DPSPRTSubsampled as CoreDPSPRTSubsampled,
    )
    builders = [
        lambda: CoreDPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=7),
        lambda: CoreDPSPRTGaussian(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, delta=1e-5, random_seed=7),
        lambda: CoreDPSPRTTuned(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, c1=1.0, c2=1.0, random_seed=7),
        lambda: CoreDPSPRTSubsampled(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=7),
    ]
    for build in builders:
        inst = build()
        z_seen = {inst._aux_z_unit}
        for _ in range(5):
            inst.reset()
            assert inst._aux_z_unit not in z_seen, (
                f"{type(inst).__name__}.reset() drew a repeated _aux_z_unit "
                "— rng is being re-seeded inside reset(), which makes trials "
                "with a reused instance perfectly correlated."
            )
            z_seen.add(inst._aux_z_unit)


def _run_functional_and_get_aux_noise(func, **kwargs):
    np.random.seed(0)
    data = (np.random.rand(50) < 0.6).astype(int)
    np.random.seed(123)
    _, _, info = func(data, **kwargs)
    return np.asarray(info["laplace_noise2"])


def test_functional_dp_sprt_laplace_uses_single_threshold_z():
    """The vectorized dp_sprt_laplace must derive its threshold noise from a
    single Z scaled by 1/t, not from independent per-step draws.

    Mathematical signature: if noise[t] = z * 2/(t*eps), then t*noise[t] is the
    constant 2*z/eps for every t.
    """
    from dpsprt.core.algorithms import dp_sprt_laplace
    lap2 = _run_functional_and_get_aux_noise(
        dp_sprt_laplace, mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0,
    )
    products = np.arange(1, len(lap2) + 1) * lap2
    assert np.allclose(products, products[0]), (
        "dp_sprt_laplace appears to use independent per-step Z draws — "
        f"products t*lap2[t] should be constant but span {products.min():.4g}..{products.max():.4g}"
    )


def test_functional_dp_sprt_tuned_uses_single_threshold_z():
    from dpsprt.core.algorithms import dp_sprt_tuned
    lap2 = _run_functional_and_get_aux_noise(
        dp_sprt_tuned, mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, c1=1.0, c2=1.0,
    )
    products = np.arange(1, len(lap2) + 1) * lap2
    assert np.allclose(products, products[0])


def test_functional_dp_sprt_subsampled_uses_single_threshold_z():
    from dpsprt.core.algorithms import dp_sprt_subsampled
    lap2 = _run_functional_and_get_aux_noise(
        dp_sprt_subsampled, mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0,
    )
    products = np.arange(1, len(lap2) + 1) * lap2
    assert np.allclose(products, products[0])


def test_subsampled_never_decides_before_first_inclusion():
    """Before the first included sample (M_n = 0) the test must not be able to
    stop, and the recorded noisy_mean must not depend on the data."""
    from dpsprt.core.sprt import DPSPRTSubsampled as CoreSub
    # eps small so inclusion is rare → many leading rejection steps.
    inst_a = CoreSub(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=0.01, random_seed=11)
    inst_b = CoreSub(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=0.01, random_seed=11)
    # Feed two different data streams; before the first inclusion both
    # instances must produce identical noisy_means and never report stopped.
    for x_a, x_b in zip([0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]):
        stop_a, _ = inst_a.add_sample(x_a)
        stop_b, _ = inst_b.add_sample(x_b)
        if inst_a.t_subsample > 0 or inst_b.t_subsample > 0:
            break
        assert not stop_a and not stop_b, "stopped before any sample was included"
        assert inst_a.noisy_means[-1] == inst_b.noisy_means[-1], (
            "noisy_mean depends on data before the first inclusion"
        )


def test_functional_subsampled_never_decides_before_first_inclusion():
    """The vectorized dp_sprt_subsampled must also suppress stopping at steps
    where no sample has been included yet."""
    from dpsprt.core.algorithms import dp_sprt_subsampled
    # Construct a regime where epsilon → 0 makes inclusion vanishingly rare so
    # that with high probability several leading steps have cumcount_included=0.
    np.random.seed(0)
    data = (np.random.rand(200) < 0.6).astype(int)
    np.random.seed(7)
    _, _, info = dp_sprt_subsampled(
        data, mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=0.005,
    )
    include_mask = info["include_mask"]
    samples_included = np.cumsum(include_mask)
    no_sample_steps = samples_included == 0
    if not no_sample_steps.any():
        pytest.skip("test parameters did not produce any pre-inclusion steps")
    cond0 = np.asarray(info["condition_0"])
    cond1 = np.asarray(info["condition_1"])
    assert not cond0[no_sample_steps].any(), "H0 stop allowed at M_n=0"
    assert not cond1[no_sample_steps].any(), "H1 stop allowed at M_n=0"


def test_subsampled_lists_aligned_in_length(bernoulli_stream):
    """DPSPRTSubsampled.subsampling_decisions and .noisy_means must share length
    so that list(zip(...)) does not misalign step indices."""
    from dpsprt.core.sprt import DPSPRTSubsampled as CoreSub
    test = CoreSub(
        mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=0.1, random_seed=3,
    )
    for x in itertools.islice(bernoulli_stream(0.6), 500):
        can_stop, _ = test.add_sample(x)
        if can_stop:
            break
    assert len(test.subsampling_decisions) == len(test.noisy_means), (
        f"length mismatch: {len(test.subsampling_decisions)} decisions "
        f"vs {len(test.noisy_means)} noisy_means"
    )
