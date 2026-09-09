"""Tests for the OutsideInterval public primitive."""

import numpy as np
import pytest

from dpsprt.api.outside_interval import OutsideInterval, OutsideIntervalParameters


def make_oi(eps=1.0, **kwargs):
    """Build an OutsideInterval whose Laplace scales implement the declared eps.

    eps_Z = 1 / (2/eps) and eps_Y = 2 / (4/eps) each contribute eps/2.
    """
    defaults = dict(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=4.0 / eps,
        threshold_noise_scale=2.0 / eps,
        epsilon=eps,
        random_seed=0,
    )
    defaults.update(kwargs)
    return OutsideInterval(**defaults)


def test_outside_interval_constructs_with_valid_params():
    assert make_oi().is_active()


def test_outside_interval_stops_on_upper_crossing():
    """Values clearly above the upper threshold stop the monitor on side=1."""
    oi = make_oi(eps=300.0, lower_threshold=lambda t: -0.1, upper_threshold=lambda t: 0.1)
    stopped, side = False, None
    for _ in range(100):
        stopped, side = oi.add_query(5.0)
        if stopped:
            break
    assert stopped
    assert side == 1


def test_outside_interval_stops_on_lower_crossing():
    oi = make_oi(
        eps=300.0, lower_threshold=lambda t: -0.1, upper_threshold=lambda t: 0.1, random_seed=1
    )
    stopped, side = False, None
    for _ in range(100):
        stopped, side = oi.add_query(-5.0)
        if stopped:
            break
    assert stopped
    assert side == -1


def test_outside_interval_does_not_leak_internal_noise():
    """get_state() must not expose internal noise variables."""
    oi = make_oi()
    oi.add_query(0.0)
    state = oi.get_state()
    forbidden = {
        "threshold_noise",
        "query_noise",
        "noisy_value",
        "_z",
        "_noisy_lower",
        "_noisy_upper",
        "_nu",
    }
    leaked = forbidden & set(state.keys())
    assert not leaked, f"leaked internal fields: {leaked}"


def test_outside_interval_state_keys():
    """get_state() must expose exactly the public fields."""
    oi = make_oi()
    assert set(oi.get_state()) == {"stopped", "side", "step", "epsilon", "sensitivity"}


def test_outside_interval_stays_stopped():
    """Once stopped, further queries return the same verdict without advancing."""
    oi = make_oi(eps=300.0, lower_threshold=lambda t: -0.1, upper_threshold=lambda t: 0.1)
    while not oi.get_state()["stopped"]:
        oi.add_query(5.0)
    step_at_stop = oi.get_state()["step"]
    for _ in range(5):
        assert oi.add_query(-5.0) == (True, 1)
    assert oi.get_state()["step"] == step_at_stop


def test_outside_interval_reset():
    """reset() reactivates the monitor and redraws Z without re-seeding."""
    oi = make_oi(eps=300.0, lower_threshold=lambda t: -0.1, upper_threshold=lambda t: 0.1)
    oi.add_query(5.0)
    assert not oi.is_active()
    oi.reset()
    assert oi.is_active()
    assert oi.get_state()["step"] == 0


def _acceptance_boundaries(seed, t0, t1):
    """Recover [lower, upper] behaviourally by bisecting on the stopping decision.

    Z depends only on the seed, so every probe rebuilds the core and sees the same Z.
    """
    from dpsprt.core.outside_interval import OutsideIntervalCore

    def stops(v):
        core = OutsideIntervalCore(
            lower_threshold=lambda t: t0,
            upper_threshold=lambda t: t1,
            query_noise_scale=1e-12,
            threshold_noise_scale=1.0,
            rng=np.random.RandomState(seed),
        )
        return core.add_query(v)

    def bisect(inside, outside):
        for _ in range(80):
            mid = 0.5 * (inside + outside)
            if stops(mid)[0]:
                outside = mid
            else:
                inside = mid
        return 0.5 * (inside + outside)

    mid = 0.5 * (t0 + t1)
    if stops(mid)[0]:
        return None
    return bisect(mid, mid - 60.0), bisect(mid, mid + 60.0)


def test_outside_interval_shares_one_z_across_both_thresholds():
    """Algorithm 2 compares against T0 - Z and T1 + Z with a single Z.

    Two independent draws would reduce the mechanism to two composed
    AboveThreshold instances and forfeit the factor-2 privacy improvement.
    One Z shifts the two boundaries by equal and opposite amounts, so the
    midpoint of the acceptance interval never moves off (T0 + T1) / 2.
    """
    t0, t1 = -3.0, 1.0
    checked = 0
    for seed in range(60):
        bounds = _acceptance_boundaries(seed, t0, t1)
        if bounds is None:
            continue
        lower, upper = bounds
        assert lower + upper == pytest.approx(t0 + t1, abs=1e-8), (
            f"seed {seed}: acceptance midpoint moved to {(lower + upper) / 2:.6f}, "
            "which means the two thresholds are reading independent noise draws"
        )
        checked += 1
    assert checked >= 30, f"only {checked} seeds produced a usable interval"


def test_declared_epsilon_must_match_the_noise_scales():
    """The declared budget is checked against eps_Z + eps_Y from the scales."""
    with pytest.raises(ValueError, match="does not match the budget"):
        OutsideInterval(
            lower_threshold=lambda t: -1.0,
            upper_threshold=lambda t: 1.0,
            query_noise_scale=2.0,
            threshold_noise_scale=1.0,
            epsilon=1.0,
        )
    # The same scales are accepted once the declared budget is honest.
    OutsideInterval(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=2.0,
        threshold_noise_scale=1.0,
        epsilon=2.0,
    )
    # And the check can be waived for non-Laplace noise.
    OutsideInterval(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=2.0,
        threshold_noise_scale=1.0,
        epsilon=1.0,
        epsilon_tol=None,
    )


def test_implied_epsilon_matches_the_dpsprt_scaling():
    """DPSPRT uses 4/(t*eps) and 2/(t*eps) at per-step sensitivity 1/t."""
    eps, t = 1.0, 7.0
    params = OutsideIntervalParameters(
        query_noise_scale=4.0 / (t * eps),
        threshold_noise_scale=2.0 / (t * eps),
        epsilon=eps,
        sensitivity=1.0 / t,
    )
    assert params.implied_epsilon() == pytest.approx(eps)
