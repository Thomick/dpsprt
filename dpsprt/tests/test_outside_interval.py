"""Tests for the OutsideInterval public primitive."""
import numpy as np
import pytest
from dpsprt.api.outside_interval import OutsideInterval, OutsideIntervalParameters


def test_outside_interval_constructs_with_valid_params():
    """OutsideInterval should construct without error for valid parameters."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=2.0,
        threshold_noise_scale=1.0,
        epsilon=1.0,
        random_seed=42,
    )
    assert oi.is_active()


def test_outside_interval_stops_on_upper_crossing():
    """Feeding values clearly above the upper threshold should trigger a stop on side=1."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -0.1,
        upper_threshold=lambda t: 0.1,
        query_noise_scale=0.01,   # tiny noise: noisy query ≈ query_value
        threshold_noise_scale=0.01,
        epsilon=10.0,
        random_seed=0,
    )
    stopped, side = False, None
    for _ in range(100):
        stopped, side = oi.add_query(5.0)   # clearly above 0.1
        if stopped:
            break
    assert stopped
    assert side == 1


def test_outside_interval_stops_on_lower_crossing():
    """Feeding values clearly below the lower threshold should trigger a stop on side=-1."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -0.1,
        upper_threshold=lambda t: 0.1,
        query_noise_scale=0.01,
        threshold_noise_scale=0.01,
        epsilon=10.0,
        random_seed=1,
    )
    stopped, side = False, None
    for _ in range(100):
        stopped, side = oi.add_query(-5.0)  # clearly below -0.1
        if stopped:
            break
    assert stopped
    assert side == -1


def test_outside_interval_does_not_leak_internal_noise():
    """get_state() must not expose internal noise variables."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=1.0,
        threshold_noise_scale=1.0,
        epsilon=1.0,
        random_seed=0,
    )
    oi.add_query(0.0)
    state = oi.get_state()
    forbidden = {"threshold_noise", "query_noise", "noisy_value", "_z",
                 "_noisy_lower", "_noisy_upper", "_nu"}
    leaked = forbidden & set(state.keys())
    assert not leaked, f"leaked internal fields: {leaked}"


def test_outside_interval_state_keys():
    """get_state() must expose exactly the public fields."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -1.0,
        upper_threshold=lambda t: 1.0,
        query_noise_scale=1.0,
        threshold_noise_scale=1.0,
        epsilon=1.0,
        random_seed=0,
    )
    state = oi.get_state()
    assert "stopped" in state
    assert "side" in state
    assert "step" in state
    assert "epsilon" in state


def test_outside_interval_stays_stopped():
    """Once stopped, all subsequent calls must return (True, same_side) immediately."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -0.1,
        upper_threshold=lambda t: 0.1,
        query_noise_scale=0.01,
        threshold_noise_scale=0.01,
        epsilon=10.0,
        random_seed=2,
    )
    # Force a stop
    stopped, first_side = False, None
    for _ in range(100):
        stopped, first_side = oi.add_query(5.0)
        if stopped:
            break
    assert stopped

    # All subsequent calls should return same result
    for _ in range(10):
        s, side = oi.add_query(0.0)
        assert s is True
        assert side == first_side


def test_outside_interval_reset():
    """reset() should allow the algorithm to run again from scratch."""
    oi = OutsideInterval(
        lower_threshold=lambda t: -0.1,
        upper_threshold=lambda t: 0.1,
        query_noise_scale=0.01,
        threshold_noise_scale=0.01,
        epsilon=10.0,
        random_seed=3,
    )
    # Stop it
    for _ in range(100):
        stopped, _ = oi.add_query(5.0)
        if stopped:
            break
    assert not oi.is_active()

    # Reset and verify active again
    oi.reset()
    assert oi.is_active()
    state = oi.get_state()
    assert state["stopped"] is False
    assert state["step"] == 0
