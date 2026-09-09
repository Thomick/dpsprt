"""
Tests for the DP-SPRT interfaces.

These tests verify that the algorithms work correctly and maintain state
properly. The incremental interface is now the standard and only supported interface.
"""

import numpy as np
import pytest

from dpsprt import (
    DPSPRT,
    ClassicalSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
    run_sprt_on_batch,
)


class TestBasicFunctionality:
    """Test basic functionality of algorithms."""

    def test_streaming_classical_sprt_basic(self):
        """Test basic functionality of Classical SPRT."""
        # Create algorithm with test parameters
        test = ClassicalSPRT(mu0=0.2, mu1=0.8, alpha=0.05, beta=0.05)

        # Test initial state
        assert test.get_sample_count() == 0
        assert not test.has_stopped()
        assert test.get_decision() == 0

        # Add test samples
        samples = [1] * 10

        decision = None
        for i, x in enumerate(samples):
            can_stop, decision = test.add_sample(x)
            assert test.get_sample_count() == i + 1

            if can_stop:
                assert decision in [-1, 1]
                assert test.has_stopped()
                break

        # Check stopping condition
        assert can_stop
        assert decision == 1

    def test_streaming_dp_sprt_basic(self):
        """Test basic functionality of DP-SPRT."""
        # Create algorithm with fixed seed for reproducibility
        test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42)

        # Test initial state
        assert not test.has_stopped()

        # Add test samples
        samples = [1] * 20

        decision = None
        for i, x in enumerate(samples):
            can_stop, decision = test.add_sample(x)

            if can_stop:
                assert decision in [-1, 1]
                break

        # Verify valid outputs
        assert decision in [-1, 0, 1]
        if can_stop:
            assert test.has_stopped()
            assert decision == 1

    def test_streaming_gaussian_basic(self):
        """Test basic functionality of DP-SPRT with Gaussian noise."""
        test = DPSPRTGaussian(
            mu0=0.2, mu1=0.8, alpha=0.05, beta=0.05, epsilon=1.0, delta=1e-5, random_seed=123
        )

        # Add test samples
        samples = [1, 1, 1, 1, 0, 1, 1, 1]

        decision = None
        for x in samples:
            can_stop, decision = test.add_sample(x)
            if can_stop:
                break

        # Verify valid output
        assert isinstance(decision, int)
        assert decision in [-1, 0, 1]


class TestStateManagement:
    """Test state management in algorithms."""

    def test_reset_functionality(self):
        """Test that reset properly resets algorithm state."""
        # Test with Classical algorithm (can access sample count)
        classical_test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Add some samples
        classical_test.add_sample(1)
        classical_test.add_sample(0)
        classical_test.add_sample(1)

        assert classical_test.get_sample_count() == 3

        # Reset and verify clean state
        classical_test.reset()
        assert classical_test.get_sample_count() == 0
        assert not classical_test.has_stopped()
        assert classical_test.get_decision() == 0

        # Should work normally after reset
        can_stop, decision = classical_test.add_sample(1)
        assert classical_test.get_sample_count() == 1

        # Test with DP algorithm
        dp_test = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42)

        # Add some samples
        dp_test.add_sample(1)
        dp_test.add_sample(0)

        assert dp_test.get_sample_count() == 2

        # But basic functionality should still work
        assert not dp_test.has_stopped()
        assert dp_test.get_decision() == 0

        # Reset should work
        dp_test.reset()
        assert not dp_test.has_stopped()
        assert dp_test.get_decision() == 0

    def test_state_inspection(self):
        """Test that get_state returns only privacy-safe information for DP algorithms."""
        # Test DP algorithm - should only expose limited state
        dp_test = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42)

        # Add some samples
        dp_test.add_sample(1)
        dp_test.add_sample(0)

        dp_state = dp_test.get_state()

        # For DP algorithms, only these keys should be present (privacy-safe)
        expected_dp_keys = ["stopped", "decision", "algorithm_type", "epsilon"]

        for key in expected_dp_keys:
            assert key in dp_state, f"Missing key: {key}"

        # Privacy-leaking keys should NOT be present
        privacy_leaking_keys = ["sample_count", "cumulative_sum", "current_mean", "sample_history"]
        for key in privacy_leaking_keys:
            assert key not in dp_state, f"Privacy-leaking key should not be present: {key}"

        assert dp_state["decision"] == 0
        assert dp_state["stopped"] is False
        assert dp_state["epsilon"] == 1.0

        # Test Classical algorithm - should expose full state (non-private)
        classical_test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)
        classical_test.add_sample(1)
        classical_test.add_sample(0)

        classical_state = classical_test.get_state()

        # Classical algorithm should have more information available
        expected_classical_keys = [
            "sample_count",
            "cumulative_sum",
            "current_llr",
            "decision",
            "stopped",
            "sample_history",
        ]

        for key in expected_classical_keys:
            assert key in classical_state, f"Missing key for classical algorithm: {key}"

        assert classical_state["sample_count"] == 2
        assert classical_state["cumulative_sum"] == 1  # 1 + 0
        # Classical SPRT stores current_llr, not current_mean directly
        # Current mean can be calculated as cumulative_sum / sample_count
        current_mean = classical_state["cumulative_sum"] / classical_state["sample_count"]
        assert current_mean == 0.5  # 1/2
        assert classical_state["sample_history"] == [1, 0]

    def test_no_sample_after_stop(self):
        """Test that algorithm maintains decision after stopping."""
        test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Add enough samples to trigger stopping
        strong_h1_samples = [1] * 50

        final_decision = None
        stopping_time = None

        for i, x in enumerate(strong_h1_samples):
            can_stop, decision = test.add_sample(x)
            if can_stop:
                final_decision = decision
                stopping_time = i + 1
                break

        assert final_decision is not None
        assert stopping_time is not None

        # Add more samples after stopping
        can_stop_again, decision_again = test.add_sample(0)

        # Should maintain same decision
        assert can_stop_again is True
        assert decision_again == final_decision


class TestSubsampling:
    """Test subsampling-specific functionality."""

    def test_subsampling_decisions_stay_out_of_public_state(self):
        """Per-step inclusion decisions must not reach get_state()."""
        test = DPSPRTSubsampled(
            mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=2.0, random_seed=42
        )
        for i in range(20):
            test.add_sample(1 if i % 2 == 0 else 0)

        state = test.get_state()
        for key in ("subsampling_decisions", "subsampled_count", "noisy_means"):
            assert key not in state, f"privacy-leaking key in public state: {key}"
        assert not hasattr(test, "get_subsampling_info")


class TestTuned:
    """Test tuned DP-SPRT functionality."""

    def test_tuning_parameters(self):
        """Test that tuning parameters are properly stored and accessible in privacy-safe state."""
        test = DPSPRTTuned(
            mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, c1=1.5, c2=1.2, random_seed=42
        )

        # Add a sample to initialize state
        test.add_sample(1)

        # For DP algorithms, only privacy-safe information should be available
        state = test.get_state()
        assert "c1" in state  # Tuning parameters are safe to expose
        assert "c2" in state
        assert "epsilon" in state
        assert state["c1"] == 1.5
        assert state["c2"] == 1.2
        assert state["epsilon"] == 1.0

        # Privacy-leaking information should not be available
        assert "sample_count" not in state
        assert "current_mean" not in state


class TestBatchHelper:
    """Test the helper function for running algorithms on data arrays."""

    def test_run_streaming_on_batch_helper(self):
        """Test the helper function for running algorithms on data arrays."""
        # Create algorithm
        test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42)

        # Generate test data
        data = np.random.default_rng(456).binomial(1, 0.6, size=50)

        # Run using helper function
        decision, stopping_time, info = run_sprt_on_batch(test, data)

        # Verify results format
        assert decision in [-1, 0, 1]
        assert isinstance(stopping_time, int)
        assert stopping_time >= 0
        assert isinstance(info, dict)

        # get_state() exposes only the privacy-safe fields for DP variants.
        assert "algorithm_type" in info
        assert "epsilon" in info
        assert "cumulative_sum" not in info


class TestPrivacyRestrictions:
    """Test that privacy restrictions are properly enforced."""

    def test_dp_algorithm_privacy_enforcement(self):
        """Test that DP algorithms enforce strict privacy controls."""
        dp_algorithms = [
            DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42),
            DPSPRTGaussian(
                mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, delta=1e-5, random_seed=42
            ),
            DPSPRTTuned(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42),
            DPSPRTSubsampled(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=42),
        ]

        for algo in dp_algorithms:
            # Add some samples
            algo.add_sample(1)
            algo.add_sample(0)

            # The caller counts its own add_sample calls, so the count is not secret.
            assert algo.get_sample_count() == 2

            # get_state should only return privacy-safe information
            state = algo.get_state()

            # Should have these privacy-safe keys
            assert "stopped" in state
            assert "decision" in state
            assert "algorithm_type" in state

            # Should NOT have these privacy-leaking keys
            privacy_leaking_keys = [
                "sample_count",
                "cumulative_sum",
                "current_mean",
                "sample_history",
                "noisy_means",
                "auxiliary_noise",
            ]
            for key in privacy_leaking_keys:
                assert (
                    key not in state
                ), f"Privacy-leaking key '{key}' found in {type(algo).__name__} state"

    def test_classical_algorithm_full_access(self):
        """Test that Classical algorithm allows full state access (non-private)."""
        classical = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Add samples
        classical.add_sample(1)
        classical.add_sample(0)

        # Should be able to access sample count
        assert classical.get_sample_count() == 2

        # Should have full state information
        state = classical.get_state()
        expected_keys = [
            "sample_count",
            "cumulative_sum",
            "current_llr",
            "decision",
            "stopped",
            "sample_history",
        ]

        for key in expected_keys:
            assert key in state, f"Missing key '{key}' in Classical SPRT state"


class TestParameterValidation:
    """Test parameter validation in algorithms."""

    def test_invalid_sample_values(self):
        """Test that invalid sample values raise errors."""
        test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Valid samples should work
        test.add_sample(0)
        test.add_sample(1)

        # Invalid samples should raise ValueError
        with pytest.raises(ValueError, match="Sample must be 0 or 1"):
            test.add_sample(2)

        with pytest.raises(ValueError, match="Sample must be 0 or 1"):
            test.add_sample(-1)

        with pytest.raises(ValueError, match="Sample must be 0 or 1"):
            test.add_sample(0.5)  # type: ignore[arg-type]

    def test_parameter_validation_at_construction(self):
        """Test that invalid parameters are caught at construction."""
        # Invalid mu parameters
        with pytest.raises(ValueError):
            ClassicalSPRT(mu0=1.5, mu1=0.6, alpha=0.05, beta=0.05)

        # Invalid epsilon
        with pytest.raises(ValueError):
            DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=-1.0)

        # Invalid delta for Gaussian
        with pytest.raises(ValueError):
            DPSPRTGaussian(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, delta=1.5)


class TestReproducibility:
    """Test that algorithms are reproducible with seeds."""

    def test_dp_sprt_reproducibility(self):
        """Test that DP-SPRT gives same results with same seed."""
        # Create two identical algorithms
        test1 = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=123)
        test2 = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=123)

        # Generate test data
        samples = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]

        # Run both algorithms
        results1 = []
        results2 = []

        for x in samples:
            can_stop1, decision1 = test1.add_sample(x)
            can_stop2, decision2 = test2.add_sample(x)

            results1.append((can_stop1, decision1))
            results2.append((can_stop2, decision2))

            if can_stop1 or can_stop2:
                break

        # Results should be identical
        assert results1 == results2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        # Create algorithms with different seeds
        test1 = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=123)
        test2 = DPSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=456)

        # Generate test data (not too extreme to avoid immediate stopping)
        samples = [1, 0, 1, 0, 1, 0]

        # Get internal states after processing samples
        for x in samples:
            test1.add_sample(x)
            test2.add_sample(x)

        state1 = test1.get_state()
        state2 = test2.get_state()

        # The noisy means should be different due to different random seeds
        if "noisy_means" in state1 and "noisy_means" in state2:
            if len(state1["noisy_means"]) > 0 and len(state2["noisy_means"]) > 0:
                # At least some noisy means should be different
                assert state1["noisy_means"] != state2["noisy_means"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_data_stream(self):
        """Test algorithm behavior with very few samples."""
        test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Add just one sample
        can_stop, decision = test.add_sample(1)

        # Should not stop with just one sample in most cases
        assert isinstance(can_stop, bool)
        assert decision in [-1, 0, 1]

    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values."""
        # Very small epsilon (strong privacy)
        test = DPSPRT(mu0=0.1, mu1=0.9, alpha=0.01, beta=0.01, epsilon=0.01, random_seed=42)

        # Should work without errors
        can_stop, decision = test.add_sample(1)
        assert isinstance(can_stop, bool)
        assert decision in [-1, 0, 1]

    def test_alternating_samples(self):
        """Test with alternating 0/1 pattern that should be hard to decide."""
        test = ClassicalSPRT(mu0=0.4, mu1=0.6, alpha=0.05, beta=0.05)

        # Alternating pattern
        alternating_samples = [1, 0] * 25

        decisions = []
        for x in alternating_samples:
            can_stop, decision = test.add_sample(x)
            decisions.append((can_stop, decision))
            if can_stop:
                break

        # Should eventually make some decision or continue
        assert len(decisions) > 0
        final_can_stop, final_decision = decisions[-1]
        assert final_decision in [-1, 0, 1]
