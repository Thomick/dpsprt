"""Shared fixtures for the test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG. Use methods like rng.binomial(...), rng.normal(...)."""
    return np.random.default_rng(42)


@pytest.fixture
def bernoulli_stream(rng):
    """Infinite Bernoulli(p) stream. Returns a factory: stream = bernoulli_stream(p); next(stream).

    Note: all calls share the module-level seeded rng; successive calls/iterations
    consume non-overlapping prefixes of the same stream.
    """
    def _stream(p):
        while True:
            yield int(rng.random() < p)
    return _stream
