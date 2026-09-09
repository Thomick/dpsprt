# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-09-08

First release on PyPI. Versions 1.0.0 and 1.1.0 existed only in the repository and were never published.

### Fixed

- `OutsideInterval` drew two independent threshold noises and compared the lower threshold against one and the upper against the other. Algorithm 2 of the paper draws a single `Z` and compares against `T0 - Z` and `T1 + Z`, and the shared realisation is what makes the mechanism `(eps_Z + eps_Y)`-DP rather than two composed AboveThreshold instances. The primitive now draws one `Z`. The five DP-SPRT classes were never affected. This changes the numerical output of `OutsideInterval`.
- The noise scales documented for `OutsideInterval` in the README and the design note implemented twice the budget they declared. `query_noise_scale = 2/eps` with `threshold_noise_scale = 1/eps` gives `eps_Z + eps_Y = 2 eps`. An even split of a total `eps` is `4 * Delta / eps` and `2 * Delta / eps`, which is what the DP-SPRT classes use at per-step sensitivity `1/t`.
- The five functions in `dpsprt.core.algorithms` called `numpy.random.seed`, which reseeded the caller's global generator. Each now uses its own `RandomState`.
- Importing `dpsprt` printed three lines to stdout in interactive sessions.
- `pyproject.toml` declared `setuptools>=61.0` while using the PEP 639 `license` string, which needs 77.

### Added

- `OutsideInterval` takes `sensitivity` and `epsilon_tol`. The constructor checks the declared `epsilon` against `eps_Z + eps_Y` implied by the noise scales and raises when they disagree. Pass `epsilon_tol=None` to waive the check for non-Laplace noise.
- `OutsideIntervalParameters.implied_epsilon()`.
- Tests for error-rate calibration, for agreement between the vectorized functions and the incremental classes, for the shared `Z`, and for the global-generator guarantee.
- A test and lint matrix on Python 3.9 through 3.13, and a release workflow using PyPI trusted publishing.
- `CHANGELOG.md`, `MANIFEST.in`. The sdist now carries the tests, examples and design notes.

### Changed

- `get_sample_count()` returns the count for every variant. It previously raised on the DP variants on the grounds that the stream length was secret, which does not hold, since the caller counts its own `add_sample` calls.
- The citation points at the AISTATS 2026 proceedings instead of the arXiv preprint.
- The classifier is `Development Status :: 4 - Beta`.
- Docstrings and the README record that noise is drawn with floating-point arithmetic and a non-cryptographic generator, so the guarantees are those of the idealized real-valued mechanisms.

### Removed

- `DPSPRTSubsampled.get_subsampling_info()`, which existed only to raise.
