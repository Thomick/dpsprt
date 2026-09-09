# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-09-08

First release on PyPI. Versions 1.0.0 and 1.1.0 existed only in the repository and were never published.

### Fixed

- `ZETA_S_VALUE` was 5.591 alongside `ZETA_S_PARAMETER = 1.1340`, but 5.591 is `zeta(1.2000)` and `zeta(1.1340) = 8.049572`. The union bound in the correctness proof needs `sum_n 1 / (n^s zeta(s)) = 1`, so the mismatched pair spent 1.44 times the nominal error budget. `zeta(s)` is a derived quantity and `s` the chosen parameter, so `ZETA_S_VALUE` is now 8.049572 and a test recomputes it from the series. Mean stopping times lengthen by about 4 percent. The same pair was corrected in the JAX code that produced the paper's figures, where the shift is far below the resolution of the published plots.
- `DPSPRTTuned` rejected `c2 < 1`, which ruled out the regime the paper's own tuned experiment uses. Privacy is unaffected by `c2`, since the noise does not change; only the `(alpha, beta)` guarantee needs `c2 >= 1`. Values below 1 are now allowed, and a test reproduces the paper's `kappa = 0.5` result.
- `OutsideInterval` drew two independent threshold noises and compared the lower threshold against one and the upper against the other. Algorithm 2 of the paper draws a single `Z` and compares against `T0 - Z` and `T1 + Z`, and the shared realisation is what makes the mechanism `(eps_Z + eps_Y)`-DP rather than two composed AboveThreshold instances. The primitive now draws one `Z`. The five DP-SPRT classes were never affected. This changes the numerical output of `OutsideInterval`.
- The noise scales documented for `OutsideInterval` in the README and the design note implemented twice the budget they declared. `query_noise_scale = 2/eps` with `threshold_noise_scale = 1/eps` gives `eps_Z + eps_Y = 2 eps`. An even split of a total `eps` is `4 * Delta / eps` and `2 * Delta / eps`, which is what the DP-SPRT classes use at per-step sensitivity `1/t`.
- The five functions in `dpsprt.core.algorithms` called `numpy.random.seed`, which reseeded the caller's global generator. Each now uses its own `RandomState`.
- Importing `dpsprt` printed three lines to stdout in interactive sessions.
- `pyproject.toml` declared `setuptools>=61.0` while using the PEP 639 `license` string, which needs 77.

### Added

- `OutsideInterval` takes `sensitivity` and `epsilon_tol`. The constructor checks the declared `epsilon` against `eps_Z + eps_Y` implied by the noise scales and raises when they disagree. Pass `epsilon_tol=None` to waive the check for non-Laplace noise.
- `OutsideIntervalParameters.implied_epsilon()`.
- Tests for error-rate calibration, for agreement between the vectorized functions and the incremental classes, for the shared `Z`, for the global-generator guarantee, for the `zeta(s)` pairing, and for the paper's tuned `kappa = 0.5` trade-off.
- A test and lint matrix on Python 3.9 through 3.13, and a release workflow using PyPI trusted publishing.
- `CHANGELOG.md`, `MANIFEST.in`. The sdist now carries the tests, examples and design notes.

### Changed

- `get_sample_count()` returns the count for every variant. It previously raised on the DP variants on the grounds that the stream length was secret, which does not hold, since the caller counts its own `add_sample` calls.
- The citation points at the AISTATS 2026 proceedings instead of the arXiv preprint.
- The classifier is `Development Status :: 4 - Beta`.
- Docstrings and the README record that noise is drawn with floating-point arithmetic and a non-cryptographic generator, so the guarantees are those of the idealized real-valued mechanisms.

### Documentation

- The module docstring defined `(eps, delta)`-DP as "eps-DP holds with probability >= (1-delta)", which is not what approximate DP says. It now states the guarantee as Theorem 1(i) does, and notes that the Gaussian variant goes through the paper's Renyi DP profile.
- Prose now uses the paper's vocabulary. `C(n, delta)` is the correction function rather than the "privacy term", `gamma` is the error allocation, `Z` and `Y_n` are the threshold and query noise, and `r` is the subsampling rate.
- The README records that `DPSPRTTuned`'s `c2` plays the role of the paper's `kappa`, that the paper's tuned experiment takes `kappa` about 0.5, and that this class rejects `c2 < 1` and so does not reproduce it.
- The README records that `DPSPRTSubsampled` divides by the realised subsample count `M_n`, following the subsampling appendix rather than the deterministic `r*n` used by the JAX code.
- `examples/stopping_times_demo.py` labelled a row "Tuned (c1=0.5)", which scales the SPRT threshold rather than the correction function. The label now says so.

### Removed

- `DPSPRTSubsampled.get_subsampling_info()`, which existed only to raise.
- `DEFAULT_MAX_SAMPLES`, exported from `dpsprt.core` and never read. No algorithm caps its sample count, so the name promised a limit that did not exist.
