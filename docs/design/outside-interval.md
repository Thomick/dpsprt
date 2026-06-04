# OutsideInterval — Public API Design

## Overview

`OutsideInterval` is a differentially private continual-observation primitive.  It monitors
a stream of query values and halts the first time a noisy query falls outside a pair of
(possibly time-varying) thresholds.

The primitive is described in Algorithm 2 of the AISTATS 2026 paper (`privacy.tex`,
OutsideInterval section).  The paper advertises it as reusable beyond DP-SPRT; this design
makes it a first-class public export of `dpsprt`.

---

## Module path

```
dpsprt.api.outside_interval.OutsideInterval
dpsprt.api.outside_interval.OutsideIntervalParameters
```

Top-level re-export: `from dpsprt import OutsideInterval`.

---

## Constructor signature

```python
OutsideInterval(
    lower_threshold: Callable[[int], float],   # τ₀(t) — lower boundary at step t
    upper_threshold: Callable[[int], float],   # τ₁(t) — upper boundary at step t
    query_noise_scale: float,                  # scale of per-query Laplace noise (> 0)
    threshold_noise_scale: float,              # scale of threshold Laplace noise (> 0)
    epsilon: float,                            # privacy parameter ε > 0
    random_seed: Optional[int] = None,
)
```

Parameter units and constraints:
- `lower_threshold(t)`, `upper_threshold(t)`: real-valued callables, `t` is 1-indexed step
  count; must satisfy `lower_threshold(t) < upper_threshold(t)` for all t.
- `query_noise_scale`: Laplace scale for per-query noise ν_t.  Set to `2/ε₂` where ε₂ is
  the per-query privacy budget.
- `threshold_noise_scale`: Laplace scale for threshold perturbations T̂₀, T̂₁ drawn once at
  construction.  Set to `1/ε₀` where ε₀ is the threshold budget; `ε₀ + ε₂ = ε` (plus any
  spare budget for the ε₁ side).
- `epsilon`: total claimed DP budget; used only for documentation and `get_state()` — the
  caller is responsible for ensuring the noise scales implement the desired ε.

---

## Methods

### `add_query(value: float) -> tuple[bool, int]`

Feed the next query result.  Returns `(stopped, side)`:
- `stopped`: True if the algorithm has halted (this or a prior call).
- `side`: 0 while running; -1 if halted because the noisy query fell below the lower
  threshold (τ₀ crossed); +1 if above the upper threshold (τ₁ crossed).

Once stopped, all subsequent calls return the same `(True, side)` immediately.

### `is_active() -> bool`

Returns `True` if the algorithm has not yet stopped.

### `reset() -> None`

Reset to the initial state and redraw the threshold noise.  The rng is NOT
re-seeded — every call to `reset()` advances it, so a reused instance produces
independent runs.

### `get_state() -> dict`

Returns a privacy-safe snapshot:
```python
{
    "stopped": bool,
    "side": int,         # -1, 0, or 1
    "step": int,         # number of add_query calls so far
    "epsilon": float,
}
```

Internal noise variables (`_noisy_lower`, `_noisy_upper`, per-step `_nu`) must NOT appear
in this dict.  A test (`test_outside_interval_does_not_leak_internal_noise`) verifies this.

---

## Privacy contract

`OutsideInterval` is ε-DP for sensitivity-1 queries (per the paper's Theorem on
OutsideInterval privacy, `privacy.tex`).  Specifically:
- Threshold noise `T̂₀ = τ₀ - Lap(threshold_noise_scale)` and
  `T̂₁ = τ₁ + Lap(threshold_noise_scale)` are drawn once at construction.
- Per-query noise `ν_t ~ Lap(query_noise_scale)` is drawn fresh at each step.
- The algorithm halts when `f_t(D) + ν_t ≤ T̂₀` or `f_t(D) + ν_t ≥ T̂₁`.

The caller must supply query values `f_t(D)` that have sensitivity ≤ 1 (i.e., changing one
record in the database changes `f_t` by at most 1).

---

## Usage examples

### (i) Standalone continual monitoring

```python
from dpsprt import OutsideInterval

# Monitor whether a noisy running statistic leaves a calibration band [−2, 2]
oi = OutsideInterval(
    lower_threshold=lambda t: -2.0,
    upper_threshold=lambda t:  2.0,
    query_noise_scale=2.0,        # Lap(2/ε) for ε=1
    threshold_noise_scale=1.0,    # Lap(1/ε) for ε=1
    epsilon=1.0,
    random_seed=42,
)

for i, query_value in enumerate(sensor_stream):
    stopped, side = oi.add_query(query_value)
    if stopped:
        print(f"Alarm at step {i+1}: crossed {'upper' if side == 1 else 'lower'} bound")
        break
```

### (ii) DP-SPRT use case

`DPSPRT`, `DPSPRTGaussian`, `DPSPRTTuned`, and `DPSPRTSubsampled` all share the
same stopping check via the stateless helper
`dpsprt.core.outside_interval.dpsprt_interval_check`:

```python
from dpsprt.core.outside_interval import dpsprt_interval_check

stopped, side = dpsprt_interval_check(
    noisy_query_h0=current_mean - mu0,
    noisy_query_h1=current_mean - mu1,
    lower_threshold=tau0_rhs,
    upper_threshold=tau1_rhs,
)
```
