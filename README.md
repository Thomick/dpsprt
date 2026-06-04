# DP-SPRT

Differentially private sequential probability ratio tests, in NumPy.
Reference implementation of the AISTATS 2026 paper
[*Differentially Private Sequential Probability Ratio Tests*](https://arxiv.org/abs/2508.06377)
(Michel, Basu, Kaufmann).

## Installation

```bash
pip install dpsprt
```

## Quick start

Each algorithm exposes an `add_sample(x)` method that returns
`(can_stop, decision)` where `decision` is `-1` (accept H₀), `+1` (accept H₁),
or `0` while sampling continues.

```python
import numpy as np
from dpsprt import DPSPRT

test = DPSPRT(mu0=0.3, mu1=0.7, alpha=0.05, beta=0.05, epsilon=1.0, random_seed=0)
data = np.random.default_rng(0).binomial(1, 0.6, size=1000)

for t, x in enumerate(data, 1):
    can_stop, decision = test.add_sample(int(x))
    if can_stop:
        print(f"Stopped after {t} samples; decision = {decision}")
        break
```

For a one-shot batch interface, use `run_streaming_sprt_on_batch`:

```python
from dpsprt.api.sprt import run_streaming_sprt_on_batch

decision, stopping_time, state = run_streaming_sprt_on_batch(test, data)
```

## Algorithms

| Class                | Privacy             | Notes                                       |
| -------------------- | ------------------- | ------------------------------------------- |
| `ClassicalSPRT`      | none                | Wald's SPRT, baseline                       |
| `DPSPRT`             | ε-DP                | Laplace noise                               |
| `DPSPRTGaussian`     | (ε, δ)-DP           | Gaussian noise (RDP)                        |
| `DPSPRTTuned`        | ε-DP                | Tunable c₁ (threshold) and c₂ ≥ 1 (privacy) |
| `DPSPRTSubsampled`   | ε-DP                | Bernoulli subsampling amplification         |

A standalone, one-shot vectorized version of each algorithm lives in
`dpsprt.core.algorithms` (`dp_sprt_laplace`, `dp_sprt_gaussian`, …).

## OutsideInterval primitive

Algorithm 2 of the paper, exposed as a stand-alone DP continual-observation
monitor.

```python
from dpsprt import OutsideInterval

eps = 1.0
oi = OutsideInterval(
    lower_threshold=lambda t: -1.0,
    upper_threshold=lambda t:  1.0,
    query_noise_scale=2.0 / eps,
    threshold_noise_scale=1.0 / eps,
    epsilon=eps,
    random_seed=0,
)

for value in sensor_stream:
    stopped, side = oi.add_query(value)
    if stopped:
        print("Upper" if side == 1 else "Lower", "bound crossed")
        break
```

## Examples

Runnable scripts under `examples/`:

- `paper_experiments.py` — reproduces the headline AISTATS 2026 figure.
- `plot_stopping_times_vs_epsilon.py` — stopping-time sweep across ε.
- `stopping_times_demo.py` — side-by-side comparison on a single stream.
- `outside_interval_demo.py` — `OutsideInterval` on a band-crossing scenario.

## Citation

```bibtex
@article{michel2025dpsprt,
  title   = {DP-SPRT: Differentially Private Sequential Probability Ratio Tests},
  author  = {Michel, Thomas and Basu, Debabrota and Kaufmann, Emilie},
  journal = {arXiv preprint arXiv:2508.06377},
  year    = {2025}
}
```
