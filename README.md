# DP-SPRT

Differentially private sequential probability ratio tests, in NumPy.
Library implementation of [*DP-SPRT: Differentially Private Sequential Probability Ratio Tests*](https://proceedings.mlr.press/v300/michel26a.html)
(Michel, Basu and Kaufmann, AISTATS 2026).
The JAX code that produced the paper's figures is [here](https://github.com/Thomick/dpsprt_paper_code).

Requires Python 3.9 and NumPy. Matplotlib is needed only by the example scripts.

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

For a one-shot batch interface, use `run_sprt_on_batch`. It calls `reset()` first,
so it starts a fresh run with a new threshold noise draw:

```python
from dpsprt import run_sprt_on_batch

decision, stopping_time, state = run_sprt_on_batch(test, data)
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
    query_noise_scale=4.0 / eps,       # Lap(4 * Delta / eps) on each query
    threshold_noise_scale=2.0 / eps,   # Lap(2 * Delta / eps) once, shared by both sides
    epsilon=eps,
    random_seed=0,
)

for value in sensor_stream:
    stopped, side = oi.add_query(value)
    if stopped:
        print("Upper" if side == 1 else "Lower", "bound crossed")
        break
```

The noise scales alone set the guarantee. `epsilon` is the budget you declare, and
the constructor rejects a declaration that disagrees with what the scales implement.
Theorem 1(i) of the paper gives that budget as `eps_Z + eps_Y`, where `Z` answers a
sensitivity-`Delta` query and `Y` a sensitivity-`2 * Delta` one, so the scales above
split a total `eps` evenly. Pass `sensitivity` if your queries are not sensitivity 1,
and `epsilon_tol=None` to waive the check for non-Laplace noise.

The caller must supply query values of sensitivity at most `Delta`.

## Examples

Runnable scripts under `examples/`:

- `paper_experiments.py` — reproduces the headline AISTATS 2026 figure.
- `plot_stopping_times_vs_epsilon.py` — stopping-time sweep across ε.
- `stopping_times_demo.py` — side-by-side comparison on a single stream.
- `outside_interval_demo.py` — `OutsideInterval` on a band-crossing scenario.

## Privacy caveat

Noise is drawn with NumPy floating-point arithmetic and a non-cryptographic
generator, so the guarantees are those of the idealized real-valued mechanisms.
The known floating-point attacks on differentially private samplers are out of
scope, and this library is not hardened against them.

## Citation

```bibtex
@InProceedings{pmlr-v300-michel26a,
  title     = {DP-SPRT: Differentially Private Sequential Probability Ratio Tests},
  author    = {Michel, Thomas and Basu, Debabrota and Kaufmann, Emilie},
  booktitle = {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  pages     = {3097--3105},
  year      = {2026},
  volume    = {300},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v300/michel26a.html}
}
```

## License

MIT. See [LICENSE](LICENSE).
