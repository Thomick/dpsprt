"""Reproduce the main experiments from the AISTATS 2026 paper.

Fixed α=β; γ = max(0.5, 1−1/ε) and r = min(1,√(ε/10)) vary automatically
with ε as implemented in the library.

Three panels matching the paper:
  1 — mean stopping time (H₁) vs ε
  2 — type I error rate vs ε
  3 — type II error rate vs ε

Usage:
    python examples/paper_experiments.py
"""

import matplotlib.pyplot as plt
import numpy as np

from dpsprt import DPSPRT, ClassicalSPRT, DPSPRTGaussian, DPSPRTSubsampled

# ---------------------------------------------------------------------------
# Experiment parameters — match paper run_all.py
# ---------------------------------------------------------------------------
MU0, MU1 = 0.3, 0.7
DELTA = 1e-5
ALPHA = BETA = 0.05  # fixed; γ and r are derived from ε internally

EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

N_TRIALS = 300  # paper uses 1000; 300 gives clear visual results
MAX_SAMPLES = 10_000
SEED = 42

STYLES = {
    "Classical SPRT": dict(color="#1f77b4", marker="o", ls="-"),
    "DP-SPRT (Laplace)": dict(color="#ff7f0e", marker="s", ls="-"),
    "DP-SPRT (Gaussian)": dict(color="#9467bd", marker="*", ls="-"),
    "DP-SPRT (Subsampled)": dict(color="#2ca02c", marker="D", ls="-"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alg(name, eps, seed):
    # γ = max(0.5, 1−1/ε) and r = min(1,√(ε/10)) are computed internally
    if name == "Classical SPRT":
        return ClassicalSPRT(mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA)
    if name == "DP-SPRT (Laplace)":
        return DPSPRT(mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA, epsilon=eps, random_seed=seed)
    if name == "DP-SPRT (Gaussian)":
        return DPSPRTGaussian(
            mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA, epsilon=eps, delta=DELTA, random_seed=seed
        )
    if name == "DP-SPRT (Subsampled)":
        return DPSPRTSubsampled(
            mu0=MU0, mu1=MU1, alpha=ALPHA, beta=BETA, epsilon=eps, random_seed=seed
        )
    raise ValueError(name)


def run_trials(name, eps, true_mu, seed):
    rng = np.random.default_rng(seed)
    target = 1 if true_mu == MU1 else -1
    times, correct = [], []
    for trial in range(N_TRIALS):
        alg = _make_alg(name, eps, seed * 10_000 + trial)
        data = rng.binomial(1, true_mu, size=MAX_SAMPLES)
        decision = 0
        for t, x in enumerate(data):
            can_stop, d = alg.add_sample(int(x))
            if can_stop:
                times.append(t + 1)
                decision = d
                break
        else:
            times.append(MAX_SAMPLES)
        correct.append(int(decision == target))
    return np.array(times, dtype=float), np.array(correct, dtype=float)


def collect():
    results = {n: {} for n in STYLES}
    total = len(STYLES) * len(EPSILONS)
    done = 0
    for alg in STYLES:
        for eps in EPSILONS:
            done += 1
            g = max(0.5, 1.0 - 1.0 / eps)
            r = min(1.0, (eps / 10.0) ** 0.5)
            print(
                f"  [{done}/{total}] {alg}  ε={eps:6.1f}  γ={g:.3f}  r={r:.3f} ...",
                end=" ",
                flush=True,
            )
            h0_t, h0_c = run_trials(alg, eps, MU0, SEED)
            h1_t, h1_c = run_trials(alg, eps, MU1, SEED + 1)
            results[alg][eps] = dict(
                h1_mean=h1_t.mean(),
                h1_q25=np.percentile(h1_t, 25),
                h1_q75=np.percentile(h1_t, 75),
                type_i=1.0 - h0_c.mean(),
                type_ii=1.0 - h1_c.mean(),
            )
            rec = results[alg][eps]
            print(f"H1_avg={rec['h1_mean']:.0f}  α̂={rec['type_i']:.3f}  β̂={rec['type_ii']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"DP-SPRT  (μ₀={MU0}, μ₁={MU1}, α=β={ALPHA},  N={N_TRIALS} trials)\n"
        r"$\gamma=\max(0.5,\,1-1/\varepsilon)$ and "
        r"$r=\min(1,\sqrt{\varepsilon/10})$ vary with $\varepsilon$ as in the paper",
        fontsize=11,
    )

    metrics = [
        ("h1_mean", "h1_q25", "h1_q75", "Mean stopping time (H₁)", False, "Stopping time vs ε"),
        ("type_i", None, None, "Type I error rate (α̂)", False, "Type I error vs ε"),
        ("type_ii", None, None, "Type II error rate (β̂)", False, "Type II error vs ε"),
    ]

    for ax, (key, q25_key, q75_key, ylabel, log_y, title) in zip(axes, metrics):
        for alg, st in STYLES.items():
            y = [results[alg][e][key] for e in EPSILONS]
            if q25_key:
                q25 = [results[alg][e][q25_key] for e in EPSILONS]
                q75 = [results[alg][e][q75_key] for e in EPSILONS]
                yerr = [
                    np.maximum(0, np.array(y) - np.array(q25)),
                    np.maximum(0, np.array(q75) - np.array(y)),
                ]
                ax.errorbar(
                    EPSILONS,
                    y,
                    yerr=yerr,
                    label=alg,
                    color=st["color"],
                    marker=st["marker"],
                    ls=st["ls"],
                    lw=1.8,
                    ms=5,
                    capsize=3,
                    elinewidth=1,
                )
            else:
                ax.plot(
                    EPSILONS,
                    y,
                    label=alg,
                    color=st["color"],
                    marker=st["marker"],
                    ls=st["ls"],
                    lw=1.8,
                    ms=5,
                )

        if key in ("type_i", "type_ii"):
            ax.axhline(ALPHA, color="black", ls=":", lw=1.5, label=f"Target = {ALPHA}")

        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Privacy budget ε", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out = "examples/paper_experiments.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running paper experiments ({len(EPSILONS)} ε values × {len(STYLES)} algorithms)")
    print(f"  α=β={ALPHA}, N={N_TRIALS} trials, max {MAX_SAMPLES} samples, seed={SEED}\n")
    results = collect()
    plot_results(results)
