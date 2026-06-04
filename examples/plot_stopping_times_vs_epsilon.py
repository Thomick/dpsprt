"""Plot mean stopping time as a function of ε for every DP-SPRT variant."""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpsprt.api.sprt import (  # noqa: E402
    ClassicalSPRT,
    DPSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
)


def run_trial(algorithm, data_stream):
    algorithm.reset()
    for i, x in enumerate(data_stream):
        can_stop, _ = algorithm.add_sample(x)
        if can_stop:
            return i + 1
    return len(data_stream)


def sweep(mu0, mu1, alpha, beta, epsilons, n_trials, max_samples, seed):
    true_p = 0.5 * (mu0 + mu1) + 0.1
    results = {
        "Classical SPRT": {},
        "DP-SPRT (Laplace)": {},
        "DP-SPRT (Gaussian)": {},
        "DP-SPRT (Tuned)": {},
        "DP-SPRT (Subsampled)": {},
    }
    for eps_idx, epsilon in enumerate(epsilons):
        print(f"ε = {epsilon}  ({eps_idx + 1}/{len(epsilons)})")
        algorithms = {
            "Classical SPRT": ClassicalSPRT(mu0, mu1, alpha, beta),
            "DP-SPRT (Laplace)": DPSPRT(mu0, mu1, alpha, beta, epsilon, random_seed=seed + eps_idx),
            "DP-SPRT (Gaussian)": DPSPRTGaussian(
                mu0, mu1, alpha, beta, epsilon, delta=1e-5, random_seed=seed + eps_idx
            ),
            "DP-SPRT (Tuned)": DPSPRTTuned(
                mu0, mu1, alpha, beta, epsilon, c1=0.5, c2=1.0, random_seed=seed + eps_idx
            ),
            "DP-SPRT (Subsampled)": DPSPRTSubsampled(
                mu0, mu1, alpha, beta, epsilon, random_seed=seed + eps_idx
            ),
        }
        for name, alg in algorithms.items():
            stops = []
            for trial in range(n_trials):
                np.random.seed(seed + eps_idx * 1000 + trial)
                data = np.random.binomial(1, true_p, size=max_samples)
                stops.append(run_trial(alg, data))
            results[name][epsilon] = stops
            print(f"  {name}: mean stop = {np.mean(stops):.1f}")
    return results


def plot(results, epsilons, output_path, mu0, mu1, alpha, beta):
    styles = {
        "Classical SPRT": {"color": "#1f77b4", "marker": "o"},
        "DP-SPRT (Laplace)": {"color": "#ff7f0e", "marker": "s"},
        "DP-SPRT (Gaussian)": {"color": "#800080", "marker": "*"},
        "DP-SPRT (Tuned)": {"color": "#2ca02c", "marker": "^"},
        "DP-SPRT (Subsampled)": {"color": "#d62728", "marker": "D"},
    }
    plt.figure(figsize=(10, 6))
    for name, style in styles.items():
        means = [np.mean(results[name][e]) for e in epsilons]
        stds = [np.std(results[name][e]) for e in epsilons]
        plt.errorbar(
            epsilons, means, yerr=stds, label=name,
            color=style["color"], marker=style["marker"], linewidth=2, markersize=6, capsize=3,
        )
    plt.xlabel("Privacy parameter ε", fontsize=12)
    plt.ylabel("Average stopping time (samples)", fontsize=12)
    plt.title(
        f"Stopping time vs ε  (μ₀={mu0}, μ₁={mu1}, α={alpha}, β={beta})",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    mu0, mu1 = 0.3, 0.7
    alpha = beta = 0.05
    epsilons = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = sweep(
        mu0, mu1, alpha, beta,
        epsilons=epsilons, n_trials=50, max_samples=100_000, seed=42,
    )
    output_path = Path(__file__).parent / "stopping_times_vs_epsilon.png"
    plot(results, epsilons, output_path, mu0, mu1, alpha, beta)


if __name__ == "__main__":
    main()
