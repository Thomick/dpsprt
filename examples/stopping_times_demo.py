"""Single-run stopping-time comparison across SPRT variants."""

import numpy as np

from dpsprt import (
    DPSPRT,
    ClassicalSPRT,
    DPSPRTGaussian,
    DPSPRTSubsampled,
    DPSPRTTuned,
)


def run_until_stop(algorithm, data_stream, max_samples):
    for i, x in enumerate(data_stream):
        if i >= max_samples:
            return None, 0
        can_stop, decision = algorithm.add_sample(x)
        if can_stop:
            return i + 1, decision
    return None, 0


def decision_label(d):
    return {-1: "H0", 0: "—", 1: "H1"}[d]


def stream(p, seed, n):
    return np.random.default_rng(seed).binomial(1, p, size=n).tolist()


def main():
    mu0, mu1 = 0.3, 0.7
    alpha = beta = 0.05
    max_samples = 10_000

    scenarios = [
        ("under H0", mu0),
        ("under H1", mu1),
        ("at midpoint", 0.5 * (mu0 + mu1)),
    ]

    for name, true_p in scenarios:
        print(f"\nStream {name} (p = {true_p}):")
        data = stream(true_p, seed=hash(name) & 0xFFFF, n=max_samples)
        algorithms = [
            ("Classical SPRT", ClassicalSPRT(mu0, mu1, alpha, beta)),
            ("DP-SPRT ε=1", DPSPRT(mu0, mu1, alpha, beta, 1.0, random_seed=42)),
            ("DP-SPRT ε=2", DPSPRT(mu0, mu1, alpha, beta, 2.0, random_seed=42)),
            ("DP-SPRT ε=5", DPSPRT(mu0, mu1, alpha, beta, 5.0, random_seed=42)),
            (
                "Tuned (threshold c1=0.5)",
                DPSPRTTuned(mu0, mu1, alpha, beta, 1.0, c1=0.5, c2=1.0, random_seed=42),
            ),
            (
                "Gaussian ε=1, δ=1e-5",
                DPSPRTGaussian(mu0, mu1, alpha, beta, 1.0, 1e-5, random_seed=42),
            ),
            ("Subsampled ε=1", DPSPRTSubsampled(mu0, mu1, alpha, beta, 1.0, random_seed=42)),
        ]
        print(f"  {'Algorithm':<24} {'Stop':>6} {'Decision':>10}")
        for label, alg in algorithms:
            stop, decision = run_until_stop(alg, data, max_samples)
            stop_str = "n/a" if stop is None else str(stop)
            print(f"  {label:<24} {stop_str:>6} {decision_label(decision):>10}")


if __name__ == "__main__":
    main()
