"""OutsideInterval demo: ε-DP detection of a band crossing on a sensor stream.

The sensor produces in-band readings (uniform in [-0.3, 0.3]) for the first
200 steps, then drifts out of the calibration band [-2, 2] for the next 200.
OutsideInterval reports the first step at which the noisy reading crosses one
of the thresholds.
"""

import numpy as np

from dpsprt import OutsideInterval


def main():
    rng = np.random.default_rng(0)
    eps = 20.0
    in_band = rng.uniform(-0.3, 0.3, size=200)
    out_of_band = rng.uniform(2.3, 2.8, size=200)
    sensor_stream = np.concatenate([in_band, out_of_band])

    oi = OutsideInterval(
        lower_threshold=lambda t: -2.0,
        upper_threshold=lambda t: 2.0,
        query_noise_scale=2.0 / eps,
        threshold_noise_scale=1.0 / eps,
        epsilon=eps,
        random_seed=42,
    )

    for i, value in enumerate(sensor_stream):
        stopped, side = oi.add_query(float(value))
        if stopped:
            direction = "upper" if side == 1 else "lower"
            phase = "in-band" if i < 200 else "out-of-band"
            print(f"Alarm at step {i + 1} ({phase} phase): {direction} bound crossed.")
            print(f"Privacy budget: ε={eps}")
            return

    print("No alarm raised in 400 steps.")


if __name__ == "__main__":
    main()
