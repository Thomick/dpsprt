"""OutsideInterval demo: eps-DP detection of a band crossing on a sensor stream.

The sensor produces in-band readings for the first 200 steps, then drifts out of
the calibration band [-2, 2] for the next 200. OutsideInterval reports the first
step at which the noisy reading crosses a threshold.

The budget is not free to choose. Query noise has a fixed scale here, since the
query is a single reading rather than a running mean whose sensitivity decays,
so the chance of a false alarm accumulates over the horizon. The bound below
picks the smallest eps that keeps that chance under 5 percent across 200 in-band
steps, and the monitor is only as private as that bound allows.
"""

import math

import numpy as np

from dpsprt import OutsideInterval

BAND = 2.0
SENSITIVITY = 1.0
IN_BAND_STEPS = 200
FALSE_ALARM_BUDGET = 0.05


def epsilon_for_horizon(band, sensitivity, steps, target):
    """Smallest eps whose per-step false-alarm probability keeps the horizon under target.

    A Lap(b) draw leaves [-band, band] with probability exp(-band/b), and an even
    split of eps puts b = 4 * sensitivity / eps on each query.
    """
    b = band / math.log(steps / target)
    return 4.0 * sensitivity / b


def main():
    rng = np.random.default_rng(0)
    eps = epsilon_for_horizon(BAND, SENSITIVITY, IN_BAND_STEPS, FALSE_ALARM_BUDGET)

    in_band = rng.uniform(-0.3, 0.3, size=IN_BAND_STEPS)
    out_of_band = rng.uniform(2.3, 2.8, size=200)
    sensor_stream = np.concatenate([in_band, out_of_band])

    oi = OutsideInterval(
        lower_threshold=lambda t: -BAND,
        upper_threshold=lambda t: BAND,
        query_noise_scale=4.0 * SENSITIVITY / eps,
        threshold_noise_scale=2.0 * SENSITIVITY / eps,
        epsilon=eps,
        sensitivity=SENSITIVITY,
        random_seed=42,
    )

    print(
        f"Budget: eps={eps:.1f} for at most {FALSE_ALARM_BUDGET:.0%} false-alarm risk "
        f"over {IN_BAND_STEPS} in-band steps."
    )

    for i, value in enumerate(sensor_stream):
        stopped, side = oi.add_query(float(value))
        if stopped:
            direction = "upper" if side == 1 else "lower"
            phase = "in-band" if i < IN_BAND_STEPS else "out-of-band"
            print(f"Alarm at step {i + 1} ({phase} phase): {direction} bound crossed.")
            return

    print("No alarm raised in 400 steps.")


if __name__ == "__main__":
    main()
