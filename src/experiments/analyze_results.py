import pandas as pd
from scipy import stats
import numpy as np


def wilson_ci(successes, n, confidence=0.95):
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    p = successes / n

    denominator = 1 + z**2 / n

    center = (
        p + z**2 / (2*n)
    ) / denominator

    margin = (
        z * np.sqrt(
            (p*(1-p) / n) + (z**2 / (4*n**2))
        )
    ) / denominator

    return center - margin, center + margin


def mean_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)

    margin = 1.96 * (std / np.sqrt(n))

    return mean, std, (mean - margin, mean + margin)


# ============================================================
# LOAD NOISE SWEEP
# ============================================================

df = pd.read_csv("noise_sweep.csv")

noise_levels = [0, 2, 4, 6, 8, 10]
modes = ["ground_truth","noisy" ,"kalman"]


# ============================================================
# RMSE BY NOISE AND MODE
# ============================================================

metrics = ["localization_rmse","minimum_clearance","convergence_time","path_length"]

for noise in noise_levels:
    for mode in modes:
        subset = df[(df["pixel_noise_std"] == noise) &(df["mode"] == mode)]

        print("\nNoise:", noise, "| Mode:", mode)

        for metric in metrics:
            if metric == "localization_rmse" and mode == "ground_truth":
                continue
            print(metric,":",mean_ci(subset[metric]))

for noise in noise_levels:

    noisy = df[(df["pixel_noise_std"] == noise) &(df["mode"] == "noisy")]

    kalman = df[(df["pixel_noise_std"] == noise) &(df["mode"] == "kalman")]

    noisy_rmse = np.mean(noisy["localization_rmse"])
    kalman_rmse = np.mean(kalman["localization_rmse"])

    if noisy_rmse > 0:
        improvement = ((noisy_rmse - kalman_rmse)/ noisy_rmse) * 100
    else:
        improvement = np.nan

    print("Noise:", noise,"| KF improvement:", improvement, "%")

# ============================================================
# GOAL + COLLISION BY NOISE AND MODE
# ============================================================

for noise in noise_levels:

    for mode in ["ground_truth", "noisy", "kalman"]:

        subset = df[(df["pixel_noise_std"] == noise) &(df["mode"] == mode)]

        n = len(subset)

        goal_count = subset["goal_reached"].sum()
        collision_count = subset["collision"].sum()

        goal_rate = goal_count / n
        collision_rate = collision_count / n

        goal_ci = wilson_ci(goal_count, n)
        collision_ci = wilson_ci(collision_count, n)

        print("\nNoise:", noise,"| Mode:", mode)

        print("Goal:",goal_rate * 100,"% | CI:",goal_ci[0] * 100,"-",goal_ci[1] * 100,"%")

        print("Collision:",collision_rate * 100,"% | CI:",collision_ci[0] * 100,"-",collision_ci[1] * 100,"%")