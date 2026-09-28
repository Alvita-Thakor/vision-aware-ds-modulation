import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy import stats
import matplotlib.pyplot as plt

def wilson_ci(successes, n, confidence=0.95):
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator

    margin = ( z * np.sqrt((p*(1-p) / n) + (z**2 / (4*n**2)))) / denominator
    lower = max(0.0,center-margin)
    upper = min(1.0,center + margin)
    return lower, upper

def mean_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    margin = 1.96 * (std / np.sqrt(n))
    return mean, std, (mean - margin, mean + margin)

def plot_sweep(
    csv_file,
    parameter_column,
    parameter_values,
    x_label,
    y_column,
    y_label,
    title,
    output_file
):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 5))

    for mode, label in [("noisy", "Noisy"), ("kalman", "Kalman")]:
        means = []
        lower = []
        upper = []

        for value in parameter_values:

            values = df[
                (df["mode"] == mode) &
                (df[parameter_column] == value)
            ][y_column].dropna()

            if len(values) == 0:
                means.append(np.nan)
                lower.append(np.nan)
                upper.append(np.nan)
                continue

            mean = values.mean()
            std = values.std()
            ci = 1.96 * std / (len(values) ** 0.5)

            means.append(mean)
            lower.append(mean - ci)
            upper.append(mean + ci)

        means = pd.Series(means)
        lower = pd.Series(lower)
        upper = pd.Series(upper)

        plt.errorbar(
            parameter_values,
            means,
            yerr=[means - lower, upper - means],
            marker="o",
            capsize=4,
            label=label
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.show()

def plot_rate_sweep(csv_file,parameter_column,parameter_values,x_label,metric_column,y_label,title,output_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 5))

    for mode, label in [("noisy", "Noisy"), ("kalman", "Kalman")]:
        rates = []
        lower = []
        upper = []

        for value in parameter_values:

            values = df[(df["mode"] == mode) &(df[parameter_column] == value)][metric_column].dropna()

            n = len(values)
            successes = values.sum()

            # Wilson 95% confidence interval
            z = 1.96
            p = successes / n

            denominator = 1 + (z**2 / n)
            centre = (p + z**2 / (2*n)) / denominator
            margin = (z * np.sqrt((p * (1-p) / n) +(z**2 / (4*n**2)))/ denominator)

            rates.append(p)
            lower.append(max(0,centre - margin))
            upper.append(min(1,centre + margin))

        rates = np.array(rates)
        lower = np.array(lower)
        upper = np.array(upper)

        # Convert to percentage
        rates *= 100
        lower *= 100
        upper *= 100
        lower = np.minimum(lower, rates)
        upper = np.maximum(upper, rates)

        plt.errorbar(parameter_values,rates,yerr=[rates - lower, upper - rates],marker="o",capsize=4,label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.show()

def print_metric_table(csv_file, parameter_column, parameter_values, metric_column, metric_name):
    df = pd.read_csv(csv_file)

    rows = []
    for value in parameter_values:
        for mode in ["noisy", "kalman"]:

            values = df[
                (df["mode"] == mode) &
                (df[parameter_column] == value)
            ][metric_column].dropna()

            mean = values.mean()
            std = values.std()
            n = len(values)

            if n == 0:
                ci = np.nan
            else:
                ci = 1.96 * std / (n ** 0.5)

            rows.append({
                "Parameter": value,
                "Mode": mode.capitalize(),
                "Mean": mean,
                "Std": std,
                "95% CI Lower": mean - ci if n > 0 else np.nan,
                "95% CI Upper": mean + ci if n > 0 else np.nan
            })

    table = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(metric_name.upper())
    print("=" * 80)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
def print_rate_table(csv_file,parameter_column,parameter_values,metric_column,metric_name):
    df = pd.read_csv(csv_file)

    rows = []

    for value in parameter_values:
        for mode in ["noisy", "kalman"]:

            values = df[(df["mode"] == mode) &(df[parameter_column] == value)][metric_column].dropna()

            n = len(values)
            successes = values.sum()

            z = 1.96
            p = successes / n
            denominator = 1 + (z**2 / n)
            centre = (p + z**2 / (2 * n)) / denominator

            margin = (z * np.sqrt((p * (1 - p) / n)+ (z**2 / (4 * n**2)))/ denominator)

            lower = max(0, centre - margin)
            upper = min(1, centre + margin)

            rows.append({"Parameter": value,"Mode": mode.capitalize(),"Rate (%)": p * 100,"95% CI Lower (%)": lower * 100,"95% CI Upper (%)": upper * 100})

    table = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(metric_name.upper())
    print("=" * 80)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

noise_levels = [0,2,4,6,8,10]
blur_levels = [0,3,5,7,9]
fps_levels = [2,4,5,10,20]
latency_levels = [0,50,100,150,200]
occlusion_levels =[0,0.25,0.50]

'''

for mode in ["noisy", "kalman"]:

    print("\nMODE:", mode)

    for noise in noise_levels:

        row = []

        for occlusion in occlusion_levels:

            subset = df[
                (df["mode"] == mode) &
                (df["pixel_noise_std"] == noise) &
                (df["occlusion_percentage"] == occlusion)
            ]

            row.append(subset["path_length"].mean())

        print("Noise", noise, ":", row)

matrix = []

for noise in noise_levels:
    row = []
    for occlusion in occlusion_levels:
        subset = df[(df["mode"] == "noisy") &(df["pixel_noise_std"] == noise) &(df["occlusion_percentage"] == occlusion)]

        row.append(subset["convergence_time"].mean())

    matrix.append(row)

matrix = np.array(matrix)

plt.figure(figsize=(7, 5))
plt.imshow(matrix, aspect="auto",vmin=0.9,vmax=1.05)
plt.xticks(range(len(occlusion_levels)),[f"{x*100:.0f}%" for x in occlusion_levels])
plt.yticks(range(len(noise_levels)),noise_levels)

plt.xlabel("Occlusion")
plt.ylabel("Pixel Noise (px)")
plt.title("Convergence Time — noisy")

plt.colorbar(label="Convergence Time (s)")

plt.tight_layout()
plt.show()

improvement = []

for noise in noise_levels:
    row = []

    for occlusion in occlusion_levels:

        noisy_subset = df[
            (df["mode"] == "noisy") &
            (df["pixel_noise_std"] == noise) &
            (df["occlusion_percentage"] == occlusion)
        ]

        kalman_subset = df[
            (df["mode"] == "kalman") &
            (df["pixel_noise_std"] == noise) &
            (df["occlusion_percentage"] == occlusion)
        ]

        noisy_mean = noisy_subset["path_length"].mean()
        kalman_mean = kalman_subset["path_length"].mean()

        if noisy_mean == 0:
            row.append(np.nan)
        else:
            row.append((noisy_mean - kalman_mean) / noisy_mean * 100)

    improvement.append(row)

improvement = np.array(improvement) *100

print(improvement)

plt.figure(figsize=(7,5))

plt.imshow(improvement, aspect="auto",vmin=-10,vmax=40)

plt.xticks(
    range(len(occlusion_levels)),
    [f"{x*100:.0f}%" for x in occlusion_levels]
)

plt.yticks(range(len(noise_levels)), noise_levels)

plt.xlabel("Occlusion")
plt.ylabel("Pixel Noise (px)")
plt.title("Kalman RMSE Improvement")

plt.colorbar(label="RMSE Improvement (%)")

plt.tight_layout()
plt.show()


CSV_FILE = "latency_sweep_new2.csv"

latency_levels = [0, 50, 100, 150, 200]

# RMSE
print_metric_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "localization_rmse",
    "Localization RMSE vs Latency"
)

plot_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "localization_rmse",
    "Localization RMSE",
    "Localization RMSE vs Latency",
    "latency_rmse.png"
)

# Minimum clearance
print_metric_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "minimum_clearance",
    "Minimum Clearance vs Latency"
)

plot_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "minimum_clearance",
    "Minimum Clearance (m)",
    "Minimum Clearance vs Latency",
    "latency_clearance.png"
)

# Convergence time
print_metric_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "convergence_time",
    "Convergence Time vs Latency"
)

plot_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "convergence_time",
    "Convergence Time (s)",
    "Convergence Time vs Latency",
    "latency_convergence.png"
)

# Path length
print_metric_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "path_length",
    "Path Length vs Latency"
)

plot_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "path_length",
    "Path Length (m)",
    "Path Length vs Latency",
    "latency_path.png"
)

# Collision rate
print_rate_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "collision",
    "Collision Rate vs Latency"
)

plot_rate_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "collision",
    "Collision Rate (%)",
    "Collision Rate vs Latency",
    "latency_collision.png"
)

# Goal reached rate
print_rate_table(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "goal_reached",
    "Goal Reached Rate vs Latency"
)

plot_rate_sweep(
    CSV_FILE,
    "latency_ms",
    latency_levels,
    "Latency (ms)",
    "goal_reached",
    "Goal Reached Rate (%)",
    "Goal Reached Rate vs Latency",
    "latency_goal_reached.png"
)
'''
# ============================================================
# STEP 3 — STATISTICAL ANALYSIS
# Noise: Paired Noisy vs Kalman comparison
# ============================================================

'''
df = pd.read_csv("noise_occlusion_sweep_new.csv")

metrics = [
    "localization_rmse",
    "minimum_clearance",
    "convergence_time",
    "path_length"
]

noise_levels = [0, 5, 10]
occlusion_levels = [0, 0.25, 0.50]

for noise in noise_levels:

    for occlusion in occlusion_levels:

        print("\n" + "=" * 80)
        print(f"Noise = {noise} px | Occlusion = {occlusion * 100:.0f}%")
        print("=" * 80)

        noisy = df[
            (df["mode"] == "noisy") &
            (df["pixel_noise_std"] == noise) &
            (df["occlusion_percentage"] == occlusion)
        ].sort_values("rng_seed")

        kalman = df[
            (df["mode"] == "kalman") &
            (df["pixel_noise_std"] == noise) &
            (df["occlusion_percentage"] == occlusion)
        ].sort_values("rng_seed")

        for metric in metrics:

            paired = pd.DataFrame({
                "noisy": noisy[metric].to_numpy(),
                "kalman": kalman[metric].to_numpy()
            }).dropna()

            noisy_values = paired["noisy"].to_numpy()
            kalman_values = paired["kalman"].to_numpy()

            print(f"\n{metric}")

            # Not enough valid pairs
            if len(paired) < 2:
                print("Not enough valid paired samples.")
                continue

            differences = kalman_values - noisy_values

            # Check for zero variance
            if differences.std(ddof=1) == 0:
                print(f"Mean difference (Kalman - Noisy): {differences.mean():.6f}")
                print("Std difference: 0.000000")
                print("Paired t-test: not applicable (zero variance)")
                continue

            t_stat, p_value = ttest_rel(
                kalman_values,
                noisy_values
            )

            print(f"Mean difference (Kalman - Noisy): {differences.mean():.6f}")
            print(f"Std difference: {differences.std(ddof=1):.6f}")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_value:.6e}")
'''

# ============================================================
# STEP 4: HOLM MULTIPLE-COMPARISON CORRECTION
# ============================================================

def holm_correction(p_values, alpha=0.05):
    """
    Holm-Bonferroni correction.
    Returns adjusted p-values and significance decisions.
    """
    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    m = len(sorted_p)

    adjusted = np.empty(m)

    for i, p in enumerate(sorted_p):
        adjusted[i] = min((m - i) * p, 1.0)

    # Ensure adjusted p-values are monotonic
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted_original_order = np.empty(m)
    adjusted_original_order[order] = adjusted

    significant = adjusted_original_order < alpha

    return adjusted_original_order, significant


# ------------------------------------------------------------
# Helper to collect valid paired-test p-values
# ------------------------------------------------------------

def collect_paired_tests(df, experiment_name, conditions):
    
    results = []

    metrics = [
        "localization_rmse",
        "minimum_clearance",
        "convergence_time",
        "path_length"
    ]

    for condition_name, condition_filter in conditions:

        noisy = df[
            (df["mode"] == "noisy") &
            condition_filter
        ].sort_values("rng_seed")

        kalman = df[
            (df["mode"] == "kalman") &
            condition_filter
        ].sort_values("rng_seed")

        for metric in metrics:

            paired = pd.DataFrame({
                "noisy": noisy[metric].to_numpy(),
                "kalman": kalman[metric].to_numpy()
            }).dropna()

            if len(paired) < 2:
                continue

            differences = (
                paired["kalman"].to_numpy()
                - paired["noisy"].to_numpy()
            )

            # Zero-variance differences → t-test not applicable
            if differences.std(ddof=1) == 0:
                continue

            t_stat, p_value = ttest_rel(
                paired["kalman"].to_numpy(),
                paired["noisy"].to_numpy()
            )

            results.append({
                "experiment": experiment_name,
                "condition": condition_name,
                "metric": metric,
                "p_value": p_value
            })

    return results


# ============================================================
# Collect tests
# ============================================================

all_results = []


# -----------------------------
# NOISE
# -----------------------------

df = pd.read_csv("noise_sweep_new.csv")

noise_conditions = []

for noise in [2, 4, 6, 8, 10]:

    condition = (
        (df["pixel_noise_std"] == noise)
    )

    noise_conditions.append(
        (f"{noise} px", condition)
    )

all_results.extend(
    collect_paired_tests(
        df,
        "Noise",
        noise_conditions
    )
)


# -----------------------------
# BLUR
# -----------------------------

df = pd.read_csv("blur_sweep_new2.csv")

blur_conditions = []

for blur in [0, 3, 5, 7, 9]:

    condition = (
        (df["blur_kernel_size"] == blur)
    )

    blur_conditions.append(
        (f"{blur} px", condition)
    )

all_results.extend(
    collect_paired_tests(
        df,
        "Blur",
        blur_conditions
    )
)


# -----------------------------
# FPS
# -----------------------------

df = pd.read_csv("fps_sweep_new2.csv")

fps_conditions = []

for fps in [2, 4, 5, 10, 20]:

    condition = (
        (df["camera_fps"] == fps)
    )

    fps_conditions.append(
        (f"{fps} FPS", condition)
    )

all_results.extend(
    collect_paired_tests(
        df,
        "FPS",
        fps_conditions
    )
)


# -----------------------------
# LATENCY
# -----------------------------

df = pd.read_csv("latency_sweep_new2.csv")

latency_conditions = []

for latency in [0, 50, 100, 150, 200]:

    condition = (
        (df["latency_ms"] == latency)
    )

    latency_conditions.append(
        (f"{latency} ms", condition)
    )

all_results.extend(
    collect_paired_tests(
        df,
        "Latency",
        latency_conditions
    )
)


# -----------------------------
# NOISE × OCCLUSION
# -----------------------------

df = pd.read_csv("noise_occlusion_sweep_new.csv")

combined_conditions = []

for noise in [0, 5, 10]:

    for occlusion in [0, 0.25, 0.50]:

        condition = (
            (df["pixel_noise_std"] == noise) &
            (df["occlusion_percentage"] == occlusion)
        )

        combined_conditions.append(
            (
                f"{noise} px + {occlusion * 100:.0f}%",
                condition
            )
        )

all_results.extend(
    collect_paired_tests(
        df,
        "Noise × Occlusion",
        combined_conditions
    )
)


# ============================================================
# APPLY HOLM CORRECTION
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 100)
print("HOLM-CORRECTED PAIRED T-TEST RESULTS")
print("=" * 100)


for experiment in results_df["experiment"].unique():

    experiment_results = results_df[
        results_df["experiment"] == experiment
    ].copy()

    adjusted_p, significant = holm_correction(
        experiment_results["p_value"].to_numpy()
    )

    experiment_results["holm_p"] = adjusted_p
    experiment_results["significant"] = significant

    print("\n" + "-" * 100)
    print(experiment)
    print("-" * 100)

    for _, row in experiment_results.iterrows():

        print(
            f"{row['condition']:15s} | "
            f"{row['metric']:20s} | "
            f"raw p = {row['p_value']:.6e} | "
            f"Holm p = {row['holm_p']:.6e} | "
            f"{'SIGNIFICANT' if row['significant'] else 'NOT SIGNIFICANT'}"
        )