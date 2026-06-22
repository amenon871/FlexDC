import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "generated_softmax_weights2.csv"
OUTPUT_DIR = "weight_distribution_dynamic_plots2"

# Plot all workload sizes now, including 16-job workload.
WORKLOAD_SIZES_TO_PLOT = [2, 4, 8, 16]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

print("Loaded:", INPUT_CSV)
print("Rows:", len(df))
print("Output folder:", OUTPUT_DIR)


def get_grid_shape(num_panels):
    """
    Choose a compact grid based on how many panels are needed.

    J=2 needs 3 panels   -> 1x3
    J=4 needs 5 panels   -> 2x3
    J=8 needs 9 panels   -> 3x3
    J=16 needs 17 panels -> 5x4
    """
    if num_panels <= 3:
        ncols = num_panels
    elif num_panels <= 9:
        ncols = 3
    else:
        ncols = 4

    nrows = math.ceil(num_panels / ncols)
    return nrows, ncols


def hide_unused_axes(axes, used_count):
    for ax in axes[used_count:]:
        ax.axis("off")


def add_reference_lines(ax, equal_weight, lower_bound, upper_bound, lower_source, upper_source):
    ax.axvline(equal_weight, linestyle="--", linewidth=1, color="black", label="Equal weight")
    ax.axvline(lower_bound, linestyle=":", linewidth=1.5, color="red", label=f"Lower bound ({lower_source})")
    ax.axvline(upper_bound, linestyle=":", linewidth=1.5, color="red", label=f"Upper bound ({upper_source})")


def plot_histogram_grid(sub, J, temperature, equal_weight, lower_bound, upper_bound, lower_source, upper_source, weight_cols):
    num_panels = 1 + len(weight_cols)
    nrows, ncols = get_grid_shape(num_panels)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    all_weights = sub[weight_cols].to_numpy().flatten()
    bins = np.linspace(0, all_weights.max(), 40)

    # First panel: all job types overlaid.
    ax = axes[0]
    for col in weight_cols:
        ax.hist(
            sub[col].dropna(),
            bins=bins,
            density=True,
            alpha=0.45,
            label=col
        )

    add_reference_lines(ax, equal_weight, lower_bound, upper_bound, lower_source, upper_source)
    ax.set_title(f"All weights overlaid\nJ={J}, T={temperature}")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

    # Remaining panels: one job type per plot.
    for i, col in enumerate(weight_cols, start=1):
        ax = axes[i]
        ax.hist(
            sub[col].dropna(),
            bins=bins,
            density=True,
            alpha=0.85
        )

        add_reference_lines(ax, equal_weight, lower_bound, upper_bound, lower_source, upper_source)
        ax.set_title(col)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    hide_unused_axes(axes, num_panels)

    fig.suptitle(f"Histogram by job type, workload size J={J}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, f"hist_grid_J{J}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cdf_grid(sub, J, temperature, equal_weight, lower_bound, upper_bound, lower_source, upper_source, weight_cols):
    num_panels = 1 + len(weight_cols)
    nrows, ncols = get_grid_shape(num_panels)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    # First panel: all job types overlaid.
    ax = axes[0]
    for col in weight_cols:
        values = np.sort(sub[col].dropna().to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)

        ax.plot(values, cdf, label=col)

    add_reference_lines(ax, equal_weight, lower_bound, upper_bound, lower_source, upper_source)
    ax.set_title(f"All weights overlaid\nJ={J}, T={temperature}")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Fraction <= weight")
    ax.legend(fontsize=7)

    # Remaining panels: one job type per plot.
    for i, col in enumerate(weight_cols, start=1):
        ax = axes[i]

        values = np.sort(sub[col].dropna().to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)

        ax.plot(values, cdf)
        add_reference_lines(ax, equal_weight, lower_bound, upper_bound, lower_source, upper_source)
        ax.set_title(col)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Fraction <= weight")
        ax.legend(fontsize=7)

    hide_unused_axes(axes, num_panels)

    fig.suptitle(f"CDF by job type, workload size J={J}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, f"cdf_grid_J{J}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()


for J in WORKLOAD_SIZES_TO_PLOT:
    sub = df[df["workload_mix_size"] == J].copy()

    if sub.empty:
        print(f"Skipping J={J}; no rows found.")
        continue

    temperature = sub["temperature"].iloc[0]
    equal_weight = 1.0 / J

    lower_bound = sub["lower_bound"].iloc[0]
    upper_bound = sub["upper_bound"].iloc[0]
    lower_source = sub["lower_bound_source"].iloc[0]
    upper_source = sub["upper_bound_source"].iloc[0]

    weight_cols = [f"Weight_{i}" for i in range(int(J))]

    plot_histogram_grid(sub, J, temperature, equal_weight, lower_bound, upper_bound, lower_source, upper_source, weight_cols)
    plot_cdf_grid(sub, J, temperature, equal_weight, lower_bound, upper_bound, lower_source, upper_source, weight_cols)


print("\nCreated plots:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(os.path.join(OUTPUT_DIR, filename))