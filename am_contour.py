import pandas as pd
import numpy as np
import matplotlib

# ============================================================
# Optional backend setup
# ============================================================
# If figures do not pop up when running as a normal Python file,
# uncomment ONE of these lines:
#
# matplotlib.use("TkAgg")
# matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data.csv"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

P_COL = "Pbar_ratio"
R_COL = "R_ratio"
WORKLOAD_COL = "workload_name"

COST_COLUMNS = {
    "Power_Cost": "Power Cost",
    "Raw_Tracking_Cost": "Raw Tracking Cost",
    "Raw_QoS_Cost": "Raw QoS Cost",
    "Raw_Total_Cost": "Raw Total Cost",
}

# This follows your older contour style more closely.
# "min" gives sharper/cliff-like behavior.
# "mean" smooths by averaging across server_count/utilization/weights.
# Other options: "max", "median"
AGGFUNC = "min"

# Plot appearance
CMAP = "viridis"
CONTOUR_LEVELS = 20
DPI = 250

# Show plots interactively as the script runs.
# Close each window to let the script continue.
SHOW_FIGURES = True

# You asked to skip individuals by default.
MAKE_INDIVIDUAL_PLOTS = False


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [P_COL, R_COL, WORKLOAD_COL] + list(COST_COLUMNS.keys())
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Workloads:", sorted(df[WORKLOAD_COL].dropna().unique()))
print("Unique P values:", df[P_COL].nunique())
print("Unique R values:", df[R_COL].nunique())
print("Aggregation:", AGGFUNC)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(":", "")
    )


def get_contour_data(data, value_col):
    """
    Build P/R grid for contour plots.

    Pbar_ratio becomes x-axis.
    R_ratio becomes y-axis.
    selected cost becomes z/color value.

    Since full data has multiple rows for the same P/R point,
    AGGFUNC collapses those rows.
    """
    pivot = data.pivot_table(
        index=R_COL,
        columns=P_COL,
        values=value_col,
        aggfunc=AGGFUNC
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    X = pivot.columns.to_numpy()
    Y = pivot.index.to_numpy()
    Z = pivot.to_numpy()

    return X, Y, Z, pivot


def summarize_group(group_name, data):
    print()
    print("====================================")
    print(group_name)
    print("Rows:", len(data))
    print("Unique P values:", data[P_COL].nunique())
    print("Unique R values:", data[R_COL].nunique())
    print("Unique P/R pairs:", data[[P_COL, R_COL]].drop_duplicates().shape[0])
    print("====================================")


def plot_contour_on_axis(ax, data, value_col, title):
    X, Y, Z, pivot = get_contour_data(data, value_col)

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=CONTOUR_LEVELS,
        cmap=CMAP
    )

    # Add contour lines to make shape/features easier to see.
    lines = ax.contour(
        X,
        Y,
        Z,
        levels=CONTOUR_LEVELS,
        colors="black",
        linewidths=0.35,
        alpha=0.45
    )

    # Mark the best observed P/R point for this cost.
    best_idx = data[value_col].idxmin()
    best_row = data.loc[best_idx]

    ax.scatter(
        best_row[P_COL],
        best_row[R_COL],
        marker="x",
        s=100,
        color="red",
        linewidths=2,
        label="min"
    )

    ax.set_title(title)
    ax.set_xlabel("Pbar ratio")
    ax.set_ylabel("R ratio")
    ax.legend(loc="best")

    return contour


def plot_combined_contours(data, group_name):
    """
    Makes one 2x2 figure with all four cost contour plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (value_col, title) in zip(axes.ravel(), COST_COLUMNS.items()):
        contour = plot_contour_on_axis(
            ax=ax,
            data=data,
            value_col=value_col,
            title=title
        )

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(value_col)

    fig.suptitle(
        f"FlexDC Cost Contours over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=16
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_combined_contours_{AGGFUNC}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    print("Saved:", save_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_individual_contour(data, group_name, value_col, title):
    """
    Optional individual contour plot.
    Disabled by default using MAKE_INDIVIDUAL_PLOTS=False.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    contour = plot_contour_on_axis(
        ax=ax,
        data=data,
        value_col=value_col,
        title=title
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(value_col)

    fig.suptitle(
        f"{title} over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=14
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_{safe_filename(value_col)}_contour_{AGGFUNC}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    print("Saved:", save_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# PLOT EACH WORKLOAD
# ============================================================

workloads = sorted(df[WORKLOAD_COL].dropna().unique())

for workload in workloads:
    workload_df = df[df[WORKLOAD_COL] == workload].copy()
    group_name = f"workload_{workload}"

    summarize_group(group_name, workload_df)

    # Main requested output: one 2x2 contour figure
    plot_combined_contours(workload_df, group_name)

    # Optional individual plots
    if MAKE_INDIVIDUAL_PLOTS:
        for value_col, title in COST_COLUMNS.items():
            plot_individual_contour(
                data=workload_df,
                group_name=group_name,
                value_col=value_col,
                title=title
            )


# ============================================================
# PLOT ALL WORKLOADS COMBINED
# ============================================================

group_name = "all_workloads_combined"

summarize_group(group_name, df)

plot_combined_contours(df, group_name)

if MAKE_INDIVIDUAL_PLOTS:
    for value_col, title in COST_COLUMNS.items():
        plot_individual_contour(
            data=df,
            group_name=group_name,
            value_col=value_col,
            title=title
        )


# ============================================================
# DONE
# ============================================================

print()
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())