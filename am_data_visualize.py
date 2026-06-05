import pandas as pd
import numpy as np
import matplotlib

# ============================================================
# Optional backend setup
# ============================================================
# If figures do not pop up when running as a normal Python file,
# uncomment ONE of these lines depending on your system:
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

# Your older contour script used min.
# mean = smoother
# min = sharper / more cliff-like
# max = worst-case / often jaggier
AGGFUNC = "min"

CMAP = "viridis"

ELEV = 25
AZIM = -135

DPI = 250

# This controls whether figures pop up interactively.
# Running as a normal .py file, plt.show() opens an interactive Matplotlib window.
SHOW_FIGURES = True

# If True, each individual cost surface also opens/saves.
# If False, only the combined 2x2 figures are made.
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


def get_surface_data(data, value_col):
    """
    Build the P/R grid.

    This follows the same idea as your original scripts:
    - pivot table
    - P values are columns
    - R values are rows
    - Z is the selected cost value

    Since the full dataset has multiple rows per P/R pair,
    AGGFUNC decides how to collapse them.
    """
    pivot = data.pivot_table(
        index=R_COL,
        columns=P_COL,
        values=value_col,
        aggfunc=AGGFUNC
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    p_values = pivot.columns.to_numpy()
    r_values = pivot.index.to_numpy()

    X, Y = np.meshgrid(p_values, r_values)
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


def plot_surface_on_axis(ax, data, value_col, title):
    X, Y, Z, pivot = get_surface_data(data, value_col)

    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=CMAP,
        linewidth=0,
        antialiased=True,
        alpha=0.95
    )

    ax.contour(
        X,
        Y,
        Z,
        zdir="z",
        offset=zmin,
        levels=15,
        cmap=CMAP,
        alpha=0.8
    )

    ax.set_title(title)
    ax.set_xlabel("Pbar ratio")
    ax.set_ylabel("R ratio")
    ax.set_zlabel("Cost")

    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=ELEV, azim=AZIM)

    return surface


def plot_individual_surface(data, group_name, value_col, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surface = plot_surface_on_axis(
        ax=ax,
        data=data,
        value_col=value_col,
        title=title
    )

    fig.colorbar(surface, ax=ax, shrink=0.6, label=value_col)

    fig.suptitle(
        f"{title} over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=14
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_{safe_filename(value_col)}_surface_{AGGFUNC}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    print("Saved:", save_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_combined_surfaces(data, group_name):
    fig = plt.figure(figsize=(16, 10))

    for i, (value_col, title) in enumerate(COST_COLUMNS.items(), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")

        surface = plot_surface_on_axis(
            ax=ax,
            data=data,
            value_col=value_col,
            title=title
        )

        fig.colorbar(surface, ax=ax, shrink=0.5)

    fig.suptitle(
        f"FlexDC Cost Surfaces over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=16
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_combined_surfaces_{AGGFUNC}.png"
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

    # Combined 2x2 figure like your original am_3d style
    plot_combined_surfaces(workload_df, group_name)

    # Optional individual figures
    if MAKE_INDIVIDUAL_PLOTS:
        for value_col, title in COST_COLUMNS.items():
            plot_individual_surface(
                data=workload_df,
                group_name=group_name,
                value_col=value_col,
                title=title
            )


# ============================================================
# PLOT ALL DATA COMBINED
# ============================================================

group_name = "all_workloads_combined"

summarize_group(group_name, df)

plot_combined_surfaces(df, group_name)

if MAKE_INDIVIDUAL_PLOTS:
    for value_col, title in COST_COLUMNS.items():
        plot_individual_surface(
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