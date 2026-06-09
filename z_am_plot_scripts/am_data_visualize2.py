import pandas as pd
import numpy as np
import matplotlib

# If figures do not pop up, uncomment ONE:
# matplotlib.use("TkAgg")
# matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("figures_3d_raw_costs_mean")
FIG_DIR.mkdir(exist_ok=True)

# Separate high-quality plots ONLY for all_workloads_combined
HQ_TOTAL_DIR = FIG_DIR / "all_workloads_combined_high_quality_individual_components"
HQ_TOTAL_DIR.mkdir(exist_ok=True)

P_COL = "Pbar_ratio"
R_COL = "R_ratio"
WORKLOAD_COL = "workload_name"

PLOT_COLUMNS = {
    "Power_Cost": "Power Cost",
    "Raw_Tracking_Cost": "Raw Tracking Cost",
    "Raw_QoS_Cost": "Raw QoS Cost",
    "Raw_Total_Cost": "Raw Total Cost",
}

AGGFUNC = "mean"

CMAP = "viridis"
ELEV = 25
AZIM = -135

DPI = 250
HIGH_QUALITY_DPI = 500

SHOW_FIGURES = True


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [P_COL, R_COL, WORKLOAD_COL] + list(PLOT_COLUMNS.keys())
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Workloads:", sorted(df[WORKLOAD_COL].dropna().unique()))
print("Unique P values:", df[P_COL].nunique())
print("Unique R values:", df[R_COL].nunique())
print("Aggregation:", AGGFUNC)


# ============================================================
# HELPERS
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

    return X, Y, Z


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
    X, Y, Z = get_surface_data(data, value_col)

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
        alpha=0.85
    )

    ax.set_title(title)
    ax.set_xlabel("Pbar ratio")
    ax.set_ylabel("R ratio")
    ax.set_zlabel("Cost")
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=ELEV, azim=AZIM)

    return surface


def plot_combined_surfaces(data, group_name):
    """
    Saves one combined 2x2 figure.
    This is used for each workload and for all workloads combined.
    """
    fig = plt.figure(figsize=(16, 10))

    for i, (value_col, title) in enumerate(PLOT_COLUMNS.items(), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")

        surface = plot_surface_on_axis(
            ax=ax,
            data=data,
            value_col=value_col,
            title=title
        )

        fig.colorbar(surface, ax=ax, shrink=0.5, label=value_col)

    fig.suptitle(
        f"Raw Cost Surfaces over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=16
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_raw_cost_surfaces_{AGGFUNC}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    print("Saved combined:", save_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_all_workloads_individual_high_quality(data):
    """
    Saves four separate high-quality figures ONLY for all workloads combined:
    - Power_Cost
    - Raw_Tracking_Cost
    - Raw_QoS_Cost
    - Raw_Total_Cost
    """
    group_name = "all_workloads_combined"

    for value_col, title in PLOT_COLUMNS.items():
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surface = plot_surface_on_axis(
            ax=ax,
            data=data,
            value_col=value_col,
            title=title
        )

        fig.colorbar(surface, ax=ax, shrink=0.65, label=value_col)

        fig.suptitle(
            f"{title} over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
            fontsize=16
        )

        fig.tight_layout()

        save_path = HQ_TOTAL_DIR / (
            f"{safe_filename(group_name)}_{safe_filename(value_col)}_high_quality_{AGGFUNC}.png"
        )

        fig.savefig(save_path, dpi=HIGH_QUALITY_DPI, bbox_inches="tight")

        print("Saved all-workloads high-quality individual figure:", save_path)

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# PLOT EACH WORKLOAD: COMBINED 4-PANEL ONLY
# ============================================================

workloads = sorted(df[WORKLOAD_COL].dropna().unique())

for workload in workloads:
    workload_df = df[df[WORKLOAD_COL] == workload].copy()
    group_name = f"workload_{workload}"

    summarize_group(group_name, workload_df)

    # Only combined 4-panel plot for each workload
    plot_combined_surfaces(workload_df, group_name)


# ============================================================
# PLOT ALL WORKLOADS COMBINED
# ============================================================

group_name = "all_workloads_combined"

summarize_group(group_name, df)

# Combined 4-panel all-workloads plot
plot_combined_surfaces(df, group_name)

# Four separate high-quality plots ONLY for all workloads combined
plot_all_workloads_individual_high_quality(df)


print()
print("Done.")
print("Combined figures saved in:", FIG_DIR.resolve())
print("All-workloads high-quality individual figures saved in:", HQ_TOTAL_DIR.resolve())