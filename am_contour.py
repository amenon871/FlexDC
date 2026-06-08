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

FIG_DIR = Path("figures_3d_softplus_costs")
FIG_DIR.mkdir(exist_ok=True)

P_COL = "Pbar_ratio"
R_COL = "R_ratio"
WORKLOAD_COL = "workload_name"

# SoftPlus/transformed cost components + total
PLOT_COLUMNS = {
    "Power_Cost": "Power Cost",
    "SoftPlus_Tracking_Cost": "SoftPlus Tracking Cost",
    "SoftPlus_QoS_Cost": "SoftPlus QoS Cost",
    "SoftPlus_Total_Cost": "SoftPlus Total Cost",
}

# Since we are ignoring server count, utilization, and weights,
# we need one value per P/R pair.
#
# "min" = best observed case at each P/R point
# "mean" = average behavior at each P/R point
# "max" = worst observed case at each P/R point
AGGFUNC = "min"

CMAP = "viridis"
ELEV = 25
AZIM = -135
DPI = 250

SHOW_FIGURES = True
MAKE_INDIVIDUAL_PLOTS = False


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [P_COL, R_COL, WORKLOAD_COL] + list(PLOT_COLUMNS.keys())
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Make sure you are using the CSV with recomputed SoftPlus columns: {CSV_PATH}"
    )

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
        f"SoftPlus Cost Surfaces over P/R Grid\n{group_name} | aggfunc={AGGFUNC}",
        fontsize=16
    )

    fig.tight_layout()

    save_path = FIG_DIR / f"{safe_filename(group_name)}_softplus_cost_surfaces_{AGGFUNC}.png"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

    print("Saved:", save_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


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

    save_path = FIG_DIR / f"{safe_filename(group_name)}_{safe_filename(value_col)}_{AGGFUNC}.png"
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
    plot_combined_surfaces(workload_df, group_name)

    if MAKE_INDIVIDUAL_PLOTS:
        for value_col, title in PLOT_COLUMNS.items():
            plot_individual_surface(workload_df, group_name, value_col, title)


# ============================================================
# PLOT ALL WORKLOADS COMBINED
# ============================================================

group_name = "all_workloads_combined"

summarize_group(group_name, df)
plot_combined_surfaces(df, group_name)

if MAKE_INDIVIDUAL_PLOTS:
    for value_col, title in PLOT_COLUMNS.items():
        plot_individual_surface(df, group_name, value_col, title)


print()
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())