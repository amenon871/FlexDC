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

FIG_DIR = Path("fatih_tracking_debug_figures")
FIG_DIR.mkdir(exist_ok=True)

DPI = 400
SHOW_FIGURES = True

# Preferred controlled slice.
# The script will fall back to the nearest available value if needed.
PREFERRED_WORKLOAD = "W11"
PREFERRED_SERVER_COUNT = 1000
PREFERRED_UTILIZATION = 0.85
PREFERRED_WEIGHT_SAMPLE_ID = 0

P_COL = "Pbar_ratio"
R_COL = "R_ratio"
P_ACTUAL_COL = "P_actual_watts"
R_ACTUAL_COL = "R_actual_watts"
SERVER_COL = "server_count"
WORKLOAD_COL = "workload_name"
UTIL_COL = "utilization"
WEIGHT_SAMPLE_COL = "Weight_Sample_ID"

SIM_TRACKING_COL = "Simulator_Tracking_Cost"
RAW_TRACKING_COL = "Raw_Tracking_Cost"
TRACKING_ERROR_90TH_COL = "Tracking_Error_90th"

P_KW_COL = "Pbar_kw_per_server"
R_KW_COL = "R_kw_per_server"

CMAP = "viridis"
ELEV = 25
AZIM = -135


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# Handle old column name if present
if SIM_TRACKING_COL not in df.columns and "Simulator_Tracking_Error" in df.columns:
    df = df.rename(columns={"Simulator_Tracking_Error": SIM_TRACKING_COL})

required_cols = [
    P_COL,
    R_COL,
    P_ACTUAL_COL,
    R_ACTUAL_COL,
    SERVER_COL,
    WORKLOAD_COL,
    UTIL_COL,
    SIM_TRACKING_COL,
    RAW_TRACKING_COL,
    TRACKING_ERROR_90TH_COL,
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Derive physical CONDOR-style axes from stored actual watts
df[P_KW_COL] = df[P_ACTUAL_COL] / df[SERVER_COL] / 1000.0
df[R_KW_COL] = df[R_ACTUAL_COL] / df[SERVER_COL] / 1000.0

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("Derived physical ranges:")
print(f"{P_KW_COL}: {df[P_KW_COL].min():.6f} to {df[P_KW_COL].max():.6f} kW/server")
print(f"{R_KW_COL}: {df[R_KW_COL].min():.6f} to {df[R_KW_COL].max():.6f} kW/server")
print()
print("CONDOR intended ranges:")
print("Pbar: 0.2 to 0.6 kW/server")
print("R:    0.0 to 0.12 kW/server")


# ============================================================
# HELPERS
# ============================================================

def save_and_show(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print("Saved:", path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


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


def choose_nearest_value(data, col, preferred):
    values = sorted(data[col].dropna().unique())

    if not values:
        raise ValueError(f"No values available for {col}")

    if preferred in values:
        return preferred

    # Numeric nearest
    if np.issubdtype(np.array(values).dtype, np.number):
        return min(values, key=lambda x: abs(float(x) - float(preferred)))

    # String fallback
    return values[0]


def build_controlled_slice(data):
    """
    Pick one controlled slice:
    fixed workload, server count, utilization, and weight sample if available.
    """
    sub = data.copy()
    chosen = {}

    # Workload
    chosen_workload = choose_nearest_value(sub, WORKLOAD_COL, PREFERRED_WORKLOAD)
    sub = sub[sub[WORKLOAD_COL] == chosen_workload].copy()
    chosen[WORKLOAD_COL] = chosen_workload

    # Server count
    chosen_server = choose_nearest_value(sub, SERVER_COL, PREFERRED_SERVER_COUNT)
    sub = sub[sub[SERVER_COL] == chosen_server].copy()
    chosen[SERVER_COL] = chosen_server

    # Utilization
    chosen_util = choose_nearest_value(sub, UTIL_COL, PREFERRED_UTILIZATION)
    sub = sub[np.isclose(sub[UTIL_COL].astype(float), float(chosen_util))].copy()
    chosen[UTIL_COL] = chosen_util

    # Weight sample, if available
    if WEIGHT_SAMPLE_COL in sub.columns:
        chosen_weight = choose_nearest_value(sub, WEIGHT_SAMPLE_COL, PREFERRED_WEIGHT_SAMPLE_ID)
        sub = sub[sub[WEIGHT_SAMPLE_COL] == chosen_weight].copy()
        chosen[WEIGHT_SAMPLE_COL] = chosen_weight
    else:
        chosen[WEIGHT_SAMPLE_COL] = "not available"

    if sub.empty:
        raise ValueError("Controlled slice is empty after filtering.")

    return sub, chosen


def get_surface_data(data, value_col):
    """
    Build surface over physical P/R axes.
    Rounding avoids floating-point uniqueness noise.
    """
    temp = data.copy()
    temp["_P_kw_plot"] = temp[P_KW_COL].round(6)
    temp["_R_kw_plot"] = temp[R_KW_COL].round(6)

    pivot = temp.pivot_table(
        index="_R_kw_plot",
        columns="_P_kw_plot",
        values=value_col,
        aggfunc="median"
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    p_values = pivot.columns.to_numpy()
    r_values = pivot.index.to_numpy()

    X, Y = np.meshgrid(p_values, r_values)
    Z = pivot.to_numpy()

    return X, Y, Z


def plot_surface_on_axis(ax, data, value_col, title, zlabel):
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
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.set_zlabel(zlabel)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=ELEV, azim=AZIM)

    return surface


def select_representative_p_values(data, max_lines=4):
    """
    Select a few P levels so line plots do not become unreadable.
    """
    unique_p = np.array(sorted(data[P_KW_COL].round(6).dropna().unique()))

    if len(unique_p) <= max_lines:
        return unique_p

    idx = np.linspace(0, len(unique_p) - 1, max_lines).round().astype(int)
    idx = sorted(set(idx.tolist()))

    return unique_p[idx]


# ============================================================
# 1. ALL-DATA GROUPED COMPARISON BY R_RATIO
# ============================================================
# This directly shows that the two tracking quantities move differently.

summary_by_r_ratio = (
    df.groupby(R_COL)[[SIM_TRACKING_COL, RAW_TRACKING_COL, TRACKING_ERROR_90TH_COL]]
    .median()
    .sort_index()
)

summary_by_r_ratio.to_csv(FIG_DIR / "tracking_metrics_by_r_ratio_median.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    summary_by_r_ratio.index,
    summary_by_r_ratio[SIM_TRACKING_COL],
    marker="o"
)
axes[0].set_title("Simulator Tracking Cost vs R Ratio")
axes[0].set_xlabel("R ratio")
axes[0].set_ylabel("Median simulator tracking cost")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    summary_by_r_ratio.index,
    summary_by_r_ratio[RAW_TRACKING_COL],
    marker="o"
)
axes[1].set_title("Raw Tracking Error Residual vs R Ratio")
axes[1].set_xlabel("R ratio")
axes[1].set_ylabel("Median Raw_Tracking_Cost")
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    "Tracking Metric Difference: Simulator Cost vs Normalized Error Residual",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "tracking_metric_difference_by_r_ratio.png")


# ============================================================
# 2. ALL-DATA GROUPED COMPARISON BY PHYSICAL R KW/SERVER
# ============================================================
# Current dataset did not truly sweep fixed kW/server values, so we bin.

df_binned = df.copy()
df_binned["R_kw_bin"] = pd.cut(df_binned[R_KW_COL], bins=10)

summary_by_r_kw_bin = (
    df_binned.groupby("R_kw_bin", observed=False)
    .agg(
        R_kw_mid=(R_KW_COL, "median"),
        Simulator_Tracking_Cost=(SIM_TRACKING_COL, "median"),
        Raw_Tracking_Cost=(RAW_TRACKING_COL, "median"),
        Tracking_Error_90th=(TRACKING_ERROR_90TH_COL, "median"),
        count=(SIM_TRACKING_COL, "size"),
    )
    .dropna()
    .sort_values("R_kw_mid")
)

summary_by_r_kw_bin.to_csv(FIG_DIR / "tracking_metrics_by_r_kw_per_server_binned.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    summary_by_r_kw_bin["R_kw_mid"],
    summary_by_r_kw_bin["Simulator_Tracking_Cost"],
    marker="o"
)
axes[0].set_title("Simulator Tracking Cost vs R kW/server")
axes[0].set_xlabel("R kW/server")
axes[0].set_ylabel("Median simulator tracking cost")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    summary_by_r_kw_bin["R_kw_mid"],
    summary_by_r_kw_bin["Raw_Tracking_Cost"],
    marker="o"
)
axes[1].set_title("Raw Tracking Error Residual vs R kW/server")
axes[1].set_xlabel("R kW/server")
axes[1].set_ylabel("Median Raw_Tracking_Cost")
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    "Tracking Metrics vs Physical R Bins",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "tracking_metric_difference_by_r_kw_per_server.png")


# ============================================================
# 3. CONTROLLED SLICE LINE PLOT
# ============================================================
# This is the most useful plot to show Fatih.
# It fixes workload/server/utilization/weight sample and varies R.

slice_df, chosen = build_controlled_slice(df)

print()
print("Controlled slice selected:")
for key, value in chosen.items():
    print(f"{key}: {value}")
print("Rows in controlled slice:", len(slice_df))
print("Unique P values:", slice_df[P_COL].nunique())
print("Unique R values:", slice_df[R_COL].nunique())
print(f"{P_KW_COL}: {slice_df[P_KW_COL].min():.6f} to {slice_df[P_KW_COL].max():.6f}")
print(f"{R_KW_COL}: {slice_df[R_KW_COL].min():.6f} to {slice_df[R_KW_COL].max():.6f}")

slice_label = (
    f"{chosen[WORKLOAD_COL]}, "
    f"{int(chosen[SERVER_COL])} servers, "
    f"util={float(chosen[UTIL_COL]):.2f}, "
    f"weight={chosen[WEIGHT_SAMPLE_COL]}"
)

selected_p_values = select_representative_p_values(slice_df, max_lines=4)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

for p_value in selected_p_values:
    sub = slice_df[np.isclose(slice_df[P_KW_COL].round(6), p_value)].copy()

    line = (
        sub.groupby(R_KW_COL)[SIM_TRACKING_COL]
        .median()
        .sort_index()
    )

    axes[0].plot(
        line.index,
        line.values,
        marker="o",
        label=f"P={p_value:.3f}"
    )

axes[0].set_title("Simulator Tracking Cost")
axes[0].set_xlabel("R kW/server")
axes[0].set_ylabel("Simulator tracking cost")
axes[0].grid(True, alpha=0.3)
axes[0].legend(title="Pbar kW/server")

for p_value in selected_p_values:
    sub = slice_df[np.isclose(slice_df[P_KW_COL].round(6), p_value)].copy()

    line = (
        sub.groupby(R_KW_COL)[RAW_TRACKING_COL]
        .median()
        .sort_index()
    )

    axes[1].plot(
        line.index,
        line.values,
        marker="o",
        label=f"P={p_value:.3f}"
    )

axes[1].set_title("Raw Tracking Error Residual")
axes[1].set_xlabel("R kW/server")
axes[1].set_ylabel("Raw_Tracking_Cost = Tracking_Error_90th - 0.3")
axes[1].grid(True, alpha=0.3)
axes[1].legend(title="Pbar kW/server")

fig.suptitle(
    f"Controlled Slice: Tracking Metrics vs R\n{slice_label}",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "controlled_slice_tracking_metrics_vs_r_kw_per_server.png")


# ============================================================
# 4. CONTROLLED SLICE 3D SURFACES
# ============================================================
# This mirrors the old P/R surface, but uses physical axes and compares
# the simulator tracking cost to the normalized residual.

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf1 = plot_surface_on_axis(
    ax=ax1,
    data=slice_df,
    value_col=SIM_TRACKING_COL,
    title="Simulator Tracking Cost",
    zlabel="Simulator tracking cost"
)
fig.colorbar(surf1, ax=ax1, shrink=0.6, label=SIM_TRACKING_COL)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf2 = plot_surface_on_axis(
    ax=ax2,
    data=slice_df,
    value_col=RAW_TRACKING_COL,
    title="Raw Tracking Error Residual",
    zlabel="Raw_Tracking_Cost"
)
fig.colorbar(surf2, ax=ax2, shrink=0.6, label=RAW_TRACKING_COL)

fig.suptitle(
    f"Controlled Slice P/R Tracking Surfaces\n{slice_label}",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "controlled_slice_tracking_surface_comparison.png")


# ============================================================
# 5. PRINT FINAL INTERPRETATION SUMMARY
# ============================================================

print()
print("====================================")
print("One-off Fatih tracking diagnostic complete")
print("Figures saved in:", FIG_DIR.resolve())
print("====================================")
print()
print("Recommended figure to show first:")
print("  controlled_slice_tracking_metrics_vs_r_kw_per_server.png")
print()
print("Interpretation:")
print("  Simulator_Tracking_Cost is the simulator-returned tracking-cost term.")
print("  Raw_Tracking_Cost is Tracking_Error_90th - 0.3, a normalized tracking-error residual.")
print("  Because Tracking_Error_90th is normalized by R, it can decrease as R increases.")
print("  Fatih's expected behavior should be checked against Simulator_Tracking_Cost.")
print()
print("Important note:")
print("  This is only a diagnostic for the existing dataset.")
print("  Future data generation should sweep Pbar and R in kW/server directly.")