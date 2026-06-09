import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("additional_analysis_figures")
FIG_DIR.mkdir(exist_ok=True)

DPI = 400
SHOW_FIGURES = True

WORKLOAD_COL = "workload_name"
SERVER_COL = "server_count"
UTIL_COL = "utilization"

RAW_TOTAL_COL = "Raw_Total_Cost"
POWER_COL = "Power_Cost"
TRACKING_COL = "Raw_Tracking_Cost"
QOS_COL = "Raw_QoS_Cost"
QOS_VIOLATION_COL = "QoS_Violation_Ratio"

P_ACTUAL_COL = "P_actual_watts"
R_ACTUAL_COL = "R_actual_watts"

# To keep scatter plots readable, sample points if dataset is large.
SCATTER_SAMPLE_SIZE = 12000
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [
    WORKLOAD_COL,
    SERVER_COL,
    UTIL_COL,
    RAW_TOTAL_COL,
    POWER_COL,
    TRACKING_COL,
    QOS_COL,
    QOS_VIOLATION_COL,
    P_ACTUAL_COL,
    R_ACTUAL_COL,
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Workloads:", sorted(df[WORKLOAD_COL].dropna().unique()))
print("Server counts:", sorted(df[SERVER_COL].dropna().unique()))
print("Utilizations:", sorted(df[UTIL_COL].dropna().unique()))


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


def sorted_workloads(data):
    return sorted(
        data[WORKLOAD_COL].dropna().unique(),
        key=lambda x: int(str(x).replace("W", "")) if str(x).replace("W", "").isdigit() else 999
    )


def get_scatter_df(data):
    if len(data) > SCATTER_SAMPLE_SIZE:
        return data.sample(SCATTER_SAMPLE_SIZE, random_state=RANDOM_SEED)
    return data.copy()


workload_order = sorted_workloads(df)
server_order = sorted(df[SERVER_COL].dropna().unique())
util_order = sorted(df[UTIL_COL].dropna().unique())


# ============================================================
# 5. COST BY SERVER COUNT
# ============================================================

# ------------------------------------------------------------
# 5A. Raw total cost distribution by server count
# ------------------------------------------------------------

data_for_boxplot = [
    df[df[SERVER_COL] == server][RAW_TOTAL_COL].dropna()
    for server in server_order
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    data_for_boxplot,
    labels=[str(int(s)) for s in server_order],
    showfliers=False
)

ax.set_title("Raw Total Cost by Server Count", fontsize=15, fontweight="bold")
ax.set_xlabel("Server count")
ax.set_ylabel("Raw total cost")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "raw_total_cost_by_server_count_boxplot.png")


# ------------------------------------------------------------
# 5B. Median raw total cost vs server count by workload
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

for workload in workload_order:
    sub = df[df[WORKLOAD_COL] == workload]

    trend = (
        sub.groupby(SERVER_COL)[RAW_TOTAL_COL]
        .median()
        .reindex(server_order)
    )

    ax.plot(
        trend.index,
        trend.values,
        marker="o",
        label=workload
    )

ax.set_title("Median Raw Total Cost vs Server Count by Workload", fontsize=15, fontweight="bold")
ax.set_xlabel("Server count")
ax.set_ylabel("Median raw total cost")
ax.grid(True, alpha=0.3)
ax.legend(title="Workload")

fig.tight_layout()
save_and_show(fig, FIG_DIR / "median_raw_total_cost_vs_server_count_by_workload.png")


# ------------------------------------------------------------
# 5C. P_actual_watts and R_actual_watts vs server count
# ------------------------------------------------------------

actual_summary = (
    df.groupby(SERVER_COL)[[P_ACTUAL_COL, R_ACTUAL_COL]]
    .mean()
    .reindex(server_order)
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    actual_summary.index,
    actual_summary[P_ACTUAL_COL],
    marker="o",
    label="Mean P actual watts"
)

ax.plot(
    actual_summary.index,
    actual_summary[R_ACTUAL_COL],
    marker="o",
    label="Mean R actual watts"
)

ax.set_title("Actual P and R Watt Values vs Server Count", fontsize=15, fontweight="bold")
ax.set_xlabel("Server count")
ax.set_ylabel("Mean watts")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
save_and_show(fig, FIG_DIR / "p_actual_and_r_actual_vs_server_count.png")


# ------------------------------------------------------------
# 5D. P_actual_watts and R_actual_watts by server count, separate panels
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    actual_summary.index,
    actual_summary[P_ACTUAL_COL],
    marker="o"
)
axes[0].set_title("Mean P Actual Watts")
axes[0].set_xlabel("Server count")
axes[0].set_ylabel("Watts")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    actual_summary.index,
    actual_summary[R_ACTUAL_COL],
    marker="o"
)
axes[1].set_title("Mean R Actual Watts")
axes[1].set_xlabel("Server count")
axes[1].set_ylabel("Watts")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Actual Watt Scaling with Server Count", fontsize=15, fontweight="bold")
fig.tight_layout()

save_and_show(fig, FIG_DIR / "p_r_actual_watts_scaling_panels.png")


# ============================================================
# 6. TRACKING VS QOS SCATTER
# ============================================================

scatter_df = get_scatter_df(df)

# ------------------------------------------------------------
# 6A. Raw tracking cost vs raw QoS cost, colored by utilization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    scatter_df[TRACKING_COL],
    scatter_df[QOS_COL],
    c=scatter_df[UTIL_COL],
    alpha=0.35,
    s=12
)

ax.set_title("Raw Tracking Cost vs Raw QoS Cost", fontsize=15, fontweight="bold")
ax.set_xlabel("Raw tracking cost")
ax.set_ylabel("Raw QoS cost")
ax.grid(True, alpha=0.3)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Utilization")

fig.tight_layout()
save_and_show(fig, FIG_DIR / "tracking_vs_qos_scatter_by_utilization.png")


# ------------------------------------------------------------
# 6B. Raw tracking cost vs QoS violation ratio, colored by utilization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    scatter_df[TRACKING_COL],
    scatter_df[QOS_VIOLATION_COL],
    c=scatter_df[UTIL_COL],
    alpha=0.35,
    s=12
)

ax.set_title("Raw Tracking Cost vs QoS Violation Ratio", fontsize=15, fontweight="bold")
ax.set_xlabel("Raw tracking cost")
ax.set_ylabel("QoS violation ratio")
ax.grid(True, alpha=0.3)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Utilization")

fig.tight_layout()
save_and_show(fig, FIG_DIR / "tracking_vs_qos_violation_scatter_by_utilization.png")


# ------------------------------------------------------------
# 6C. Correlation table for tracking and QoS
# ------------------------------------------------------------

corr_cols = [
    TRACKING_COL,
    QOS_COL,
    QOS_VIOLATION_COL,
    RAW_TOTAL_COL,
]

corr = df[corr_cols].corr()
corr_path = FIG_DIR / "tracking_qos_correlation_table.csv"
corr.to_csv(corr_path)

print()
print("Tracking/QoS correlation table:")
print(corr)
print("Saved:", corr_path)


# ============================================================
# 7. RAW TOTAL VS EACH COMPONENT
# ============================================================

component_cols = [
    POWER_COL,
    TRACKING_COL,
    QOS_COL,
]

component_names = {
    POWER_COL: "Power Cost",
    TRACKING_COL: "Raw Tracking Cost",
    QOS_COL: "Raw QoS Cost",
}

scatter_df = get_scatter_df(df)

fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

last_scatter = None

for ax, component_col in zip(axes, component_cols):
    last_scatter = ax.scatter(
        scatter_df[component_col],
        scatter_df[RAW_TOTAL_COL],
        c=scatter_df[UTIL_COL],
        alpha=0.35,
        s=12
    )

    ax.set_title(f"{component_names[component_col]} vs Raw Total Cost")
    ax.set_xlabel(component_names[component_col])
    ax.set_ylabel("Raw total cost")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Raw Total Cost vs Individual Components",
    fontsize=16,
    fontweight="bold"
)

# Leave room on the right for a dedicated colorbar
fig.subplots_adjust(right=0.88, wspace=0.28, top=0.86)

# Dedicated colorbar axis on the FAR RIGHT
cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.62])
cbar = fig.colorbar(last_scatter, cax=cbar_ax)
cbar.set_label("Utilization")

save_and_show(fig, FIG_DIR / "raw_total_cost_vs_components_scatter.png")

# ------------------------------------------------------------
# 7B. Correlation with raw total cost
# ------------------------------------------------------------

total_corr = (
    df[[POWER_COL, TRACKING_COL, QOS_COL, RAW_TOTAL_COL]]
    .corr()[RAW_TOTAL_COL]
    .drop(RAW_TOTAL_COL)
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    [component_names.get(idx, idx) for idx in total_corr.index],
    total_corr.values
)

ax.set_title("Correlation with Raw Total Cost", fontsize=15, fontweight="bold")
ax.set_xlabel("Pearson correlation")
ax.set_ylabel("Component")
ax.grid(True, axis="x", alpha=0.3)

for i, value in enumerate(total_corr.values):
    ax.text(
        value,
        i,
        f" {value:.3f}",
        va="center",
        fontsize=10
    )

fig.tight_layout()
save_and_show(fig, FIG_DIR / "component_correlation_with_raw_total_cost.png")


# ============================================================
# 9. QOS VIOLATION HEATMAP / BAR PLOTS
# ============================================================

# ------------------------------------------------------------
# 9A. QoS violation ratio heatmap: workload x utilization
# ------------------------------------------------------------

qos_heatmap = (
    df.pivot_table(
        index=WORKLOAD_COL,
        columns=UTIL_COL,
        values=QOS_VIOLATION_COL,
        aggfunc="mean"
    )
    .reindex(index=workload_order, columns=util_order)
)

qos_heatmap_path = FIG_DIR / "qos_violation_ratio_by_workload_utilization.csv"
qos_heatmap.to_csv(qos_heatmap_path)

fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(qos_heatmap.values, aspect="auto")

ax.set_xticks(np.arange(len(qos_heatmap.columns)))
ax.set_yticks(np.arange(len(qos_heatmap.index)))

ax.set_xticklabels([str(x) for x in qos_heatmap.columns])
ax.set_yticklabels(qos_heatmap.index)

ax.set_xlabel("Utilization")
ax.set_ylabel("Workload")
ax.set_title("Mean QoS Violation Ratio by Workload and Utilization", fontsize=15, fontweight="bold")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean QoS violation ratio")

for i in range(qos_heatmap.shape[0]):
    for j in range(qos_heatmap.shape[1]):
        value = qos_heatmap.iloc[i, j]
        if pd.notna(value):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

fig.tight_layout()
save_and_show(fig, FIG_DIR / "qos_violation_ratio_heatmap_workload_utilization.png")


# ------------------------------------------------------------
# 9B. QoS violation ratio by workload
# ------------------------------------------------------------

qos_by_workload = (
    df.groupby(WORKLOAD_COL)[QOS_VIOLATION_COL]
    .mean()
    .reindex(workload_order)
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    qos_by_workload.index.astype(str),
    qos_by_workload.values
)

ax.set_title("Mean QoS Violation Ratio by Workload", fontsize=15, fontweight="bold")
ax.set_xlabel("Workload")
ax.set_ylabel("Mean QoS violation ratio")
ax.grid(True, axis="y", alpha=0.3)

for i, value in enumerate(qos_by_workload.values):
    ax.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

fig.tight_layout()
save_and_show(fig, FIG_DIR / "mean_qos_violation_ratio_by_workload.png")


# ------------------------------------------------------------
# 9C. QoS violation ratio by server count
# ------------------------------------------------------------

qos_by_server = (
    df.groupby(SERVER_COL)[QOS_VIOLATION_COL]
    .mean()
    .reindex(server_order)
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    [str(int(s)) for s in qos_by_server.index],
    qos_by_server.values
)

ax.set_title("Mean QoS Violation Ratio by Server Count", fontsize=15, fontweight="bold")
ax.set_xlabel("Server count")
ax.set_ylabel("Mean QoS violation ratio")
ax.grid(True, axis="y", alpha=0.3)

for i, value in enumerate(qos_by_server.values):
    ax.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

fig.tight_layout()
save_and_show(fig, FIG_DIR / "mean_qos_violation_ratio_by_server_count.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("====================================")
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())
print("====================================")