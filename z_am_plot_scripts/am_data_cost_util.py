import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("cost_by_utilization_figures")
FIG_DIR.mkdir(exist_ok=True)

UTIL_COL = "utilization"
WORKLOAD_COL = "workload_name"

COST_COLUMNS = [
    "Raw_Total_Cost",
    "Power_Cost",
    "Raw_Tracking_Cost",
    "Raw_QoS_Cost",
]

DISPLAY_NAMES = {
    "Raw_Total_Cost": "Raw Total Cost",
    "Power_Cost": "Power Cost",
    "Raw_Tracking_Cost": "Raw Tracking Cost",
    "Raw_QoS_Cost": "Raw QoS Cost",
}

DPI = 400
SHOW_FIGURES = True

# Use median for main trends because tracking/total can have large tails.
MAIN_AGG = "median"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [UTIL_COL, WORKLOAD_COL] + COST_COLUMNS
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Utilizations:", sorted(df[UTIL_COL].dropna().unique()))
print("Workloads:", sorted(df[WORKLOAD_COL].dropna().unique()))


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


def sorted_workloads(data):
    return sorted(
        data[WORKLOAD_COL].dropna().unique(),
        key=lambda x: int(str(x).replace("W", "")) if str(x).replace("W", "").isdigit() else 999
    )


def save_and_show(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print("Saved:", path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def aggregate_by_utilization(data, cost_col, agg):
    if agg == "mean":
        return data.groupby(UTIL_COL)[cost_col].mean().sort_index()

    if agg == "median":
        return data.groupby(UTIL_COL)[cost_col].median().sort_index()

    if agg == "max":
        return data.groupby(UTIL_COL)[cost_col].max().sort_index()

    if agg == "min":
        return data.groupby(UTIL_COL)[cost_col].min().sort_index()

    raise ValueError(f"Unsupported aggregation: {agg}")


# ============================================================
# SUMMARY TABLES
# ============================================================

summary_by_utilization = (
    df.groupby(UTIL_COL)[COST_COLUMNS]
    .agg(["mean", "median", "std", "min", "max"])
)

summary_path = FIG_DIR / "cost_by_utilization_summary.csv"
summary_by_utilization.to_csv(summary_path)

print()
print("Saved utilization summary:", summary_path)

summary_by_workload_util = (
    df.groupby([WORKLOAD_COL, UTIL_COL])[COST_COLUMNS]
    .agg(["mean", "median"])
)

summary_workload_util_path = FIG_DIR / "cost_by_workload_utilization_summary.csv"
summary_by_workload_util.to_csv(summary_workload_util_path)

print("Saved workload/utilization summary:", summary_workload_util_path)


# ============================================================
# PLOT 1: RAW TOTAL COST VS UTILIZATION BY WORKLOAD
# ============================================================

workload_order = sorted_workloads(df)

fig, ax = plt.subplots(figsize=(11, 6))

for workload in workload_order:
    sub = df[df[WORKLOAD_COL] == workload]

    trend = (
        sub.groupby(UTIL_COL)["Raw_Total_Cost"]
        .agg(MAIN_AGG)
        .sort_index()
    )

    ax.plot(
        trend.index,
        trend.values,
        marker="o",
        label=workload
    )

ax.set_title(
    f"Raw Total Cost vs Utilization by Workload ({MAIN_AGG})",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Utilization")
ax.set_ylabel(f"{MAIN_AGG.capitalize()} raw total cost")
ax.grid(True, alpha=0.3)
ax.legend(title="Workload")

fig.tight_layout()
save_and_show(fig, FIG_DIR / f"raw_total_cost_vs_utilization_by_workload_{MAIN_AGG}.png")


# ============================================================
# PLOT 2: ALL RAW COST COMPONENTS VS UTILIZATION, ALL DATA COMBINED
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

for cost_col in COST_COLUMNS:
    trend = aggregate_by_utilization(
        data=df,
        cost_col=cost_col,
        agg=MAIN_AGG
    )

    ax.plot(
        trend.index,
        trend.values,
        marker="o",
        label=DISPLAY_NAMES[cost_col]
    )

ax.set_title(
    f"Raw Cost Components vs Utilization, All Workloads Combined ({MAIN_AGG})",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Utilization")
ax.set_ylabel(f"{MAIN_AGG.capitalize()} cost value")
ax.grid(True, alpha=0.3)
ax.legend(title="Cost component")

fig.tight_layout()
save_and_show(fig, FIG_DIR / f"all_cost_components_vs_utilization_{MAIN_AGG}.png")


# ============================================================
# PLOT 3: 2x2 COMPONENT TRENDS BY WORKLOAD
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, cost_col in zip(axes.ravel(), COST_COLUMNS):
    for workload in workload_order:
        sub = df[df[WORKLOAD_COL] == workload]

        trend = (
            sub.groupby(UTIL_COL)[cost_col]
            .agg(MAIN_AGG)
            .sort_index()
        )

        ax.plot(
            trend.index,
            trend.values,
            marker="o",
            label=workload
        )

    ax.set_title(DISPLAY_NAMES[cost_col])
    ax.set_xlabel("Utilization")
    ax.set_ylabel(f"{MAIN_AGG.capitalize()} cost")
    ax.grid(True, alpha=0.3)

axes[0, 0].legend(title="Workload", fontsize=8)

fig.suptitle(
    f"Raw Cost Components vs Utilization by Workload ({MAIN_AGG})",
    fontsize=16,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / f"cost_components_vs_utilization_by_workload_{MAIN_AGG}.png")


# ============================================================
# PLOT 4: RAW TOTAL COST BOXPLOT BY UTILIZATION
# ============================================================

util_order = sorted(df[UTIL_COL].dropna().unique())

data_for_boxplot = [
    df[df[UTIL_COL] == util]["Raw_Total_Cost"].dropna()
    for util in util_order
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    data_for_boxplot,
    labels=[str(u) for u in util_order],
    showfliers=False
)

ax.set_title("Raw Total Cost Distribution by Utilization", fontsize=15, fontweight="bold")
ax.set_xlabel("Utilization")
ax.set_ylabel("Raw total cost")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "raw_total_cost_by_utilization_boxplot_no_outliers.png")


# ============================================================
# PLOT 5: QOS VIOLATION RATIO VS UTILIZATION, IF AVAILABLE
# ============================================================

if "QoS_Violation_Ratio" in df.columns:
    fig, ax = plt.subplots(figsize=(11, 6))

    for workload in workload_order:
        sub = df[df[WORKLOAD_COL] == workload]

        trend = (
            sub.groupby(UTIL_COL)["QoS_Violation_Ratio"]
            .mean()
            .sort_index()
        )

        ax.plot(
            trend.index,
            trend.values,
            marker="o",
            label=workload
        )

    ax.set_title(
        "QoS Violation Ratio vs Utilization by Workload",
        fontsize=15,
        fontweight="bold"
    )
    ax.set_xlabel("Utilization")
    ax.set_ylabel("Mean QoS violation ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Workload")

    fig.tight_layout()
    save_and_show(fig, FIG_DIR / "qos_violation_ratio_vs_utilization_by_workload.png")


# ============================================================
# DONE
# ============================================================

print()
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())