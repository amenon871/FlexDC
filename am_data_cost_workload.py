import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("cost_by_workload_figures")
FIG_DIR.mkdir(exist_ok=True)

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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [WORKLOAD_COL] + COST_COLUMNS
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
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


# ============================================================
# SUMMARY TABLE BY WORKLOAD
# ============================================================

summary = (
    df.groupby(WORKLOAD_COL)[COST_COLUMNS]
    .agg(["mean", "median", "std", "min", "max"])
)

summary_path = FIG_DIR / "cost_by_workload_summary.csv"
summary.to_csv(summary_path)

print()
print("Saved workload summary:", summary_path)


# ============================================================
# PLOT 1: RAW TOTAL COST BY WORKLOAD
# ============================================================

workload_order = sorted_workloads(df)

data_for_boxplot = [
    df[df[WORKLOAD_COL] == workload]["Raw_Total_Cost"].dropna()
    for workload in workload_order
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    data_for_boxplot,
    labels=workload_order,
    showfliers=False
)

ax.set_title("Raw Total Cost by Workload", fontsize=15, fontweight="bold")
ax.set_xlabel("Workload")
ax.set_ylabel("Raw total cost")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "raw_total_cost_by_workload_boxplot.png")


# ============================================================
# PLOT 2: ALL COST COMPONENTS BY WORKLOAD, 2x2
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, cost_col in zip(axes.ravel(), COST_COLUMNS):
    data_for_boxplot = [
        df[df[WORKLOAD_COL] == workload][cost_col].dropna()
        for workload in workload_order
    ]

    ax.boxplot(
        data_for_boxplot,
        labels=workload_order,
        showfliers=False
    )

    ax.set_title(DISPLAY_NAMES[cost_col])
    ax.set_xlabel("Workload")
    ax.set_ylabel("Cost value")
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "Raw Cost Components by Workload",
    fontsize=16,
    fontweight="bold"
)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "all_cost_components_by_workload_boxplots.png")


# ============================================================
# PLOT 3: MEAN RAW COST COMPONENTS BY WORKLOAD
# ============================================================

mean_summary = (
    df.groupby(WORKLOAD_COL)[COST_COLUMNS]
    .mean()
    .reindex(workload_order)
    .rename(columns=DISPLAY_NAMES)
)

fig, ax = plt.subplots(figsize=(12, 6))

mean_summary.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Mean Raw Cost Components by Workload", fontsize=15, fontweight="bold")
ax.set_xlabel("Workload")
ax.set_ylabel("Mean cost value")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="Component")

plt.xticks(rotation=0)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "mean_cost_components_by_workload.png")


# ============================================================
# PLOT 4: MEDIAN RAW COST COMPONENTS BY WORKLOAD
# ============================================================

median_summary = (
    df.groupby(WORKLOAD_COL)[COST_COLUMNS]
    .median()
    .reindex(workload_order)
    .rename(columns=DISPLAY_NAMES)
)

fig, ax = plt.subplots(figsize=(12, 6))

median_summary.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Median Raw Cost Components by Workload", fontsize=15, fontweight="bold")
ax.set_xlabel("Workload")
ax.set_ylabel("Median cost value")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="Component")

plt.xticks(rotation=0)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "median_cost_components_by_workload.png")


# ============================================================
# DONE
# ============================================================

print()
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())