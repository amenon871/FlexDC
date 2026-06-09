import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("cost_distribution_figures")
FIG_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = FIG_DIR / "mean_absolute_component_magnitude_by_workload.png"
DPI = 400
SHOW_FIGURES = True

WORKLOAD_COL = "workload_name"

COMPONENT_COLUMNS = [
    "Power_Cost",
    "Raw_Tracking_Cost",
    "Raw_QoS_Cost",
]

DISPLAY_NAMES = {
    "Power_Cost": "Power",
    "Raw_Tracking_Cost": "Tracking",
    "Raw_QoS_Cost": "QoS",
}

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

required_cols = [WORKLOAD_COL] + COMPONENT_COLUMNS
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# COMPUTE BY WORKLOAD
# ============================================================

summary = (
    df.groupby(WORKLOAD_COL)[COMPONENT_COLUMNS]
    .apply(lambda x: x.abs().mean())
    .rename(columns=DISPLAY_NAMES)
)

summary = summary.sort_index()

print("Mean absolute component magnitude by workload:")
print(summary)

summary.to_csv(FIG_DIR / "mean_absolute_component_magnitude_by_workload.csv")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

summary.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Workload")
ax.set_ylabel("Mean absolute cost contribution")
ax.set_title(
    "Average Raw Cost Component Magnitudes by Workload",
    fontsize=15,
    fontweight="bold"
)

ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="Component")

plt.xticks(rotation=0)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")

print("Saved:", OUTPUT_PATH)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)