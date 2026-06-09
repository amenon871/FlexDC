import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("summary_figures")
FIG_DIR.mkdir(exist_ok=True)

OUTPUT_PNG = FIG_DIR / "cost_summary_table.png"
OUTPUT_CSV = FIG_DIR / "cost_summary_table.csv"

DPI = 400

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

cost_cols = [
    "Power_Cost",
    "Raw_Tracking_Cost",
    "Raw_QoS_Cost",
    "Raw_Total_Cost",
]

missing = [col for col in cost_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# COMPUTE SUMMARY TABLE
# ============================================================

summary = df[cost_cols].describe().T[
    ["mean", "std", "min", "max"]
]

summary = summary.rename(
    index={
        "Power_Cost": "Power Cost",
        "Raw_Tracking_Cost": "Raw Tracking Cost",
        "Raw_QoS_Cost": "Raw QoS Cost",
        "Raw_Total_Cost": "Raw Total Cost",
    },
    columns={
        "mean": "Mean",
        "std": "Std. Dev.",
        "min": "Min",
        "max": "Max",
    }
)

# Round for presentation
summary_display = summary.round(3)

# Save CSV too, in case you want to paste into PowerPoint manually
summary_display.to_csv(OUTPUT_CSV)

print("Cost summary:")
print(summary_display)
print()
print("Saved CSV:", OUTPUT_CSV)

# ============================================================
# MAKE TABLE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

table = ax.table(
    cellText=summary_display.values,
    rowLabels=summary_display.index,
    colLabels=summary_display.columns,
    cellLoc="center",
    rowLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.6)

# Bold header row and row labels
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
    if col == -1:
        cell.set_text_props(weight="bold")

ax.set_title(
    "Raw Cost Component Summary",
    fontsize=15,
    fontweight="bold",
    pad=20
)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")

print("Saved table figure:", OUTPUT_PNG)

plt.show()