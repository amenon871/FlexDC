import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "flexdc_all_data_fixed_with_softplus.csv"

FIG_DIR = Path("cost_distribution_figures")
FIG_DIR.mkdir(exist_ok=True)

DPI = 400
SHOW_FIGURES = True

COST_COLUMNS = [
    "Power_Cost",
    "Raw_Tracking_Cost",
    "Raw_QoS_Cost",
    "Raw_Total_Cost",
]

DISPLAY_NAMES = {
    "Power_Cost": "Power Cost",
    "Raw_Tracking_Cost": "Raw Tracking Cost",
    "Raw_QoS_Cost": "Raw QoS Cost",
    "Raw_Total_Cost": "Raw Total Cost",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

missing = [col for col in COST_COLUMNS if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Loaded:", CSV_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# PREPARE DATA
# ============================================================

plot_df = df[COST_COLUMNS].copy()
plot_df = plot_df.rename(columns=DISPLAY_NAMES)

summary = plot_df.describe().T[
    ["min", "25%", "50%", "75%", "max", "mean", "std"]
].round(4)

summary_path = FIG_DIR / "cost_component_distribution_summary.csv"
summary.to_csv(summary_path)

print()
print("Cost distribution summary:")
print(summary)
print()
print("Saved summary CSV:", summary_path)


# ============================================================
# PLOT 1: RAW BOX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plot_df.boxplot(
    ax=ax,
    showfliers=True,
    grid=False
)

ax.set_title("Raw Cost Component Distributions", fontsize=15, fontweight="bold")
ax.set_ylabel("Cost value")
ax.set_xlabel("Cost component")
ax.grid(True, axis="y", alpha=0.3)

plt.xticks(rotation=20, ha="right")
fig.tight_layout()

output_path = FIG_DIR / "cost_component_boxplot.png"
fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
print("Saved:", output_path)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# PLOT 2: BOX PLOT WITHOUT OUTLIERS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plot_df.boxplot(
    ax=ax,
    showfliers=False,
    grid=False
)

ax.set_title("Raw Cost Component Distributions Without Outliers", fontsize=15, fontweight="bold")
ax.set_ylabel("Cost value")
ax.set_xlabel("Cost component")
ax.grid(True, axis="y", alpha=0.3)

plt.xticks(rotation=20, ha="right")
fig.tight_layout()

output_path = FIG_DIR / "cost_component_boxplot_no_outliers.png"
fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
print("Saved:", output_path)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# PLOT 3: VIOLIN PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

data_for_violin = [
    plot_df[col].dropna().values
    for col in plot_df.columns
]

ax.violinplot(
    data_for_violin,
    showmeans=True,
    showmedians=True,
    showextrema=True
)

ax.set_xticks(range(1, len(plot_df.columns) + 1))
ax.set_xticklabels(plot_df.columns, rotation=20, ha="right")

ax.set_title("Raw Cost Component Distribution Shapes", fontsize=15, fontweight="bold")
ax.set_ylabel("Cost value")
ax.set_xlabel("Cost component")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()

output_path = FIG_DIR / "cost_component_violinplot.png"
fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
print("Saved:", output_path)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# PLOT 4: HISTOGRAMS, ONE PANEL PER COST COMPONENT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, col in zip(axes.ravel(), plot_df.columns):
    ax.hist(plot_df[col].dropna(), bins=50)
    ax.set_title(col)
    ax.set_xlabel("Cost value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

fig.suptitle("Raw Cost Component Histograms", fontsize=15, fontweight="bold")
fig.tight_layout()

output_path = FIG_DIR / "cost_component_histograms.png"
fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
print("Saved:", output_path)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# DONE
# ============================================================

print()
print("Done.")
print("Figures saved in:", FIG_DIR.resolve())