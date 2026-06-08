import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


# ============================================================
# CONFIG
# ============================================================

MANIFEST_DIR = Path(".")
FIG_DIR = Path("manifest_execution_figures")
FIG_DIR.mkdir(exist_ok=True)

WORKER_MANIFEST_PATTERN = "dataset_worker_manifest*.csv"

DPI = 300
SHOW_FIGURES = True


# ============================================================
# HELPERS
# ============================================================

def extract_workload(row):
    """
    Extract workload name like W8, W9, W10 from workload_config or output_dir.
    """
    for col in ["workload_config", "output_dir"]:
        if col in row and pd.notna(row[col]):
            match = re.search(r"W(\d+)", str(row[col]))
            if match:
                return f"W{match.group(1)}"

    return "unknown"


def extract_server_count(row):
    """
    Extract server count from experiment_config or output_dir.
    """
    for col in ["experiment_config", "output_dir"]:
        if col in row and pd.notna(row[col]):
            match = re.search(r"exp_servers_(\d+)", str(row[col]))
            if match:
                return int(match.group(1))

    return np.nan


def save_and_show(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print("Saved:", path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def annotate_heatmap(ax, data):
    """
    Add values inside heatmap cells.
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9
                )


# ============================================================
# LOAD MANIFESTS
# ============================================================

manifest_files = sorted(MANIFEST_DIR.glob(WORKER_MANIFEST_PATTERN))

if not manifest_files:
    raise FileNotFoundError(
        f"No files found matching {WORKER_MANIFEST_PATTERN} in {MANIFEST_DIR.resolve()}"
    )

frames = []

for file_path in manifest_files:
    try:
        df_one = pd.read_csv(file_path)
        df_one["manifest_file"] = file_path.name
        frames.append(df_one)
        print(f"Loaded {file_path.name}: {len(df_one)} rows")
    except Exception as e:
        print(f"FAILED to load {file_path.name}: {e}")

df = pd.concat(frames, ignore_index=True)

print()
print("Combined manifest rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# CLEAN / DERIVE FIELDS
# ============================================================

required_cols = [
    "elapsed_minutes",
    "grid_search_rows",
    "status",
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in manifests: {missing}")

df["workload_name"] = df.apply(extract_workload, axis=1)
df["server_count"] = df.apply(extract_server_count, axis=1)

df["elapsed_minutes"] = pd.to_numeric(df["elapsed_minutes"], errors="coerce")
df["grid_search_rows"] = pd.to_numeric(df["grid_search_rows"], errors="coerce")

df["rows_per_minute"] = df["grid_search_rows"] / df["elapsed_minutes"]
df["seconds_per_row"] = (df["elapsed_minutes"] * 60) / df["grid_search_rows"]

# Sort categories nicely
workload_order = sorted(
    df["workload_name"].dropna().unique(),
    key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 999
)

server_order = sorted(df["server_count"].dropna().unique())

df["workload_name"] = pd.Categorical(
    df["workload_name"],
    categories=workload_order,
    ordered=True
)

print()
print("Status counts:")
print(df["status"].value_counts())

print()
print("Workloads:", workload_order)
print("Server counts:", server_order)

print()
print("Runtime summary:")
print(
    df[["elapsed_minutes", "rows_per_minute", "seconds_per_row"]]
    .describe()
    .T
)

# Save cleaned manifest summary
summary_csv = FIG_DIR / "manifest_execution_summary.csv"
df.to_csv(summary_csv, index=False)
print()
print("Saved cleaned summary:", summary_csv)


# ============================================================
# TABLE-LIKE SUMMARY BY WORKLOAD / SERVER COUNT
# ============================================================

runtime_pivot = df.pivot_table(
    index="workload_name",
    columns="server_count",
    values="elapsed_minutes",
    aggfunc="mean",
    observed=False
)

throughput_pivot = df.pivot_table(
    index="workload_name",
    columns="server_count",
    values="rows_per_minute",
    aggfunc="mean",
    observed=False
)

seconds_per_row_pivot = df.pivot_table(
    index="workload_name",
    columns="server_count",
    values="seconds_per_row",
    aggfunc="mean",
    observed=False
)

runtime_pivot = runtime_pivot.reindex(index=workload_order, columns=server_order)
throughput_pivot = throughput_pivot.reindex(index=workload_order, columns=server_order)
seconds_per_row_pivot = seconds_per_row_pivot.reindex(index=workload_order, columns=server_order)

runtime_pivot.to_csv(FIG_DIR / "runtime_minutes_by_workload_server.csv")
throughput_pivot.to_csv(FIG_DIR / "throughput_rows_per_minute_by_workload_server.csv")
seconds_per_row_pivot.to_csv(FIG_DIR / "seconds_per_row_by_workload_server.csv")


# ============================================================
# PLOT 1: HEATMAP OF RUNTIME MINUTES
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(runtime_pivot.values, aspect="auto")

ax.set_xticks(np.arange(len(runtime_pivot.columns)))
ax.set_yticks(np.arange(len(runtime_pivot.index)))

ax.set_xticklabels(runtime_pivot.columns.astype(int))
ax.set_yticklabels(runtime_pivot.index.astype(str))

ax.set_xlabel("Server count")
ax.set_ylabel("Workload")
ax.set_title("Dataset Generation Runtime by Workload and Server Count")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Elapsed minutes")

annotate_heatmap(ax, runtime_pivot)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "runtime_minutes_heatmap.png")


# ============================================================
# PLOT 2: HEATMAP OF THROUGHPUT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(throughput_pivot.values, aspect="auto")

ax.set_xticks(np.arange(len(throughput_pivot.columns)))
ax.set_yticks(np.arange(len(throughput_pivot.index)))

ax.set_xticklabels(throughput_pivot.columns.astype(int))
ax.set_yticklabels(throughput_pivot.index.astype(str))

ax.set_xlabel("Server count")
ax.set_ylabel("Workload")
ax.set_title("Dataset Generation Throughput by Workload and Server Count")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Rows per minute")

annotate_heatmap(ax, throughput_pivot)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "throughput_rows_per_minute_heatmap.png")


# ============================================================
# PLOT 3: RUNTIME VS SERVER COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for workload in workload_order:
    sub = (
        df[df["workload_name"] == workload]
        .sort_values("server_count")
    )

    ax.plot(
        sub["server_count"],
        sub["elapsed_minutes"],
        marker="o",
        label=str(workload)
    )

ax.set_xlabel("Server count")
ax.set_ylabel("Elapsed minutes")
ax.set_title("Runtime vs Server Count by Workload")
ax.legend(title="Workload")
ax.grid(True, alpha=0.3)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "runtime_vs_server_count_by_workload.png")


# ============================================================
# PLOT 4: THROUGHPUT VS SERVER COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for workload in workload_order:
    sub = (
        df[df["workload_name"] == workload]
        .sort_values("server_count")
    )

    ax.plot(
        sub["server_count"],
        sub["rows_per_minute"],
        marker="o",
        label=str(workload)
    )

ax.set_xlabel("Server count")
ax.set_ylabel("Rows per minute")
ax.set_title("Throughput vs Server Count by Workload")
ax.legend(title="Workload")
ax.grid(True, alpha=0.3)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "throughput_vs_server_count_by_workload.png")


# ============================================================
# PLOT 5: RUNTIME BY WORKLOAD, AVERAGED ACROSS SERVER COUNTS
# ============================================================

runtime_by_workload = (
    df.groupby("workload_name", observed=False)["elapsed_minutes"]
    .mean()
    .reindex(workload_order)
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(runtime_by_workload.index.astype(str), runtime_by_workload.values)

ax.set_xlabel("Workload")
ax.set_ylabel("Mean elapsed minutes")
ax.set_title("Mean Runtime by Workload")
ax.grid(True, axis="y", alpha=0.3)

for i, value in enumerate(runtime_by_workload.values):
    ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "mean_runtime_by_workload.png")


# ============================================================
# PLOT 6: RUNTIME BY SERVER COUNT, AVERAGED ACROSS WORKLOADS
# ============================================================

runtime_by_server = (
    df.groupby("server_count")["elapsed_minutes"]
    .mean()
    .reindex(server_order)
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(runtime_by_server.index.astype(str), runtime_by_server.values)

ax.set_xlabel("Server count")
ax.set_ylabel("Mean elapsed minutes")
ax.set_title("Mean Runtime by Server Count")
ax.grid(True, axis="y", alpha=0.3)

for i, value in enumerate(runtime_by_server.values):
    ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "mean_runtime_by_server_count.png")


# ============================================================
# PLOT 7: SLOWEST RUNS
# ============================================================

slowest = (
    df.sort_values("elapsed_minutes", ascending=False)
    .head(10)
    .copy()
)

slowest["label"] = (
    slowest["workload_name"].astype(str)
    + " / "
    + slowest["server_count"].astype(int).astype(str)
    + " servers"
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.barh(slowest["label"], slowest["elapsed_minutes"])
ax.invert_yaxis()

ax.set_xlabel("Elapsed minutes")
ax.set_ylabel("Run")
ax.set_title("Top 10 Slowest Dataset Generation Runs")
ax.grid(True, axis="x", alpha=0.3)

for i, value in enumerate(slowest["elapsed_minutes"]):
    ax.text(value, i, f" {value:.1f}", va="center", fontsize=9)

fig.tight_layout()
save_and_show(fig, FIG_DIR / "top_10_slowest_runs.png")


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("====================================")
print("Execution-time plotting complete")
print("Figures saved in:", FIG_DIR.resolve())
print("====================================")

print()
print("Total completed runs:", (df["status"] == "completed").sum())
print("Total rows generated:", int(df["grid_search_rows"].sum()))
print("Total serial runtime minutes:", df["elapsed_minutes"].sum())
print("Total serial runtime hours:", df["elapsed_minutes"].sum() / 60)
print("Mean runtime per worker:", df["elapsed_minutes"].mean())
print("Median runtime per worker:", df["elapsed_minutes"].median())
print("Slowest run:")
print(
    df.loc[
        df["elapsed_minutes"].idxmax(),
        ["workload_name", "server_count", "elapsed_minutes", "grid_search_rows", "rows_per_minute"]
    ]
)