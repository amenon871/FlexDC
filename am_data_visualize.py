"""
Visualize the FlexDC rangefind64 combined raw/diagnostic outputs.

Put this file in the FlexDC repo root and run:

    python -u visualize_rangefind64_outputs.py

Default expected inputs, relative to repo root:

    combined_outputs/rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_raw.csv
    combined_outputs/rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_diagnostics.csv

The script creates a focused set of plots for Fatih/Kerim:
- validation report
- physical P/R grid coverage
- 3D surfaces + heatmaps for simulator RSR total, power, tracking, QoS observables
- line plots that isolate P and R effects
- diagnostic plots for normalized tracking error, if diagnostics CSV is available

Metric naming reminder:
- Raw CSV has raw simulator RSR outputs and QoS observables.
- Diagnostics CSV has normalized tracking error, residuals, and SoftPlus-style diagnostic terms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_RAW_CSV = Path("combined_outputs") / "rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_raw.csv"
DEFAULT_DIAGNOSTICS_CSV = Path("combined_outputs") / "rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_diagnostics.csv"
DEFAULT_OUT_DIR = Path("figures_rangefind64")

EXPECTED_WORKLOADS = ["W9", "W10", "W_train_2app", "W_inference_2app"]
EXPECTED_SERVER_COUNT = 64
EXPECTED_POLICY = "AQA"

P_COL = "Pbar_kw_per_server"
R_COL = "R_kw_per_server"
WORKLOAD_COL = "Source_Workload_Name"
UTIL_COL = "utilization"
SERVER_COL = "server_count"

MERGE_KEYS = [
    "Source_Output_Dir",
    "Source_Workload_Name",
    "Iteration",
    "Weight_Sample_ID",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "P_actual_watts",
    "R_actual_watts",
    "Pbar_ratio",
    "R_ratio",
    "server_count",
    "utilization",
]

KEY_RAW_METRICS = {
    "Simulator_RSR_Total_Cost": "Simulator RSR total cost",
    "Simulator_Power_Cost": "Simulator power cost",
    "Simulator_Tracking_Cost": "Simulator tracking cost",
    "QoS_Delay_Probability_Sum": "QoS delay probability sum",
    "QoS_Violation_Ratio": "QoS violation ratio",
}

KEY_DIAGNOSTIC_METRICS = {
    "Normalized_Tracking_Error_90th": "Normalized tracking error 90th percentile",
    "Normalized_Tracking_Error_90th_Residual": "Normalized tracking error residual",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "Diagnostic FlexDC SoftPlus QoS cost",
}


# ============================================================
# Helpers
# ============================================================

def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(":", "")
        .replace("%", "pct")
        .lower()
    )


def rounded_unique(values: Iterable[float], decimals: int = 10) -> list[float]:
    out = sorted(pd.Series(values).dropna().astype(float).round(decimals).unique().tolist())
    return [float(x) for x in out]


def parse_json_array(value, fallback=None):
    if fallback is None:
        fallback = []
    if pd.isna(value):
        return fallback
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else fallback
    except Exception:
        return fallback


def ensure_file(path: Path, label: str) -> Path:
    if path.exists():
        return path
    root_fallback = Path(path.name)
    if root_fallback.exists():
        print(f"WARNING: {label} not found at {path}; using repo-root fallback {root_fallback}")
        return root_fallback
    raise FileNotFoundError(f"{label} not found: {path}")


def save_fig(fig: plt.Figure, out_path: Path, dpi: int = 180) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_ylim(values: pd.Series, pad_fraction: float = 0.08):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmin, vmax):
        return (vmin - 1, vmax + 1)
    pad = (vmax - vmin) * pad_fraction
    return (vmin - pad, vmax + pad)


# ============================================================
# Load + validation
# ============================================================

def load_data(raw_csv: Path, diagnostics_csv: Path | None):
    raw = pd.read_csv(raw_csv)
    diagnostics = None
    merged = raw.copy()

    if diagnostics_csv is not None and diagnostics_csv.exists():
        diagnostics = pd.read_csv(diagnostics_csv)
        missing_keys = [k for k in MERGE_KEYS if k not in raw.columns or k not in diagnostics.columns]
        if missing_keys:
            raise ValueError(f"Cannot merge raw and diagnostics; missing keys: {missing_keys}")
        merged = raw.merge(
            diagnostics,
            on=MERGE_KEYS,
            how="left",
            validate="one_to_one",
            suffixes=("", "_diag"),
        )

    return raw, diagnostics, merged


def validate_data(raw: pd.DataFrame, diagnostics: pd.DataFrame | None, merged: pd.DataFrame) -> str:
    messages = []

    raw_required = [
        "Source_Output_Dir", "Source_Workload_Name", "Iteration", "Weight_Sample_ID",
        "Policy_Name", "Node_Count_Control", "Pbar_kw_per_server", "R_kw_per_server",
        "P_actual_watts", "R_actual_watts", "Pbar_ratio", "R_ratio",
        "Pbar_denominator_watts", "R_denominator_watts", "server_count", "utilization",
        "workload_mix_size", "workload_mix", "Weights", "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost", "Simulator_Tracking_Cost", "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio", "QoS_Delay_Probabilities",
    ]
    missing = [c for c in raw_required if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw CSV missing required columns: {missing}")

    messages.append(f"Raw rows: {len(raw)}")
    messages.append(f"Raw columns: {len(raw.columns)}")
    messages.append(f"Workloads: {sorted(raw[WORKLOAD_COL].unique().tolist())}")
    messages.append(f"Server counts: {sorted(raw[SERVER_COL].unique().tolist())}")
    messages.append(f"Policies: {sorted(raw['Policy_Name'].unique().tolist())}")
    messages.append(f"Utilizations: {rounded_unique(raw[UTIL_COL])}")
    messages.append(f"Pbar kW/server values: {rounded_unique(raw[P_COL])}")
    messages.append(f"R kW/server values: {rounded_unique(raw[R_COL])}")

    workloads = set(raw[WORKLOAD_COL].dropna().unique().tolist())
    if workloads != set(EXPECTED_WORKLOADS):
        raise ValueError(f"Unexpected workloads: expected {EXPECTED_WORKLOADS}, got {sorted(workloads)}")
    if sorted(raw[SERVER_COL].dropna().unique().tolist()) != [EXPECTED_SERVER_COUNT]:
        raise ValueError(f"Expected only server_count={EXPECTED_SERVER_COUNT}")
    if sorted(raw["Policy_Name"].dropna().unique().tolist()) != [EXPECTED_POLICY]:
        raise ValueError(f"Expected only policy {EXPECTED_POLICY}")

    bad_pr = raw[raw[P_COL] <= raw[R_COL]]
    if len(bad_pr):
        raise ValueError(f"Found {len(bad_pr)} rows with Pbar <= R")
    messages.append("Pbar > R check: PASS")

    p_expected = (raw[P_COL] * 1000 * raw[SERVER_COL]).round().astype(int)
    r_expected = (raw[R_COL] * 1000 * raw[SERVER_COL]).round().astype(int)
    if int((p_expected != raw["P_actual_watts"].astype(int)).sum()) != 0:
        raise ValueError("P_actual_watts conversion mismatch")
    if int((r_expected != raw["R_actual_watts"].astype(int)).sum()) != 0:
        raise ValueError("R_actual_watts conversion mismatch")
    messages.append("kW/server to watts conversion: PASS")

    rsr_error = (raw["Simulator_RSR_Total_Cost"] - (raw["Simulator_Power_Cost"] + raw["Simulator_Tracking_Cost"])).abs()
    if float(rsr_error.max()) > 1e-8:
        raise ValueError(f"Simulator_RSR_Total_Cost mismatch, max error={float(rsr_error.max())}")
    messages.append(f"Simulator_RSR_Total_Cost = power + tracking: PASS, max error={float(rsr_error.max()):.3e}")

    weight_sum_errors = []
    qos_sum_errors = []
    mix_weight_mismatch = 0
    for _, row in raw.iterrows():
        weights = [float(x) for x in parse_json_array(row["Weights"])]
        weight_sum_errors.append(abs(sum(weights) - 1.0))

        qos_probs = [float(x) for x in parse_json_array(row["QoS_Delay_Probabilities"])]
        qos_sum_errors.append(abs(sum(qos_probs) - float(row["QoS_Delay_Probability_Sum"])))

        workload_mix = parse_json_array(row["workload_mix"])
        mix_weights = [float(item[6]) for item in workload_mix if isinstance(item, list) and len(item) >= 7]
        if len(mix_weights) != len(weights) or any(abs(a - b) > 1e-9 for a, b in zip(mix_weights, weights)):
            mix_weight_mismatch += 1

    if max(weight_sum_errors) > 1e-8:
        raise ValueError(f"Weight sums fail, max error={max(weight_sum_errors)}")
    if max(qos_sum_errors) > 1e-8:
        raise ValueError(f"QoS probability sums fail, max error={max(qos_sum_errors)}")
    if mix_weight_mismatch:
        raise ValueError(f"Weights mismatch workload_mix rows: {mix_weight_mismatch}")
    messages.append(f"Weights parse/sum check: PASS, max sum error={max(weight_sum_errors):.3e}")
    messages.append(f"QoS_Delay_Probability_Sum check: PASS, max error={max(qos_sum_errors):.3e}")
    messages.append("Weights column matches workload_mix embedded weights: PASS")

    messages.append("Rows by workload:")
    for workload, count in raw.groupby(WORKLOAD_COL).size().sort_index().items():
        messages.append(f"  {workload}: {count}")

    if diagnostics is None:
        messages.append("Diagnostics CSV not provided: diagnostic plots skipped")
        return "\n".join(messages) + "\n"

    messages.append(f"Diagnostics rows: {len(diagnostics)}")
    messages.append(f"Diagnostics columns: {len(diagnostics.columns)}")
    if len(diagnostics) != len(raw):
        raise ValueError("Diagnostics row count does not match raw row count")

    residual_error = (merged["Normalized_Tracking_Error_90th_Residual"] - (merged["Normalized_Tracking_Error_90th"] - 0.3)).abs()
    if float(residual_error.max()) > 1e-8:
        raise ValueError(f"Tracking residual mismatch, max error={float(residual_error.max())}")
    messages.append(f"Normalized tracking residual check: PASS, max error={float(residual_error.max()):.3e}")

    total_error = (
        merged["Diagnostic_PaperLike_Total_Cost"]
        - (
            merged["Simulator_Power_Cost"]
            + merged["Diagnostic_PaperLike_SoftPlus_Tracking_Cost"]
            + merged["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]
        )
    ).abs()
    if float(total_error.max()) > 1e-8:
        raise ValueError(f"Diagnostic total mismatch, max error={float(total_error.max())}")
    messages.append(f"Diagnostic paper-like total check: PASS, max error={float(total_error.max()):.3e}")

    return "\n".join(messages) + "\n"


# ============================================================
# Plot helpers
# ============================================================

def aggregate_grid(df: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    temp = df[[P_COL, R_COL, metric]].copy()
    temp[P_COL] = temp[P_COL].astype(float).round(10)
    temp[R_COL] = temp[R_COL].astype(float).round(10)
    return temp.pivot_table(index=R_COL, columns=P_COL, values=metric, aggfunc=agg).sort_index(axis=0).sort_index(axis=1)


def plot_grid_coverage(raw: pd.DataFrame, out_dir: Path):
    temp = raw.copy()
    temp[P_COL] = temp[P_COL].round(10)
    temp[R_COL] = temp[R_COL].round(10)
    counts = temp.pivot_table(index=R_COL, columns=P_COL, values="Iteration", aggfunc="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(counts.values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in counts.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels([f"{y:.3f}" for y in counts.index])
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.set_title("P/R grid coverage: rows per physical bid pair")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Rows")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, int(counts.iloc[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / "00_pr_grid_coverage_counts.png")


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, out_dir: Path, agg: str):
    pivot = aggregate_grid(df, metric, agg)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.set_title(f"{title}\nAll workloads, agg={agg}")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(title)
    fig.tight_layout()
    save_fig(fig, out_dir / f"01_heatmap_all_{safe_name(metric)}_{agg}.png")


def plot_surface(df: pd.DataFrame, metric: str, title: str, out_dir: Path, agg: str):
    pivot = aggregate_grid(df, metric, agg)
    X, Y = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
    Z = np.ma.masked_invalid(pivot.to_numpy(float))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    finite = np.asarray(pivot.values, dtype=float)
    if np.isfinite(finite).any():
        zmin, zmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        ax.contour(X, Y, Z, zdir="z", offset=zmin, levels=12, cmap="viridis", alpha=0.75)
        ax.set_zlim(zmin, zmax)
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.set_zlabel(title)
    ax.set_title(f"{title} surface over physical P/R\nAll workloads, agg={agg}")
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surface, ax=ax, shrink=0.6, label=title)
    fig.tight_layout()
    save_fig(fig, out_dir / f"02_surface_all_{safe_name(metric)}_{agg}.png", dpi=220)


def plot_vs_r_by_workload(df: pd.DataFrame, metric: str, title: str, out_dir: Path, agg: str):
    workloads = sorted(df[WORKLOAD_COL].dropna().unique().tolist())
    p_values = rounded_unique(df[P_COL])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    ylim = metric_ylim(df[metric])
    for ax, workload in zip(axes, workloads):
        sub_w = df[df[WORKLOAD_COL] == workload]
        for p in p_values:
            sub = sub_w[np.isclose(sub_w[P_COL], p)]
            if sub.empty:
                continue
            line = sub.groupby(R_COL)[metric].agg(agg).sort_index()
            ax.plot(line.index, line.values, marker="o", linewidth=1.5, label=f"P={p:.3f}")
        ax.set_title(workload)
        ax.set_xlabel("R kW/server")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pbar kW/server", loc="center right", bbox_to_anchor=(1.08, 0.5))
    fig.suptitle(f"{title} vs R, lines by Pbar\nagg={agg} across utilization and weights", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    save_fig(fig, out_dir / f"03_line_vs_r_by_workload_{safe_name(metric)}_{agg}.png")


def plot_vs_p_by_workload(df: pd.DataFrame, metric: str, title: str, out_dir: Path, agg: str):
    workloads = sorted(df[WORKLOAD_COL].dropna().unique().tolist())
    r_values = rounded_unique(df[R_COL])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    ylim = metric_ylim(df[metric])
    for ax, workload in zip(axes, workloads):
        sub_w = df[df[WORKLOAD_COL] == workload]
        for r in r_values:
            sub = sub_w[np.isclose(sub_w[R_COL], r)]
            if sub.empty:
                continue
            line = sub.groupby(P_COL)[metric].agg(agg).sort_index()
            ax.plot(line.index, line.values, marker="o", linewidth=1.5, label=f"R={r:.3f}")
        ax.set_title(workload)
        ax.set_xlabel("Pbar kW/server")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="R kW/server", loc="center right", bbox_to_anchor=(1.08, 0.5))
    fig.suptitle(f"{title} vs Pbar, lines by R\nagg={agg} across utilization and weights", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    save_fig(fig, out_dir / f"04_line_vs_p_by_workload_{safe_name(metric)}_{agg}.png")


def plot_qos_by_utilization(raw: pd.DataFrame, out_dir: Path, agg: str):
    for metric, title in [
        ("QoS_Delay_Probability_Sum", "QoS delay probability sum"),
        ("QoS_Violation_Ratio", "QoS violation ratio"),
    ]:
        workloads = sorted(raw[WORKLOAD_COL].dropna().unique().tolist())
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        axes = axes.ravel()
        for ax, workload in zip(axes, workloads):
            sub_w = raw[raw[WORKLOAD_COL] == workload]
            for util in rounded_unique(sub_w[UTIL_COL]):
                sub = sub_w[np.isclose(sub_w[UTIL_COL], util)]
                line = sub.groupby(P_COL)[metric].agg(agg).sort_index()
                ax.plot(line.index, line.values, marker="o", linewidth=1.5, label=f"util={util:.2f}")
            ax.set_title(workload)
            ax.set_xlabel("Pbar kW/server")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Utilization", loc="center right", bbox_to_anchor=(1.08, 0.5))
        fig.suptitle(f"{title} vs Pbar by utilization\nagg={agg} across R and weights", fontsize=15)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])
        save_fig(fig, out_dir / f"05_qos_vs_p_by_utilization_{safe_name(metric)}_{agg}.png")


def plot_scatter(raw: pd.DataFrame, out_dir: Path):
    pairs = [
        ("Simulator_Power_Cost", "Simulator_RSR_Total_Cost", "Power cost", "RSR total cost"),
        ("Simulator_Tracking_Cost", "Simulator_RSR_Total_Cost", "Tracking cost", "RSR total cost"),
        ("R_kw_per_server", "Simulator_Tracking_Cost", "R kW/server", "Tracking cost"),
        ("QoS_Delay_Probability_Sum", "QoS_Violation_Ratio", "QoS delay sum", "QoS violation ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, pairs):
        for workload, sub in raw.groupby(WORKLOAD_COL):
            ax.scatter(sub[x_col], sub[y_col], s=10, alpha=0.4, label=workload)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Workload", loc="center right", bbox_to_anchor=(1.10, 0.5))
    fig.suptitle("Component and metric relationships", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    save_fig(fig, out_dir / "06_component_relationship_scatter.png")


def plot_workload_summary(raw: pd.DataFrame, out_dir: Path, agg: str):
    summary = raw.groupby(WORKLOAD_COL)[list(KEY_RAW_METRICS.keys())].agg(agg).sort_index()
    summary.to_csv(out_dir / f"summary_raw_metrics_by_workload_{agg}.csv")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()
    for ax, metric in zip(axes, KEY_RAW_METRICS.keys()):
        values = summary[metric]
        ax.bar(values.index, values.values)
        ax.set_title(KEY_RAW_METRICS[metric])
        ax.set_ylabel(agg)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle(f"Raw metric summary by workload, agg={agg}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir / f"07_workload_metric_summary_bars_{agg}.png")


def write_readme(out_dir: Path):
    readme = """
# Rangefind64 visualization outputs

Recommended order to inspect:

1. `00_pr_grid_coverage_counts.png`
   Confirms physical P/R grid coverage and skipped invalid pair.

2. `03_line_vs_r_by_workload_simulator_tracking_cost_median.png`
   Best plot for the tracking-cost vs R question.

3. `04_line_vs_p_by_workload_simulator_power_cost_median.png`
   Shows simulator power-cost behavior versus Pbar.

4. `05_qos_vs_p_by_utilization_qos_delay_probability_sum_median.png`
   Shows QoS behavior versus Pbar by utilization.

5. `02_surface_all_simulator_rsr_total_cost_median.png`
   Main combined 3D surface over physical P/R.

There is no `QoS_Cost` in the raw combined CSV because the simulator returns raw RSR monetary components as power cost and tracking cost. QoS is computed separately from delay probabilities. To avoid repeating the earlier mistake, the raw CSV stores QoS observables (`QoS_Delay_Probability_Sum`, `QoS_Violation_Ratio`) and the diagnostics CSV stores residual/SoftPlus-style QoS terms.
""".strip()
    write_text(out_dir / "README_figures.md", readme + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize rangefind64 FlexDC combined outputs.")
    parser.add_argument("--raw-csv", type=str, default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--diagnostics-csv", type=str, default=str(DEFAULT_DIAGNOSTICS_CSV))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--agg", type=str, default="median", choices=["median", "mean", "min", "max"])
    args = parser.parse_args()

    raw_csv = ensure_file(Path(args.raw_csv), "raw CSV")
    diag_input = Path(args.diagnostics_csv) if args.diagnostics_csv else None
    diagnostics_csv = None
    if diag_input is not None:
        try:
            diagnostics_csv = ensure_file(diag_input, "diagnostics CSV")
        except FileNotFoundError:
            print(f"WARNING: diagnostics CSV not found: {diag_input}. Diagnostic plots skipped.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, diagnostics, merged = load_data(raw_csv, diagnostics_csv)
    report = validate_data(raw, diagnostics, merged)
    print(report)
    write_text(out_dir / "00_validation_report.txt", report)

    # Summary CSVs
    raw.groupby([WORKLOAD_COL, UTIL_COL]).agg(
        rows=("Iteration", "count"),
        simulator_rsr_total_cost_median=("Simulator_RSR_Total_Cost", "median"),
        simulator_power_cost_median=("Simulator_Power_Cost", "median"),
        simulator_tracking_cost_median=("Simulator_Tracking_Cost", "median"),
        qos_delay_probability_sum_median=("QoS_Delay_Probability_Sum", "median"),
        qos_violation_ratio_median=("QoS_Violation_Ratio", "median"),
    ).reset_index().to_csv(out_dir / "summary_by_workload_utilization.csv", index=False)

    raw.groupby([P_COL, R_COL]).agg(
        rows=("Iteration", "count"),
        simulator_rsr_total_cost_median=("Simulator_RSR_Total_Cost", "median"),
        simulator_power_cost_median=("Simulator_Power_Cost", "median"),
        simulator_tracking_cost_median=("Simulator_Tracking_Cost", "median"),
        qos_delay_probability_sum_median=("QoS_Delay_Probability_Sum", "median"),
        qos_violation_ratio_median=("QoS_Violation_Ratio", "median"),
    ).reset_index().to_csv(out_dir / "summary_by_physical_p_r.csv", index=False)

    # Focused plots
    plot_grid_coverage(raw, out_dir)
    plot_workload_summary(raw, out_dir, args.agg)
    plot_scatter(raw, out_dir)
    plot_qos_by_utilization(raw, out_dir, args.agg)

    for metric, title in KEY_RAW_METRICS.items():
        plot_heatmap(raw, metric, title, out_dir, args.agg)
        plot_surface(raw, metric, title, out_dir, args.agg)

    plot_vs_r_by_workload(raw, "Simulator_Tracking_Cost", "Simulator tracking cost", out_dir, args.agg)
    plot_vs_p_by_workload(raw, "Simulator_Power_Cost", "Simulator power cost", out_dir, args.agg)
    plot_vs_p_by_workload(raw, "Simulator_RSR_Total_Cost", "Simulator RSR total cost", out_dir, args.agg)

    if diagnostics is not None:
        for metric, title in KEY_DIAGNOSTIC_METRICS.items():
            if metric in merged.columns:
                plot_heatmap(merged, metric, title, out_dir, args.agg)
                plot_surface(merged, metric, title, out_dir, args.agg)
        if "Normalized_Tracking_Error_90th" in merged.columns:
            plot_vs_r_by_workload(merged, "Normalized_Tracking_Error_90th", "Normalized tracking error 90th percentile", out_dir, args.agg)

    write_readme(out_dir)
    print("\nDone.")
    print(f"Figures and summaries saved in: {out_dir.resolve()}")
    print(f"Recommended first plot: {out_dir / '03_line_vs_r_by_workload_simulator_tracking_cost_median.png'}")


if __name__ == "__main__":
    main()
