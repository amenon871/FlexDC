r"""
Targeted P/R meeting plots for FlexDC data collection.

Run from the FlexDC repo root, for example:

python -u .\plot_targeted_pr_meeting.py `
  --raw-csv .\combined_output_targetedpr\060112026targeted_pr_test_combined_grid_search_results.csv `
  --diagnostics-csv .\combined_output_targetedpr\060112026targeted_pr_test_combined_grid_search_diagnostics.csv `
  --out-dir .\figures_targeted_pr_meeting

This script is designed for the new targeted P/R sweep where each workload has its own
Pmin/Pmax-derived Pbar range and each Pbar has its own valid R interval.

It focuses on the meeting-safe plots:
- generated workload-specific P/R feasible grid
- Simulator_Power_Cost (raw)
- Simulator_Tracking_Cost (raw)
- implied mean absolute tracking error in watts
- QoS_Violation_Ratio / QoS_Delay_Probability_Sum (raw QoS behavior)
- Diagnostic_FlexDC_SoftPlus_QoS_Cost (diagnostic transformed QoS penalty, clearly labeled)

It avoids mixed total-cost plots as the main story.

Important presentation choice:
Because Pbar/R ranges are workload-specific, the main P/R plots are generated separately
per workload. Cross-workload plots are limited to summary bars/boxplots.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # required by matplotlib for 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


RAW_DEFAULT = "combined_output_targetedpr/060112026targeted_pr_test_combined_grid_search_results.csv"
DIAG_DEFAULT = "combined_output_targetedpr/060112026targeted_pr_test_combined_grid_search_diagnostics.csv"

CORE_METRICS = [
    "Simulator_Power_Cost",
    "Simulator_Tracking_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
]

DIAG_METRICS = [
    "Normalized_Tracking_Error_90th",
    "Normalized_Tracking_Error_90th_Residual",
]

NUMERIC_COLS = [
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "P_actual_watts",
    "R_actual_watts",
    "Pbar_ratio",
    "R_ratio",
    "Pbar_denominator_watts",
    "R_denominator_watts",
    "server_count",
    "utilization",
    "Simulator_RSR_Total_Cost",
    "Simulator_Power_Cost",
    "Simulator_Tracking_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
    "Normalized_Tracking_Error_90th",
    "Normalized_Tracking_Error_90th_Residual",
    "QoS_Delay_Probability_Residual_Sum",
    "Diagnostic_PaperLike_SoftPlus_Tracking_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_PaperLike_Total_Cost",
    "Pmin_kw_per_server",
    "Pmax_kw_per_server",
    "Pbar_lower_bound_kw_per_server",
    "Pbar_upper_bound_kw_per_server",
    "PR_upper_bound_kw_per_server",
    "R_lower_bound_kw_per_server",
    "P_plus_R_kw_per_server",
    "P_minus_R_kw_per_server",
    "R_max_for_this_Pbar_kw_per_server",
    "Pbar_points",
    "R_points_per_Pbar",
]


# -----------------------------
# Loading / cleaning helpers
# -----------------------------


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p

    # Useful fallback if user passes a path but is already in a directory containing the file.
    p2 = Path(p.name)
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Could not find file: {path_text}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_workload_col(df: pd.DataFrame) -> str:
    candidates = ["Workload_Name", "Source_Workload_Name", "workload_name", "Workload"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find workload identifier column. Expected one of: "
        + ", ".join(candidates)
    )


def collapse_weight_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the combined file still has Weight_0, Weight_1, ... columns, collapse them
    into a single JSON-list column named weights. If weights already exists, leave it.
    """
    if "weights" in df.columns:
        return df

    weight_cols = [col for col in df.columns if re.fullmatch(r"Weight_\d+", str(col))]
    if not weight_cols:
        return df

    weight_cols = sorted(weight_cols, key=lambda c: int(c.split("_")[1]))
    first_idx = df.columns.get_loc(weight_cols[0])

    def row_weights(row):
        values = []
        for col in weight_cols:
            val = row[col]
            if pd.isna(val):
                continue
            values.append(float(val))
        return json.dumps(values)

    weights = df.apply(row_weights, axis=1)
    df = df.drop(columns=weight_cols)
    df.insert(first_idx, "weights", weights)
    return df


def merge_raw_diag(raw: pd.DataFrame, diag: pd.DataFrame | None) -> pd.DataFrame:
    raw = normalize_cols(collapse_weight_columns_if_needed(raw)).copy()
    if diag is None:
        return raw

    diag = normalize_cols(diag).copy()

    preferred_keys = [
        "Source_Output_Dir",
        "Iteration",
        "Weight_Sample_ID",
        "Workload_Name",
        "Workload_Config",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "server_count",
        "utilization",
    ]
    keys = [col for col in preferred_keys if col in raw.columns and col in diag.columns]

    diag_add_cols = [
        col for col in diag.columns
        if col not in raw.columns or col in keys
    ]
    diag_small = diag[diag_add_cols].copy()

    if keys:
        merged = raw.merge(diag_small, on=keys, how="left", suffixes=("", "_diag"))
    else:
        # Fallback: row-wise append if files came from the same combine script/order.
        merged = raw.reset_index(drop=True)
        diag_small = diag_small.reset_index(drop=True)
        for col in diag_small.columns:
            if col not in merged.columns:
                merged[col] = diag_small[col]

    return normalize_cols(merged)


def agg_df(df: pd.DataFrame, group_cols: list[str], metric: str, agg: str) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False)[metric].agg(agg).reset_index()


def metric_label(metric: str) -> str:
    labels = {
        "Simulator_RSR_Total_Cost": "RSR cost (raw power + raw tracking only)",
        "Simulator_Power_Cost": "Power cost (raw)",
        "Simulator_Tracking_Cost": "Tracking cost (raw)",
        "QoS_Delay_Probability_Sum": "QoS delay probability sum (raw)",
        "QoS_Violation_Ratio": "QoS violation ratio (raw)",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "QoS cost (diagnostic SoftPlus)",
        "Diagnostic_PaperLike_SoftPlus_Tracking_Cost": "Tracking cost (diagnostic SoftPlus)",
        "Diagnostic_PaperLike_Total_Cost": "Diagnostic total cost (mixed; do not present as raw total)",
        "Normalized_Tracking_Error_90th": "90th percentile normalized tracking error",
        "Implied_Mean_Absolute_Tracking_Error_Watts": "Implied mean absolute tracking error (W)",
    }
    return labels.get(metric, metric)


def title_label(metric: str) -> str:
    """Short labels for plot titles so they fit on slides."""
    labels = {
        "Simulator_RSR_Total_Cost": "RSR cost",
        "Simulator_Power_Cost": "Power cost",
        "Simulator_Tracking_Cost": "Tracking cost",
        "QoS_Delay_Probability_Sum": "QoS delay probability",
        "QoS_Violation_Ratio": "QoS violation ratio",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "SoftPlus QoS cost",
        "Diagnostic_PaperLike_SoftPlus_Tracking_Cost": "SoftPlus tracking cost",
        "Diagnostic_PaperLike_Total_Cost": "Diagnostic total cost",
        "Normalized_Tracking_Error_90th": "90th pct. normalized tracking error",
        "Implied_Mean_Absolute_Tracking_Error_Watts": "Mean absolute tracking error",
    }
    return labels.get(metric, metric)


def first_numeric_value(df: pd.DataFrame, col: str) -> float | None:
    """Return the first non-null numeric value for a column, or None."""
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def format_factor(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "?"
    return f"{value:.3g}"


def infer_sweep_metadata(sub: pd.DataFrame) -> dict[str, float | None]:
    """
    Infer the actual sweep settings from diagnostic columns.
    This makes labels follow the CSV, so runs with 0.9/1.0/1.2 are labeled correctly
    and older runs with 0.9/1.1/1.3 are also labeled correctly.
    """
    pmin = first_numeric_value(sub, "Pmin_kw_per_server")
    pmax = first_numeric_value(sub, "Pmax_kw_per_server")
    pbar_lower = first_numeric_value(sub, "Pbar_lower_bound_kw_per_server")
    pbar_upper = first_numeric_value(sub, "Pbar_upper_bound_kw_per_server")
    pr_upper = first_numeric_value(sub, "PR_upper_bound_kw_per_server")
    r_lower = first_numeric_value(sub, "R_lower_bound_kw_per_server")
    pbar_points = first_numeric_value(sub, "Pbar_points")
    r_points = first_numeric_value(sub, "R_points_per_Pbar")

    return {
        "pmin": pmin,
        "pmax": pmax,
        "pbar_lower": pbar_lower,
        "pbar_upper": pbar_upper,
        "pr_upper": pr_upper,
        "r_lower": r_lower,
        "pbar_lower_factor": (pbar_lower / pmin) if pmin and pbar_lower is not None else None,
        "pbar_upper_factor": (pbar_upper / pmax) if pmax and pbar_upper is not None else None,
        "pr_upper_factor": (pr_upper / pmax) if pmax and pr_upper is not None else None,
        "pbar_points": pbar_points,
        "r_points_per_pbar": r_points,
    }


def sweep_label(sub: pd.DataFrame) -> str:
    m = infer_sweep_metadata(sub)
    lf = format_factor(m["pbar_lower_factor"])
    uf = format_factor(m["pbar_upper_factor"])
    pf = format_factor(m["pr_upper_factor"])
    rl = format_factor(m["r_lower"])
    pp = int(m["pbar_points"]) if m["pbar_points"] is not None and np.isfinite(m["pbar_points"]) else "?"
    rp = int(m["r_points_per_pbar"]) if m["r_points_per_pbar"] is not None and np.isfinite(m["r_points_per_pbar"]) else "?"
    return (
        f"Pbar: {lf}×Pmin to {uf}×Pmax; "
        f"R: {rl} to min(Pbar, {pf}×Pmax−Pbar); "
        f"{pp} Pbar × {rp} R/Pbar"
    )


def color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-9)
        return vmin - pad, vmax + pad
    return vmin, vmax


# -----------------------------
# Derived columns / validation
# -----------------------------


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "P_plus_R_kw_per_server" not in df.columns:
        df["P_plus_R_kw_per_server"] = df["Pbar_kw_per_server"] + df["R_kw_per_server"]

    if "P_minus_R_kw_per_server" not in df.columns:
        df["P_minus_R_kw_per_server"] = df["Pbar_kw_per_server"] - df["R_kw_per_server"]

    # final_output(): tracking_cost = piE * R_watts * mean(abs(normalized_tracking_error)) * hour.
    # piE = 0.1 / (3600 * 1000), hour = 3600, so coefficient is 0.0001.
    # R_watts * mean(abs(normalized_tracking_error)) = mean absolute tracking miss in watts.
    if "Simulator_Tracking_Cost" in df.columns:
        df["Implied_Mean_Absolute_Tracking_Error_Watts"] = df["Simulator_Tracking_Cost"] * 10000.0

    return df


def validate_pr_constraints(df: pd.DataFrame, workload_col: str, out_dir: Path) -> pd.DataFrame:
    rows = []
    for workload, sub in df.groupby(workload_col):
        row = {"Workload": workload, "Rows": len(sub)}
        row["Unique_Pbar"] = sub["Pbar_kw_per_server"].nunique()
        row["Unique_PR_pairs"] = sub[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates().shape[0]

        if "Pmin_kw_per_server" in sub.columns:
            row["Pmin_kw_per_server"] = sub["Pmin_kw_per_server"].dropna().iloc[0]
        else:
            row["Pmin_kw_per_server"] = np.nan

        if "Pmax_kw_per_server" in sub.columns:
            row["Pmax_kw_per_server"] = sub["Pmax_kw_per_server"].dropna().iloc[0]
        else:
            row["Pmax_kw_per_server"] = np.nan

        m = infer_sweep_metadata(sub)
        pr_upper = m["pr_upper"] if m["pr_upper"] is not None else np.nan

        row["Pbar_lower_factor"] = m["pbar_lower_factor"]
        row["Pbar_upper_factor"] = m["pbar_upper_factor"]
        row["PR_upper_factor"] = m["pr_upper_factor"]
        row["R_lower_kw_per_server"] = m["r_lower"]
        row["PR_upper_bound_kw_per_server"] = pr_upper
        row["Min_P_minus_R"] = sub["P_minus_R_kw_per_server"].min()
        row["Max_P_plus_R"] = sub["P_plus_R_kw_per_server"].max()
        row["Invalid_P_minus_R_count"] = int((sub["P_minus_R_kw_per_server"] < -1e-8).sum())
        if pd.isna(pr_upper):
            row["Invalid_P_plus_R_count"] = np.nan
        else:
            row["Invalid_P_plus_R_count"] = int((sub["P_plus_R_kw_per_server"] > pr_upper + 1e-8).sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "00_pr_constraint_validation_summary.csv", index=False)
    return summary


# -----------------------------
# Plot helpers
# -----------------------------


def plot_pr_grid_boundary(sub: pd.DataFrame, workload: str, workload_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    pts = sub[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates().sort_values(
        ["Pbar_kw_per_server", "R_kw_per_server"]
    )
    ax.scatter(pts["Pbar_kw_per_server"], pts["R_kw_per_server"], s=28, alpha=0.8, label="Generated P/R points")

    p_min = pts["Pbar_kw_per_server"].min()
    p_max = pts["Pbar_kw_per_server"].max()
    p_grid = np.linspace(p_min, p_max, 300)

    # R <= P line.
    ax.plot(p_grid, p_grid, linestyle="--", linewidth=1.2, label="R = Pbar boundary")

    # P + R <= pr_upper_factor * Pmax boundary if available.
    m = infer_sweep_metadata(sub)
    if m["pr_upper"] is not None:
        pr_upper = float(m["pr_upper"])
        pr_factor = format_factor(m["pr_upper_factor"])
        ax.plot(
            p_grid,
            pr_upper - p_grid,
            linestyle="--",
            linewidth=1.2,
            label=f"Pbar + R = {pr_factor}×Pmax boundary",
        )

    ax.set_title(f"{workload}: P/R grid")
    fig.text(0.5, 0.01, sweep_label(sub), ha="center", fontsize=8)
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(workload_dir / "01_pr_grid_and_constraints.png", dpi=220)
    plt.close(fig)


def plot_metric_scatter_surface(sub: pd.DataFrame, workload: str, workload_dir: Path, metric: str, agg: str, make_3d: bool = True, show_3d: bool = False) -> None:
    if metric not in sub.columns:
        return

    g = agg_df(sub, ["Pbar_kw_per_server", "R_kw_per_server"], metric, agg).dropna()
    if g.empty:
        return

    values = g[metric].to_numpy(dtype=float)
    vmin, vmax = color_limits(values)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        g["Pbar_kw_per_server"],
        g["R_kw_per_server"],
        c=values,
        cmap=cmap,
        norm=norm,
        s=70,
        alpha=0.95,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(metric_label(metric))

    ax.set_title(f"{workload}: {title_label(metric)}")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(workload_dir / f"02_scatter_pr_{safe_name(metric)}.png", dpi=220)
    plt.close(fig)

    # 3D trisurf is valid here because targeted R values form an irregular triangular/curved feasible domain.
    # We do NOT use rectangular surface interpolation because R values differ for each Pbar.
    if make_3d and len(g) >= 3 and g[metric].nunique(dropna=True) > 1:
        fig = plt.figure(figsize=(10, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        x = g["Pbar_kw_per_server"].to_numpy()
        y = g["R_kw_per_server"].to_numpy()
        z = g[metric].to_numpy()
        tri = Triangulation(x, y)
        z_min, z_max = color_limits(z)
        cmap3d = plt.get_cmap("viridis")
        norm3d = Normalize(vmin=z_min, vmax=z_max)

        # IMPORTANT: build the 3D surface manually instead of relying on plot_trisurf(cmap=...).
        # On some Matplotlib/Windows backends, plot_trisurf renders as a single blue surface
        # even when a colorbar exists. Poly3DCollection lets us explicitly assign one
        # viridis facecolor per triangle, based on the metric value for that triangle.
        triangles = tri.triangles
        z_faces = np.mean(z[triangles], axis=1)
        verts = [list(zip(x[t], y[t], z[t])) for t in triangles]
        facecolors = cmap3d(norm3d(z_faces))
        surf = Poly3DCollection(
            verts,
            facecolors=facecolors,
            edgecolors=(0.25, 0.25, 0.25, 0.18),
            linewidths=0.25,
            alpha=0.97,
        )
        ax3.add_collection3d(surf)
        ax3.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax3.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
        ax3.set_zlim(float(z_min), float(z_max))
        ax3.autoscale_view()

        mappable = ScalarMappable(norm=norm3d, cmap=cmap3d)
        mappable.set_array(z_faces)
        fig.colorbar(mappable, ax=ax3, shrink=0.65, pad=0.12, label=metric_label(metric))
        ax3.set_title(f"{workload}: {title_label(metric)}")
        ax3.set_xlabel("Pbar kW/server")
        ax3.set_ylabel("R kW/server")
        ax3.set_zlabel(metric_label(metric))
        fig.tight_layout()
        fig.savefig(workload_dir / f"03_3d_trisurf_pr_{safe_name(metric)}.png", dpi=220)
        if not show_3d:
            plt.close(fig)


def plot_lines_vs_r_by_pbar(sub: pd.DataFrame, workload: str, workload_dir: Path, metric: str, agg: str, gamma_line: bool = False) -> None:
    if metric not in sub.columns:
        return

    g = agg_df(sub, ["Pbar_kw_per_server", "R_kw_per_server"], metric, agg).dropna()
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    for pbar, gp in g.groupby("Pbar_kw_per_server"):
        gp = gp.sort_values("R_kw_per_server")
        ax.plot(gp["R_kw_per_server"], gp[metric], marker="o", linewidth=1.6, label=f"P={pbar:.4f}")

    if gamma_line:
        ax.axhline(0.3, linestyle="--", linewidth=1.4, label="tracking constraint gamma=0.3")

    ax.set_title(f"{workload}: {title_label(metric)} vs R")
    ax.set_xlabel("R kW/server")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(workload_dir / f"04_lines_vs_r_by_pbar_{safe_name(metric)}.png", dpi=220)
    plt.close(fig)


def plot_cost_components_vs_pbar(sub: pd.DataFrame, workload: str, workload_dir: Path, agg: str) -> None:
    metrics = ["Simulator_Power_Cost", "Simulator_Tracking_Cost"]
    if "Diagnostic_FlexDC_SoftPlus_QoS_Cost" in sub.columns:
        metrics.append("Diagnostic_FlexDC_SoftPlus_QoS_Cost")
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        return

    g = sub.groupby("Pbar_kw_per_server", dropna=False)[metrics].agg(agg).reset_index().sort_values("Pbar_kw_per_server")

    fig, ax = plt.subplots(figsize=(10, 7))
    for metric in metrics:
        ax.plot(g["Pbar_kw_per_server"], g[metric], marker="o", label=metric_label(metric))
    ax.set_title(f"{workload}: components vs Pbar")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(workload_dir / "05_cost_components_vs_pbar.png", dpi=220)
    plt.close(fig)


def plot_tracking_qos_tradeoff(sub: pd.DataFrame, workload: str, workload_dir: Path, agg: str) -> None:
    required = ["Simulator_Tracking_Cost", "QoS_Violation_Ratio", "Pbar_kw_per_server", "R_kw_per_server"]
    if any(c not in sub.columns for c in required):
        return

    g = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[
        ["Simulator_Tracking_Cost", "QoS_Violation_Ratio", "QoS_Delay_Probability_Sum"]
    ].agg(agg).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        g["Simulator_Tracking_Cost"],
        g["QoS_Violation_Ratio"],
        c=g["R_kw_per_server"],
        s=60,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("R kW/server")
    ax.set_title(f"{workload}: tracking vs QoS")
    ax.set_xlabel("Tracking cost (raw)")
    ax.set_ylabel("QoS violation ratio (raw)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(workload_dir / "06_tracking_vs_qos_violation_colored_by_r.png", dpi=220)
    plt.close(fig)


def plot_best_rows_table(sub: pd.DataFrame, workload: str, workload_dir: Path, agg: str) -> None:
    metrics = [
        "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Normalized_Tracking_Error_90th",
        "Implied_Mean_Absolute_Tracking_Error_Watts",
    ]
    metrics = [m for m in metrics if m in sub.columns]

    g = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[metrics].agg(agg).reset_index()

    lowest_cost = g.sort_values("Simulator_Tracking_Cost").head(10) if "Simulator_Tracking_Cost" in g.columns else pd.DataFrame()
    lowest_qos = g.sort_values("QoS_Violation_Ratio").head(10) if "QoS_Violation_Ratio" in g.columns else pd.DataFrame()

    if not lowest_cost.empty:
        lowest_cost.to_csv(workload_dir / "07_top10_lowest_tracking_cost_pr_points.csv", index=False)
    if not lowest_qos.empty:
        lowest_qos.to_csv(workload_dir / "08_top10_lowest_qos_violation_pr_points.csv", index=False)


def plot_cross_workload_summary(df: pd.DataFrame, workload_col: str, out_dir: Path, agg: str) -> None:
    metrics = [
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Implied_Mean_Absolute_Tracking_Error_Watts",
    ]
    metrics = [m for m in metrics if m in df.columns]

    summary = df.groupby(workload_col, dropna=False)[metrics].agg(agg).reset_index()
    summary.to_csv(out_dir / "00_cross_workload_metric_summary.csv", index=False)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        tmp = summary.sort_values(metric)
        ax.bar(tmp[workload_col].astype(str), tmp[metric])
        ax.set_title(f"Cross-workload: {title_label(metric)}")
        ax.set_xlabel("Workload")
        ax.set_ylabel(metric_label(metric))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"00_cross_workload_bar_{safe_name(metric)}.png", dpi=220)
        plt.close(fig)


# -----------------------------
# Reporting
# -----------------------------


def write_report(df: pd.DataFrame, workload_col: str, validation: pd.DataFrame, out_dir: Path, agg: str) -> None:
    lines = []
    lines.append("TARGETED P/R MEETING PLOT REPORT")
    lines.append("================================")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Aggregation used for plots: {agg}")
    lines.append(f"Workload column: {workload_col}")
    lines.append(f"Workloads: {sorted(df[workload_col].astype(str).unique().tolist())}")
    lines.append("")
    lines.append("Detected sweep settings by workload:")
    for workload, sub in df.groupby(workload_col):
        lines.append(f"- {workload}: {sweep_label(sub)}")
    lines.append("")

    lines.append("Why workload-separated plots are recommended:")
    lines.append("- In the targeted sweep, Pbar and R ranges are computed from each workload's own Pmin/Pmax.")
    lines.append("- The actual factors are read from the diagnostics CSV; this run should show pbar_upper_factor=1.0 and pr_upper_factor=1.2 if generated with your latest command.")
    lines.append("- For meeting plots, use separate P/R surfaces/line plots per workload, then use cross-workload bars only for high-level comparison.")
    lines.append("")

    lines.append("P/R constraint validation summary:")
    lines.append(validation.to_string(index=False))
    lines.append("")

    corr_cols = [
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Implied_Mean_Absolute_Tracking_Error_Watts",
    ]
    if "Normalized_Tracking_Error_90th" in df.columns:
        corr_cols.append("Normalized_Tracking_Error_90th")
    corr_cols = [c for c in corr_cols if c in df.columns]

    lines.append("Per-workload correlation matrices:")
    for workload, sub in df.groupby(workload_col):
        lines.append("")
        lines.append(f"[{workload}]")
        lines.append(sub[corr_cols].corr(numeric_only=True).to_string())

    lines.append("")
    lines.append("Suggested plots to show Fatih/Kerim first:")
    lines.append("1. Per workload: 01_pr_grid_and_constraints.png")
    lines.append("2. Per workload: 03_3d_trisurf_pr_Simulator_Power_Cost.png")
    lines.append("3. Per workload: 03_3d_trisurf_pr_Simulator_Tracking_Cost.png")
    lines.append("4. Per workload: 04_lines_vs_r_by_pbar_Implied_Mean_Absolute_Tracking_Error_Watts.png")
    lines.append("5. Per workload: 02_scatter_pr_QoS_Violation_Ratio.png")
    lines.append("6. Optional: 02_scatter_pr_Diagnostic_FlexDC_SoftPlus_QoS_Cost.png or 03_3d_trisurf_pr_Diagnostic_FlexDC_SoftPlus_QoS_Cost.png")
    lines.append("")

    lines.append("Interpretation warnings:")
    lines.append("- Do not judge the R effect using an all-workload aggregate P/R heatmap; P/R ranges differ by workload.")
    lines.append("- Use controlled plots: fixed workload, fixed Pbar, x-axis=R.")
    lines.append("- Simulator_Tracking_Cost is the simulator monetary tracking term.")
    lines.append("- Implied_Mean_Absolute_Tracking_Error_Watts = Simulator_Tracking_Cost * 10000 under the simulator's piE/hour convention.")
    lines.append("- QoS_Violation_Ratio is raw QoS behavior; Diagnostic_FlexDC_SoftPlus_QoS_Cost is a transformed diagnostic penalty.")
    lines.append("- Avoid mixed total-cost plots unless explicitly explaining that they are diagnostic, not raw.")

    (out_dir / "00_meeting_plot_report.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FlexDC targeted P/R meeting plots.")
    parser.add_argument("--raw-csv", default=RAW_DEFAULT)
    parser.add_argument("--diagnostics-csv", default=DIAG_DEFAULT)
    parser.add_argument("--out-dir", default="figures_targeted_pr_meeting")
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D triangular surface plots.")
    parser.add_argument("--show-3d", action="store_true", help="Open saved 3D Matplotlib figures in interactive windows so they can be rotated.")
    args = parser.parse_args()

    raw_path = resolve_path(args.raw_csv)
    diag_path = None
    try:
        diag_path = resolve_path(args.diagnostics_csv)
    except FileNotFoundError:
        print(f"Diagnostics file not found, continuing with raw only: {args.diagnostics_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    diag = pd.read_csv(diag_path) if diag_path is not None else None

    df = merge_raw_diag(raw, diag)
    df = add_derived_columns(df)
    workload_col = find_workload_col(df)

    validation = validate_pr_constraints(df, workload_col, out_dir)
    plot_cross_workload_summary(df, workload_col, out_dir, args.agg)

    for workload, sub in df.groupby(workload_col, dropna=False):
        workload_name = safe_name(workload)
        workload_dir = out_dir / workload_name
        workload_dir.mkdir(parents=True, exist_ok=True)

        sub = sub.copy()
        plot_pr_grid_boundary(sub, str(workload), workload_dir)

        surface_metrics = [
            "Simulator_Power_Cost",
            "Simulator_Tracking_Cost",
            "QoS_Delay_Probability_Sum",
            "QoS_Violation_Ratio",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            "Implied_Mean_Absolute_Tracking_Error_Watts",
        ]
        surface_metrics = [m for m in surface_metrics if m in sub.columns]

        for metric in surface_metrics:
            plot_metric_scatter_surface(
                sub,
                str(workload),
                workload_dir,
                metric,
                args.agg,
                make_3d=(not args.skip_3d),
                show_3d=args.show_3d,
            )

        line_metrics = [
            "Simulator_Tracking_Cost",
            "Implied_Mean_Absolute_Tracking_Error_Watts",
            "QoS_Delay_Probability_Sum",
            "QoS_Violation_Ratio",
        ]
        line_metrics = [m for m in line_metrics if m in sub.columns]

        for metric in line_metrics:
            plot_lines_vs_r_by_pbar(
                sub,
                str(workload),
                workload_dir,
                metric,
                args.agg,
                gamma_line=(metric == "Normalized_Tracking_Error_90th"),
            )

        plot_cost_components_vs_pbar(sub, str(workload), workload_dir, args.agg)
        plot_tracking_qos_tradeoff(sub, str(workload), workload_dir, args.agg)
        plot_best_rows_table(sub, str(workload), workload_dir, args.agg)

    write_report(df, workload_col, validation, out_dir, args.agg)

    if args.show_3d and not args.skip_3d:
        print("3D figures are open. Close the Matplotlib windows to finish.")
        plt.show()

    print("Done.")
    print(f"Raw CSV: {raw_path}")
    if diag_path is not None:
        print(f"Diagnostics CSV: {diag_path}")
    print(f"Output dir: {out_dir}")
    print(f"Read report: {out_dir / '00_meeting_plot_report.txt'}")
    print("Recommended first plots are listed in the report.")


if __name__ == "__main__":
    main()
