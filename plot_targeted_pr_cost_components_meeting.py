r"""
Meeting plots for the targeted P/R FlexDC data collection.

Run from the FlexDC repo root, for example:

python -u .\plot_targeted_pr_cost_components_meeting.py `
  --raw-csv .\combined_output_targetedpr\targeted_pr_test_combined_grid_search_results.csv `
  --diagnostics-csv .\combined_output_targetedpr\targeted_pr_test_combined_grid_search_diagnostics.csv `
  --out-dir .\figures_targeted_pr_cost_components_meeting

Optional interactive 3D viewer windows:

python -u .\plot_targeted_pr_cost_components_meeting.py `
  --raw-csv .\combined_output_targetedpr\targeted_pr_test_combined_grid_search_results.csv `
  --diagnostics-csv .\combined_output_targetedpr\targeted_pr_test_combined_grid_search_diagnostics.csv `
  --out-dir .\figures_targeted_pr_cost_components_meeting `
  --show-3d

Important interpretation:
- Simulator_RSR_Total_Cost = Simulator_Power_Cost + Simulator_Tracking_Cost only.
- It does NOT include QoS.
- This script therefore creates Presentation_Total_Cost = Power + Tracking + QoS cost when the diagnostic QoS cost column exists.
- QoS_Delay_Probability_Sum and QoS_Violation_Ratio are raw QoS/feasibility metrics, not monetary/SoftPlus costs.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


RAW_DEFAULT = "combined_output_targetedpr/targeted_pr_test_combined_grid_search_results.csv"
DIAG_DEFAULT = "combined_output_targetedpr/targeted_pr_test_combined_grid_search_diagnostics.csv"

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
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_PaperLike_SoftPlus_Tracking_Cost",
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
# Loading / cleaning
# -----------------------------


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    p2 = Path(p.name)
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find file: {path_text}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_workload_col(df: pd.DataFrame) -> str:
    for c in ["Workload_Name", "Source_Workload_Name", "workload_name", "Workload"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find Workload_Name / workload identifier column.")


def collapse_weight_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "weights" in df.columns:
        return df

    weight_cols = [c for c in df.columns if re.fullmatch(r"Weight_\d+", str(c))]
    if not weight_cols:
        return df

    weight_cols = sorted(weight_cols, key=lambda c: int(c.split("_")[1]))
    insert_at = df.columns.get_loc(weight_cols[0])

    def row_to_weights(row):
        vals = []
        for c in weight_cols:
            v = row[c]
            if pd.isna(v):
                continue
            vals.append(float(v))
        return json.dumps(vals)

    weights = df.apply(row_to_weights, axis=1)
    df = df.drop(columns=weight_cols)
    df.insert(insert_at, "weights", weights)
    return df


def merge_raw_diag(raw: pd.DataFrame, diag: pd.DataFrame | None) -> pd.DataFrame:
    raw = collapse_weight_columns_if_needed(normalize_numeric(raw))
    if diag is None:
        return raw

    diag = normalize_numeric(diag)

    # Prefer exact merge keys. These exist in the current output if the workload-id patch is applied.
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
    keys = [c for c in preferred_keys if c in raw.columns and c in diag.columns]

    # Add only columns that are not already in raw, plus keys for joining.
    diag_cols = keys + [c for c in diag.columns if c not in raw.columns]
    diag_small = diag[diag_cols].copy()

    if keys:
        merged = raw.merge(diag_small, on=keys, how="left", validate="one_to_one")
    else:
        # Fallback for older combined outputs made in identical row order.
        merged = raw.reset_index(drop=True).copy()
        diag_small = diag_small.reset_index(drop=True)
        for c in diag_small.columns:
            if c not in merged.columns:
                merged[c] = diag_small[c]

    return normalize_numeric(merged)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "P_plus_R_kw_per_server" not in df.columns:
        df["P_plus_R_kw_per_server"] = df["Pbar_kw_per_server"] + df["R_kw_per_server"]
    if "P_minus_R_kw_per_server" not in df.columns:
        df["P_minus_R_kw_per_server"] = df["Pbar_kw_per_server"] - df["R_kw_per_server"]

    if "Simulator_Tracking_Cost" in df.columns:
        # final_output(): tracking_cost = piE * R_watts * mean(abs(normalized_error)) * hour.
        # piE = 0.1/(3600*1000), hour=3600 => cost = 0.0001 * mean_abs_tracking_miss_watts.
        df["Implied_Mean_Absolute_Tracking_Error_Watts"] = df["Simulator_Tracking_Cost"] * 10000.0

    if "Diagnostic_FlexDC_SoftPlus_QoS_Cost" in df.columns:
        df["QoS_Cost"] = df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]
        df["Presentation_Total_Cost"] = (
            df["Simulator_Power_Cost"]
            + df["Simulator_Tracking_Cost"]
            + df["QoS_Cost"]
        )
    else:
        df["QoS_Cost"] = np.nan
        df["Presentation_Total_Cost"] = df["Simulator_Power_Cost"] + df["Simulator_Tracking_Cost"]

    return df


def agg_pr(df: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    return (
        df.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[metric]
        .agg(agg)
        .reset_index()
        .dropna()
        .sort_values(["Pbar_kw_per_server", "R_kw_per_server"])
    )


def metric_label(metric: str) -> str:
    labels = {
        "Presentation_Total_Cost": "Total cost = power + tracking + QoS cost",
        "Simulator_Power_Cost": "Power cost",
        "Simulator_Tracking_Cost": "Tracking cost",
        "QoS_Cost": "QoS cost (diagnostic SoftPlus)",
        "QoS_Delay_Probability_Sum": "Raw QoS delay probability sum",
        "QoS_Violation_Ratio": "QoS violation ratio",
        "Normalized_Tracking_Error_90th": "90th percentile normalized tracking error",
        "Implied_Mean_Absolute_Tracking_Error_Watts": "Implied mean absolute tracking error (W)",
        "Simulator_RSR_Total_Cost": "RSR cost = power + tracking only",
    }
    return labels.get(metric, metric)


# -----------------------------
# Plotting
# -----------------------------


def save_or_show_3d(fig, output_path: Path, show_3d: bool, interactive_figs: list) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    if show_3d:
        interactive_figs.append(fig)
    else:
        plt.close(fig)


def plot_pr_grid(sub: pd.DataFrame, workload: str, out_dir: Path) -> None:
    pts = sub[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pts["Pbar_kw_per_server"], pts["R_kw_per_server"], s=34, alpha=0.85, label="generated P/R points")

    p_grid = np.linspace(pts["Pbar_kw_per_server"].min(), pts["Pbar_kw_per_server"].max(), 300)
    ax.plot(p_grid, p_grid, linestyle="--", linewidth=1.2, label="R = P")

    if "PR_upper_bound_kw_per_server" in sub.columns and sub["PR_upper_bound_kw_per_server"].notna().any():
        pr_upper = float(sub["PR_upper_bound_kw_per_server"].dropna().iloc[0])
        ax.plot(p_grid, pr_upper - p_grid, linestyle="--", linewidth=1.2, label="P + R = 1.3 Pmax")

    title_extra = ""
    if "Pmin_kw_per_server" in sub.columns and sub["Pmin_kw_per_server"].notna().any():
        pmin = float(sub["Pmin_kw_per_server"].dropna().iloc[0])
        pmax = float(sub["Pmax_kw_per_server"].dropna().iloc[0])
        title_extra = f"\nPmin={pmin:.4f}, Pmax={pmax:.4f} kW/server"

    ax.set_title(f"Targeted P/R grid: {workload}{title_extra}")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_pr_grid_constraints.png", dpi=220)
    plt.close(fig)


def plot_pr_scatter(sub: pd.DataFrame, workload: str, out_dir: Path, metric: str, agg: str) -> None:
    if metric not in sub.columns:
        return
    g = agg_pr(sub, metric, agg)
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(g["Pbar_kw_per_server"], g["R_kw_per_server"], c=g[metric], s=70, alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(metric_label(metric))
    ax.set_title(f"{metric_label(metric)} over P/R: {workload} ({agg})")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"02_scatter_{safe_name(metric)}.png", dpi=220)
    plt.close(fig)


def plot_3d_metric(sub: pd.DataFrame, workload: str, out_dir: Path, metric: str, agg: str, show_3d: bool, interactive_figs: list) -> None:
    if metric not in sub.columns:
        return
    g = agg_pr(sub, metric, agg)
    if len(g) < 3 or g[metric].nunique(dropna=True) <= 1:
        return

    x = g["Pbar_kw_per_server"].to_numpy()
    y = g["R_kw_per_server"].to_numpy()
    z = g[metric].to_numpy()
    tri = Triangulation(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Use the metric value itself to color the surface.
    # Without cmap/vmin/vmax, Matplotlib renders plot_trisurf as a single default blue surface
    # even though a colorbar is present. shade=False keeps the colormap visible instead of
    # letting 3D lighting make the surface look mostly one color.
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    surf = ax.plot_trisurf(
        tri,
        z,
        cmap="viridis",
        vmin=z_min,
        vmax=z_max,
        linewidth=0.2,
        antialiased=True,
        alpha=0.95,
        shade=False,
    )
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12, label=metric_label(metric))
    ax.set_title(f"3D surface: {metric_label(metric)}\n{workload} ({agg})")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.set_zlabel(metric_label(metric))
    save_or_show_3d(fig, out_dir / f"03_3d_{safe_name(metric)}.png", show_3d, interactive_figs)


def plot_lines_vs_r(sub: pd.DataFrame, workload: str, out_dir: Path, metric: str, agg: str) -> None:
    if metric not in sub.columns:
        return
    g = agg_pr(sub, metric, agg)
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    for pbar, gp in g.groupby("Pbar_kw_per_server"):
        gp = gp.sort_values("R_kw_per_server")
        ax.plot(gp["R_kw_per_server"], gp[metric], marker="o", linewidth=1.5, label=f"P={pbar:.4f}")

    if metric == "Normalized_Tracking_Error_90th":
        ax.axhline(0.3, linestyle="--", linewidth=1.4, label="gamma=0.3")

    ax.set_title(f"{metric_label(metric)} vs R at fixed Pbar: {workload} ({agg})")
    ax.set_xlabel("R kW/server")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"04_lines_vs_r_{safe_name(metric)}.png", dpi=220)
    plt.close(fig)


def plot_cost_components_lines(sub: pd.DataFrame, workload: str, out_dir: Path, agg: str) -> None:
    metrics = ["Simulator_Power_Cost", "Simulator_Tracking_Cost", "QoS_Cost", "Presentation_Total_Cost"]
    metrics = [m for m in metrics if m in sub.columns and sub[m].notna().any()]
    if not metrics:
        return

    g = sub.groupby("Pbar_kw_per_server", dropna=False)[metrics].agg(agg).reset_index().sort_values("Pbar_kw_per_server")

    fig, ax = plt.subplots(figsize=(10, 7))
    for metric in metrics:
        ax.plot(g["Pbar_kw_per_server"], g[metric], marker="o", label=metric_label(metric))
    ax.set_title(f"Cost components vs Pbar: {workload} ({agg} over R/weights)")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_cost_components_vs_pbar.png", dpi=220)
    plt.close(fig)


def plot_tradeoff(sub: pd.DataFrame, workload: str, out_dir: Path, agg: str) -> None:
    required = ["Presentation_Total_Cost", "QoS_Violation_Ratio", "Pbar_kw_per_server", "R_kw_per_server"]
    if any(c not in sub.columns for c in required):
        return

    g = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[
        ["Presentation_Total_Cost", "QoS_Violation_Ratio", "QoS_Delay_Probability_Sum", "Simulator_Tracking_Cost"]
    ].agg(agg).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        g["Presentation_Total_Cost"],
        g["QoS_Violation_Ratio"],
        c=g["R_kw_per_server"],
        s=64,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("R kW/server")
    ax.set_title(f"Total cost vs QoS tradeoff: {workload} ({agg})")
    ax.set_xlabel("Total cost = power + tracking + QoS cost")
    ax.set_ylabel("QoS violation ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "06_total_cost_vs_qos_tradeoff_colored_by_r.png", dpi=220)
    plt.close(fig)


def write_top_tables(sub: pd.DataFrame, out_dir: Path, agg: str) -> None:
    metrics = [
        "Presentation_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Normalized_Tracking_Error_90th",
        "Implied_Mean_Absolute_Tracking_Error_Watts",
    ]
    metrics = [m for m in metrics if m in sub.columns]
    g = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[metrics].agg(agg).reset_index()

    if "Presentation_Total_Cost" in g.columns:
        g.sort_values("Presentation_Total_Cost").head(15).to_csv(out_dir / "07_top15_lowest_total_cost_pr_points.csv", index=False)
    if "QoS_Violation_Ratio" in g.columns:
        g.sort_values("QoS_Violation_Ratio").head(15).to_csv(out_dir / "08_top15_lowest_qos_violation_pr_points.csv", index=False)
    if "Simulator_Tracking_Cost" in g.columns:
        g.sort_values("Simulator_Tracking_Cost").head(15).to_csv(out_dir / "09_top15_lowest_tracking_cost_pr_points.csv", index=False)


def plot_cross_workload_summary(df: pd.DataFrame, workload_col: str, out_dir: Path, agg: str) -> None:
    metrics = [
        "Presentation_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Normalized_Tracking_Error_90th",
    ]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    summary = df.groupby(workload_col, dropna=False)[metrics].agg(agg).reset_index()
    summary.to_csv(out_dir / "00_cross_workload_metric_summary.csv", index=False)


def validate_pr(df: pd.DataFrame, workload_col: str, out_dir: Path) -> None:
    rows = []
    for workload, sub in df.groupby(workload_col):
        pr_upper = np.nan
        if "PR_upper_bound_kw_per_server" in sub.columns and sub["PR_upper_bound_kw_per_server"].notna().any():
            pr_upper = float(sub["PR_upper_bound_kw_per_server"].dropna().iloc[0])
        elif "Pmax_kw_per_server" in sub.columns and sub["Pmax_kw_per_server"].notna().any():
            pr_upper = 1.3 * float(sub["Pmax_kw_per_server"].dropna().iloc[0])

        rows.append({
            "Workload": workload,
            "Rows": len(sub),
            "Unique_Pbar": sub["Pbar_kw_per_server"].nunique(),
            "Unique_PR_pairs": sub[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates().shape[0],
            "Min_P_minus_R": sub["P_minus_R_kw_per_server"].min(),
            "Max_P_plus_R": sub["P_plus_R_kw_per_server"].max(),
            "PR_upper_bound_kw_per_server": pr_upper,
            "Invalid_P_minus_R_count": int((sub["P_minus_R_kw_per_server"] < -1e-8).sum()),
            "Invalid_P_plus_R_count": int((sub["P_plus_R_kw_per_server"] > pr_upper + 1e-8).sum()) if not np.isnan(pr_upper) else np.nan,
        })
    pd.DataFrame(rows).to_csv(out_dir / "00_pr_constraint_validation_summary.csv", index=False)


def write_report(df: pd.DataFrame, workload_col: str, out_dir: Path, agg: str) -> None:
    lines = []
    lines.append("TARGETED P/R COST-COMPONENT MEETING REPORT")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Aggregation: {agg}")
    lines.append(f"Workloads: {sorted(df[workload_col].astype(str).unique().tolist())}")
    lines.append("")
    lines.append("What to show first:")
    lines.append("1. 01_pr_grid_constraints.png for each workload: proves the targeted sweep is correct.")
    lines.append("2. 03_3d_Presentation_Total_Cost.png for each workload: total cost = power + tracking + QoS cost.")
    lines.append("3. 03_3d_Simulator_Power_Cost.png, 03_3d_Simulator_Tracking_Cost.png, 03_3d_QoS_Cost.png: separate cost components.")
    lines.append("4. 04_lines_vs_r_Simulator_Tracking_Cost.png: controlled R effect at fixed Pbar.")
    lines.append("5. 04_lines_vs_r_QoS_Cost.png and 04_lines_vs_r_QoS_Violation_Ratio.png: QoS effect of P/R choices.")
    lines.append("6. 06_total_cost_vs_qos_tradeoff_colored_by_r.png: shows cost/QoS tradeoff.")
    lines.append("")
    lines.append("Column interpretation:")
    lines.append("- Simulator_RSR_Total_Cost is NOT full total cost. It is only power + tracking.")
    lines.append("- Presentation_Total_Cost is created by this script as power + tracking + Diagnostic_FlexDC_SoftPlus_QoS_Cost.")
    lines.append("- QoS_Cost is Diagnostic_FlexDC_SoftPlus_QoS_Cost when available. It is a SoftPlus diagnostic cost, not a raw probability.")
    lines.append("- QoS_Delay_Probability_Sum and QoS_Violation_Ratio are raw QoS/feasibility metrics.")
    lines.append("- Normalized_Tracking_Error_90th is a constraint diagnostic, not the same as tracking cost.")
    lines.append("")
    lines.append("Why plots are workload-separated:")
    lines.append("- P/R ranges are derived from each workload's own Pmin/Pmax, so cross-workload P/R surfaces would mix different feasible domains.")
    lines.append("- Use cross-workload summaries only for high-level comparisons.")
    (out_dir / "00_meeting_report.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate targeted P/R cost-component meeting plots.")
    parser.add_argument("--raw-csv", default=RAW_DEFAULT)
    parser.add_argument("--diagnostics-csv", default=DIAG_DEFAULT)
    parser.add_argument("--out-dir", default="figures_targeted_pr_cost_components_meeting")
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    parser.add_argument("--show-3d", action="store_true", help="Open interactive Matplotlib 3D windows after saving the 3D plots.")
    parser.add_argument("--skip-3d", action="store_true", help="Do not generate 3D plots.")
    args = parser.parse_args()

    raw_path = resolve_path(args.raw_csv)
    diag_path = None
    try:
        diag_path = resolve_path(args.diagnostics_csv)
    except FileNotFoundError:
        print(f"Diagnostics CSV not found; continuing without diagnostic QoS cost: {args.diagnostics_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    diag = pd.read_csv(diag_path) if diag_path is not None else None

    df = merge_raw_diag(raw, diag)
    df = add_derived_columns(df)
    workload_col = find_workload_col(df)

    validate_pr(df, workload_col, out_dir)
    plot_cross_workload_summary(df, workload_col, out_dir, args.agg)

    # These are the meeting-relevant cost components.
    cost_metrics = [
        "Presentation_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Cost",
    ]
    cost_metrics = [m for m in cost_metrics if m in df.columns and df[m].notna().any()]

    diagnostic_metrics = [
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "Normalized_Tracking_Error_90th",
        "Implied_Mean_Absolute_Tracking_Error_Watts",
    ]
    diagnostic_metrics = [m for m in diagnostic_metrics if m in df.columns and df[m].notna().any()]

    interactive_figs = []

    for workload, sub in df.groupby(workload_col, dropna=False):
        workload_dir = out_dir / safe_name(workload)
        workload_dir.mkdir(parents=True, exist_ok=True)
        sub = sub.copy()

        plot_pr_grid(sub, str(workload), workload_dir)

        for metric in cost_metrics + diagnostic_metrics:
            plot_pr_scatter(sub, str(workload), workload_dir, metric, args.agg)
            plot_lines_vs_r(sub, str(workload), workload_dir, metric, args.agg)

            if not args.skip_3d and metric in cost_metrics:
                plot_3d_metric(sub, str(workload), workload_dir, metric, args.agg, args.show_3d, interactive_figs)

        plot_cost_components_lines(sub, str(workload), workload_dir, args.agg)
        plot_tradeoff(sub, str(workload), workload_dir, args.agg)
        write_top_tables(sub, workload_dir, args.agg)

    write_report(df, workload_col, out_dir, args.agg)

    print("Done.")
    print(f"Raw CSV: {raw_path}")
    if diag_path is not None:
        print(f"Diagnostics CSV: {diag_path}")
    print(f"Output dir: {out_dir}")
    print(f"Read report: {out_dir / '00_meeting_report.txt'}")

    if args.show_3d and interactive_figs:
        print("Opening interactive 3D viewer windows. Close them to end the script.")
        plt.show(block=True)


if __name__ == "__main__":
    main()
