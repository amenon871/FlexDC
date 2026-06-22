"""
Additional targeted P/R meeting plots for the final FlexDC AQA dataset schema.

This is meant to be run separately from am_visualize_final_dataset.py.
It adds the older meeting-style plots, updated for the new column names:
  - Mtrack_Cost instead of Simulator_Tracking_Cost
  - Ctrack_Epsilon_90th / Ctrack_Weighted_Cost instead of old normalized tracking names
  - Diagnostic_FullPaperObjective_Cost instead of old paperlike total name
  - weight diagnostics from Weight_* bound columns

Run from the FlexDC repo root, for example:

python am_plot_additional_targeted_pr_new_schema.py \
  --results-csv combined_output_final_aqa_s250_1000_u060_080_prchunk10/final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv \
  --diagnostics-csv combined_output_final_aqa_s250_1000_u060_080_prchunk10/final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_diagnostics.csv \
  --out-dir final_aqa_additional_targeted_pr_plots \
  --agg median
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_RESULTS = "combined_output_final_aqa_s250_1000_u060_080_prchunk10/final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv"
DEFAULT_DIAG = "combined_output_final_aqa_s250_1000_u060_080_prchunk10/final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_diagnostics.csv"

NUMERIC_COLS = [
    "Pbar_kw_per_server", "R_kw_per_server", "P_actual_watts", "R_actual_watts",
    "Pbar_ratio", "R_ratio", "server_count", "utilization", "workload_mix_size",
    "Simulator_RSR_Total_Cost", "Simulator_Power_Cost", "Mtrack_Cost",
    "Mtrack_Error_MeanAbs_Normalized", "Mtrack_Error_MeanAbs_Watts",
    "QoS_Delay_Probability_Sum", "QoS_Violation_Ratio",
    "Pmin_kw_per_server", "Pmax_kw_per_server", "Pbar_lower_bound_kw_per_server",
    "Pbar_upper_bound_kw_per_server", "PR_upper_bound_kw_per_server",
    "R_lower_bound_kw_per_server", "R_max_for_this_Pbar_kw_per_server",
    "P_plus_R_kw_per_server", "P_minus_R_kw_per_server",
    "Pbar_points", "R_points_per_Pbar", "PR_Total_Pairs_Before_Chunk",
    "PR_Chunk_Index", "PR_Num_Chunks", "PR_Pairs_In_Chunk",
    "Weight_Equal_Value", "Weight_Relative_Lower_Bound", "Weight_Relative_Upper_Bound",
    "Weight_Server_Lower_Bound", "Weight_Upper_From_Lower_Bound",
    "Weight_Final_Lower_Bound", "Weight_Final_Upper_Bound", "Weight_Min", "Weight_Max",
    "Ctrack_Epsilon_90th", "Ctrack_Gamma", "Ctrack_Residual", "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost", "Diagnostic_FullPaperObjective_Cost",
]

SURFACE_METRICS = [
    "Mtrack_Cost",
    "Ctrack_Epsilon_90th",
    "QoS_Violation_Ratio",
    "Diagnostic_FullPaperObjective_Cost",
]

LINE_METRICS = [
    "Mtrack_Cost",
    "Ctrack_Epsilon_90th",
    "QoS_Violation_Ratio",
]


def safe_name(x) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") or "unknown"


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    p2 = Path(p.name)
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find: {path_text}")


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_merge(results_csv: Path, diagnostics_csv: Path) -> pd.DataFrame:
    raw = normalize_numeric(pd.read_csv(results_csv))
    diag = normalize_numeric(pd.read_csv(diagnostics_csv))

    keys = [
        "Source_Output_Dir", "Iteration", "Weight_Sample_ID", "Workload_Name",
        "Workload_Config", "Pbar_kw_per_server", "R_kw_per_server",
        "P_actual_watts", "R_actual_watts", "server_count", "utilization",
    ]
    keys = [c for c in keys if c in raw.columns and c in diag.columns]

    diag_cols = keys + [c for c in diag.columns if c not in raw.columns]
    df = raw.merge(diag[diag_cols], on=keys, how="left", validate="one_to_one")

    if "P_plus_R_kw_per_server" not in df.columns:
        df["P_plus_R_kw_per_server"] = df["Pbar_kw_per_server"] + df["R_kw_per_server"]
    if "P_minus_R_kw_per_server" not in df.columns:
        df["P_minus_R_kw_per_server"] = df["Pbar_kw_per_server"] - df["R_kw_per_server"]

    return normalize_numeric(df)


def metric_label(metric: str) -> str:
    labels = {
        "Simulator_RSR_Total_Cost": "RSR cost",
        "Simulator_Power_Cost": "Power cost",
        "Mtrack_Cost": "Monetary tracking cost",
        "Mtrack_Error_MeanAbs_Watts": "Mean absolute tracking error (W)",
        "QoS_Delay_Probability_Sum": "QoS delay probability sum",
        "QoS_Violation_Ratio": "QoS violation ratio",
        "Ctrack_Epsilon_90th": "90th percentile tracking error",
        "Ctrack_Weighted_Cost": "Weighted Ctrack SoftPlus cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "SoftPlus QoS cost",
        "Diagnostic_FullPaperObjective_Cost": "Diagnostic full paper objective",
    }
    return labels.get(metric, metric)


def color_limits(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1e-9)
        return lo - pad, hi + pad
    return lo, hi


def agg_grid(sub: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    return (
        sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[metric]
        .agg(agg)
        .reset_index()
        .dropna()
        .sort_values(["Pbar_kw_per_server", "R_kw_per_server"])
    )


def first(sub: pd.DataFrame, col: str):
    if col not in sub.columns:
        return None
    vals = pd.to_numeric(sub[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[0])


def make_group_dir(out_dir: Path, workload, server, util=None) -> Path:
    if util is None:
        d = out_dir / f"{safe_name(workload)}__S{int(server)}"
    else:
        d = out_dir / f"{safe_name(workload)}__S{int(server)}__U{safe_name(util)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_pr_grid(sub: pd.DataFrame, workload, server, out_dir: Path) -> None:
    pts = sub[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates()
    if pts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(pts["Pbar_kw_per_server"], pts["R_kw_per_server"], s=32, alpha=0.85, label="Generated P/R points")

    pmin = pts["Pbar_kw_per_server"].min()
    pmax = pts["Pbar_kw_per_server"].max()
    pgrid = np.linspace(pmin, pmax, 300)
    ax.plot(pgrid, pgrid, linestyle="--", linewidth=1.3, label="R = Pbar")

    pr_upper = first(sub, "PR_upper_bound_kw_per_server")
    if pr_upper is not None:
        ax.plot(pgrid, pr_upper - pgrid, linestyle="--", linewidth=1.3, label="Pbar + R upper boundary")

    ax.set_title(f"{workload}, S={int(server)}: P/R feasible grid")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_PR_grid_and_constraints.png", dpi=160)
    plt.close(fig)


def plot_scatter_and_3d(sub: pd.DataFrame, metric: str, workload, server, util, out_dir: Path, agg: str, make_3d: bool) -> None:
    if metric not in sub.columns:
        return
    g = agg_grid(sub, metric, agg)
    if len(g) < 3 or g[metric].nunique(dropna=True) < 1:
        return

    x = g["Pbar_kw_per_server"].to_numpy(float)
    y = g["R_kw_per_server"].to_numpy(float)
    z = g[metric].to_numpy(float)
    vmin, vmax = color_limits(z)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=70, alpha=0.95)
    fig.colorbar(sc, ax=ax, label=metric_label(metric))
    ax.set_title(f"{workload}, S={int(server)}, U={util}: {metric_label(metric)}")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("R kW/server")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"02_scatter_PR_{safe_name(metric)}.png", dpi=160)
    plt.close(fig)

    if not make_3d:
        return

    tri = Triangulation(x, y)
    triangles = tri.triangles
    z_faces = np.mean(z[triangles], axis=1)
    verts = [list(zip(x[t], y[t], z[t])) for t in triangles]

    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    surf = Poly3DCollection(
        verts,
        facecolors=cmap(norm(z_faces)),
        edgecolors=(0.25, 0.25, 0.25, 0.18),
        linewidths=0.25,
        alpha=0.97,
    )
    ax3.add_collection3d(surf)
    ax3.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax3.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    ax3.set_zlim(float(vmin), float(vmax))
    ax3.autoscale_view()
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(z_faces)
    fig.colorbar(mappable, ax=ax3, shrink=0.65, pad=0.12, label=metric_label(metric))
    ax3.set_title(f"{workload}, S={int(server)}, U={util}: {metric_label(metric)}")
    ax3.set_xlabel("Pbar kW/server")
    ax3.set_ylabel("R kW/server")
    ax3.set_zlabel(metric_label(metric))
    fig.tight_layout()
    fig.savefig(out_dir / f"03_3D_PR_{safe_name(metric)}.png", dpi=160)
    plt.close(fig)


def plot_lines_vs_r(sub: pd.DataFrame, metric: str, workload, server, util, out_dir: Path, agg: str) -> None:
    if metric not in sub.columns:
        return
    g = agg_grid(sub, metric, agg)
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    for pbar, gp in g.groupby("Pbar_kw_per_server"):
        gp = gp.sort_values("R_kw_per_server")
        ax.plot(gp["R_kw_per_server"], gp[metric], marker="o", linewidth=1.5, label=f"P={pbar:.4f}")

    if metric == "Ctrack_Epsilon_90th":
        ax.axhline(0.3, linestyle="--", linewidth=1.4, color="red", label="gamma = 0.3")

    ax.set_title(f"{workload}, S={int(server)}, U={util}: {metric_label(metric)} vs R")
    ax.set_xlabel("R kW/server")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"04_lines_vs_R_by_Pbar_{safe_name(metric)}.png", dpi=160)
    plt.close(fig)


def plot_components_vs_pbar(sub: pd.DataFrame, workload, server, util, out_dir: Path, agg: str) -> None:
    metrics = [
        "Simulator_Power_Cost",
        "Mtrack_Cost",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    ]
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        return

    g = sub.groupby("Pbar_kw_per_server", dropna=False)[metrics].agg(agg).reset_index().sort_values("Pbar_kw_per_server")

    fig, ax = plt.subplots(figsize=(10, 7))
    for metric in metrics:
        ax.plot(g["Pbar_kw_per_server"], g[metric], marker="o", linewidth=1.6, label=metric_label(metric))
    ax.set_title(f"{workload}, S={int(server)}, U={util}: cost components vs Pbar")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("Cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_cost_components_vs_Pbar.png", dpi=160)
    plt.close(fig)


def plot_tracking_qos_tradeoff(sub: pd.DataFrame, workload, server, out_dir: Path, agg: str) -> None:
    needed = ["Mtrack_Cost", "QoS_Violation_Ratio", "R_kw_per_server", "utilization"]
    if any(c not in sub.columns for c in needed):
        return

    g = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server", "utilization"], dropna=False)[
        ["Mtrack_Cost", "QoS_Violation_Ratio", "QoS_Delay_Probability_Sum"]
    ].agg(agg).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(g["Mtrack_Cost"], g["QoS_Violation_Ratio"], c=g["R_kw_per_server"], s=60, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="R kW/server")
    ax.set_title(f"{workload}, S={int(server)}: tracking vs QoS")
    ax.set_xlabel("Mtrack_Cost")
    ax.set_ylabel("QoS_Violation_Ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "06_tracking_vs_QoS_colored_by_R.png", dpi=160)
    plt.close(fig)


def save_top_tables(sub: pd.DataFrame, out_dir: Path, agg: str) -> None:
    metrics = [m for m in [
        "Simulator_RSR_Total_Cost", "Mtrack_Cost", "QoS_Violation_Ratio",
        "QoS_Delay_Probability_Sum", "Ctrack_Epsilon_90th", "Diagnostic_FullPaperObjective_Cost",
    ] if m in sub.columns]
    if not metrics:
        return
    g = sub.groupby(["utilization", "Pbar_kw_per_server", "R_kw_per_server"], dropna=False)[metrics].agg(agg).reset_index()
    if "Mtrack_Cost" in g.columns:
        g.sort_values("Mtrack_Cost").head(20).to_csv(out_dir / "07_top20_lowest_Mtrack_Cost.csv", index=False)
    if "QoS_Violation_Ratio" in g.columns:
        g.sort_values("QoS_Violation_Ratio").head(20).to_csv(out_dir / "08_top20_lowest_QoS_Violation_Ratio.csv", index=False)
    if "Diagnostic_FullPaperObjective_Cost" in g.columns:
        g.sort_values("Diagnostic_FullPaperObjective_Cost").head(20).to_csv(out_dir / "09_top20_lowest_Diagnostic_FullPaperObjective_Cost.csv", index=False)


def plot_cross_workload_summary(df: pd.DataFrame, out_dir: Path, agg: str) -> None:
    metrics = [m for m in [
        "Simulator_Power_Cost", "Mtrack_Cost", "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio", "Ctrack_Weighted_Cost", "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    ] if m in df.columns]
    if not metrics:
        return

    group_cols = ["Workload_Name", "server_count"]
    summary = df.groupby(group_cols, dropna=False)[metrics].agg(agg).reset_index()
    summary.to_csv(out_dir / "00_cross_workload_server_metric_summary.csv", index=False)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        tmp = summary.copy()
        tmp["label"] = tmp["Workload_Name"].astype(str) + "\nS=" + tmp["server_count"].astype(int).astype(str)
        tmp = tmp.sort_values(metric)
        ax.bar(tmp["label"], tmp[metric])
        ax.set_title(f"Cross workload/server: {metric_label(metric)}")
        ax.set_xlabel("Workload and server count")
        ax.set_ylabel(metric_label(metric))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"00_cross_workload_server_bar_{safe_name(metric)}.png", dpi=160)
        plt.close(fig)


def write_report(df: pd.DataFrame, out_dir: Path, agg: str) -> None:
    lines = []
    lines.append("ADDITIONAL TARGETED P/R PLOT REPORT")
    lines.append("===================================")
    lines.append("")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Aggregation: {agg}")
    lines.append(f"Workloads: {sorted(df['Workload_Name'].astype(str).unique())}")
    lines.append(f"Server counts: {sorted(pd.to_numeric(df['server_count'], errors='coerce').dropna().astype(int).unique())}")
    lines.append(f"Utilizations: {sorted(pd.to_numeric(df['utilization'], errors='coerce').dropna().unique())}")
    lines.append("")
    lines.append("Plot organization:")
    lines.append("- 01_PR_grid_and_constraints: one per workload/server")
    lines.append("- 02_scatter_PR_* and 03_3D_PR_*: one per workload/server/utilization")
    lines.append("- 04_lines_vs_R_by_Pbar_*: controlled R-sensitivity plots")
    lines.append("- 05_cost_components_vs_Pbar: raw and diagnostic cost components")
    lines.append("- 06_tracking_vs_QoS_colored_by_R: tradeoff scatter")
    lines.append("")
    lines.append("Column mapping used:")
    lines.append("- Raw tracking monetary term: Mtrack_Cost")
    lines.append("- Mean tracking miss in watts: Mtrack_Error_MeanAbs_Watts")
    lines.append("- Paper tracking constraint metric: Ctrack_Epsilon_90th")
    lines.append("- Weighted tracking SoftPlus penalty: Ctrack_Weighted_Cost")
    lines.append("- Full diagnostic paper objective: Diagnostic_FullPaperObjective_Cost")
    (out_dir / "00_additional_targeted_pr_plot_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS)
    parser.add_argument("--diagnostics-csv", default=DEFAULT_DIAG)
    parser.add_argument("--out-dir", default="final_aqa_additional_targeted_pr_plots")
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    parser.add_argument("--skip-3d", action="store_true")
    args = parser.parse_args()

    results_csv = resolve_path(args.results_csv)
    diagnostics_csv = resolve_path(args.diagnostics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(results_csv, diagnostics_csv)
    plot_cross_workload_summary(df, out_dir, args.agg)

    for (workload, server), sub_ws in df.groupby(["Workload_Name", "server_count"], dropna=False):
        group_dir = make_group_dir(out_dir, workload, server)
        plot_pr_grid(sub_ws, workload, server, group_dir)
        plot_tracking_qos_tradeoff(sub_ws, workload, server, group_dir, args.agg)
        save_top_tables(sub_ws, group_dir, args.agg)

        for util, sub in sub_ws.groupby("utilization", dropna=False):
            util_dir = make_group_dir(out_dir, workload, server, util)
            for metric in SURFACE_METRICS:
                plot_scatter_and_3d(sub, metric, workload, server, util, util_dir, args.agg, make_3d=not args.skip_3d)
            for metric in LINE_METRICS:
                plot_lines_vs_r(sub, metric, workload, server, util, util_dir, args.agg)
            plot_components_vs_pbar(sub, workload, server, util, util_dir, args.agg)

    write_report(df, out_dir, args.agg)
    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Report: {out_dir / '00_additional_targeted_pr_plot_report.txt'}")


if __name__ == "__main__":
    main()
