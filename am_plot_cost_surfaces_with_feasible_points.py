#!/usr/bin/env python3
"""
Plot paper-consistent 3D P/R cost surfaces and mark configurations that pass
tracking + QoS constraints.

Main use case:
  - surface data: the broad combined FlexDC grid CSVs
  - point data: a local validation/sweep CSV such as mini_dense_w2_local_sweep_all_rows.csv

v7 additions:
  * overlays pass-both-constraints points on the 3D surfaces
  * can recompute paper-form FlexDC objective columns from raw logged metrics at plot time
  * writes a CSV summary of the plotted pass points
  * still saves side-by-side isometric PNGs and does not open interactive windows
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


CORE_METRICS = [
    "Simulator_RSR_Total_Cost",
    "Ctrack_Epsilon_90th",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]

PAPER_METRICS = [
    "Simulator_RSR_Total_Cost",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]

EXTRA_METRICS = [
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
    "Max_QoS_Delay_Probability",
]

METRIC_LABELS = {
    "Simulator_RSR_Total_Cost": "M_RSR / simulator RSR cost",
    "Ctrack_Epsilon_90th": "p90 normalized tracking error",
    "Ctrack_Weighted_Cost": "FlexDC weighted tracking penalty",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "FlexDC weighted QoS penalty",
    "Diagnostic_FullPaperObjective_Cost": "FlexDC full paper objective",
    "QoS_Delay_Probability_Sum": "QoS delay probability sum",
    "QoS_Violation_Ratio": "QoS violation ratio",
    "Max_QoS_Delay_Probability": "Max job-type QoS probability",
}

MERGE_KEY_CANDIDATES = [
    ["Source_Output_Dir", "Iteration"],
    ["Iteration", "Weight_Sample_ID", "Workload_Name", "Pbar_kw_per_server", "R_kw_per_server", "server_count", "utilization"],
    ["Iteration", "Weight_Sample_ID"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side isometric P/R surfaces and overlay points that pass "
            "tracking and QoS constraints."
        )
    )
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv or all-rows CSV")
    parser.add_argument("--diagnostics-csv", default=None, help="Combined grid_search_diagnostics.csv. Optional if metrics already in results CSV.")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--workload", default="all", help="Workload name, comma-separated workload names, or all")
    parser.add_argument("--workloads", default=None, help="Optional comma-separated workload names; overrides --workload")
    parser.add_argument("--server-count", type=int, required=True)
    parser.add_argument("--aggregation-scope", choices=["fixed_utilization", "all_utilizations"], default="fixed_utilization")
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--expected-rows-per-pr", type=int, default=None)
    parser.add_argument("--agg", choices=["mean", "min", "median", "max"], default="mean")
    parser.add_argument("--metrics", default="paper", help="Comma-separated metrics, or paper/core/all")
    parser.add_argument("--surface-z-axis", choices=["auto", "shared", "per_workload"], default="auto")
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-135.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show-3d", action="store_true", help="Accepted for compatibility; ignored.")

    # Constraint overlays.
    parser.add_argument("--points-csv", default=None, help="CSV containing candidate points to overlay, e.g. mini_dense_w2_local_sweep_all_rows.csv")
    parser.add_argument("--points-csv-list", default=None, help="Semicolon-separated list of point CSVs")
    parser.add_argument("--points-label", default="Pass both constraints", help="Legend label for overlaid pass points")
    parser.add_argument("--point-filter", choices=["both_pass", "all", "tracking_pass", "qos_pass"], default="both_pass")
    parser.add_argument(
        "--constraint-logic",
        choices=["current", "maxprob", "both"],
        default="current",
        help=(
            "current: tracking <= threshold and QoS_Violation_Ratio <= threshold; "
            "maxprob: tracking <= threshold and Max_QoS_Delay_Probability <= threshold; "
            "both: require both QoS checks."
        ),
    )
    parser.add_argument("--tracking-threshold", type=float, default=0.3)
    parser.add_argument("--qos-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--qos-prob-threshold", type=float, default=0.1)
    parser.add_argument("--annotate-points", action="store_true", help="Annotate overlaid points with P/R text; can clutter figures.")
    parser.add_argument("--plot-all-points-faint", action="store_true", help="Also overlay all candidate points faintly for context.")

    # Paper-form recompute. Useful when point CSVs come straight from the wizard and
    # still contain gradient_config diagnostic cost columns.
    parser.add_argument("--recompute-paper-objective", action="store_true", help="Recompute Ctrack/CQoS/full objective from raw epsilon90 and QoS probabilities.")
    parser.add_argument("--paper-psi", type=float, default=1.0)
    parser.add_argument("--paper-mu", type=float, default=10.0)
    parser.add_argument("--paper-gamma", type=float, default=0.3)
    parser.add_argument("--paper-beta", type=float, default=20.0)
    parser.add_argument("--paper-rho", type=float, default=2.0)
    parser.add_argument("--paper-delta", type=float, default=0.1)

    return parser.parse_args()


def split_list(value: str | None, sep: str = ",") -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(sep) if item.strip()]


def safe_name(value) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    return value.strip("_") or "value"


def stable_softplus(x):
    return np.logaddexp(0.0, x)


def parse_probability_vector(value) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        return [float(x) for x in parsed]
    except Exception:
        pass
    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
    try:
        parsed = ast.literal_eval(cleaned)
        return [float(x) for x in parsed]
    except Exception:
        return []


def add_max_qos_probability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Max_QoS_Delay_Probability" not in out.columns and "QoS_Delay_Probabilities" in out.columns:
        out["Max_QoS_Delay_Probability"] = out["QoS_Delay_Probabilities"].apply(
            lambda x: max(parse_probability_vector(x)) if parse_probability_vector(x) else np.nan
        )
    return out


def recompute_paper_objective(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = add_max_qos_probability(df)
    if "Ctrack_Epsilon_90th" not in out.columns or "Simulator_RSR_Total_Cost" not in out.columns:
        return out
    if "QoS_Delay_Probabilities" not in out.columns:
        return out

    eps = out["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    ctrack = args.paper_psi * stable_softplus(args.paper_mu * (eps - args.paper_gamma))

    qos_cost = []
    qos_sum = []
    for value in out["QoS_Delay_Probabilities"]:
        probs = np.asarray(parse_probability_vector(value), dtype=float)
        if probs.size == 0:
            qos_cost.append(np.nan)
            qos_sum.append(np.nan)
        else:
            qos_cost.append(args.paper_beta * stable_softplus(args.paper_rho * (probs - args.paper_delta)).sum())
            qos_sum.append(float(probs.sum()))

    out["Ctrack_Weighted_Cost"] = ctrack
    out["Diagnostic_FlexDC_SoftPlus_QoS_Cost"] = np.asarray(qos_cost, dtype=float)
    out["QoS_Delay_Probability_Sum"] = np.asarray(qos_sum, dtype=float)
    out["Diagnostic_FullPaperObjective_Cost"] = (
        out["Simulator_RSR_Total_Cost"].astype(float)
        + out["Ctrack_Weighted_Cost"].astype(float)
        + out["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].astype(float)
    )
    return out


def choose_merge_keys(results: pd.DataFrame, diagnostics: pd.DataFrame) -> list[str] | None:
    for keys in MERGE_KEY_CANDIDATES:
        if all(k in results.columns and k in diagnostics.columns for k in keys):
            return keys
    return None


def load_surface_data(results_csv: Path, diagnostics_csv: Path | None, args: argparse.Namespace) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    if diagnostics_csv is not None:
        diagnostics = pd.read_csv(diagnostics_csv)
        keys = choose_merge_keys(results, diagnostics)
        if keys is None:
            if len(results) != len(diagnostics):
                raise ValueError("No merge keys found and results/diagnostics row counts differ.")
            extra = diagnostics[[c for c in diagnostics.columns if c not in results.columns]]
            df = pd.concat([results.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        else:
            extra_cols = keys + [c for c in diagnostics.columns if c not in results.columns]
            df = results.merge(diagnostics[extra_cols], on=keys, how="left", validate="one_to_one")
    else:
        df = results

    df = add_max_qos_probability(df)
    if args.recompute_paper_objective:
        df = recompute_paper_objective(df, args)
    return df


def load_points(args: argparse.Namespace) -> pd.DataFrame:
    paths = []
    if args.points_csv:
        paths.append(args.points_csv)
    paths.extend(split_list(args.points_csv_list, sep=";"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["Point_Source_CSV"] = str(p)
        frames.append(df)
    points = pd.concat(frames, ignore_index=True)
    points = add_max_qos_probability(points)
    if args.recompute_paper_objective:
        points = recompute_paper_objective(points, args)
    return points


def filter_surface(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df[df["server_count"].astype(int) == int(args.server_count)].copy()
    if args.aggregation_scope == "fixed_utilization":
        if args.utilization is None:
            raise ValueError("--utilization is required for fixed_utilization")
        out = out[np.isclose(out["utilization"].astype(float), float(args.utilization))].copy()
    if out.empty:
        raise ValueError("No surface rows remain after filtering.")
    return out


def filter_points(points: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if points.empty:
        return points
    out = points.copy()
    if "server_count" in out.columns:
        out = out[out["server_count"].astype(int) == int(args.server_count)].copy()
    if args.aggregation_scope == "fixed_utilization" and args.utilization is not None and "utilization" in out.columns:
        out = out[np.isclose(out["utilization"].astype(float), float(args.utilization))].copy()

    if "Ctrack_Epsilon_90th" in out.columns:
        tracking_pass = out["Ctrack_Epsilon_90th"].astype(float) <= float(args.tracking_threshold)
    elif "Tracking_Pass" in out.columns:
        tracking_pass = out["Tracking_Pass"].astype(bool)
    else:
        tracking_pass = pd.Series(False, index=out.index)

    if args.constraint_logic == "current":
        if "QoS_Violation_Ratio" in out.columns:
            qos_pass = out["QoS_Violation_Ratio"].astype(float) <= float(args.qos_ratio_threshold)
        elif "QoS_Pass_CurrentLogic" in out.columns:
            qos_pass = out["QoS_Pass_CurrentLogic"].astype(bool)
        else:
            qos_pass = pd.Series(False, index=out.index)
    elif args.constraint_logic == "maxprob":
        if "Max_QoS_Delay_Probability" in out.columns:
            qos_pass = out["Max_QoS_Delay_Probability"].astype(float) <= float(args.qos_prob_threshold)
        else:
            qos_pass = pd.Series(False, index=out.index)
    else:  # both
        if "QoS_Violation_Ratio" in out.columns:
            qos_ratio_pass = out["QoS_Violation_Ratio"].astype(float) <= float(args.qos_ratio_threshold)
        else:
            qos_ratio_pass = pd.Series(False, index=out.index)
        if "Max_QoS_Delay_Probability" in out.columns:
            qos_max_pass = out["Max_QoS_Delay_Probability"].astype(float) <= float(args.qos_prob_threshold)
        else:
            qos_max_pass = pd.Series(False, index=out.index)
        qos_pass = qos_ratio_pass & qos_max_pass

    out["Plot_Tracking_Pass"] = tracking_pass
    out["Plot_QoS_Pass"] = qos_pass
    out["Plot_Both_Pass"] = tracking_pass & qos_pass

    if args.point_filter == "both_pass":
        out = out[out["Plot_Both_Pass"]].copy()
    elif args.point_filter == "tracking_pass":
        out = out[out["Plot_Tracking_Pass"]].copy()
    elif args.point_filter == "qos_pass":
        out = out[out["Plot_QoS_Pass"]].copy()
    elif args.point_filter == "all":
        pass
    return out


def choose_workloads(df: pd.DataFrame, workload_arg: str, workloads_arg: str | None) -> list[str]:
    available = sorted(str(x) for x in df["Workload_Name"].dropna().unique())
    request = workloads_arg if workloads_arg is not None else workload_arg
    if request is None or str(request).strip().lower() == "all":
        return available
    requested = split_list(request)
    missing = [w for w in requested if w not in available]
    if missing:
        raise ValueError("Requested workload(s) not found: " + ", ".join(missing) + "\nAvailable: " + ", ".join(available))
    return requested


def choose_metrics(df: pd.DataFrame, metric_arg: str) -> list[str]:
    arg = metric_arg.lower().strip()
    if arg == "paper":
        candidates = PAPER_METRICS
    elif arg == "core":
        candidates = CORE_METRICS
    elif arg == "all":
        candidates = list(dict.fromkeys(PAPER_METRICS + CORE_METRICS + EXTRA_METRICS))
    else:
        candidates = split_list(metric_arg)
    present = [m for m in candidates if m in df.columns]
    missing = [m for m in candidates if m not in df.columns]
    if missing:
        print("Warning: skipping missing metric columns:", missing)
    if not present:
        raise ValueError("No requested metrics were found in the data.")
    return present


def aggregate_surface(df: pd.DataFrame, workload: str, metric: str, agg: str) -> pd.DataFrame:
    sub = df[df["Workload_Name"].astype(str) == workload].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Pbar_kw_per_server", "R_kw_per_server", metric, "rows_per_pr"])
    group_cols = ["Pbar_kw_per_server", "R_kw_per_server"]
    grouped = sub.groupby(group_cols, as_index=False)
    if agg == "mean":
        val = grouped[metric].mean()
    elif agg == "min":
        val = grouped[metric].min()
    elif agg == "median":
        val = grouped[metric].median()
    elif agg == "max":
        val = grouped[metric].max()
    else:
        raise ValueError(agg)
    counts = grouped.size().rename(columns={"size": "rows_per_pr"})
    return val.merge(counts, on=group_cols, how="left")


def warn_expected_rows(surface: pd.DataFrame, workload: str, metric: str, expected: int | None) -> None:
    if expected is None or surface.empty:
        return
    bad = surface[surface["rows_per_pr"] != expected]
    if not bad.empty:
        print(
            f"Warning: {workload}/{metric}: {len(bad)} groups do not have rows_per_pr={expected}; "
            f"observed={sorted(surface['rows_per_pr'].dropna().unique())}"
        )


def plot_side_by_side(
    surfaces: dict[str, pd.DataFrame],
    points: pd.DataFrame,
    metric: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path:
    workloads = list(surfaces.keys())
    n = len(workloads)
    ncols = min(n, 4)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(max(5.4 * ncols, 7.0), max(4.7 * nrows, 4.7)), constrained_layout=True)

    all_z = []
    for surface in surfaces.values():
        if not surface.empty:
            all_z.extend(surface[metric].astype(float).to_numpy())
    if args.surface_z_axis in ["auto", "shared"] and all_z:
        zmin = float(np.nanmin(all_z))
        zmax = float(np.nanmax(all_z))
        if np.isclose(zmin, zmax):
            zmax = zmin + 1e-9
    else:
        zmin = zmax = None

    mappable = None
    handles_for_legend = []
    for idx, workload in enumerate(workloads, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        surface = surfaces[workload]
        ax.set_title(workload, fontsize=10)
        ax.set_xlabel("Pbar kW/server", labelpad=7)
        ax.set_ylabel("R kW/server", labelpad=7)
        ax.set_zlabel(METRIC_LABELS.get(metric, metric), labelpad=7)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.tick_params(axis="both", labelsize=8)

        if surface.empty or len(surface) < 3:
            ax.text2D(0.05, 0.5, "Not enough surface points", transform=ax.transAxes)
            continue

        x = surface["Pbar_kw_per_server"].astype(float).to_numpy()
        y = surface["R_kw_per_server"].astype(float).to_numpy()
        z = surface[metric].astype(float).to_numpy()
        tri = Triangulation(x, y)
        if args.surface_z_axis == "per_workload":
            local_zmin = float(np.nanmin(z))
            local_zmax = float(np.nanmax(z))
            if np.isclose(local_zmin, local_zmax):
                local_zmax = local_zmin + 1e-9
        else:
            local_zmin, local_zmax = zmin, zmax
        surf = ax.plot_trisurf(
            tri,
            z,
            cmap="viridis",
            linewidth=0.15,
            antialiased=True,
            alpha=0.88,
            vmin=local_zmin,
            vmax=local_zmax,
        )
        mappable = surf

        # Optional context: all points from the point CSV for this workload.
        if args.plot_all_points_faint and not points.empty and "Workload_Name" in points.columns:
            allp = points[points["Workload_Name"].astype(str) == workload]
            if not allp.empty and metric in allp.columns:
                ax.scatter(
                    allp["Pbar_kw_per_server"].astype(float),
                    allp["R_kw_per_server"].astype(float),
                    allp[metric].astype(float),
                    marker="o",
                    s=18,
                    alpha=0.22,
                    color="gray",
                    depthshade=False,
                    label="Candidate points",
                )

        # Pass points overlay.
        if not points.empty and "Workload_Name" in points.columns:
            pp = points[points["Workload_Name"].astype(str) == workload].copy()
            if not pp.empty and metric in pp.columns:
                px = pp["Pbar_kw_per_server"].astype(float)
                py = pp["R_kw_per_server"].astype(float)
                pz = pp[metric].astype(float)
                sc = ax.scatter(
                    px,
                    py,
                    pz,
                    marker="*",
                    s=120,
                    color="red",
                    edgecolors="black",
                    linewidths=0.7,
                    depthshade=False,
                    label=args.points_label,
                )
                handles_for_legend.append(sc)
                # Light vertical stems to make the points easier to see on 3D surfaces.
                if len(px) <= 60:
                    stem_bottom = local_zmin if local_zmin is not None else float(np.nanmin(z))
                    for x0, y0, z0 in zip(px, py, pz):
                        ax.plot([x0, x0], [y0, y0], [stem_bottom, z0], color="black", alpha=0.35, linewidth=0.6)
                if args.annotate_points:
                    for _, row in pp.iterrows():
                        ax.text(
                            float(row["Pbar_kw_per_server"]),
                            float(row["R_kw_per_server"]),
                            float(row[metric]),
                            f"P={float(row['Pbar_kw_per_server']):.3f}\nR={float(row['R_kw_per_server']):.3f}",
                            fontsize=7,
                        )

    title_metric = METRIC_LABELS.get(metric, metric)
    util_text = f", U={args.utilization}" if args.aggregation_scope == "fixed_utilization" else ", all utilizations"
    title = (
        f"Side-by-side isometric P/R surfaces: {title_metric}\n"
        f"N={args.server_count}{util_text}, aggregation={args.agg}"
    )
    if args.points_csv or args.points_csv_list:
        title += f", overlay={args.point_filter} ({args.constraint_logic} constraints)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if mappable is not None and args.surface_z_axis in ["auto", "shared"]:
        fig.colorbar(mappable, ax=fig.axes, shrink=0.62, pad=0.02, label=title_metric)
    if handles_for_legend:
        # Put a figure-level legend so it does not repeat on every panel.
        fig.legend(handles=[handles_for_legend[0]], labels=[args.points_label], loc="lower center", ncol=1, frameon=True)

    metric_safe = safe_name(METRIC_LABELS.get(metric, metric))
    util_safe = safe_name(args.utilization if args.utilization is not None else "allU")
    point_suffix = "_with_constraint_points" if (args.points_csv or args.points_csv_list) else ""
    out_path = out_dir / f"side_by_side_isometric_{metric_safe}_N{args.server_count}_U{util_safe}_{args.agg}{point_suffix}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.show_3d:
        print("Note: --show-3d is ignored; this script saves PNGs only.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surface_df = load_surface_data(Path(args.results_csv), Path(args.diagnostics_csv) if args.diagnostics_csv else None, args)
    surface_df = filter_surface(surface_df, args)
    workloads = choose_workloads(surface_df, args.workload, args.workloads)
    metrics = choose_metrics(surface_df, args.metrics)

    raw_points = load_points(args)
    raw_points = filter_points(raw_points, args)
    if not raw_points.empty:
        # Keep only workloads being plotted.
        raw_points = raw_points[raw_points["Workload_Name"].astype(str).isin(workloads)].copy()

    # Save the exact points that were overlaid.
    overlay_summary_path = out_dir / "constraint_points_overlay_summary.csv"
    if not raw_points.empty:
        cols = [
            c for c in [
                "Point_Source_CSV", "Output_Prefix", "Output_Dir", "Workload_Name", "server_count", "utilization",
                "Pbar_kw_per_server", "R_kw_per_server", "Weight_Sample_ID",
                "Weight_0", "Weight_1", "Weight_2", "Weight_3",
                "Ctrack_Epsilon_90th", "QoS_Violation_Ratio", "Max_QoS_Delay_Probability",
                "QoS_Delay_Probabilities", "Simulator_RSR_Total_Cost",
                "Ctrack_Weighted_Cost", "Diagnostic_FlexDC_SoftPlus_QoS_Cost", "Diagnostic_FullPaperObjective_Cost",
                "Plot_Tracking_Pass", "Plot_QoS_Pass", "Plot_Both_Pass",
            ] if c in raw_points.columns
        ]
        raw_points[cols].to_csv(overlay_summary_path, index=False)
        print(f"Overlay points after filtering: {len(raw_points)}")
        print("Saved:", overlay_summary_path)
    else:
        print("No overlay points after filtering.")

    print("Selected workloads:")
    for w in workloads:
        print("  ", w)
    print("Selected metrics:")
    for m in metrics:
        print("  ", m)
    print("Surface rows after filtering:", len(surface_df))

    for metric in metrics:
        surfaces = {}
        for workload in workloads:
            surface = aggregate_surface(surface_df, workload, metric, args.agg)
            warn_expected_rows(surface, workload, metric, args.expected_rows_per_pr)
            surfaces[workload] = surface
            surface_csv = out_dir / (
                f"aggregated_surface_{safe_name(workload)}_{safe_name(metric)}_"
                f"N{args.server_count}_U{safe_name(args.utilization if args.utilization is not None else 'allU')}_{args.agg}.csv"
            )
            surface.to_csv(surface_csv, index=False)
        fig_path = plot_side_by_side(surfaces, raw_points, metric, args, out_dir)
        print("Saved:", fig_path)

    print("Done.")


if __name__ == "__main__":
    main()
