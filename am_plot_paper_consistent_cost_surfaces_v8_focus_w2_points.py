#!/usr/bin/env python3
"""
Focused W2 3D P/R surface plot with feasible-point overlay.

v8 purpose:
  * Plot ONE workload/utilization at a time instead of four side-by-side panels.
  * Open an interactive 3D matplotlib window when --show-3d is passed.
  * Overlay points that pass tracking + QoS constraints from a points CSV.
  * Place pass-point markers ON the displayed surface by default, not below it.

Default pass-both logic:
  Ctrack_Epsilon_90th <= 0.3 and QoS_Violation_Ratio <= 0.1

Default points CSV is the mini dense W2 sweep path from Achuthan's PC.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_POINTS_CSV = r"C:\Users\Achuthan Menon\Desktop\Research Work\flexdc-sim-main\src\peacsim\mini_dense_w2_local_sweep_all_rows.csv"

FOCUS_PRESETS = {
    "w2-lu": {
        "workload": "W2-short-qos5_4.5_4_3.5",
        "utilization": 0.60,
        "label": "W2-LU / Winf-LU, traditional signal",
    },
    "w2-hu": {
        "workload": "W2-short-qos5555",
        "utilization": 0.80,
        "label": "W2-HU / Winf-HU, traditional signal",
    },
    "custom": {
        "workload": None,
        "utilization": None,
        "label": "Custom workload",
    },
}

METRIC_LABELS = {
    "Simulator_RSR_Total_Cost": "M_RSR / simulator RSR cost",
    "Simulator_Power_Cost": "Power settlement cost",
    "Mtrack_Cost": "Monetary mean-tracking term",
    "Ctrack_Epsilon_90th": "p90 normalized tracking error",
    "Ctrack_Weighted_Cost": "Weighted tracking penalty",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "Weighted QoS penalty",
    "Diagnostic_FullPaperObjective_Cost": "FlexDC full paper objective",
    "QoS_Delay_Probability_Sum": "QoS probability sum",
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
        description="Focused interactive 3D W2 P/R surface plot with feasible-point markers."
    )
    parser.add_argument("--results-csv", required=True, help="Surface CSV: broad grid results or local all-rows CSV")
    parser.add_argument("--diagnostics-csv", default=None, help="Optional diagnostics CSV if surface metrics are not already in results CSV")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--focus",
        choices=["w2-lu", "w2-hu", "custom"],
        default="w2-lu",
        help="Convenience preset for the W2 workload/utilization to plot.",
    )
    parser.add_argument("--workload", default=None, help="Required only for --focus custom; overrides preset workload if provided.")
    parser.add_argument("--utilization", type=float, default=None, help="Overrides preset utilization if provided.")
    parser.add_argument("--server-count", type=int, default=1000)

    parser.add_argument(
        "--metrics",
        default="Diagnostic_FullPaperObjective_Cost",
        help="Comma-separated metrics to plot. Use one metric per command for easiest interactive viewing.",
    )
    parser.add_argument("--agg", choices=["mean", "min", "median", "max"], default="mean")
    parser.add_argument("--expected-rows-per-pr", type=int, default=None)

    parser.add_argument("--points-csv", default=DEFAULT_POINTS_CSV, help="Candidate/mini-dense points CSV to overlay")
    parser.add_argument(
        "--point-filter",
        choices=["both_pass", "tracking_pass", "qos_pass", "all"],
        default="both_pass",
        help="Which rows from --points-csv to overlay.",
    )
    parser.add_argument(
        "--constraint-logic",
        choices=["current", "maxprob", "both"],
        default="current",
        help="current uses QoS_Violation_Ratio; maxprob uses Max_QoS_Delay_Probability; both requires both QoS checks.",
    )
    parser.add_argument("--tracking-threshold", type=float, default=0.3)
    parser.add_argument("--qos-ratio-threshold", type=float, default=0.1)
    parser.add_argument("--qos-prob-threshold", type=float, default=0.1)

    parser.add_argument(
        "--point-z-mode",
        choices=["surface", "point_metric"],
        default="surface",
        help=(
            "surface: put stars on the displayed surface using interpolated surface z. "
            "point_metric: put stars at the candidate row's own metric value."
        ),
    )
    parser.add_argument(
        "--point-z-offset-frac",
        type=float,
        default=0.018,
        help="Small upward z offset as a fraction of z-range so stars are visible above the surface.",
    )
    parser.add_argument("--point-size", type=float, default=180)
    parser.add_argument("--plot-all-points-faint", action="store_true")
    parser.add_argument("--annotate-points", action="store_true")

    parser.add_argument("--surface-alpha", type=float, default=0.82)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-135.0)
    parser.add_argument("--fig-width", type=float, default=10.5)
    parser.add_argument("--fig-height", type=float, default=7.5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show-3d", action="store_true", help="Open interactive matplotlib 3D window after saving.")
    parser.add_argument("--no-save", action="store_true", help="Only show interactive window, do not save PNG.")

    parser.add_argument("--recompute-paper-objective", action="store_true", help="Recompute paper-form Ctrack/CQoS/full objective from raw columns.")
    parser.add_argument("--paper-psi", type=float, default=1.0)
    parser.add_argument("--paper-mu", type=float, default=10.0)
    parser.add_argument("--paper-gamma", type=float, default=0.3)
    parser.add_argument("--paper-beta", type=float, default=20.0)
    parser.add_argument("--paper-rho", type=float, default=2.0)
    parser.add_argument("--paper-delta", type=float, default=0.1)
    return parser.parse_args()


def configure_matplotlib(show_3d: bool):
    import matplotlib
    if show_3d:
        # Try a real interactive backend on Windows. If it fails, matplotlib will
        # fall back to whatever is available and plt.show() may still work.
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    return plt, Triangulation, LinearTriInterpolator


def split_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


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
    needed = {"Ctrack_Epsilon_90th", "Simulator_RSR_Total_Cost", "QoS_Delay_Probabilities"}
    if not needed.issubset(out.columns):
        return out

    eps = out["Ctrack_Epsilon_90th"].astype(float).to_numpy()
    out["Ctrack_Weighted_Cost"] = args.paper_psi * stable_softplus(args.paper_mu * (eps - args.paper_gamma))

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
    out["Diagnostic_FlexDC_SoftPlus_QoS_Cost"] = np.asarray(qos_cost, dtype=float)
    out["QoS_Delay_Probability_Sum"] = np.asarray(qos_sum, dtype=float)
    out["Diagnostic_FullPaperObjective_Cost"] = (
        out["Simulator_RSR_Total_Cost"].astype(float)
        + out["Ctrack_Weighted_Cost"].astype(float)
        + out["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].astype(float)
    )
    return out


def choose_merge_keys(results: pd.DataFrame, diagnostics: pd.DataFrame) -> Optional[list[str]]:
    for keys in MERGE_KEY_CANDIDATES:
        if all(k in results.columns and k in diagnostics.columns for k in keys):
            return keys
    return None


def load_surface_data(args: argparse.Namespace) -> pd.DataFrame:
    results = pd.read_csv(args.results_csv)
    if args.diagnostics_csv:
        diagnostics = pd.read_csv(args.diagnostics_csv)
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


def resolve_focus(args: argparse.Namespace) -> tuple[str, float, str]:
    preset = FOCUS_PRESETS[args.focus]
    workload = args.workload or preset["workload"]
    utilization = args.utilization if args.utilization is not None else preset["utilization"]
    label = preset["label"]
    if workload is None or utilization is None:
        raise ValueError("For --focus custom, pass both --workload and --utilization.")
    return str(workload), float(utilization), str(label)


def filter_one_workload(df: pd.DataFrame, workload: str, utilization: float, server_count: int) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Workload_Name"].astype(str) == workload]
    out = out[out["server_count"].astype(int) == int(server_count)]
    out = out[np.isclose(out["utilization"].astype(float), float(utilization))]
    return out.copy()


def aggregate_surface(df: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    group_cols = ["Pbar_kw_per_server", "R_kw_per_server"]
    grouped = df.groupby(group_cols, as_index=False)
    if agg == "mean":
        values = grouped[metric].mean()
    elif agg == "min":
        values = grouped[metric].min()
    elif agg == "median":
        values = grouped[metric].median()
    elif agg == "max":
        values = grouped[metric].max()
    else:
        raise ValueError(agg)
    counts = grouped.size().rename(columns={"size": "rows_per_pr"})
    return values.merge(counts, on=group_cols, how="left")


def load_points(args: argparse.Namespace, workload: str, utilization: float) -> pd.DataFrame:
    if not args.points_csv:
        return pd.DataFrame()
    points_path = Path(args.points_csv)
    if not points_path.exists():
        print(f"Warning: points CSV not found: {points_path}")
        return pd.DataFrame()
    points = pd.read_csv(points_path)
    points = add_max_qos_probability(points)
    if args.recompute_paper_objective:
        points = recompute_paper_objective(points, args)
    points = filter_one_workload(points, workload, utilization, args.server_count)
    if points.empty:
        return points

    if "Ctrack_Epsilon_90th" in points.columns:
        tracking_pass = points["Ctrack_Epsilon_90th"].astype(float) <= float(args.tracking_threshold)
    elif "Tracking_Pass" in points.columns:
        tracking_pass = points["Tracking_Pass"].astype(bool)
    else:
        tracking_pass = pd.Series(False, index=points.index)

    if args.constraint_logic == "current":
        if "QoS_Violation_Ratio" in points.columns:
            qos_pass = points["QoS_Violation_Ratio"].astype(float) <= float(args.qos_ratio_threshold)
        elif "QoS_Pass_CurrentLogic" in points.columns:
            qos_pass = points["QoS_Pass_CurrentLogic"].astype(bool)
        else:
            qos_pass = pd.Series(False, index=points.index)
    elif args.constraint_logic == "maxprob":
        qos_pass = points["Max_QoS_Delay_Probability"].astype(float) <= float(args.qos_prob_threshold)
    else:
        ratio_pass = points["QoS_Violation_Ratio"].astype(float) <= float(args.qos_ratio_threshold) if "QoS_Violation_Ratio" in points.columns else pd.Series(False, index=points.index)
        prob_pass = points["Max_QoS_Delay_Probability"].astype(float) <= float(args.qos_prob_threshold) if "Max_QoS_Delay_Probability" in points.columns else pd.Series(False, index=points.index)
        qos_pass = ratio_pass & prob_pass

    points["Plot_Tracking_Pass"] = tracking_pass
    points["Plot_QoS_Pass"] = qos_pass
    points["Plot_Both_Pass"] = tracking_pass & qos_pass

    if args.point_filter == "both_pass":
        points = points[points["Plot_Both_Pass"]].copy()
    elif args.point_filter == "tracking_pass":
        points = points[points["Plot_Tracking_Pass"]].copy()
    elif args.point_filter == "qos_pass":
        points = points[points["Plot_QoS_Pass"]].copy()
    elif args.point_filter == "all":
        pass
    return points


def nearest_surface_z(surface: pd.DataFrame, px: np.ndarray, py: np.ndarray, metric: str) -> np.ndarray:
    sx = surface["Pbar_kw_per_server"].astype(float).to_numpy()
    sy = surface["R_kw_per_server"].astype(float).to_numpy()
    sz = surface[metric].astype(float).to_numpy()
    out = []
    for x, y in zip(px, py):
        dist = (sx - x) ** 2 + (sy - y) ** 2
        out.append(float(sz[int(np.nanargmin(dist))]))
    return np.asarray(out, dtype=float)


def surface_interpolated_z(surface: pd.DataFrame, metric: str, px: np.ndarray, py: np.ndarray, Triangulation, LinearTriInterpolator) -> np.ndarray:
    x = surface["Pbar_kw_per_server"].astype(float).to_numpy()
    y = surface["R_kw_per_server"].astype(float).to_numpy()
    z = surface[metric].astype(float).to_numpy()
    if len(surface) < 3:
        return nearest_surface_z(surface, px, py, metric)
    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, z)
    zi = interp(px, py)
    zi = np.asarray(zi, dtype=float)
    bad = ~np.isfinite(zi)
    if bad.any():
        zi[bad] = nearest_surface_z(surface, px[bad], py[bad], metric)
    return zi


def plot_metric(surface: pd.DataFrame, points: pd.DataFrame, metric: str, workload: str, utilization: float, focus_label: str, args: argparse.Namespace, plt, Triangulation, LinearTriInterpolator) -> Optional[Path]:
    if metric not in surface.columns:
        print(f"Skipping missing metric: {metric}")
        return None
    if surface.empty or len(surface) < 3:
        raise ValueError("Not enough surface points after filtering. Check workload/utilization/server-count.")

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    x = surface["Pbar_kw_per_server"].astype(float).to_numpy()
    y = surface["R_kw_per_server"].astype(float).to_numpy()
    z = surface[metric].astype(float).to_numpy()
    tri = Triangulation(x, y)

    surf = ax.plot_trisurf(
        tri,
        z,
        cmap="viridis",
        linewidth=0.2,
        antialiased=True,
        alpha=float(args.surface_alpha),
    )
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=METRIC_LABELS.get(metric, metric))

    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    zrange = max(zmax - zmin, 1e-9)
    offset = float(args.point_z_offset_frac) * zrange

    if args.plot_all_points_faint and not points.empty:
        all_points = load_points_all_context(args, workload, utilization)
        if not all_points.empty:
            px_all = all_points["Pbar_kw_per_server"].astype(float).to_numpy()
            py_all = all_points["R_kw_per_server"].astype(float).to_numpy()
            pz_all = surface_interpolated_z(surface, metric, px_all, py_all, Triangulation, LinearTriInterpolator) + offset * 0.25
            ax.scatter(px_all, py_all, pz_all, marker="o", s=20, color="gray", alpha=0.25, depthshade=False, label="Candidate points")

    if not points.empty:
        px = points["Pbar_kw_per_server"].astype(float).to_numpy()
        py = points["R_kw_per_server"].astype(float).to_numpy()
        if args.point_z_mode == "surface" or metric not in points.columns:
            pz_base = surface_interpolated_z(surface, metric, px, py, Triangulation, LinearTriInterpolator)
        else:
            pz_base = points[metric].astype(float).to_numpy()
        pz = pz_base + offset
        ax.scatter(
            px,
            py,
            pz,
            marker="*",
            s=float(args.point_size),
            color="red",
            edgecolors="black",
            linewidths=0.8,
            depthshade=False,
            label=f"{args.point_filter} points",
        )
        for x0, y0, z0, z_surface in zip(px, py, pz, pz_base):
            ax.plot([x0, x0], [y0, y0], [zmin, z0], color="black", alpha=0.35, linewidth=0.8)
        if args.annotate_points:
            for _, row in points.iterrows():
                ax.text(
                    float(row["Pbar_kw_per_server"]),
                    float(row["R_kw_per_server"]),
                    float(surface_interpolated_z(surface, metric, np.array([row["Pbar_kw_per_server"]], dtype=float), np.array([row["R_kw_per_server"]], dtype=float), Triangulation, LinearTriInterpolator)[0] + offset),
                    f"P={float(row['Pbar_kw_per_server']):.4f}\nR={float(row['R_kw_per_server']):.4f}",
                    fontsize=8,
                )
    else:
        print("No overlay points after filtering.")

    ax.set_xlabel("Pbar kW/server", labelpad=10)
    ax.set_ylabel("R kW/server", labelpad=10)
    ax.set_zlabel(METRIC_LABELS.get(metric, metric), labelpad=10)
    ax.set_title(
        f"{focus_label}\n{METRIC_LABELS.get(metric, metric)} | N={args.server_count}, U={utilization}, aggregation={args.agg}",
        fontsize=13,
        fontweight="bold",
    )
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.legend(loc="upper left")

    out_path = None
    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metric_safe = safe_name(METRIC_LABELS.get(metric, metric))
        out_path = out_dir / f"focused_3d_{safe_name(workload)}_U{safe_name(utilization)}_{metric_safe}_{args.agg}_points.png"
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
        print("Saved:", out_path)
    if args.show_3d:
        print("Opening interactive 3D window. Rotate with the mouse; close the window to continue.")
        plt.show()
    else:
        plt.close(fig)
    return out_path


def load_points_all_context(args: argparse.Namespace, workload: str, utilization: float) -> pd.DataFrame:
    # Context points before pass/fail filtering, used only for faint optional overlay.
    points_path = Path(args.points_csv)
    if not points_path.exists():
        return pd.DataFrame()
    pts = pd.read_csv(points_path)
    pts = add_max_qos_probability(pts)
    if args.recompute_paper_objective:
        pts = recompute_paper_objective(pts, args)
    pts = filter_one_workload(pts, workload, utilization, args.server_count)
    return pts


def main() -> None:
    args = parse_args()
    workload, utilization, focus_label = resolve_focus(args)
    plt, Triangulation, LinearTriInterpolator = configure_matplotlib(args.show_3d)

    surface_df = load_surface_data(args)
    surface_df = filter_one_workload(surface_df, workload, utilization, args.server_count)
    if surface_df.empty:
        available = pd.read_csv(args.results_csv, nrows=1000)
        raise ValueError(
            f"No surface rows for workload={workload}, utilization={utilization}, server_count={args.server_count}."
        )

    metrics = split_list(args.metrics)
    if not metrics:
        metrics = ["Diagnostic_FullPaperObjective_Cost"]

    points = load_points(args, workload, utilization)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not points.empty:
        summary_cols = [
            c for c in [
                "Output_Prefix", "Output_Dir", "Workload_Name", "server_count", "utilization",
                "Pbar_kw_per_server", "R_kw_per_server", "Weight_Sample_ID",
                "Weight_0", "Weight_1", "Weight_2", "Weight_3",
                "Ctrack_Epsilon_90th", "QoS_Violation_Ratio", "Max_QoS_Delay_Probability",
                "QoS_Delay_Probabilities", "Simulator_RSR_Total_Cost", "Diagnostic_FullPaperObjective_Cost",
                "Plot_Tracking_Pass", "Plot_QoS_Pass", "Plot_Both_Pass",
            ] if c in points.columns
        ]
        summary_path = out_dir / f"focused_overlay_points_{safe_name(workload)}_U{safe_name(utilization)}.csv"
        points[summary_cols].to_csv(summary_path, index=False)
        print(f"Overlay points after filtering: {len(points)}")
        print("Saved:", summary_path)
    else:
        print("Overlay points after filtering: 0")

    print("Focused workload:", workload)
    print("Focused utilization:", utilization)
    print("Surface rows after filtering:", len(surface_df))
    print("Pbar range:", surface_df["Pbar_kw_per_server"].min(), "to", surface_df["Pbar_kw_per_server"].max())
    print("R range:", surface_df["R_kw_per_server"].min(), "to", surface_df["R_kw_per_server"].max())
    print("Metrics:", metrics)

    for metric in metrics:
        surface = aggregate_surface(surface_df, metric, args.agg)
        if args.expected_rows_per_pr is not None:
            bad = surface[surface["rows_per_pr"] != int(args.expected_rows_per_pr)]
            if not bad.empty:
                print(
                    f"Warning: {len(bad)} P/R groups have rows_per_pr != {args.expected_rows_per_pr}; "
                    f"observed={sorted(surface['rows_per_pr'].dropna().unique())}"
                )
        surface_csv = out_dir / f"focused_surface_{safe_name(workload)}_U{safe_name(utilization)}_{safe_name(metric)}_{args.agg}.csv"
        surface.to_csv(surface_csv, index=False)
        print("Saved:", surface_csv)
        plot_metric(surface, points, metric, workload, utilization, focus_label, args, plt, Triangulation, LinearTriInterpolator)

    print("Done.")


if __name__ == "__main__":
    main()
