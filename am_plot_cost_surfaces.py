"""Plot paper-consistent FlexDC cost surfaces with side-by-side workload panels.

v6 publication-label variant changes from v5:
  * Adds --metrics sa for recomputed FlexDC objective columns.
  * Uses presentation labels without exposing implementation-column prefixes in figure text.
  * Defaults P/R aggregation to mean across weight vectors.

v5 changes from the earlier plotting workflow:
  * Does not open interactive 3-view windows.
  * Accepts --show-3d for backward compatibility, but ignores it.
  * Adds side-by-side isometric 3D surface figures comparing workloads.

Expected inputs are the combined grid_search_results.csv and
combined grid_search_diagnostics.csv files produced by the FlexDC data wizard.
"""

from __future__ import annotations

import argparse
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
from matplotlib import cm
from matplotlib.tri import Triangulation


CORE_METRICS = [
    "Simulator_RSR_Total_Cost",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]

SA_METRICS = [
    "Simulator_RSR_Total_Cost",
    "SA_Ctrack_Weighted_Cost",
    "SA_Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "SA_Diagnostic_FullPaperObjective_Cost",
]

EXTRA_METRICS = [
    "Ctrack_Epsilon_90th",
    "QoS_Delay_Probability_Sum",
]

METRIC_LABELS = {
    "Simulator_RSR_Total_Cost": "M_RSR / simulator RSR cost",
    "Ctrack_Weighted_Cost": "FlexDC weighted tracking penalty",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "FlexDC weighted QoS penalty",
    "Diagnostic_FullPaperObjective_Cost": "FlexDC full paper objective",
    "Ctrack_Epsilon_90th": "p90 normalized tracking error",
    "QoS_Delay_Probability_Sum": "QoS delay probability sum",
    # Recomputed FlexDC objective columns using the configured optimization constants.
    # Presentation labels intentionally match the paper/objective terminology.
    "SA_Ctrack_Weighted_Cost": "FlexDC weighted tracking penalty",
    "SA_Diagnostic_FlexDC_SoftPlus_QoS_Cost": "FlexDC weighted QoS penalty",
    "SA_Diagnostic_FullPaperObjective_Cost": "FlexDC full paper objective",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side isometric P/R cost-surface plots for one or more workloads."
    )
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv")
    parser.add_argument("--diagnostics-csv", required=True, help="Combined grid_search_diagnostics.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures and aggregate CSVs")

    # Backward-compatible single-workload option. Use --workload all for all workloads.
    parser.add_argument("--workload", default="all", help="Workload name, comma-separated names, or 'all'.")
    parser.add_argument("--workloads", default=None, help="Optional comma-separated workload names. Overrides --workload.")

    parser.add_argument("--server-count", type=int, required=True)
    parser.add_argument(
        "--aggregation-scope",
        choices=["fixed_utilization", "all_utilizations"],
        default="fixed_utilization",
    )
    parser.add_argument("--utilization", type=float, default=None)
    parser.add_argument("--expected-rows-per-pr", type=int, default=None)
    parser.add_argument(
        "--agg",
        choices=["min", "mean", "median", "max"],
        default="mean",
        help="How to aggregate multiple weight-vector rows at each P/R pair. Default: mean.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(CORE_METRICS),
        help="Comma-separated metric columns to plot, or 'core', 'sa', 'all'.",
    )
    parser.add_argument(
        "--surface-z-axis",
        choices=["auto", "shared", "per_workload"],
        default="auto",
        help="Color/z scaling mode. auto/shared uses one metric scale across panels; per_workload scales each panel separately.",
    )
    parser.add_argument("--elev", type=float, default=25.0, help="3D elevation angle.")
    parser.add_argument("--azim", type=float, default=-135.0, help="3D azimuth angle.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show-3d", action="store_true", help="Accepted for compatibility; no windows are opened in v5.")
    return parser.parse_args()


def split_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    value = value.strip("_")
    return value or "value"


def choose_workloads(df: pd.DataFrame, workload_arg: str, workloads_arg: str | None) -> list[str]:
    available = sorted(str(x) for x in df["Workload_Name"].dropna().unique())
    request = workloads_arg if workloads_arg is not None else workload_arg

    if request is None or str(request).strip().lower() == "all":
        return available

    requested = split_list(request)
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(
            "Requested workload(s) not found: "
            + ", ".join(missing)
            + "\nAvailable workloads:\n  "
            + "\n  ".join(available)
        )
    return requested


def choose_metrics(df: pd.DataFrame, metric_arg: str) -> list[str]:
    if metric_arg.lower() == "core":
        candidates = CORE_METRICS
    elif metric_arg.lower() == "sa":
        candidates = SA_METRICS
    elif metric_arg.lower() == "all":
        candidates = CORE_METRICS + EXTRA_METRICS
    else:
        candidates = split_list(metric_arg)

    present = [m for m in candidates if m in df.columns]
    missing = [m for m in candidates if m not in df.columns]
    if missing:
        print("Warning: skipping missing metric columns:", missing)
    if not present:
        raise ValueError("No requested metric columns were found in merged data.")
    return present


def load_and_merge(results_csv: Path, diagnostics_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    # The combined files should have matching row order from the same wizard outputs.
    # Merge by a conservative set of keys where possible; otherwise fall back to row-wise concat.
    possible_keys = [
        "Iteration",
        "Weight_Sample_ID",
        "Workload_Name",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "server_count",
        "utilization",
    ]
    keys = [key for key in possible_keys if key in results.columns and key in diagnostics.columns]

    if keys:
        # Avoid duplicate non-key columns from diagnostics if already present in results.
        diag_extra_cols = keys + [c for c in diagnostics.columns if c not in results.columns]
        merged = results.merge(diagnostics[diag_extra_cols], on=keys, how="left", validate="one_to_one")
        if len(merged) != len(results):
            raise ValueError("Merged row count changed unexpectedly. Check combined CSV alignment.")
        return merged

    if len(results) != len(diagnostics):
        raise ValueError(
            "Results and diagnostics have different row counts and no common merge keys were found."
        )
    diag_extra = diagnostics[[c for c in diagnostics.columns if c not in results.columns]]
    return pd.concat([results.reset_index(drop=True), diag_extra.reset_index(drop=True)], axis=1)


def filter_data(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df[df["server_count"].astype(int) == int(args.server_count)].copy()

    if args.aggregation_scope == "fixed_utilization":
        if args.utilization is None:
            raise ValueError("--utilization is required when --aggregation-scope fixed_utilization")
        filtered = filtered[np.isclose(filtered["utilization"].astype(float), float(args.utilization))].copy()

    if filtered.empty:
        raise ValueError("No rows remain after server/utilization filtering.")
    return filtered


def aggregate_surface(df: pd.DataFrame, workload: str, metric: str, agg: str) -> pd.DataFrame:
    sub = df[df["Workload_Name"].astype(str) == workload].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Pbar_kw_per_server", "R_kw_per_server", metric, "rows_per_pr"])

    group_cols = ["Pbar_kw_per_server", "R_kw_per_server"]
    grouped = sub.groupby(group_cols, as_index=False)
    if agg == "min":
        agg_df = grouped[metric].min()
    elif agg == "mean":
        agg_df = grouped[metric].mean()
    elif agg == "median":
        agg_df = grouped[metric].median()
    elif agg == "max":
        agg_df = grouped[metric].max()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    counts = grouped.size().rename(columns={"size": "rows_per_pr"})
    return agg_df.merge(counts, on=group_cols, how="left")


def warn_expected_rows(surface_df: pd.DataFrame, workload: str, metric: str, expected_rows_per_pr: int | None) -> None:
    if expected_rows_per_pr is None or surface_df.empty:
        return
    bad = surface_df[surface_df["rows_per_pr"] != expected_rows_per_pr]
    if not bad.empty:
        print(
            f"Warning: {workload} / {metric}: {len(bad)} P/R groups do not have "
            f"expected rows_per_pr={expected_rows_per_pr}. "
            f"Observed counts: {sorted(surface_df['rows_per_pr'].unique())}"
        )


def plot_side_by_side(
    surfaces: dict[str, pd.DataFrame],
    metric: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path:
    workloads = list(surfaces.keys())
    n = len(workloads)
    if n == 0:
        raise ValueError("No workloads to plot.")

    # Keep panels readable.
    ncols = min(n, 4)
    nrows = int(math.ceil(n / ncols))
    fig_w = max(5.0 * ncols, 7.0)
    fig_h = max(4.4 * nrows, 4.4)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    all_z = []
    for df in surfaces.values():
        if not df.empty:
            all_z.extend(df[metric].to_numpy(dtype=float))
    if all_z and args.surface_z_axis in ["auto", "shared"]:
        zmin, zmax = float(np.nanmin(all_z)), float(np.nanmax(all_z))
        if np.isclose(zmin, zmax):
            zmax = zmin + 1e-9
    else:
        zmin = zmax = None

    mappable = None
    for idx, workload in enumerate(workloads, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        surf = surfaces[workload]
        if surf.empty or len(surf) < 3:
            ax.text2D(0.05, 0.5, "Not enough points", transform=ax.transAxes)
            ax.set_title(workload)
            continue

        x = surf["Pbar_kw_per_server"].to_numpy(dtype=float)
        y = surf["R_kw_per_server"].to_numpy(dtype=float)
        z = surf[metric].to_numpy(dtype=float)

        tri = Triangulation(x, y)
        if args.surface_z_axis == "per_workload":
            local_zmin, local_zmax = float(np.nanmin(z)), float(np.nanmax(z))
            if np.isclose(local_zmin, local_zmax):
                local_zmax = local_zmin + 1e-9
        else:
            local_zmin, local_zmax = zmin, zmax

        plot = ax.plot_trisurf(
            tri,
            z,
            cmap="viridis",
            linewidth=0.15,
            antialiased=True,
            alpha=0.95,
            vmin=local_zmin,
            vmax=local_zmax,
        )
        mappable = plot
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(workload, fontsize=10)
        ax.set_xlabel("Pbar kW/server", labelpad=7)
        ax.set_ylabel("R kW/server", labelpad=7)
        ax.set_zlabel(METRIC_LABELS.get(metric, metric), labelpad=6)
        ax.tick_params(axis="both", labelsize=8)

    title_metric = METRIC_LABELS.get(metric, metric)
    util_text = f", U={args.utilization}" if args.aggregation_scope == "fixed_utilization" else ", all utilizations"
    fig.suptitle(
        f"Side-by-side isometric P/R surfaces: {title_metric}\n"
        f"N={args.server_count}{util_text}, aggregation={args.agg}",
        fontsize=14,
        fontweight="bold",
    )

    if mappable is not None and args.surface_z_axis in ["auto", "shared"]:
        fig.colorbar(mappable, ax=fig.axes, shrink=0.62, pad=0.02, label=title_metric)

    metric_safe = safe_name(METRIC_LABELS.get(metric, metric))
    util_safe = safe_name(args.utilization if args.utilization is not None else "allU")
    out_path = out_dir / f"side_by_side_isometric_{metric_safe}_N{args.server_count}_U{util_safe}_{args.agg}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.show_3d:
        print("Note: --show-3d is ignored in v5. No interactive windows will be opened.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(Path(args.results_csv), Path(args.diagnostics_csv))
    df = filter_data(df, args)
    workloads = choose_workloads(df, args.workload, args.workloads)
    metrics = choose_metrics(df, args.metrics)

    print("Selected workloads:")
    for w in workloads:
        print("  ", w)
    print("Selected metrics:")
    for m in metrics:
        print("  ", m)
    print(f"Rows after filtering: {len(df)}")

    for metric in metrics:
        surfaces: dict[str, pd.DataFrame] = {}
        for workload in workloads:
            surf = aggregate_surface(df, workload, metric, args.agg)
            warn_expected_rows(surf, workload, metric, args.expected_rows_per_pr)
            surfaces[workload] = surf

            csv_name = (
                f"aggregated_surface_{safe_name(workload)}_{safe_name(metric)}_"
                f"N{args.server_count}_U{safe_name(args.utilization if args.utilization is not None else 'allU')}_{args.agg}.csv"
            )
            surf.to_csv(out_dir / csv_name, index=False)

        fig_path = plot_side_by_side(surfaces, metric, args, out_dir)
        print("Saved:", fig_path)

    print("Done.")


if __name__ == "__main__":
    main()
