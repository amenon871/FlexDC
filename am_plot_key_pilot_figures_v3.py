"""
Minimal, presentation-quality figures for the FlexDC pilot dataset.

This script intentionally makes only four useful raw-output figures for one
controlled workload / server-count / utilization slice:

  01_tracking_response_vs_R_by_Pbar.png
      Raw Mtrack_Cost and Mtrack_Error_MeanAbs_Normalized versus R.

  02_monetary_response_vs_R_by_Pbar.png
      Simulator_Power_Cost and Simulator_RSR_Total_Cost versus R.

  03_qos_response_vs_R_by_Pbar.png
      QoS_Delay_Probability_Sum and QoS_Violation_Ratio versus R.

  04_interactive_3D_Mtrack_Cost.png
      Rotatable raw Mtrack_Cost surface over physical (Pbar, R).

Every line fixes Pbar and varies R. Values are median across the accepted
weight vectors for each P/R point. The script defaults to the controlled slice
with the largest observed raw Mtrack_Cost range so the first figures are
actually informative.

Run from the FlexDC repository root:

python .\\am_plot_key_pilot_figures_v3.py `
  --results-csv .\\combined_output_final_aqa_s250_1000_u060_080_prchunk10\\final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv `
  --out-dir .\\key_pilot_figures `
  --show-3d

Optional explicit slice:

python .\\am_plot_key_pilot_figures_v3.py `
  --results-csv .\\...\\final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv `
  --workload "W2-short-qos2-2.5-2.5-3" `
  --server-count 1000 `
  --utilization 0.6 `
  --out-dir .\\key_pilot_figures `
  --show-3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


REQUIRED_COLUMNS = [
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
    "Weight_Sample_ID",
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "Mtrack_Error_MeanAbs_Normalized",
    "Simulator_RSR_Total_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
]

NUMERIC_COLUMNS = [
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
    "Weight_Sample_ID",
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "Mtrack_Error_MeanAbs_Normalized",
    "Simulator_RSR_Total_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
]


METRIC_LABELS = {
    "Mtrack_Cost": r"Raw monetary tracking cost $M_{\mathrm{track}}$\n(simulator monetary units)",
    "Mtrack_Error_MeanAbs_Normalized": "Mean absolute normalized tracking error\n(unitless)",
    "Simulator_Power_Cost": r"Power/reserve monetary term $M_{\mathrm{power}}$\n(simulator monetary units)",
    "Simulator_RSR_Total_Cost": r"Raw RSR monetary cost $M_{\mathrm{RSR}}$\n(simulator monetary units)",
    "QoS_Delay_Probability_Sum": r"Raw delay-probability sum $\sum_j P^{D,j}$",
    "QoS_Violation_Ratio": "QoS violation ratio\n(fraction of job types)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create key FlexDC pilot-dataset figures.")
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv path.")
    parser.add_argument("--out-dir", default="key_pilot_figures", help="Directory for saved figures.")
    parser.add_argument("--workload", default=None, help="Workload_Name. Omit with the two settings below for auto-selection.")
    parser.add_argument("--server-count", type=int, default=None, help="Server count for the controlled slice.")
    parser.add_argument("--utilization", type=float, default=None, help="Utilization for the controlled slice.")
    parser.add_argument("--agg", choices=["median", "mean"], default="median", help="Weight aggregation used only for visualization.")
    parser.add_argument("--num-pbar-lines", type=int, default=5, help="Number of readable representative Pbar lines. Default: 5.")
    parser.add_argument(
        "--pbar-values",
        default=None,
        help="Optional comma-separated Pbar values to plot exactly, e.g. '0.1782,0.247867,0.317533'.",
    )
    parser.add_argument("--all-pbars", action="store_true", help="Plot all Pbar lines instead of representative lines.")
    parser.add_argument(
        "--three-d-metric",
        choices=["Mtrack_Cost", "Simulator_RSR_Total_Cost", "Simulator_Power_Cost"],
        default="Mtrack_Cost",
        help="Raw metric used in the rotatable 3D view.",
    )
    parser.add_argument("--show-3d", action="store_true", help="Open the rotatable 3D Matplotlib window after saving it.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    fallback = Path.cwd() / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find: {path_text}")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {missing}")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate_slice(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    metrics = [
        "Mtrack_Cost",
        "Mtrack_Error_MeanAbs_Normalized",
        "Simulator_Power_Cost",
        "Simulator_RSR_Total_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
    ]
    return (
        df.groupby(["Pbar_kw_per_server", "R_kw_per_server"], as_index=False)[metrics]
        .agg(agg)
        .sort_values(["Pbar_kw_per_server", "R_kw_per_server"])
        .reset_index(drop=True)
    )


def select_best_slice(df: pd.DataFrame, agg: str) -> tuple[str, int, float]:
    candidates = []
    for (workload, server_count, utilization), sub in df.groupby(
        ["Workload_Name", "server_count", "utilization"], sort=True
    ):
        grid = aggregate_slice(sub, agg)
        score = float(grid["Mtrack_Cost"].max() - grid["Mtrack_Cost"].min())
        candidates.append((score, str(workload), int(server_count), float(utilization)))
    if not candidates:
        raise ValueError("No valid controlled slices were found.")
    _, workload, server_count, utilization = max(candidates, key=lambda x: x[0])
    return workload, server_count, utilization


def choose_slice(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str, int, float, bool]:
    supplied = [args.workload is not None, args.server_count is not None, args.utilization is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide all of --workload, --server-count, and --utilization, or omit all three for automatic selection.")

    auto_selected = not any(supplied)
    if auto_selected:
        workload, server_count, utilization = select_best_slice(df, args.agg)
    else:
        workload = str(args.workload)
        server_count = int(args.server_count)
        utilization = float(args.utilization)

    sub = df[
        (df["Workload_Name"] == workload)
        & (df["server_count"].astype(int) == server_count)
        & np.isclose(df["utilization"].astype(float), utilization, atol=1e-9)
    ].copy()
    if sub.empty:
        raise ValueError("The requested controlled slice has no rows.")
    return sub, workload, server_count, utilization, auto_selected


def validate(sub: pd.DataFrame, grid: pd.DataFrame) -> None:
    max_identity_error = (
        sub["Simulator_RSR_Total_Cost"]
        - (sub["Simulator_Power_Cost"] + sub["Mtrack_Cost"])
    ).abs().max()
    negative_mtrack = int((sub["Mtrack_Cost"] < -1e-12).sum())
    counts = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"]).size()

    print("\nValidation")
    print("----------")
    print(f"Selected rows: {len(sub):,}")
    print(f"P/R points: {len(grid):,}")
    print(f"Accepted weights per P/R point: {sorted(counts.unique().tolist())}")
    print(f"Negative raw Mtrack_Cost rows: {negative_mtrack}")
    print(f"max |M_RSR - (M_power + M_track)|: {max_identity_error:.3e}")

    if negative_mtrack:
        raise ValueError("Raw Mtrack_Cost should not be negative. Stop and inspect the results CSV.")
    if max_identity_error > 1e-8:
        raise ValueError("Simulator_RSR_Total_Cost does not equal Power + Mtrack. Stop and inspect the results CSV.")


def pbar_sensitivity_ranges(grid: pd.DataFrame) -> pd.DataFrame:
    return (
        grid.groupby("Pbar_kw_per_server", as_index=False)["Mtrack_Cost"]
        .agg(lambda values: float(values.max() - values.min()))
        .rename(columns={"Mtrack_Cost": "mtrack_range"})
        .sort_values("mtrack_range", ascending=False)
    )


def select_pbars(grid: pd.DataFrame, args: argparse.Namespace) -> list[float]:
    available = np.sort(grid["Pbar_kw_per_server"].dropna().unique()).astype(float)

    if args.pbar_values:
        requested = [float(value.strip()) for value in args.pbar_values.split(",") if value.strip()]
        selected = []
        for value in requested:
            closest = float(available[np.argmin(np.abs(available - value))])
            if not np.isclose(value, closest, atol=1e-6):
                raise ValueError(f"Requested Pbar={value} is unavailable. Available: {available.tolist()}")
            selected.append(closest)
        return sorted(set(selected))

    if args.all_pbars or args.num_pbar_lines >= len(available):
        return available.tolist()

    # Select Pbar values whose raw Mtrack response changes most across R.
    # This makes the line figure informative rather than showing many flat curves.
    ranked = pbar_sensitivity_ranges(grid).head(max(1, args.num_pbar_lines))
    return sorted(ranked["Pbar_kw_per_server"].astype(float).tolist())


def context_text(workload: str, server_count: int, utilization: float, agg: str) -> str:
    return f"{workload} | {server_count:,} servers | utilization = {utilization:.2f} | {agg} across accepted weights"


def label_for_pbar(pbar: float) -> str:
    return rf"$\bar{{P}}={pbar:.3f}$"


def plot_metric_lines(ax, grid: pd.DataFrame, pbar_values: list[float], metric: str) -> None:
    for pbar in pbar_values:
        line = grid[np.isclose(grid["Pbar_kw_per_server"], pbar, atol=1e-9)].sort_values("R_kw_per_server")
        ax.plot(
            line["R_kw_per_server"],
            line[metric],
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            label=label_for_pbar(pbar),
        )


def style_axis(ax, title: str, y_label: str, y_limits: tuple[float, float] | None = None) -> None:
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel(r"Reserve bid $R$ (kW/server)", fontsize=14, labelpad=8)
    ax.set_ylabel(y_label, fontsize=13, labelpad=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25, linewidth=0.7)
    if y_limits:
        ax.set_ylim(*y_limits)


def add_legend(fig, axes, pbar_values: list[float]) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=r"Representative baseline bids $\bar{P}$ (kW/server)",
        loc="center left",
        bbox_to_anchor=(0.79, 0.50),
        frameon=True,
        fontsize=11,
        title_fontsize=11,
    )


def save_two_panel(
    grid: pd.DataFrame,
    pbar_values: list[float],
    metric_a: str,
    metric_b: str,
    title_a: str,
    title_b: str,
    overall_title: str,
    context: str,
    footnote: str,
    output_path: Path,
    y_limits_b: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 11.5), sharex=False)

    plot_metric_lines(axes[0], grid, pbar_values, metric_a)
    style_axis(axes[0], title_a, METRIC_LABELS[metric_a])

    plot_metric_lines(axes[1], grid, pbar_values, metric_b)
    style_axis(axes[1], title_b, METRIC_LABELS[metric_b], y_limits=y_limits_b)

    fig.suptitle(overall_title, fontsize=22, y=0.98)
    fig.text(0.5, 0.935, context, ha="center", fontsize=13)
    fig.text(0.40, 0.035, footnote, ha="center", fontsize=11)
    add_legend(fig, axes, pbar_values)
    fig.subplots_adjust(left=0.12, right=0.75, top=0.88, bottom=0.10, hspace=0.36)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_3d(
    grid: pd.DataFrame,
    metric: str,
    title_context: str,
    output_path: Path,
    show: bool,
) -> None:
    x = grid["Pbar_kw_per_server"].to_numpy(dtype=float)
    y = grid["R_kw_per_server"].to_numpy(dtype=float)
    z = grid[metric].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_trisurf(Triangulation(x, y), z, linewidth=0.25, antialiased=True, alpha=0.88)
    ax.scatter(x, y, z, s=18, depthshade=False)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label=METRIC_LABELS[metric])

    ax.set_xlabel(r"Baseline power bid $\bar{P}$ (kW/server)", fontsize=12, labelpad=14)
    ax.set_ylabel(r"Reserve bid $R$ (kW/server)", fontsize=12, labelpad=14)
    ax.set_zlabel(METRIC_LABELS[metric], fontsize=11, labelpad=16)
    ax.view_init(elev=26, azim=-135)
    ax.set_title(f"Raw {metric} over the feasible physical P/R region\n{title_context}", fontsize=17, pad=20)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        try:
            fig.canvas.manager.set_window_title(f"Rotate this view: {metric}")
        except Exception:
            pass
        print("\n3D window is open. Drag to rotate, choose a screenshot angle, then close it to finish.")
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    results = load_results(resolve_path(args.results_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub, workload, server_count, utilization, automatic = choose_slice(results, args)
    grid = aggregate_slice(sub, args.agg)
    validate(sub, grid)

    pbar_values = select_pbars(grid, args)
    context = context_text(workload, server_count, utilization, args.agg)

    print("\nFigure selection")
    print("----------------")
    print(f"Workload: {workload}")
    print(f"Server count: {server_count}")
    print(f"Utilization: {utilization:.2f}")
    print(f"Automatic slice selection: {automatic}")
    print(f"Pbar lines: {[round(value, 6) for value in pbar_values]}")

    outputs = []
    tracking_path = out_dir / "01_tracking_response_vs_R_by_Pbar.png"
    save_two_panel(
        grid,
        pbar_values,
        "Mtrack_Cost",
        "Mtrack_Error_MeanAbs_Normalized",
        r"Raw monetary tracking cost versus $R$",
        r"Mean absolute normalized tracking error versus $R$",
        r"Tracking response to reserve bid $R$",
        context,
        "The first panel is the raw nonnegative simulator monetary term. The second panel shows the normalized error component that contributes to it.",
        tracking_path,
    )
    outputs.append(tracking_path)

    monetary_path = out_dir / "02_monetary_response_vs_R_by_Pbar.png"
    save_two_panel(
        grid,
        pbar_values,
        "Simulator_Power_Cost",
        "Simulator_RSR_Total_Cost",
        r"Power/reserve monetary term versus $R$",
        r"Total raw RSR monetary cost versus $R$",
        r"Raw monetary response to reserve bid $R$",
        context,
        r"$M_{\mathrm{RSR}} = M_{\mathrm{power}} + M_{\mathrm{track}}$. The plotted values are raw simulator monetary terms.",
        monetary_path,
    )
    outputs.append(monetary_path)

    qos_path = out_dir / "03_qos_response_vs_R_by_Pbar.png"
    save_two_panel(
        grid,
        pbar_values,
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        r"Raw delay-probability sum versus $R$",
        r"QoS violation ratio versus $R$",
        r"Raw QoS response to reserve bid $R$",
        context,
        "The delay-probability sum is a raw summary across job types; the violation ratio is the fraction of job types exceeding their QoS threshold.",
        qos_path,
        y_limits_b=(-0.05, 1.05),
    )
    outputs.append(qos_path)

    metric = args.three_d_metric
    surface_path = out_dir / f"04_interactive_3D_{metric}.png"
    plot_interactive_3d(grid, metric, context, surface_path, args.show_3d)
    outputs.append(surface_path)

    print("\nCreated files")
    print("-------------")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
