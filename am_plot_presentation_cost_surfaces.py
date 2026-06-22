"""
Presentation-quality FlexDC cost figures for one controlled dataset slice.

Creates exactly four line plots and four rotatable 3D surfaces:

  01_Mtrack_Cost_vs_R_by_Pbar.png
  02_Simulator_Power_Cost_vs_R_by_Pbar.png
  03_Ctrack_Weighted_Cost_vs_R_by_Pbar.png
  04_Diagnostic_FlexDC_SoftPlus_QoS_Cost_vs_R_by_Pbar.png
  05_3D_Mtrack_Cost.png
  06_3D_Simulator_Power_Cost.png
  07_3D_Ctrack_Weighted_Cost.png
  08_3D_Diagnostic_FlexDC_SoftPlus_QoS_Cost.png
  00_metric_scale_summary.csv

The script deliberately keeps raw monetary outputs separate from the paper-style
tracking/QoS diagnostics:

  Mtrack_Cost = raw simulator monetary tracking term.
  Simulator_Power_Cost = raw power/reserve monetary term.
  Ctrack_Weighted_Cost = psi * SoftPlus(mu * (epsilon_90 - gamma)).
  Diagnostic_FlexDC_SoftPlus_QoS_Cost = beta * sum_j SoftPlus(rho * (P_j - delta)).

Every figure fixes workload, server count, and utilization. For each physical
(Pbar, R) point, it uses the requested summary across accepted weight vectors
(default: median). Thus, the figures show the P/R response while summarizing
workload-weight variation only for readability.

Run from the FlexDC repository root, for example:

python .\\am_plot_presentation_cost_surfaces.py `
  --results-csv .\\combined_output_final_aqa_s250_1000_u060_080_prchunk10\\final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv `
  --diagnostics-csv .\\combined_output_final_aqa_s250_1000_u060_080_prchunk10\\final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_diagnostics.csv `
  --workload "W2-short-qos2-2.5-2.5-3" `
  --server-count 1000 `
  --utilization 0.60 `
  --out-dir .\\presentation_cost_figures `
  --show-3d

Close each 3D figure window to advance to the next rotatable surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import numpy as np
import pandas as pd


JOIN_KEYS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
]

RESULT_REQUIRED = JOIN_KEYS + [
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "Simulator_RSR_Total_Cost",
    "QoS_Delay_Probabilities",
]

DIAGNOSTIC_REQUIRED = JOIN_KEYS + [
    "Ctrack_Epsilon_90th",
    "Ctrack_Gamma",
    "Ctrack_Mu",
    "Ctrack_Psi",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
]

METRICS = [
    {
        "column": "Mtrack_Cost",
        "short": "Mtrack_Cost",
        "title": r"Raw monetary tracking cost $M_{\mathrm{track}}$",
        "ylabel": r"Raw monetary tracking cost $M_{\mathrm{track}}$\n(simulator monetary units)",
        "family": "Raw simulator monetary term",
        "formula": r"$M_{\mathrm{track}} = \sum_{\mathrm{phases}} \pi^E R\,\mathrm{mean}(|e|)H$",
        "footnote": "Raw simulator monetary term. It is nonnegative; changes reflect both reserve magnitude and tracking error.",
        "line_scale": "log",
    },
    {
        "column": "Simulator_Power_Cost",
        "short": "Simulator_Power_Cost",
        "title": r"Power/reserve monetary term $M_{\mathrm{power}}$",
        "ylabel": r"Power/reserve monetary term $M_{\mathrm{power}}$\n(simulator monetary units)",
        "family": "Raw simulator monetary term",
        "formula": r"$M_{\mathrm{power}} = (\pi^P P - \pi^R R)H$",
        "footnote": "Raw simulator monetary term. At fixed Pbar, increasing R increases reserve revenue and therefore lowers this value.",
        "line_scale": "linear",
    },
    {
        "column": "Ctrack_Weighted_Cost",
        "short": "Ctrack_Weighted_Cost",
        "title": r"Paper-style weighted tracking penalty $\psi C_{\mathrm{track}}$",
        "ylabel": r"Weighted tracking penalty $\psi C_{\mathrm{track}}$\n(objective / penalty units)",
        "family": "Paper-style diagnostic penalty",
        "formula": r"$\psi C_{\mathrm{track}} = \psi\,\mathrm{SoftPlus}(\mu(\epsilon^r-\gamma))$",
        "footnote": "Diagnostic penalty, not a monetary simulator output. It reacts to the 90th-percentile normalized tracking error.",
        "line_scale": "log",
    },
    {
        "column": "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "short": "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "title": r"Paper-style weighted QoS penalty $\beta C_{\mathrm{QoS}}$",
        "ylabel": r"Weighted QoS penalty $\beta C_{\mathrm{QoS}}$\n(objective / penalty units)",
        "family": "Paper-style diagnostic penalty",
        "formula": r"$\beta C_{\mathrm{QoS}} = \beta\sum_j\mathrm{SoftPlus}(\rho(P^{D,j}-\delta))$",
        "footnote": "Diagnostic penalty, not a monetary simulator output. It aggregates QoS pressure over job types.",
        "line_scale": "linear",
    },
]

NUMERIC_COLUMNS = {
    "Iteration",
    "Weight_Sample_ID",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "Simulator_RSR_Total_Cost",
    "Ctrack_Epsilon_90th",
    "Ctrack_Gamma",
    "Ctrack_Mu",
    "Ctrack_Psi",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create key FlexDC cost lines and interactive 3D surfaces.")
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv path.")
    parser.add_argument("--diagnostics-csv", required=True, help="Combined grid_search_diagnostics.csv path.")
    parser.add_argument("--out-dir", default="presentation_cost_figures", help="Output directory.")
    parser.add_argument("--workload", default="W2-short-qos2-2.5-2.5-3", help="Controlled workload name.")
    parser.add_argument("--server-count", type=int, default=1000, help="Controlled server count.")
    parser.add_argument("--utilization", type=float, default=0.60, help="Controlled utilization.")
    parser.add_argument("--agg", choices=["median", "mean"], default="median", help="Weight summary used only for visualization.")
    parser.add_argument(
        "--num-pbar-lines",
        type=int,
        default=10,
        help="Number of evenly spaced Pbar lines to show. Default 10 shows all current Pbar levels.",
    )
    parser.add_argument(
        "--z-axis-mode",
        choices=["zero", "data"],
        default="zero",
        help="'zero' begins each surface at 0; 'data' zooms to the selected metric's observed range.",
    )
    parser.add_argument(
        "--qos-beta",
        type=float,
        default=0.5,
        help="Beta used to verify Diagnostic_FlexDC_SoftPlus_QoS_Cost. Default matches gradient_descent.ini.",
    )
    parser.add_argument(
        "--qos-rho",
        type=float,
        default=2.0,
        help="Rho used to verify Diagnostic_FlexDC_SoftPlus_QoS_Cost. Default matches gradient_descent.ini.",
    )
    parser.add_argument(
        "--qos-threshold",
        type=float,
        default=0.1,
        help="QoS delay-probability threshold delta used in diagnostics. Default matches gradient_descent.ini.",
    )
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Open each 3D surface in turn so it can be rotated before closing.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate
    fallback = Path.cwd() / candidate.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find: {path_text}")


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}")


def load_and_merge(results_path: Path, diagnostics_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    diagnostics = pd.read_csv(diagnostics_path)
    require_columns(results, RESULT_REQUIRED, "Results")
    require_columns(diagnostics, DIAGNOSTIC_REQUIRED, "Diagnostics")

    for frame in (results, diagnostics):
        for column in NUMERIC_COLUMNS.intersection(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    diag_keep = JOIN_KEYS + [
        "Ctrack_Epsilon_90th",
        "Ctrack_Gamma",
        "Ctrack_Mu",
        "Ctrack_Psi",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    ]
    merged = results.merge(
        diagnostics[diag_keep],
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(results) or len(merged) != len(diagnostics):
        raise ValueError(
            f"Results/diagnostics merge mismatch: results={len(results)}, diagnostics={len(diagnostics)}, merged={len(merged)}"
        )
    return merged


def parse_probability_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        raise ValueError(f"Cannot parse QoS_Delay_Probabilities value: {value!r}")
    array = np.asarray(parsed, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"Expected a non-empty one-dimensional QoS probability array, got: {parsed!r}")
    return array


def select_slice(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    sub = df[
        (df["Workload_Name"] == args.workload)
        & (df["server_count"].astype(int) == int(args.server_count))
        & np.isclose(df["utilization"].astype(float), float(args.utilization), atol=1e-9)
    ].copy()
    if sub.empty:
        available = (
            df[["Workload_Name", "server_count", "utilization"]]
            .drop_duplicates()
            .sort_values(["Workload_Name", "server_count", "utilization"])
        )
        raise ValueError(
            "Requested controlled slice has no rows. Available slices:\n"
            + available.to_string(index=False)
        )
    return sub


def aggregate_slice(sub: pd.DataFrame, agg: str) -> pd.DataFrame:
    metric_columns = [spec["column"] for spec in METRICS]
    return (
        sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"], as_index=False)[metric_columns]
        .agg(agg)
        .sort_values(["Pbar_kw_per_server", "R_kw_per_server"])
        .reset_index(drop=True)
    )


def validate_metric_lineage(sub: pd.DataFrame, args: argparse.Namespace) -> None:
    rsr_error = (
        sub["Simulator_RSR_Total_Cost"]
        - (sub["Simulator_Power_Cost"] + sub["Mtrack_Cost"])
    ).abs().max()
    if rsr_error > 1e-8:
        raise ValueError(
            f"Raw monetary identity failed: max |M_RSR - (M_power + M_track)| = {rsr_error:.3e}"
        )

    if int((sub["Mtrack_Cost"] < -1e-12).sum()) != 0:
        raise ValueError("Mtrack_Cost contains negative values, which should not occur for the raw monetary tracking term.")

    ctrack_expected = (
        sub["Ctrack_Psi"]
        * np.logaddexp(0, sub["Ctrack_Mu"] * (sub["Ctrack_Epsilon_90th"] - sub["Ctrack_Gamma"]))
    )
    ctrack_error = (ctrack_expected - sub["Ctrack_Weighted_Cost"]).abs().max()
    if ctrack_error > 1e-8:
        raise ValueError(f"Ctrack diagnostic identity failed: max error = {ctrack_error:.3e}")

    qos_expected = sub["QoS_Delay_Probabilities"].map(
        lambda value: args.qos_beta * float(np.logaddexp(0, args.qos_rho * (parse_probability_array(value) - args.qos_threshold)).sum())
    )
    qos_error = (qos_expected - sub["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]).abs().max()
    if qos_error > 1e-8:
        raise ValueError(
            "QoS diagnostic identity failed. "
            f"max error={qos_error:.3e}. Check --qos-beta, --qos-rho, and --qos-threshold against the gradient configuration."
        )

    counts = sub.groupby(["Pbar_kw_per_server", "R_kw_per_server"]).size()
    print("\nValidation")
    print("----------")
    print(f"Rows in selected slice: {len(sub):,}")
    print(f"P/R points: {len(counts):,}")
    print(f"Accepted weights per P/R point: {sorted(counts.unique().tolist())}")
    print(f"max |M_RSR - (M_power + M_track)|: {rsr_error:.3e}")
    print(f"max Ctrack reconstruction error: {ctrack_error:.3e}")
    print(f"max Cqos reconstruction error: {qos_error:.3e}")


def choose_pbars(grid: pd.DataFrame, n_lines: int) -> list[float]:
    available = np.sort(grid["Pbar_kw_per_server"].unique().astype(float))
    if n_lines <= 0 or n_lines >= len(available):
        return available.tolist()
    indexes = np.unique(np.round(np.linspace(0, len(available) - 1, n_lines)).astype(int))
    return available[indexes].tolist()


def pbar_label(value: float) -> str:
    return rf"$\bar{{P}}={value:.3f}$"


def controlled_context(args: argparse.Namespace) -> str:
    return (
        f"{args.workload} | {args.server_count:,} servers | utilization = {args.utilization:.2f} "
        f"| {args.agg} across accepted weights"
    )


def metric_z_limits(values: np.ndarray, mode: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Metric has no finite values for z-axis scaling.")
    observed_min = float(finite.min())
    observed_max = float(finite.max())
    if mode == "zero":
        lower = 0.0
        upper = observed_max * 1.05 if observed_max > 0 else 1.0
    else:
        span = observed_max - observed_min
        pad = 0.05 * span if span > 0 else max(abs(observed_max) * 0.05, 1.0)
        lower = observed_min - pad
        upper = observed_max + pad
    return lower, upper


def save_scale_summary(grid: pd.DataFrame, args: argparse.Namespace, out_path: Path) -> pd.DataFrame:
    rows = []
    for spec in METRICS:
        values = grid[spec["column"]].to_numpy(dtype=float)
        zmin, zmax = metric_z_limits(values, args.z_axis_mode)
        max_row = grid.loc[grid[spec["column"]].idxmax()]
        min_row = grid.loc[grid[spec["column"]].idxmin()]
        rows.append(
            {
                "Metric_Column": spec["column"],
                "Metric_Family": spec["family"],
                "Formula": spec["formula"],
                "Observed_Min": float(np.nanmin(values)),
                "Observed_Median": float(np.nanmedian(values)),
                "Observed_Max": float(np.nanmax(values)),
                "Observed_Range": float(np.nanmax(values) - np.nanmin(values)),
                "Suggested_Z_Min": zmin,
                "Suggested_Z_Max": zmax,
                "Maximum_At_Pbar_kw_per_server": float(max_row["Pbar_kw_per_server"]),
                "Maximum_At_R_kw_per_server": float(max_row["R_kw_per_server"]),
                "Minimum_At_Pbar_kw_per_server": float(min_row["Pbar_kw_per_server"]),
                "Minimum_At_R_kw_per_server": float(min_row["R_kw_per_server"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print("\nMetric scale summary")
    print("--------------------")
    print(summary[["Metric_Column", "Observed_Min", "Observed_Median", "Observed_Max", "Suggested_Z_Min", "Suggested_Z_Max"]].round(4).to_string(index=False))
    return summary


def plot_metric_lines(
    grid: pd.DataFrame,
    pbar_values: list[float],
    spec: dict,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    cmap = plt.get_cmap("viridis")
    pbar_min, pbar_max = min(pbar_values), max(pbar_values)
    norm = Normalize(vmin=pbar_min, vmax=pbar_max if pbar_max > pbar_min else pbar_min + 1.0)

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    for pbar in pbar_values:
        line = grid[np.isclose(grid["Pbar_kw_per_server"], pbar, atol=1e-9)].sort_values("R_kw_per_server")
        ax.plot(
            line["R_kw_per_server"],
            line[spec["column"]],
            marker="o",
            markersize=5.5,
            linewidth=2.25,
            color=cmap(norm(pbar)),
            label=pbar_label(pbar),
        )

    ax.set_title(f"{spec['title']} versus reserve bid $R$", fontsize=20, pad=16)
    ax.set_xlabel(r"Reserve bid $R$ (kW/server)", fontsize=15, labelpad=10)
    ax.set_ylabel(spec["ylabel"], fontsize=14, labelpad=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.28, linewidth=0.75)

    values = grid[spec["column"]].to_numpy(dtype=float)
    if spec.get("line_scale") == "log":
        positive = values[np.isfinite(values) & (values > 0)]
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) * 0.75, float(positive.max()) * 1.25)
        scale_note = "Logarithmic y-axis: each vertical step represents a multiplicative change."
    else:
        zmin, zmax = metric_z_limits(values, args.z_axis_mode)
        ax.set_ylim(zmin, zmax)
        scale_note = "Linear y-axis in the metric's native scale."

    ax.legend(
        title=r"Fixed baseline bid $\bar{P}$\n(kW/server)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        title_fontsize=10,
    )
    fig.text(0.50, 0.925, controlled_context(args), ha="center", fontsize=12)
    fig.text(0.47, 0.050, spec["footnote"], ha="center", fontsize=10.5)
    fig.text(0.47, 0.027, scale_note, ha="center", fontsize=9.5)
    fig.text(
        0.47,
        0.006,
        "Each curve fixes $\\bar{P}$ and varies $R$. Lines end at their workload-specific feasible maximum reserve.",
        ha="center",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.74, top=0.86, bottom=0.14)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_surface(
    grid: pd.DataFrame,
    spec: dict,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    x = grid["Pbar_kw_per_server"].to_numpy(dtype=float)
    y = grid["R_kw_per_server"].to_numpy(dtype=float)
    z = grid[spec["column"]].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    zmin, zmax = metric_z_limits(z, args.z_axis_mode)
    norm = Normalize(vmin=zmin, vmax=zmax)
    cmap = plt.get_cmap("viridis")
    tri = Triangulation(x, y)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")

    # Passing cmap and norm to plot_trisurf is essential. Without these, Matplotlib
    # uses its default solid blue surface even though a colorbar may be drawn.
    # Color each triangular face by the mean metric value at its three vertices.
    # This prevents Matplotlib's default single-blue surface from being shown.
    face_values = z[tri.triangles].mean(axis=1)
    face_colors = cmap(norm(face_values))
    face_colors[:, 3] = 0.94

    surface = ax.plot_trisurf(
        tri,
        z,
        color="white",
        shade=False,
        linewidth=0.18,
        edgecolor=(0.12, 0.12, 0.12, 0.23),
        antialiased=True,
    )
    surface.set_facecolors(face_colors)
    ax.scatter(
        x,
        y,
        z,
        c=z,
        cmap=cmap,
        norm=norm,
        s=24,
        edgecolors="black",
        linewidths=0.18,
        depthshade=False,
    )

    color_mapper = ScalarMappable(norm=norm, cmap=cmap)
    color_mapper.set_array(z)
    cbar = fig.colorbar(color_mapper, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label(spec["ylabel"], fontsize=11, labelpad=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r"Baseline power bid $\bar{P}$ (kW/server)", fontsize=13, labelpad=16)
    ax.set_ylabel(r"Reserve bid $R$ (kW/server)", fontsize=13, labelpad=16)
    ax.set_zlabel(spec["ylabel"], fontsize=11, labelpad=18)
    ax.set_zlim(zmin, zmax)
    ax.tick_params(axis="both", labelsize=10)
    ax.view_init(elev=28, azim=-132)
    fig.suptitle(f"{spec['title']} over the feasible physical P/R region", fontsize=17, y=0.975)
    fig.text(0.50, 0.935, controlled_context(args), ha="center", fontsize=11)
    fig.text(
        0.50,
        0.020,
        "Color and height encode the same metric. Each surface has its own vertical and color scale; compare shapes within a metric, not height across metrics.",
        ha="center",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.03, right=0.90, top=0.86, bottom=0.08)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")

    if args.show_3d:
        try:
            fig.canvas.manager.set_window_title(f"Rotate surface: {spec['short']}")
        except Exception:
            pass
        print(f"\nInteractive 3D window: {spec['short']}. Rotate, screenshot, then close this window to continue.")
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_csv)
    diagnostics_path = resolve_path(args.diagnostics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(results_path, diagnostics_path)
    sub = select_slice(merged, args)
    validate_metric_lineage(sub, args)
    grid = aggregate_slice(sub, args.agg)
    pbar_values = choose_pbars(grid, args.num_pbar_lines)

    print("\nControlled plotting slice")
    print("-------------------------")
    print(f"Workload: {args.workload}")
    print(f"Server count: {args.server_count}")
    print(f"Utilization: {args.utilization:.2f}")
    print(f"P/R points: {len(grid)}")
    print(f"Pbar lines: {[round(x, 6) for x in pbar_values]}")
    print(f"Z-axis mode: {args.z_axis_mode}")

    scale_path = out_dir / "00_metric_scale_summary.csv"
    save_scale_summary(grid, args, scale_path)

    outputs: list[Path] = [scale_path]
    for index, spec in enumerate(METRICS, start=1):
        line_path = out_dir / f"{index:02d}_{spec['short']}_vs_R_by_Pbar.png"
        plot_metric_lines(grid, pbar_values, spec, args, line_path)
        outputs.append(line_path)

    for index, spec in enumerate(METRICS, start=5):
        surface_path = out_dir / f"{index:02d}_3D_{spec['short']}.png"
        plot_interactive_surface(grid, spec, args, surface_path)
        outputs.append(surface_path)

    print("\nCreated files")
    print("-------------")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
