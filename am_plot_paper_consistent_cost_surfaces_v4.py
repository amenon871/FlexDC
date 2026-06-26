"""
Paper- and code-consistent FlexDC 3D cost surfaces.

This script intentionally distinguishes two aggregation modes:

1) fixed_utilization
   Fixes workload, server count, and utilization; aggregates only across the
   accepted weight vectors at each physical (Pbar, R) point.
   Use this for causal/interpretable presentation figures.

2) median_over_utilizations
   Fixes workload and server count; aggregates across both utilization values
   and accepted weights. This reproduces the aggregation used by the earlier
   am_visualize_final_dataset.py 3D figures.
   Use this only when you explicitly want the old aggregate view.

It creates only the four 3D surface graph types that were previously numbered 05–08.
Each metric is saved from three fixed camera angles, for a total of 12 PNG files:

- 05_3D_Mtrack_Cost_01_perspective.png
- 05_3D_Mtrack_Cost_02_front_right.png
- 05_3D_Mtrack_Cost_03_high_overview.png
- ...and the corresponding three views for metrics 06–08.

No CSV audit file is created. If an older run left
00_metric_context_and_scale_audit.csv in the output directory, this script
removes that legacy file before plotting.

The script validates the corresponding cost identities against the CSV before
plotting. It also supports --expected-rows-per-pr, including intentional
single-row-per-P/R datasets via --expected-rows-per-pr 1.
"""

from __future__ import annotations

import argparse
import configparser
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
    "Mtrack_Error_MeanAbs_Watts",
    "Mtrack_Price_Coefficient_piE",
    "Mtrack_Hour_Seconds",
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
        "file": "Mtrack_Cost",
        "title": r"Raw monetary tracking cost $M_{\mathrm{track}}$",
        "zlabel": r"$M_{\mathrm{track}}$ (simulator monetary units)",
        "formula": r"$M_{\mathrm{track}} = \sum_{\mathrm{phases}} \pi^E R\,\mathrm{mean}(|e|)H$",
        "line_log": True,
        "surface_note": "Raw simulator monetary term; nonnegative by construction.",
    },
    {
        "column": "Simulator_Power_Cost",
        "file": "Simulator_Power_Cost",
        "title": r"Power/reserve monetary term $M_{\mathrm{power}}$",
        "zlabel": r"$M_{\mathrm{power}}$ (simulator monetary units)",
        "formula": r"$M_{\mathrm{power}} = (\pi^P P - \pi^R R)H$",
        "line_log": False,
        "surface_note": "Raw simulator monetary term; increasing R earns reserve revenue at fixed Pbar.",
    },
    {
        "column": "Ctrack_Weighted_Cost",
        "file": "Ctrack_Weighted_Cost",
        "title": r"Weighted tracking penalty $\psi C_{\mathrm{track}}$",
        "zlabel": r"$\psi C_{\mathrm{track}}$ (penalty units)",
        "formula": r"$\psi\,\mathrm{SoftPlus}(\mu(\epsilon^r-\gamma))$",
        "line_log": True,
        "surface_note": "Derived paper-style penalty; not a simulator monetary cost.",
    },
    {
        "column": "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "file": "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "title": r"Weighted QoS penalty $\beta C_{\mathrm{QoS}}$",
        "zlabel": r"$\beta C_{\mathrm{QoS}}$ (penalty units)",
        "formula": r"$\beta\sum_j \mathrm{SoftPlus}(\rho(P^{D,j}-\delta))$",
        "line_log": False,
        "surface_note": "Derived QoS penalty aggregated across job types; not a monetary cost.",
    },
]


SURFACE_VIEWS = [
    {
        "file_suffix": "01_perspective",
        "label": "Perspective view",
        "elev": 28,
        "azim": -132,
    },
    {
        "file_suffix": "02_front_right",
        "label": "Front-right view",
        "elev": 24,
        "azim": -48,
    },
    {
        "file_suffix": "03_high_overview",
        "label": "High overview",
        "elev": 64,
        "azim": -132,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-consistent FlexDC 3D cost surfaces (three fixed views per metric).")
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv path.")
    parser.add_argument("--diagnostics-csv", required=True, help="Combined grid_search_diagnostics.csv path.")
    parser.add_argument("--out-dir", default="paper_consistent_cost_figures", help="Output directory.")
    parser.add_argument("--workload", default="W2-short-qos2-2.5-2.5-3", help="Workload_Name to plot.")
    parser.add_argument("--server-count", type=int, default=1000, help="server_count to plot.")
    parser.add_argument(
        "--aggregation-scope",
        choices=["fixed_utilization", "median_over_utilizations"],
        default="fixed_utilization",
        help=(
            "fixed_utilization: median across weights at one fixed U. "
            "median_over_utilizations: median across both U values and weights; matches older visualizer aggregation."
        ),
    )
    parser.add_argument("--utilization", type=float, default=0.60, help="Required context for fixed_utilization mode.")
    parser.add_argument("--agg", choices=["median", "mean"], default="median", help="Aggregation function.")
    parser.add_argument(
        "--expected-rows-per-pr",
        type=int,
        default=None,
        help=(
            "Expected source rows per physical P/R point. Defaults to 101 for "
            "fixed_utilization and 202 for median_over_utilizations. Use 1 for "
            "single-row datasets."
        ),
    )
    parser.add_argument(
        "--surface-z-axis",
        choices=["auto", "zero", "observed"],
        default="auto",
        help=(
            "auto reproduces Matplotlib's native data-driven z limits, as in the earlier visualizer. "
            "zero forces z=0. observed uses exact observed min/max with a small pad."
        ),
    )
    parser.add_argument(
        "--gradient-config",
        default="configs/gradient_descent/gradient_descent.ini",
        help="Gradient config used to recover beta, rho, and QoS threshold for QoS identity validation.",
    )
    parser.add_argument("--qos-beta", type=float, default=None, help="Optional beta override.")
    parser.add_argument("--qos-rho", type=float, default=None, help="Optional rho override.")
    parser.add_argument("--qos-threshold", type=float, default=None, help="Optional QoS threshold delta override.")
    parser.add_argument("--show-3d", action="store_true", help="Open each rendered 3D view before continuing.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    fallback = Path.cwd() / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find: {path_text}")


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {missing}")


def load_merged(results_path: Path, diagnostics_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    diagnostics = pd.read_csv(diagnostics_path)
    require_columns(results, RESULT_REQUIRED, "Results")
    require_columns(diagnostics, DIAGNOSTIC_REQUIRED, "Diagnostics")

    numeric = {
        "Iteration",
        "Weight_Sample_ID",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "server_count",
        "utilization",
        "Simulator_Power_Cost",
        "Mtrack_Cost",
        "Simulator_RSR_Total_Cost",
        "Mtrack_Error_MeanAbs_Watts",
        "Mtrack_Price_Coefficient_piE",
        "Mtrack_Hour_Seconds",
        "Ctrack_Epsilon_90th",
        "Ctrack_Gamma",
        "Ctrack_Mu",
        "Ctrack_Psi",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    }
    for frame in (results, diagnostics):
        for column in numeric.intersection(frame.columns):
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


def parse_probabilities(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    values = np.asarray(value, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid QoS_Delay_Probabilities value: {value!r}")
    return values


def read_qos_parameters(args: argparse.Namespace) -> tuple[float, float, float, str]:
    beta = args.qos_beta
    rho = args.qos_rho
    threshold = args.qos_threshold
    source = "CLI overrides"

    config_path = Path(args.gradient_config)
    if (beta is None or rho is None or threshold is None) and config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        if beta is None:
            beta = config.getfloat("gradient_driver", "beta")
        if rho is None:
            rho = config.getfloat("gradient_driver", "rho")
        if threshold is None:
            threshold = config.getfloat("calculate_gradient", "QOS_THRESHOLD")
        source = str(config_path)

    # This fallback matches the archived active gradient_descent.ini used by the pilot.
    if beta is None:
        beta = 0.5
    if rho is None:
        rho = 2.0
    if threshold is None:
        threshold = 0.1
    if source == "CLI overrides" and not config_path.exists():
        source = "fallback defaults beta=0.5, rho=2.0, delta=0.1"
    return float(beta), float(rho), float(threshold), source


def validate_lineage(df: pd.DataFrame, beta: float, rho: float, threshold: float) -> None:
    rsr_error = (
        df["Simulator_RSR_Total_Cost"]
        - df["Simulator_Power_Cost"]
        - df["Mtrack_Cost"]
    ).abs().max()
    if rsr_error > 1e-8:
        raise ValueError(f"RSR identity failed: max error={rsr_error:.3e}")

    mtrack_expected = (
        df["Mtrack_Price_Coefficient_piE"]
        * df["Mtrack_Error_MeanAbs_Watts"]
        * df["Mtrack_Hour_Seconds"]
    )
    mtrack_error = (mtrack_expected - df["Mtrack_Cost"]).abs().max()
    if mtrack_error > 1e-8:
        raise ValueError(f"Mtrack reconstruction failed: max error={mtrack_error:.3e}")

    ctrack_expected = df["Ctrack_Psi"] * np.logaddexp(
        0,
        df["Ctrack_Mu"] * (df["Ctrack_Epsilon_90th"] - df["Ctrack_Gamma"]),
    )
    ctrack_error = (ctrack_expected - df["Ctrack_Weighted_Cost"]).abs().max()
    if ctrack_error > 1e-8:
        raise ValueError(f"Ctrack reconstruction failed: max error={ctrack_error:.3e}")

    qos_expected = df["QoS_Delay_Probabilities"].map(
        lambda item: beta * np.logaddexp(0, rho * (parse_probabilities(item) - threshold)).sum()
    )
    qos_error = (qos_expected - df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]).abs().max()
    if qos_error > 1e-8:
        raise ValueError(
            f"Cqos reconstruction failed: max error={qos_error:.3e}. "
            "Verify beta/rho/qos threshold arguments against the configuration used for generation."
        )

    if (df["Mtrack_Cost"] < -1e-12).any():
        raise ValueError("Mtrack_Cost contains negative values; this contradicts simulator.final_output().")

    print("\nFormula validation passed")
    print("-------------------------")
    print(f"max |M_RSR - (M_power + M_track)|: {rsr_error:.3e}")
    print(f"max Mtrack reconstruction error:   {mtrack_error:.3e}")
    print(f"max Ctrack reconstruction error:   {ctrack_error:.3e}")
    print(f"max Cqos reconstruction error:     {qos_error:.3e}")


def expected_rows_per_pr(args: argparse.Namespace) -> int:
    """Return the expected source-row count for one physical P/R point."""
    if args.expected_rows_per_pr is not None:
        if args.expected_rows_per_pr <= 0:
            raise ValueError("--expected-rows-per-pr must be a positive integer.")
        return int(args.expected_rows_per_pr)
    return 101 if args.aggregation_scope == "fixed_utilization" else 202


def select_and_aggregate(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    selected = df[
        (df["Workload_Name"] == args.workload)
        & (df["server_count"].astype(int) == int(args.server_count))
    ].copy()
    if args.aggregation_scope == "fixed_utilization":
        selected = selected[np.isclose(selected["utilization"].astype(float), args.utilization, atol=1e-9)].copy()
        if expected_rows_per_pr(args) == 1:
            scope_label = f"One source row per P/R point; utilization fixed at {args.utilization:.2f}"
        else:
            scope_label = f"{args.agg} across accepted weights; utilization fixed at {args.utilization:.2f}"
    else:
        if expected_rows_per_pr(args) == 1:
            scope_label = "One source row per P/R point"
        else:
            scope_label = f"{args.agg} across utilizations and accepted weights"

    if selected.empty:
        raise ValueError("No rows matched the requested workload/server/utilization selection.")

    metric_columns = [item["column"] for item in METRICS]
    grouped = (
        selected.groupby(["Pbar_kw_per_server", "R_kw_per_server"], as_index=False)[metric_columns]
        .agg(args.agg)
        .sort_values(["Pbar_kw_per_server", "R_kw_per_server"])
        .reset_index(drop=True)
    )
    point_counts = selected.groupby(["Pbar_kw_per_server", "R_kw_per_server"]).size().rename("rows_aggregated").reset_index()
    return grouped, scope_label, point_counts



def observed_limits(values: np.ndarray, mode: str) -> tuple[float, float]:
    """Return z limits.

    auto intentionally reproduces the approximately 1/48 data-range padding
    that Axes3D applied in the earlier am_visualize_final_dataset.py surfaces.
    This makes matching workload/server/aggregation figures directly comparable.
    """
    values = values[np.isfinite(values)]
    observed_min = float(values.min())
    observed_max = float(values.max())
    span = observed_max - observed_min
    if span <= 0:
        span = max(abs(observed_max) * 0.05, 0.01)

    if mode == "auto":
        pad = span / 48.0
        return observed_min - pad, observed_max + pad
    if mode == "zero":
        return 0.0, observed_max * 1.05 if observed_max > 0 else 1.0

    pad = 0.04 * span
    return observed_min - pad, observed_max + pad



def plot_surface(
    grid: pd.DataFrame,
    spec: dict,
    scope_label: str,
    args: argparse.Namespace,
    view: dict,
    out_path: Path,
) -> None:
    """Render and save one camera angle for one 3D metric surface."""
    x = grid["Pbar_kw_per_server"].to_numpy(float)
    y = grid["R_kw_per_server"].to_numpy(float)
    z = grid[spec["column"]].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]

    metric_min, metric_max = float(z.min()), float(z.max())
    color_norm = Normalize(
        vmin=metric_min,
        vmax=metric_max if metric_max > metric_min else metric_min + 1.0,
    )
    cmap = plt.get_cmap("viridis")
    tri = Triangulation(x, y)
    face_values = z[tri.triangles].mean(axis=1)
    face_colors = cmap(color_norm(face_values))
    face_colors[:, 3] = 0.94

    fig = plt.figure(figsize=(14.5, 10.5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_trisurf(
        tri,
        z,
        color="white",
        shade=False,
        linewidth=0.20,
        edgecolor=(0.08, 0.08, 0.08, 0.22),
        antialiased=True,
    )
    surface.set_facecolors(face_colors)
    ax.scatter(
        x,
        y,
        z,
        c=z,
        cmap=cmap,
        norm=color_norm,
        s=22,
        edgecolors=(0.05, 0.05, 0.05, 0.70),
        linewidths=0.18,
        depthshade=False,
    )

    lower, upper = observed_limits(z, args.surface_z_axis)
    ax.set_zlim(lower, upper)

    mapper = ScalarMappable(norm=color_norm, cmap=cmap)
    mapper.set_array(z)
    cbar = fig.colorbar(mapper, ax=ax, shrink=0.66, pad=0.07)
    cbar.set_label(spec["zlabel"], fontsize=11, labelpad=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r"Baseline power bid $\bar{P}$ (kW/server)", fontsize=13, labelpad=15)
    ax.set_ylabel(r"Reserve bid $R$ (kW/server)", fontsize=13, labelpad=15)
    ax.set_zlabel(spec["zlabel"], fontsize=11, labelpad=16)
    ax.tick_params(labelsize=10)
    ax.set_box_aspect((1.15, 1.00, 0.76))
    ax.view_init(elev=view["elev"], azim=view["azim"])

    fig.suptitle(f"3D surface: {spec['title']}", fontsize=18, y=0.955)
    fig.text(0.49, 0.895, f"{scope_label} | {view['label']}", ha="center", fontsize=11)
    fig.text(
        0.49,
        0.020,
        "Height and color show the same metric. Markers are measured P/R points; the surface only interpolates between them.",
        ha="center",
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.03, right=0.90, top=0.92, bottom=0.08)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")

    if args.show_3d:
        try:
            fig.canvas.manager.set_window_title(f"{spec['file']} — {view['label']}")
        except Exception:
            pass
        print(f"Interactive surface open: {spec['file']} — {view['label']}. Close to continue.")
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_csv)
    diagnostics_path = resolve_path(args.diagnostics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta, rho, threshold, qos_source = read_qos_parameters(args)
    print(f"QoS validation parameters: beta={beta}, rho={rho}, delta={threshold} ({qos_source})")

    merged = load_merged(results_path, diagnostics_path)
    validate_lineage(merged, beta, rho, threshold)
    grid, scope_label, point_counts = select_and_aggregate(merged, args)

    unique_counts = sorted(point_counts["rows_aggregated"].unique().tolist())
    expected_rows = expected_rows_per_pr(args)

    print("\nPlot context")
    print("------------")
    print(f"Workload: {args.workload}")
    print(f"Server count: {args.server_count}")
    print(f"Aggregation scope: {args.aggregation_scope}")
    print(f"Description: {scope_label}")
    print(f"Physical P/R points: {len(grid)}")
    print(f"Rows aggregated per P/R point: {unique_counts} (expected [{expected_rows}])")
    if unique_counts != [expected_rows]:
        raise ValueError(
            f"Unexpected number of source rows per P/R point: "
            f"found {unique_counts}, expected [{expected_rows}]."
        )

    legacy_audit_path = out_dir / "00_metric_context_and_scale_audit.csv"
    if legacy_audit_path.exists():
        legacy_audit_path.unlink()
        print(f"Removed legacy audit CSV: {legacy_audit_path}")

    created_paths: list[Path] = []
    for index, spec in enumerate(METRICS, start=5):
        for view in SURFACE_VIEWS:
            surface_path = out_dir / f"{index:02d}_3D_{spec['file']}_{view['file_suffix']}.png"
            plot_surface(grid, spec, scope_label, args, view, surface_path)
            created_paths.append(surface_path)

    print("\nCreated 3D surface figures")
    print("--------------------------")
    for path in created_paths:
        print(path)
    print(f"Total PNG files created: {len(created_paths)}")


if __name__ == "__main__":
    main()
