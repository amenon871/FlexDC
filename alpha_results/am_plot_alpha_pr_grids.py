"""
Generate targeted P/R-grid plots for the alpha experiment result folders.

Expected input folders (under --input-root):
    alpha_results_1p1
    alpha_results_1p2
    ...
    alpha_results_2p0

Each folder should contain:
    alpha_<value>_w1_s64_u070_equal_AQA_combined_grid_search_results.csv
    alpha_<value>_w1_s64_u070_equal_AQA_combined_grid_search_diagnostics.csv

For every alpha experiment, this script:
  * reads the measured (Pbar, R) grid points from the combined results CSV,
  * reads the P/R feasibility bounds from the diagnostics CSV,
  * draws the R = P boundary,
  * draws the P + R = alpha * Pmax boundary using the CSV's
    PR_upper_bound_kw_per_server value,
  * saves one PNG image.

No CSV files are written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VALUES = ["1p1", "1p2", "1p3", "1p4", "1p5", "1p6", "1p7", "1p8", "1p9", "2p0"]

REQUIRED_RESULT_COLUMNS = [
    "Pbar_kw_per_server",
    "R_kw_per_server",
]

REQUIRED_DIAGNOSTIC_COLUMNS = [
    "Pmax_kw_per_server",
    "Pbar_lower_bound_kw_per_server",
    "Pbar_upper_bound_kw_per_server",
    "PR_upper_bound_kw_per_server",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one targeted P/R-grid plot per alpha result folder."
    )
    parser.add_argument(
        "--input-root",
        default=".",
        help=(
            "Folder containing alpha_results_1p1, alpha_results_1p2, ..., "
            "alpha_results_2p0. Default: current directory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="alpha_pr_grid_plots",
        help="Folder where the PNG plots will be saved.",
    )
    parser.add_argument(
        "--workload",
        default="W1-train",
        help="Workload_Name to select from each CSV.",
    )
    parser.add_argument(
        "--server-count",
        type=int,
        default=64,
        help="server_count to select from each CSV.",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.70,
        help="utilization to select from each CSV.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=DEFAULT_VALUES,
        help=(
            "Alpha folder suffixes to process. "
            "Default: 1p1 1p2 1p3 1p4 1p5 1p6 1p7 1p8 1p9 2p0."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG resolution. Default: 240.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each figure after saving it.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if an expected input file is missing.",
    )
    return parser.parse_args()


def alpha_from_value(value: str) -> float:
    """Convert a folder suffix such as '1p1' into alpha=1.1."""
    try:
        alpha = float(value.replace("p", "."))
    except ValueError as exc:
        raise ValueError(
            f"Could not parse alpha from '{value}'. Use values such as 1p1 or 2p0."
        ) from exc

    if alpha <= 0:
        raise ValueError(f"Alpha must be positive; got {alpha}.")
    return alpha


def prefix_for_value(value: str) -> str:
    return f"alpha_{value}_w1_s64_u070_equal_AQA"


def require_columns(df: pd.DataFrame, columns: list[str], label: str, csv_path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing {missing}: {csv_path}")


def filter_context(df: pd.DataFrame, args: argparse.Namespace, csv_label: str) -> pd.DataFrame:
    filtered = df.copy()

    if "Workload_Name" in filtered.columns:
        filtered = filtered[filtered["Workload_Name"].astype(str) == args.workload]

    if "server_count" in filtered.columns:
        server_count = pd.to_numeric(filtered["server_count"], errors="coerce")
        filtered = filtered[server_count == int(args.server_count)]

    if "utilization" in filtered.columns:
        utilization = pd.to_numeric(filtered["utilization"], errors="coerce")
        filtered = filtered[np.isclose(utilization, args.utilization, atol=1e-9)]

    if filtered.empty:
        raise ValueError(
            f"No {csv_label} rows matched workload={args.workload!r}, "
            f"server_count={args.server_count}, utilization={args.utilization:.2f}."
        )

    return filtered


def read_constant(frame: pd.DataFrame, column: str, fallback: float | None = None) -> float:
    """Read a single numeric configuration value, tolerating repeated identical rows."""
    if column not in frame.columns:
        if fallback is not None:
            return float(fallback)
        raise ValueError(f"Diagnostics CSV is missing required boundary column: {column}")

    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        if fallback is not None:
            return float(fallback)
        raise ValueError(f"No usable numeric value found in diagnostics column: {column}")

    minimum = float(values.min())
    maximum = float(values.max())
    if not np.isclose(minimum, maximum, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"Diagnostics column {column!r} is not constant for the selected context: "
            f"min={minimum}, max={maximum}."
        )

    return float(np.median(values))


def load_pr_grid(
    results_csv: Path,
    diagnostics_csv: Path,
    args: argparse.Namespace,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    results = pd.read_csv(results_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    require_columns(results, REQUIRED_RESULT_COLUMNS, "Results", results_csv)
    require_columns(diagnostics, REQUIRED_DIAGNOSTIC_COLUMNS, "Diagnostics", diagnostics_csv)

    results = filter_context(results, args, "results")
    diagnostics = filter_context(diagnostics, args, "diagnostics")

    for column in REQUIRED_RESULT_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    points = (
        results[REQUIRED_RESULT_COLUMNS]
        .dropna()
        .drop_duplicates()
        .sort_values(REQUIRED_RESULT_COLUMNS)
        .reset_index(drop=True)
    )

    if points.empty:
        raise ValueError(f"No valid P/R points found in {results_csv}")

    pbar_min = read_constant(
        diagnostics,
        "Pbar_lower_bound_kw_per_server",
        fallback=float(points["Pbar_kw_per_server"].min()),
    )
    pbar_max = read_constant(
        diagnostics,
        "Pbar_upper_bound_kw_per_server",
        fallback=float(points["Pbar_kw_per_server"].max()),
    )
    pmax = read_constant(diagnostics, "Pmax_kw_per_server", fallback=pbar_max)
    pr_upper = read_constant(
        diagnostics,
        "PR_upper_bound_kw_per_server",
        fallback=alpha * pmax,
    )

    # P/R-grid feasibility checks. A small tolerance accounts for CSV rounding.
    pbar = points["Pbar_kw_per_server"].to_numpy(float)
    reserve = points["R_kw_per_server"].to_numpy(float)
    tolerance = 1e-7

    if (reserve > pbar + tolerance).any():
        raise ValueError("Found a P/R point above the R = P feasibility boundary.")

    if (pbar + reserve > pr_upper + tolerance).any():
        raise ValueError(
            "Found a P/R point above the P + R feasibility boundary stored in the diagnostics CSV."
        )

    expected_upper = alpha * pmax
    if not np.isclose(pr_upper, expected_upper, rtol=1e-7, atol=1e-7):
        print(
            "WARNING: diagnostics PR upper bound does not exactly equal alpha * Pmax: "
            f"{pr_upper:.8g} vs {alpha:.1f} * {pmax:.8g} = {expected_upper:.8g}. "
            "The plotted orange line uses the diagnostics value."
        )

    metadata = {
        "pbar_min": pbar_min,
        "pbar_max": pbar_max,
        "pmax": pmax,
        "pr_upper": pr_upper,
        "point_count": float(len(points)),
    }
    return points, metadata


def plot_pr_grid(
    points: pd.DataFrame,
    metadata: dict[str, float],
    alpha: float,
    value: str,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    x_points = points["Pbar_kw_per_server"].to_numpy(float)
    y_points = points["R_kw_per_server"].to_numpy(float)

    pbar_lower = min(metadata["pbar_min"], float(x_points.min()))
    pbar_upper = max(metadata["pbar_max"], float(x_points.max()))
    pr_upper = metadata["pr_upper"]

    # Draw the two boundaries across the full configured Pbar range.
    x_boundary = np.linspace(pbar_lower, pbar_upper, 500)
    r_equals_p = x_boundary
    pr_boundary = pr_upper - x_boundary
    valid_pr = pr_boundary >= 0.0

    # Axis limits include all points and both boundaries with modest whitespace.
    max_y = max(
        float(y_points.max()),
        float(r_equals_p.max()),
        float(pr_boundary[valid_pr].max()) if valid_pr.any() else 0.0,
    )
    x_span = max(pbar_upper - pbar_lower, 0.05)
    y_span = max(max_y, 0.05)

    x_pad = 0.05 * x_span
    y_pad = 0.055 * y_span

    fig, ax = plt.subplots(figsize=(12.5, 9.5))

    ax.scatter(
        x_points,
        y_points,
        color="tab:blue",
        s=42,
        alpha=0.78,
        label="Generated P/R points",
        zorder=3,
    )
    ax.plot(
        x_boundary,
        r_equals_p,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=r"$R = P$ boundary",
        zorder=2,
    )
    ax.plot(
        x_boundary[valid_pr],
        pr_boundary[valid_pr],
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        alpha=0.90,
        label=rf"$P + R = {alpha:.1f}\,P_{{\max}}$ boundary",
        zorder=2,
    )

    ax.set_title(
        f"Generated targeted P/R grid: {args.workload} "
        rf"($\alpha={alpha:.1f}$; run {value})",
        fontsize=16,
        pad=8,
    )
    ax.set_xlabel(r"$\bar{P}$ (kW/server)", fontsize=13)
    ax.set_ylabel(r"$R$ (kW/server)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.28, linewidth=1.0)

    ax.set_xlim(pbar_lower - x_pad, pbar_upper + x_pad)
    ax.set_ylim(-y_pad, max_y + y_pad)

    ax.legend(loc="center right", fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")

    if args.show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = []
    skipped = []

    print("Alpha P/R grid plot configuration")
    print("---------------------------------")
    print(f"Input root:  {input_root.resolve()}")
    print(f"Output dir:  {out_dir.resolve()}")
    print(f"Workload:    {args.workload}")
    print(f"Servers:     {args.server_count}")
    print(f"Utilization: {args.utilization:.2f}")
    print()

    for value in args.values:
        alpha = alpha_from_value(value)
        prefix = prefix_for_value(value)
        run_dir = input_root / f"alpha_results_{value}"
        results_csv = run_dir / f"{prefix}_combined_grid_search_results.csv"
        diagnostics_csv = run_dir / f"{prefix}_combined_grid_search_diagnostics.csv"
        output_path = out_dir / f"alpha_{value}_P_R_grid.png"

        missing = [path for path in (results_csv, diagnostics_csv) if not path.exists()]
        if missing:
            message = f"Skipping alpha_{value}; missing: " + ", ".join(str(path) for path in missing)
            if args.strict:
                raise FileNotFoundError(message)
            print(f"WARNING: {message}")
            skipped.append(value)
            continue

        print(f"Plotting alpha_{value} (alpha={alpha:.1f})")
        points, metadata = load_pr_grid(results_csv, diagnostics_csv, args, alpha)
        plot_pr_grid(points, metadata, alpha, value, args, output_path)

        print(
            f"  points={int(metadata['point_count'])}, "
            f"Pmax={metadata['pmax']:.6g}, "
            f"P+R upper={metadata['pr_upper']:.6g}"
        )
        print(f"  saved: {output_path}")
        created.append(output_path)

    print()
    print(f"Created {len(created)} PNG plot(s).")
    if skipped:
        print(f"Skipped {len(skipped)} run(s): {', '.join(skipped)}")

    if not created:
        sys.exit("No plots were created. Check --input-root and the alpha_results_* folders.")


if __name__ == "__main__":
    main()
