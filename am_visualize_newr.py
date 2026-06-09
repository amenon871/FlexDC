import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPANDED_FILENAME = "rangefind64_p020_084_r001_084_AQA_exp_servers_64_combined_raw.csv"

DEFAULT_SEARCH_DIRS = [
    Path("combined_outputs_new"),
    Path("combined_outputs"),
    Path("combined_outputs_new_2"),
    Path("."),
]

DEFAULT_OUT_DIR = Path("useful_presentation_figures")

WORKLOAD_ORDER = [
    "W10",
    "W9",
    "W_inference_2app",
    "W_train_2app",
]

REQUIRED_COLUMNS = [
    "Source_Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "P_actual_watts",
    "R_actual_watts",
    "server_count",
    "utilization",
    "Weights",
    "Simulator_Power_Cost",
    "Simulator_Tracking_Cost",
    "Simulator_RSR_Total_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
]


def find_raw_csv() -> Path:
    for folder in DEFAULT_SEARCH_DIRS:
        candidate = folder / EXPANDED_FILENAME
        if candidate.exists():
            return candidate

    matches = list(Path(".").glob(f"**/{EXPANDED_FILENAME}"))
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    matches = list(Path(".").glob("**/*r001_084*combined_raw.csv"))
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        "Could not find expanded-R combined raw CSV. "
        "Pass it explicitly with --raw-csv."
    )


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Pbar_kw_per_server"] = df["Pbar_kw_per_server"].astype(float).round(10)
    df["R_kw_per_server"] = df["R_kw_per_server"].astype(float).round(10)
    df["utilization"] = df["utilization"].astype(float).round(10)
    return df


def validate_df(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV has zero rows.")

    invalid = df[df["Pbar_kw_per_server"] <= df["R_kw_per_server"]]
    if len(invalid) > 0:
        raise ValueError(f"Found {len(invalid)} rows with Pbar <= R.")

    expected_p = np.round(
        df["Pbar_kw_per_server"] * 1000 * df["server_count"].astype(float)
    ).astype(int)
    expected_r = np.round(
        df["R_kw_per_server"] * 1000 * df["server_count"].astype(float)
    ).astype(int)

    p_bad = (df["P_actual_watts"].astype(int) != expected_p).sum()
    r_bad = (df["R_actual_watts"].astype(int) != expected_r).sum()

    if p_bad or r_bad:
        raise ValueError(f"kW/server conversion mismatch: P_bad={p_bad}, R_bad={r_bad}")

    total_error = (
        df["Simulator_RSR_Total_Cost"]
        - df["Simulator_Power_Cost"]
        - df["Simulator_Tracking_Cost"]
    ).abs().max()

    if total_error > 1e-8:
        raise ValueError(
            "RSR total cost does not equal power cost + tracking cost. "
            f"Max error={total_error}"
        )


def ordered_workloads(df: pd.DataFrame) -> list[str]:
    existing = list(df["Source_Workload_Name"].dropna().unique())
    ordered = [w for w in WORKLOAD_ORDER if w in existing]
    ordered += sorted([w for w in existing if w not in ordered])
    return ordered


def make_heatmap(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    output_path: Path,
    annotate: bool = False,
    integer_labels: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    matrix = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])

    if annotate:
        for yi, row_value in enumerate(pivot.index):
            for xi, col_value in enumerate(pivot.columns):
                value = pivot.loc[row_value, col_value]
                if pd.notna(value):
                    label = f"{int(value)}" if integer_labels else f"{value:.3f}"
                    ax.text(xi, yi, label, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_01_valid_grid(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        df.groupby(["R_kw_per_server", "Pbar_kw_per_server"])
        .size()
        .reset_index(name="row_count")
    )

    pivot = counts.pivot(
        index="R_kw_per_server",
        columns="Pbar_kw_per_server",
        values="row_count",
    ).sort_index().sort_index(axis=1)

    output_path = out_dir / "01_valid_pr_grid_counts.png"

    make_heatmap(
        pivot=pivot,
        title="Valid P/R grid coverage\nExpanded R sweep; missing cells are skipped by Pbar > R",
        xlabel="Pbar kW/server",
        ylabel="R kW/server",
        colorbar_label="Rows per valid P/R point",
        output_path=output_path,
        annotate=True,
        integer_labels=True,
    )

    return output_path


def plot_02_tracking_sensitivity_heatmap(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    """
    For each workload and Pbar, compute how much tracking cost changes
    from the smallest valid R to the largest valid R.
    This directly summarizes the R effect.
    """
    grouped = (
        df.groupby(["Source_Workload_Name", "Pbar_kw_per_server", "R_kw_per_server"], as_index=False)
        ["Simulator_Tracking_Cost"]
        .agg(agg)
    )

    rows = []

    for (workload, pbar), sub in grouped.groupby(["Source_Workload_Name", "Pbar_kw_per_server"]):
        sub = sub.sort_values("R_kw_per_server")
        if len(sub) < 2:
            continue

        r_min = float(sub["R_kw_per_server"].iloc[0])
        r_max = float(sub["R_kw_per_server"].iloc[-1])
        c_min = float(sub["Simulator_Tracking_Cost"].iloc[0])
        c_max = float(sub["Simulator_Tracking_Cost"].iloc[-1])

        delta_cost = c_max - c_min
        slope = delta_cost / (r_max - r_min) if r_max > r_min else np.nan

        rows.append({
            "Source_Workload_Name": workload,
            "Pbar_kw_per_server": pbar,
            "delta_tracking_cost": delta_cost,
            "tracking_cost_slope_per_kw_server": slope,
            "r_min": r_min,
            "r_max": r_max,
        })

    result = pd.DataFrame(rows)

    workloads = ordered_workloads(df)
    pivot = result.pivot(
        index="Source_Workload_Name",
        columns="Pbar_kw_per_server",
        values="tracking_cost_slope_per_kw_server",
    )

    pivot = pivot.reindex(workloads).sort_index(axis=1)

    output_path = out_dir / "02_tracking_sensitivity_to_r_by_workload_and_pbar.png"

    make_heatmap(
        pivot=pivot,
        title=(
            "Tracking-cost sensitivity to R\n"
            "Slope from lowest to highest valid R at each workload/Pbar"
        ),
        xlabel="Pbar kW/server",
        ylabel="Workload",
        colorbar_label="Δ tracking cost / Δ R",
        output_path=output_path,
        annotate=True,
        integer_labels=False,
    )

    csv_path = out_dir / "02_tracking_sensitivity_to_r_by_workload_and_pbar.csv"
    result.to_csv(csv_path, index=False)

    return output_path


def plot_03_tracking_delta_curves(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    """
    Instead of plotting raw tracking cost, plot increase from each curve's
    lowest valid R. This makes the R effect clearer and removes baseline scale.
    """
    grouped = (
        df.groupby(
            ["Source_Workload_Name", "Pbar_kw_per_server", "R_kw_per_server"],
            as_index=False,
        )["Simulator_Tracking_Cost"]
        .agg(agg)
    )

    workloads = ordered_workloads(df)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax_idx, workload in enumerate(workloads[:4]):
        ax = axes[ax_idx]
        sub = grouped[grouped["Source_Workload_Name"] == workload]

        for pbar in sorted(sub["Pbar_kw_per_server"].unique()):
            line = sub[sub["Pbar_kw_per_server"] == pbar].sort_values("R_kw_per_server")
            if len(line) < 2:
                continue

            baseline = float(line["Simulator_Tracking_Cost"].iloc[0])
            delta = line["Simulator_Tracking_Cost"] - baseline

            ax.plot(
                line["R_kw_per_server"],
                delta,
                marker="o",
                linewidth=1.8,
                label=f"P={pbar:.3f}",
            )

        ax.set_title(workload)
        ax.set_xlabel("R kW/server")
        ax.set_ylabel("Increase in tracking cost from lowest valid R")
        ax.grid(True, alpha=0.25)

    for idx in range(len(workloads), len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Pbar kW/server",
        loc="center right",
        bbox_to_anchor=(1.08, 0.5),
    )

    fig.suptitle(
        "Tracking cost increase as R grows\n"
        f"Expanded R sweep; {agg} across utilization and weights",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 0.90, 0.94])

    output_path = out_dir / "03_tracking_cost_increase_vs_r.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_04_qos_high_util_heatmaps(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    """
    QoS is not part of RSR total in the raw CSV, so plot it separately.
    Use max utilization because QoS effects are most likely to appear near saturation.
    """
    metric = "QoS_Violation_Ratio"
    max_util = float(df["utilization"].max())

    high = df[df["utilization"] == max_util].copy()

    grouped = (
        high.groupby(
            ["Source_Workload_Name", "R_kw_per_server", "Pbar_kw_per_server"],
            as_index=False,
        )[metric]
        .agg(agg)
    )

    workloads = ordered_workloads(df)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), sharex=False, sharey=False)
    axes = axes.flatten()

    all_values = grouped[metric].to_numpy(dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    vmin = float(np.nanmin(finite_values)) if len(finite_values) else None
    vmax = float(np.nanmax(finite_values)) if len(finite_values) else None

    last_im = None

    for ax_idx, workload in enumerate(workloads[:4]):
        ax = axes[ax_idx]
        sub = grouped[grouped["Source_Workload_Name"] == workload]

        pivot = sub.pivot(
            index="R_kw_per_server",
            columns="Pbar_kw_per_server",
            values=metric,
        ).sort_index().sort_index(axis=1)

        matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")

        last_im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(workload)
        ax.set_xlabel("Pbar kW/server")
        ax.set_ylabel("R kW/server")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45, ha="right")

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])

    for idx in range(len(workloads), len(axes)):
        axes[idx].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85)
        cbar.set_label(metric)

    fig.suptitle(
        f"QoS violation ratio over P/R at utilization={max_util}\n"
        "QoS shown separately from raw RSR cost",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 0.92, 0.94])

    output_path = out_dir / "04_qos_violation_heatmaps_high_utilization.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_05_tracking_qos_tradeoff(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    """
    Useful meeting plot: shows whether P/R settings with higher tracking cost
    also tend to create QoS issues.
    """
    grouped = (
        df.groupby(
            [
                "Source_Workload_Name",
                "Pbar_kw_per_server",
                "R_kw_per_server",
                "utilization",
            ],
            as_index=False,
        )
        .agg({
            "Simulator_Tracking_Cost": agg,
            "QoS_Violation_Ratio": agg,
        })
    )

    workloads = ordered_workloads(df)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), sharex=False, sharey=False)
    axes = axes.flatten()

    last_scatter = None

    for ax_idx, workload in enumerate(workloads[:4]):
        ax = axes[ax_idx]
        sub = grouped[grouped["Source_Workload_Name"] == workload]

        last_scatter = ax.scatter(
            sub["Simulator_Tracking_Cost"],
            sub["QoS_Violation_Ratio"],
            c=sub["utilization"],
            s=35,
            alpha=0.75,
            cmap="viridis",
        )

        ax.set_title(workload)
        ax.set_xlabel("Simulator tracking cost")
        ax.set_ylabel("QoS violation ratio")
        ax.grid(True, alpha=0.25)

    for idx in range(len(workloads), len(axes)):
        axes[idx].axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist(), shrink=0.85)
        cbar.set_label("Utilization")

    fig.suptitle(
        "Tracking cost vs QoS tradeoff\n"
        f"Each point is a workload/P/R/utilization setting, aggregated by {agg} over weights",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 0.92, 0.94])

    output_path = out_dir / "05_tracking_vs_qos_tradeoff.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate useful FlexDC presentation plots without RSR-total focus."
    )

    parser.add_argument(
        "--raw-csv",
        type=str,
        default=None,
        help="Path to expanded combined raw CSV. If omitted, searches common folders.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )

    parser.add_argument(
        "--agg",
        type=str,
        default="median",
        choices=["median", "mean", "min"],
        help="Aggregation across weights/repeated settings. Default: median.",
    )

    args = parser.parse_args()

    raw_csv = Path(args.raw_csv) if args.raw_csv else find_raw_csv()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {raw_csv}")
    df = pd.read_csv(raw_csv)
    df = clean_df(df)
    validate_df(df)

    print(f"Rows: {len(df)}")
    print(f"Workloads: {sorted(df['Source_Workload_Name'].unique())}")
    print(f"Pbar values: {[round(x, 4) for x in sorted(df['Pbar_kw_per_server'].unique())]}")
    print(f"R values: {[round(x, 4) for x in sorted(df['R_kw_per_server'].unique())]}")
    print(f"Utilizations: {[round(x, 4) for x in sorted(df['utilization'].unique())]}")
    print(f"Output directory: {out_dir}")

    generated = [
        plot_01_valid_grid(df, out_dir),
        plot_02_tracking_sensitivity_heatmap(df, out_dir, args.agg),
        plot_03_tracking_delta_curves(df, out_dir, args.agg),
        plot_04_qos_high_util_heatmaps(df, out_dir, args.agg),
        plot_05_tracking_qos_tradeoff(df, out_dir, args.agg),
    ]

    print("\nGenerated plots:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()