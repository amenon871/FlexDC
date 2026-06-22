import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


DEFAULT_RESULTS_FILENAME = "final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_results.csv"
DEFAULT_DIAGNOSTICS_FILENAME = "final_aqa_s250_1000_u060_080_prchunk10_combined_grid_search_diagnostics.csv"
DEFAULT_OUT_DIR = Path("final_aqa_presentation_figures")

DEFAULT_SEARCH_DIRS = [
    Path("."),
    Path("combined_output_final_aqa_s250_1000_u060_080_prchunk10"),
    Path("06112026combined_output_targetedpr"),
    Path("src") / "peacsim" / "output" / "optimization",
]

BASE_REQUIRED_RESULTS_COLUMNS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Policy_Name",
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "P_actual_watts",
    "R_actual_watts",
    "Pbar_ratio",
    "R_ratio",
    "server_count",
    "utilization",
    "workload_mix_size",
    "Simulator_RSR_Total_Cost",
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
]

BASE_REQUIRED_DIAGNOSTIC_COLUMNS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
    "PR_Total_Pairs_Before_Chunk",
    "PR_Chunk_Index",
    "PR_Num_Chunks",
    "PR_Chunk_Start_Index",
    "PR_Chunk_End_Index_Exclusive",
    "PR_Pairs_In_Chunk",
    "Weight_Equal_Value",
    "Weight_Relative_Lower_Bound",
    "Weight_Relative_Upper_Bound",
    "Weight_Server_Lower_Bound",
    "Weight_Upper_From_Lower_Bound",
    "Weight_Final_Lower_Bound",
    "Weight_Final_Upper_Bound",
    "Weight_Min",
    "Weight_Max",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]

SURFACE_METRICS = [
    "Simulator_RSR_Total_Cost",
    "Simulator_Power_Cost",
    "Mtrack_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]


def slugify(value) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "value"


def find_csv(explicit_path: str | None, default_filename: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find explicit CSV path: {path}")
        return path

    for folder in DEFAULT_SEARCH_DIRS:
        candidate = folder / default_filename
        if candidate.exists():
            return candidate

    matches = sorted(Path(".").glob(f"**/{default_filename}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {default_filename}. Pass it explicitly with --results-csv or --diagnostics-csv."
    )


def ensure_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def parse_weights_value(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return [float(x) for x in parsed]
    except Exception as exc:
        raise ValueError(f"Could not parse weights value: {value}") from exc


def expand_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_weight_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"Weight_\d+", str(c))],
        key=lambda c: int(c.split("_")[1]),
    )
    if existing_weight_cols:
        return df

    if "weights" not in df.columns:
        raise ValueError("No Weight_i columns and no JSON weights column found.")

    parsed = df["weights"].apply(parse_weights_value)
    max_len = int(parsed.apply(len).max())

    insert_at = df.columns.get_loc("weights") + 1
    for i in range(max_len):
        values = parsed.apply(lambda w: float(w[i]) if i < len(w) else np.nan)
        df.insert(insert_at + i, f"Weight_{i}", values)

    df["Weight_Vector_Length"] = parsed.apply(len)
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Pbar_kw_per_server", "R_kw_per_server", "utilization"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(10)
    if "server_count" in df.columns:
        df["server_count"] = df["server_count"].astype(int)
    return df


def load_and_merge(results_csv: Path, diagnostics_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    ensure_columns(results, BASE_REQUIRED_RESULTS_COLUMNS, "results CSV")
    ensure_columns(diagnostics, BASE_REQUIRED_DIAGNOSTIC_COLUMNS, "diagnostics CSV")

    results = clean_numeric_columns(expand_weights(results))
    diagnostics = clean_numeric_columns(diagnostics)

    merge_keys = [
        "Source_Output_Dir",
        "Iteration",
        "Weight_Sample_ID",
        "Workload_Name",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "server_count",
        "utilization",
    ]

    diag_extra_cols = [c for c in diagnostics.columns if c not in results.columns or c in merge_keys]
    diagnostics_subset = diagnostics[diag_extra_cols].copy()

    merged = results.merge(
        diagnostics_subset,
        on=merge_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_diagnostics"),
    )

    return merged


def validate_dataset(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Merged dataset has zero rows.")

    audit = {}
    audit["rows"] = len(df)
    audit["workloads"] = sorted(df["Workload_Name"].dropna().unique().tolist())
    audit["server_counts"] = sorted(df["server_count"].dropna().unique().tolist())
    audit["utilizations"] = sorted(df["utilization"].dropna().unique().tolist())

    expected_p = np.round(df["Pbar_kw_per_server"] * 1000 * df["server_count"]).astype(int)
    expected_r = np.round(df["R_kw_per_server"] * 1000 * df["server_count"]).astype(int)
    audit["p_actual_watts_mismatches"] = int((df["P_actual_watts"].astype(int) != expected_p).sum())
    audit["r_actual_watts_mismatches"] = int((df["R_actual_watts"].astype(int) != expected_r).sum())

    rsr_error = (
        df["Simulator_RSR_Total_Cost"]
        - df["Simulator_Power_Cost"]
        - df["Mtrack_Cost"]
    ).abs()
    audit["max_rsr_total_error"] = float(rsr_error.max())

    weight_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"Weight_\d+", str(c))],
        key=lambda c: int(c.split("_")[1]),
    )
    weights = df[weight_cols]
    weight_sum = weights.sum(axis=1, skipna=True)
    weight_min_calc = weights.min(axis=1, skipna=True)
    weight_max_calc = weights.max(axis=1, skipna=True)

    audit["max_weight_sum_error"] = float((weight_sum - 1.0).abs().max())
    audit["max_weight_min_metadata_error"] = float((weight_min_calc - df["Weight_Min"]).abs().max())
    audit["max_weight_max_metadata_error"] = float((weight_max_calc - df["Weight_Max"]).abs().max())
    audit["weight_lower_bound_violations"] = int((df["Weight_Min"] < df["Weight_Final_Lower_Bound"] - 1e-8).sum())
    audit["weight_upper_bound_violations"] = int((df["Weight_Max"] > df["Weight_Final_Upper_Bound"] + 1e-8).sum())

    pr_pairs = df.groupby(["Workload_Name", "server_count"])[["Pbar_kw_per_server", "R_kw_per_server"]].apply(
        lambda x: len(x.drop_duplicates())
    )
    audit["pr_pairs_per_workload_server"] = pr_pairs.to_dict()

    chunk_counts = df.groupby(["Workload_Name", "server_count"])["PR_Chunk_Index"].nunique()
    audit["chunks_per_workload_server"] = chunk_counts.to_dict()

    weights_per_setting = df.groupby([
        "Workload_Name",
        "server_count",
        "utilization",
        "Pbar_kw_per_server",
        "R_kw_per_server",
    ])["Weight_Sample_ID"].nunique()
    audit["weight_samples_per_setting_min"] = int(weights_per_setting.min())
    audit["weight_samples_per_setting_max"] = int(weights_per_setting.max())

    hard_failures = []
    if audit["p_actual_watts_mismatches"]:
        hard_failures.append("P_actual_watts conversion mismatch")
    if audit["r_actual_watts_mismatches"]:
        hard_failures.append("R_actual_watts conversion mismatch")
    if audit["max_rsr_total_error"] > 1e-6:
        hard_failures.append("Simulator_RSR_Total_Cost != Simulator_Power_Cost + Mtrack_Cost")
    if audit["max_weight_sum_error"] > 1e-8:
        hard_failures.append("weight vectors do not sum to 1")
    if audit["weight_lower_bound_violations"] or audit["weight_upper_bound_violations"]:
        hard_failures.append("weight bound violation")
    if audit["weight_samples_per_setting_min"] != audit["weight_samples_per_setting_max"]:
        hard_failures.append("inconsistent number of weight samples per P/R/utilization setting")

    if hard_failures:
        raise ValueError(f"Dataset validation failed: {hard_failures}\nAudit: {audit}")

    return audit


def save_audit_summary(audit: dict, out_dir: Path) -> None:
    lines = []
    for key, value in audit.items():
        lines.append({"check": key, "value": str(value)})
    pd.DataFrame(lines).to_csv(out_dir / "00_dataset_audit_summary.csv", index=False)


def ordered_workloads(df: pd.DataFrame) -> list[str]:
    return sorted(df["Workload_Name"].dropna().unique().tolist())


def make_heatmap_from_pivot(
    pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    output_path: Path,
    annotate: bool = False,
    integer_labels: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    matrix = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])

    if annotate:
        for yi, row_value in enumerate(pivot.index):
            for xi, col_value in enumerate(pivot.columns):
                value = pivot.loc[row_value, col_value]
                if pd.notna(value):
                    label = f"{int(value)}" if integer_labels else f"{value:.3g}"
                    ax.text(xi, yi, label, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_01_row_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        df.groupby(["Workload_Name", "server_count", "utilization"])
        .size()
        .reset_index(name="row_count")
    )
    counts["group"] = (
        counts["Workload_Name"].astype(str)
        + "\nS=" + counts["server_count"].astype(str)
        + ", U=" + counts["utilization"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.bar(counts["group"], counts["row_count"])
    ax.set_title("Row coverage by workload, server count, and utilization")
    ax.set_xlabel("Workload / server / utilization")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = out_dir / "01_row_counts_by_Workload_Name_server_count_utilization.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    counts.to_csv(out_dir / "01_row_counts_by_Workload_Name_server_count_utilization.csv", index=False)
    return output_path


def plot_02_pr_chunk_coverage(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        df.groupby(["Workload_Name", "server_count", "PR_Chunk_Index"])
        .size()
        .reset_index(name="row_count")
    )
    counts["group"] = counts["Workload_Name"].astype(str) + " | S=" + counts["server_count"].astype(str)
    pivot = counts.pivot(index="group", columns="PR_Chunk_Index", values="row_count").sort_index().sort_index(axis=1)

    output_path = out_dir / "02_PR_Chunk_Index_row_count_coverage.png"
    make_heatmap_from_pivot(
        pivot,
        title="P/R chunk coverage by workload/server group",
        xlabel="PR_Chunk_Index",
        ylabel="Workload | server_count",
        colorbar_label="Rows per chunk",
        output_path=output_path,
        annotate=True,
        integer_labels=True,
    )
    counts.to_csv(out_dir / "02_PR_Chunk_Index_row_count_coverage.csv", index=False)
    return output_path


def plot_03_pr_grid_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = (
        df.groupby(["R_kw_per_server", "Pbar_kw_per_server"])
        .size()
        .reset_index(name="row_count")
    )
    pivot = counts.pivot(index="R_kw_per_server", columns="Pbar_kw_per_server", values="row_count").sort_index().sort_index(axis=1)
    pivot.columns = [f"{x:.3f}" for x in pivot.columns]
    pivot.index = [f"{x:.3f}" for x in pivot.index]

    output_path = out_dir / "03_Pbar_kw_per_server_R_kw_per_server_grid_row_counts.png"
    make_heatmap_from_pivot(
        pivot,
        title="P/R grid row counts across combined dataset",
        xlabel="Pbar_kw_per_server",
        ylabel="R_kw_per_server",
        colorbar_label="Rows",
        output_path=output_path,
        annotate=False,
    )
    counts.to_csv(out_dir / "03_Pbar_kw_per_server_R_kw_per_server_grid_row_counts.csv", index=False)
    return output_path


def plot_04_weight_bounds(df: pd.DataFrame, out_dir: Path) -> Path:
    grouped = (
        df.groupby(["Workload_Name", "server_count"], as_index=False)
        .agg({
            "Weight_Equal_Value": "first",
            "Weight_Relative_Lower_Bound": "first",
            "Weight_Relative_Upper_Bound": "first",
            "Weight_Server_Lower_Bound": "first",
            "Weight_Upper_From_Lower_Bound": "first",
            "Weight_Final_Lower_Bound": "first",
            "Weight_Final_Upper_Bound": "first",
            "Weight_Min": "min",
            "Weight_Max": "max",
        })
    )
    grouped["group"] = grouped["Workload_Name"].astype(str) + "\nS=" + grouped["server_count"].astype(str)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    x = np.arange(len(grouped))
    ax.scatter(x, grouped["Weight_Equal_Value"], marker="o", s=70, label="Weight_Equal_Value")
    ax.scatter(x, grouped["Weight_Final_Lower_Bound"], marker="v", s=80, label="Weight_Final_Lower_Bound")
    ax.scatter(x, grouped["Weight_Final_Upper_Bound"], marker="^", s=80, label="Weight_Final_Upper_Bound")
    ax.scatter(x, grouped["Weight_Min"], marker="x", s=90, label="Observed min Weight_Min")
    ax.scatter(x, grouped["Weight_Max"], marker="x", s=90, label="Observed max Weight_Max")

    for i, row in grouped.iterrows():
        ax.vlines(i, row["Weight_Final_Lower_Bound"], row["Weight_Final_Upper_Bound"], alpha=0.3, linewidth=4)

    ax.set_title("Final weight bounds and observed weight range")
    ax.set_xlabel("Workload / server_count")
    ax.set_ylabel("Weight value")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["group"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    output_path = out_dir / "04_weight_bounds_and_observed_Weight_Min_Weight_Max.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    grouped.to_csv(out_dir / "04_weight_bounds_and_observed_Weight_Min_Weight_Max.csv", index=False)
    return output_path


def plot_05_weight_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    weight_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"Weight_\d+", str(c))],
        key=lambda c: int(c.split("_")[1]),
    )
    groups = list(df.groupby(["Workload_Name", "server_count"]))
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, ((workload, server_count), sub) in enumerate(groups):
        ax = axes[ax_idx]
        values = sub[weight_cols].to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        ax.hist(values, bins=40, density=True, alpha=0.75)

        equal_value = float(sub["Weight_Equal_Value"].iloc[0])
        final_lower = float(sub["Weight_Final_Lower_Bound"].iloc[0])
        final_upper = float(sub["Weight_Final_Upper_Bound"].iloc[0])

        ax.axvline(equal_value, linestyle="--", linewidth=1.5, color="black", label="Weight_Equal_Value")
        ax.axvline(final_lower, linestyle=":", linewidth=2.0, color="red", label="Weight_Final_Lower_Bound")
        ax.axvline(final_upper, linestyle=":", linewidth=2.0, color="red", label="Weight_Final_Upper_Bound")
        ax.set_title(f"{workload}, server_count={server_count}")
        ax.set_xlabel("Weight_i value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Generated weight distribution with final bounds", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = out_dir / "05_weight_distribution_with_Weight_Final_bounds.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_06_mtrack_delta_curves(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    metric = "Mtrack_Cost"
    grouped = (
        df.groupby(["Workload_Name", "server_count", "Pbar_kw_per_server", "R_kw_per_server"], as_index=False)[metric]
        .agg(agg)
    )

    groups = list(grouped.groupby(["Workload_Name", "server_count"]))
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, 5.0 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, ((workload, server_count), sub) in enumerate(groups):
        ax = axes[ax_idx]
        for pbar in sorted(sub["Pbar_kw_per_server"].unique()):
            line = sub[sub["Pbar_kw_per_server"] == pbar].sort_values("R_kw_per_server")
            if len(line) < 2:
                continue
            baseline = float(line[metric].iloc[0])
            delta = line[metric] - baseline
            ax.plot(line["R_kw_per_server"], delta, marker="o", linewidth=1.4, label=f"Pbar={pbar:.3f}")

        ax.set_title(f"{workload}, server_count={server_count}")
        ax.set_xlabel("R_kw_per_server")
        ax.set_ylabel(f"Increase in {metric} from lowest R")
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.04, 0.5), fontsize=8)
    fig.suptitle(f"{metric} sensitivity to R, aggregated by {agg}", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    output_path = out_dir / f"06_{metric}_increase_vs_R_kw_per_server_by_Workload_Name_server_count.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_07_qos_high_util_heatmaps(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    metric = "QoS_Violation_Ratio"
    max_util = float(df["utilization"].max())
    high = df[df["utilization"] == max_util].copy()
    grouped = (
        high.groupby(["Workload_Name", "server_count", "R_kw_per_server", "Pbar_kw_per_server"], as_index=False)[metric]
        .agg(agg)
    )

    groups = list(grouped.groupby(["Workload_Name", "server_count"]))
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, 5.3 * nrows), squeeze=False)
    axes = axes.flatten()

    all_values = grouped[metric].to_numpy(dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    vmin = float(np.nanmin(finite_values)) if len(finite_values) else None
    vmax = float(np.nanmax(finite_values)) if len(finite_values) else None
    last_im = None

    for ax_idx, ((workload, server_count), sub) in enumerate(groups):
        ax = axes[ax_idx]
        pivot = sub.pivot(index="R_kw_per_server", columns="Pbar_kw_per_server", values=metric).sort_index().sort_index(axis=1)
        matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        last_im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{workload}, server_count={server_count}")
        ax.set_xlabel("Pbar_kw_per_server")
        ax.set_ylabel("R_kw_per_server")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])

    for ax in axes[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85)
        cbar.set_label(metric)
    fig.suptitle(f"{metric} over P/R at utilization={max_util}, aggregated by {agg}", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    output_path = out_dir / f"07_{metric}_heatmap_highest_utilization_{max_util}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_08_mtrack_vs_qos_tradeoff(df: pd.DataFrame, out_dir: Path, agg: str) -> Path:
    grouped = (
        df.groupby(["Workload_Name", "server_count", "Pbar_kw_per_server", "R_kw_per_server", "utilization"], as_index=False)
        .agg({"Mtrack_Cost": agg, "QoS_Violation_Ratio": agg})
    )

    groups = list(grouped.groupby(["Workload_Name", "server_count"]))
    n = len(groups)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.0, 5.0 * nrows), squeeze=False)
    axes = axes.flatten()
    last_scatter = None

    for ax_idx, ((workload, server_count), sub) in enumerate(groups):
        ax = axes[ax_idx]
        last_scatter = ax.scatter(
            sub["Mtrack_Cost"],
            sub["QoS_Violation_Ratio"],
            c=sub["utilization"],
            s=34,
            alpha=0.75,
            cmap="viridis",
        )
        ax.set_title(f"{workload}, server_count={server_count}")
        ax.set_xlabel("Mtrack_Cost")
        ax.set_ylabel("QoS_Violation_Ratio")
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.tolist(), shrink=0.85)
        cbar.set_label("utilization")
    fig.suptitle(f"Mtrack_Cost vs QoS_Violation_Ratio, aggregated by {agg} over weights", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    output_path = out_dir / "08_Mtrack_Cost_vs_QoS_Violation_Ratio_tradeoff.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_heatmaps(df: pd.DataFrame, out_dir: Path, metrics: Iterable[str], agg: str) -> list[Path]:
    heatmap_dir = out_dir / "09_PR_metric_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        metric_dir = heatmap_dir / slugify(metric)
        metric_dir.mkdir(parents=True, exist_ok=True)
        grouped = (
            df.groupby(["Workload_Name", "server_count", "R_kw_per_server", "Pbar_kw_per_server"], as_index=False)[metric]
            .agg(agg)
        )
        for (workload, server_count), sub in grouped.groupby(["Workload_Name", "server_count"]):
            pivot = sub.pivot(index="R_kw_per_server", columns="Pbar_kw_per_server", values=metric).sort_index().sort_index(axis=1)
            pivot.columns = [f"{x:.3f}" for x in pivot.columns]
            pivot.index = [f"{x:.3f}" for x in pivot.index]
            output_path = metric_dir / f"heatmap_{slugify(metric)}_{agg}_{slugify(workload)}_server_count_{server_count}.png"
            make_heatmap_from_pivot(
                pivot,
                title=f"{metric} over P/R\n{workload}, server_count={server_count}, {agg} over utilization/weights",
                xlabel="Pbar_kw_per_server",
                ylabel="R_kw_per_server",
                colorbar_label=metric,
                output_path=output_path,
            )
            generated.append(output_path)
    return generated


def plot_3d_surfaces(df: pd.DataFrame, out_dir: Path, metrics: Iterable[str], agg: str) -> list[Path]:
    surface_dir = out_dir / "10_PR_metric_3D_surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for metric in metrics:
        if metric not in df.columns:
            continue
        metric_dir = surface_dir / slugify(metric)
        metric_dir.mkdir(parents=True, exist_ok=True)
        grouped = (
            df.groupby(["Workload_Name", "server_count", "Pbar_kw_per_server", "R_kw_per_server"], as_index=False)[metric]
            .agg(agg)
        )

        for (workload, server_count), sub in grouped.groupby(["Workload_Name", "server_count"]):
            sub = sub.dropna(subset=["Pbar_kw_per_server", "R_kw_per_server", metric])
            if len(sub) < 3:
                continue
            x = sub["Pbar_kw_per_server"].to_numpy(dtype=float)
            y = sub["R_kw_per_server"].to_numpy(dtype=float)
            z = sub[metric].to_numpy(dtype=float)

            fig = plt.figure(figsize=(10.5, 7.8))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, antialiased=True, alpha=0.95)
            ax.scatter(x, y, z, s=12, alpha=0.35)
            ax.set_title(f"3D surface: {metric}\n{workload}, server_count={server_count}, {agg} over utilization/weights")
            ax.set_xlabel("Pbar_kw_per_server")
            ax.set_ylabel("R_kw_per_server")
            ax.set_zlabel(metric)
            ax.view_init(elev=28, azim=-135)
            fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12, label=metric)
            fig.tight_layout()

            output_path = metric_dir / f"3d_surface_{slugify(metric)}_{agg}_{slugify(workload)}_server_count_{server_count}.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            generated.append(output_path)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Visualize final chunked FlexDC dataset with new column schema.")
    parser.add_argument("--results-csv", type=str, default=None, help="Combined grid_search_results CSV.")
    parser.add_argument("--diagnostics-csv", type=str, default=None, help="Combined grid_search_diagnostics CSV.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output plot directory.")
    parser.add_argument("--agg", type=str, default="median", choices=["median", "mean", "min"], help="Aggregation over repeated settings.")
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D surface generation.")
    parser.add_argument("--make-metric-heatmaps", action="store_true", help="Also generate per-metric 2D P/R heatmaps for every workload/server group. This can create many files.")
    args = parser.parse_args()

    results_csv = find_csv(args.results_csv, DEFAULT_RESULTS_FILENAME)
    diagnostics_csv = find_csv(args.diagnostics_csv, DEFAULT_DIAGNOSTICS_FILENAME)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results: {results_csv}")
    print(f"Reading diagnostics: {diagnostics_csv}")
    print(f"Output directory: {out_dir}")

    df = load_and_merge(results_csv, diagnostics_csv)
    audit = validate_dataset(df)
    save_audit_summary(audit, out_dir)

    print("\nDataset audit summary")
    print("---------------------")
    for key, value in audit.items():
        print(f"{key}: {value}")

    metrics = [m for m in SURFACE_METRICS if m in df.columns]
    generated = []
    generated.append(plot_01_row_counts(df, out_dir))
    generated.append(plot_02_pr_chunk_coverage(df, out_dir))
    generated.append(plot_03_pr_grid_counts(df, out_dir))
    generated.append(plot_04_weight_bounds(df, out_dir))
    generated.append(plot_05_weight_distribution(df, out_dir))
    generated.append(plot_06_mtrack_delta_curves(df, out_dir, args.agg))
    generated.append(plot_07_qos_high_util_heatmaps(df, out_dir, args.agg))
    generated.append(plot_08_mtrack_vs_qos_tradeoff(df, out_dir, args.agg))
    if args.make_metric_heatmaps:
        generated.extend(plot_metric_heatmaps(df, out_dir, metrics, args.agg))
    if not args.skip_3d:
        generated.extend(plot_3d_surfaces(df, out_dir, metrics, args.agg))

    print("\nGenerated files:")
    for path in generated:
        print(f"  {path}")
    print(f"\nAudit CSV: {out_dir / '00_dataset_audit_summary.csv'}")


if __name__ == "__main__":
    main()
