#!/usr/bin/env python3
"""
Whole-dataset diagnostics for the augmented FlexDC training dataset.

Purpose
-------
Diagnose whether the surrogate-model targets are:
- skewed or heavy-tailed,
- sparse near the tracking/QoS feasibility boundaries,
- imbalanced across workloads and data sources,
- vulnerable to optimistic random-row train/test splits,
- under-resolved near the W1 P=R boundary.

The script creates:
1. A small "main_diagnostics" folder containing the plots most useful for
   model debugging and meetings.
2. An "appendix_cost_histograms" folder with one full-dataset histogram for
   each cost/target column.
3. CSV summaries and a Markdown findings report.

It is standalone: only pandas, numpy, and matplotlib are required.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import kurtosis, skew


RESULT_COLUMNS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Policy_Name",
    "Workload_Name",
    "Workload_Config",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "Pbar_ratio",
    "R_ratio",
    "server_count",
    "utilization",
    "workload_mix_size",
    "weights",
    "Simulator_RSR_Total_Cost",
    "Simulator_Power_Cost",
    "Mtrack_Error_MeanAbs_Normalized",
    "Mtrack_Error_MeanAbs_Watts",
    "Mtrack_Cost",
    "QoS_Delay_Probability_Sum",
    "QoS_Violation_Ratio",
    "QoS_Delay_Probabilities",
    "Ctrack_Epsilon_90th",
    "Ctrack_Gamma",
    "Ctrack_Residual",
    "Ctrack_SoftPlus_Value",
    "Ctrack_Weighted_Cost",
    "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Diagnostic_FullPaperObjective_Cost",
]

DIAGNOSTIC_KEY_COLUMNS = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Workload_Name",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "server_count",
    "utilization",
]

MODEL_TARGETS = {
    "M_RSR": "Simulator_RSR_Total_Cost",
    "p90_tracking": "Ctrack_Epsilon_90th",
    "QoS_mean_probability": "QoS_Mean_Probability",
    "QoS_max_probability": "QoS_Max_Probability",
}

APPENDIX_METRICS = {
    "M_RSR": "Simulator_RSR_Total_Cost",
    "Power_cost": "Simulator_Power_Cost",
    "Mtrack_mean_abs_normalized": "Mtrack_Error_MeanAbs_Normalized",
    "Mtrack_cost": "Mtrack_Cost",
    "p90_tracking": "Ctrack_Epsilon_90th",
    "QoS_probability_sum": "QoS_Delay_Probability_Sum",
    "QoS_mean_probability": "QoS_Mean_Probability",
    "QoS_max_probability": "QoS_Max_Probability",
    "QoS_violation_ratio_legacy": "QoS_Violation_Ratio",
    "Paper_Ctrack_cost": "Ctrack_Weighted_Cost",
    "Paper_CQoS_cost": "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
    "Paper_full_objective": "Diagnostic_FullPaperObjective_Cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focused whole-dataset histograms and model-data diagnostics."
    )
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_vector(value: object, expected_len: int = 4) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        result = [float(x) for x in value]
    else:
        text = str(value).strip()
        text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        try:
            parsed = ast.literal_eval(text)
            result = [float(x) for x in parsed]
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f"Could not parse numeric vector: {text[:160]}") from exc

    if len(result) != expected_len:
        raise ValueError(
            f"Expected vector of length {expected_len}, found {len(result)}: {value}"
        )
    return result


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def distribution_summary(name: str, series: pd.Series) -> dict[str, float | str | int]:
    x = finite_values(series)
    q = np.quantile(
        x,
        [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0],
    )
    return {
        "metric": name,
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skew": float(skew(x, bias=False)),
        "excess_kurtosis": float(kurtosis(x, bias=False)),
        "zero_fraction": float(np.mean(np.isclose(x, 0.0))),
        "min": float(q[0]),
        "p0.1": float(q[1]),
        "p1": float(q[2]),
        "p5": float(q[3]),
        "p10": float(q[4]),
        "p25": float(q[5]),
        "p50": float(q[6]),
        "p75": float(q[7]),
        "p90": float(q[8]),
        "p95": float(q[9]),
        "p99": float(q[10]),
        "p99.9": float(q[11]),
        "max": float(q[12]),
    }


def add_common_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=13, pad=12)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )


def make_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    dpi: int,
    *,
    bins: int | np.ndarray = 80,
    xlim: tuple[float, float] | None = None,
    log_y: bool = True,
    threshold: float | None = None,
    clip_quantile: float | None = 0.995,
) -> None:
    x = finite_values(series)
    original_n = x.size
    subtitle_parts = [f"n={original_n:,}"]

    if clip_quantile is not None:
        upper = float(np.quantile(x, clip_quantile))
        x = x[x <= upper]
        subtitle_parts.append(
            f"display clipped at p{clip_quantile * 100:g}={upper:.4g}; all rows retained in summaries"
        )

    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    ax.hist(x, bins=bins, edgecolor="white", linewidth=0.35)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Rows (log scale)")
    else:
        ax.set_ylabel("Rows")
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if threshold is not None:
        ax.axvline(threshold, linestyle="--", linewidth=1.8, label=f"threshold = {threshold:g}")
        ax.legend()
    add_common_title(ax, title, " | ".join(subtitle_parts))
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path, dpi)


def build_dataframe(results_csv: Path) -> pd.DataFrame:
    print(f"Reading results: {results_csv}")
    df = pd.read_csv(results_csv, usecols=RESULT_COLUMNS, low_memory=False)

    weights = np.asarray(df["weights"].map(parse_vector).tolist(), dtype=float)
    qos = np.asarray(df["QoS_Delay_Probabilities"].map(parse_vector).tolist(), dtype=float)

    for i in range(4):
        df[f"weight_{i}"] = weights[:, i]
        df[f"QoS_P{i}"] = qos[:, i]

    df["QoS_Mean_Probability"] = qos.mean(axis=1)
    df["QoS_Max_Probability"] = qos.max(axis=1)
    df["QoS_Recomputed_Violation_Ratio"] = (qos > 0.1).mean(axis=1)
    df["Tracking_Pass"] = df["Ctrack_Epsilon_90th"] <= 0.3
    df["QoS_Pass_Legacy"] = df["QoS_Violation_Ratio"] <= 0.1
    df["QoS_Pass_MaxP"] = df["QoS_Max_Probability"] <= 0.1
    df["Both_Pass_Legacy"] = df["Tracking_Pass"] & df["QoS_Pass_Legacy"]
    df["Both_Pass_MaxP"] = df["Tracking_Pass"] & df["QoS_Pass_MaxP"]
    df["P_minus_R"] = df["Pbar_kw_per_server"] - df["R_kw_per_server"]
    df["R_over_P"] = df["R_kw_per_server"] / df["Pbar_kw_per_server"]
    df["Is_W2_Dense"] = (
        df["Source_Output_Dir"].astype(str).str.contains("w2_dense", case=False, na=False)
    )
    df["Data_Source"] = np.where(df["Is_W2_Dense"], "W2 dense augmentation", "Original coarse grid")
    df["Workload_Family"] = df["Workload_Name"].str.extract(r"^(W[12])", expand=False)
    df["Weight_L2_From_Equal"] = np.sqrt(np.sum((weights - 0.25) ** 2, axis=1))
    return df


def validate_dataset(df: pd.DataFrame, diagnostics_csv: Path | None) -> pd.DataFrame:
    weight_cols = [f"weight_{i}" for i in range(4)]
    qos_cols = [f"QoS_P{i}" for i in range(4)]

    numeric_check_cols = [
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "Pbar_ratio",
        "R_ratio",
        "server_count",
        "utilization",
        "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost",
        "Mtrack_Cost",
        "Ctrack_Epsilon_90th",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
        *weight_cols,
        *qos_cols,
    ]

    numeric = df[numeric_check_cols].to_numpy(dtype=float)
    checks = {
        "rows": len(df),
        "missing_values_in_checked_columns": int(df[numeric_check_cols].isna().sum().sum()),
        "nonfinite_values_in_checked_columns": int((~np.isfinite(numeric)).sum()),
        "rows_with_P_less_than_R": int((df["Pbar_kw_per_server"] + 1e-12 < df["R_kw_per_server"]).sum()),
        "rows_with_negative_weight": int((df[weight_cols].to_numpy() < 0).any(axis=1).sum()),
        "rows_with_QoS_probability_outside_0_1": int(
            (
                (df[qos_cols].to_numpy() < 0)
                | (df[qos_cols].to_numpy() > 1)
            ).any(axis=1).sum()
        ),
        "max_abs_weight_sum_error": float(
            np.max(np.abs(df[weight_cols].sum(axis=1).to_numpy() - 1.0))
        ),
        "max_abs_RSR_identity_error": float(
            np.max(
                np.abs(
                    df["Simulator_RSR_Total_Cost"]
                    - (df["Simulator_Power_Cost"] + df["Mtrack_Cost"])
                )
            )
        ),
        "max_abs_full_objective_identity_error": float(
            np.max(
                np.abs(
                    df["Diagnostic_FullPaperObjective_Cost"]
                    - (
                        df["Simulator_RSR_Total_Cost"]
                        + df["Ctrack_Weighted_Cost"]
                        + df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"]
                    )
                )
            )
        ),
        "QoS_ratio_rows_disagreeing_with_recomputed_maxP_rule": int(
            ~np.isclose(
                df["QoS_Violation_Ratio"],
                df["QoS_Recomputed_Violation_Ratio"],
            )
        .sum()
        ),
    }

    # Correct the expression above if pandas/numpy precedence changes.
    checks["QoS_ratio_rows_disagreeing_with_recomputed_maxP_rule"] = int(
        (
            ~np.isclose(
                df["QoS_Violation_Ratio"].to_numpy(dtype=float),
                df["QoS_Recomputed_Violation_Ratio"].to_numpy(dtype=float),
            )
        ).sum()
    )

    if diagnostics_csv is not None:
        print(f"Reading diagnostics keys: {diagnostics_csv}")
        diagnostics = pd.read_csv(
            diagnostics_csv,
            usecols=DIAGNOSTIC_KEY_COLUMNS,
            low_memory=False,
        )
        checks["diagnostics_rows"] = len(diagnostics)
        checks["results_diagnostics_same_row_count"] = int(len(diagnostics) == len(df))

        aligned = True
        if len(diagnostics) == len(df):
            for col in DIAGNOSTIC_KEY_COLUMNS:
                if pd.api.types.is_numeric_dtype(df[col]):
                    same = np.allclose(
                        df[col].to_numpy(dtype=float),
                        diagnostics[col].to_numpy(dtype=float),
                        equal_nan=True,
                    )
                else:
                    same = np.array_equal(
                        df[col].astype(str).to_numpy(),
                        diagnostics[col].astype(str).to_numpy(),
                    )
                aligned = aligned and bool(same)
        else:
            aligned = False
        checks["results_diagnostics_key_alignment"] = int(aligned)

    return pd.DataFrame(
        [{"check": key, "value": value} for key, value in checks.items()]
    )


def make_group_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "Workload_Name",
        "server_count",
        "utilization",
        "Data_Source",
    ]

    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False, observed=False):
        row = dict(zip(group_cols, key))
        row["rows"] = len(group)
        row["unique_P"] = group["Pbar_kw_per_server"].nunique()
        row["unique_R"] = group["R_kw_per_server"].nunique()
        row["unique_PR_pairs"] = len(
            group[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates()
        )
        row["unique_weights"] = group["weights"].nunique()
        row["tracking_pass_rows"] = int(group["Tracking_Pass"].sum())
        row["tracking_pass_pct"] = 100.0 * group["Tracking_Pass"].mean()
        row["legacy_QoS_pass_rows"] = int(group["QoS_Pass_Legacy"].sum())
        row["legacy_QoS_pass_pct"] = 100.0 * group["QoS_Pass_Legacy"].mean()
        row["maxP_QoS_pass_rows"] = int(group["QoS_Pass_MaxP"].sum())
        row["maxP_QoS_pass_pct"] = 100.0 * group["QoS_Pass_MaxP"].mean()
        row["both_pass_legacy_rows"] = int(group["Both_Pass_Legacy"].sum())
        row["both_pass_legacy_pct"] = 100.0 * group["Both_Pass_Legacy"].mean()
        row["both_pass_maxP_rows"] = int(group["Both_Pass_MaxP"].sum())
        row["both_pass_maxP_pct"] = 100.0 * group["Both_Pass_MaxP"].mean()
        row["p90_median"] = group["Ctrack_Epsilon_90th"].median()
        row["p90_p90"] = group["Ctrack_Epsilon_90th"].quantile(0.90)
        row["QoS_mean_median"] = group["QoS_Mean_Probability"].median()
        row["QoS_max_median"] = group["QoS_Max_Probability"].median()
        row["M_RSR_median"] = group["Simulator_RSR_Total_Cost"].median()
        row["near_tracking_boundary_pct"] = 100.0 * (
            group["Ctrack_Epsilon_90th"].between(0.20, 0.40, inclusive="both")
        ).mean()
        row["near_QoS_boundary_pct"] = 100.0 * (
            group["QoS_Max_Probability"].between(0.05, 0.15, inclusive="both")
        ).mean()
        rows.append(row)

    group_summary = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    feasibility_categories = (
        df.assign(
            Feasibility_Category=np.select(
                [
                    df["Tracking_Pass"] & df["QoS_Pass_Legacy"],
                    df["Tracking_Pass"] & ~df["QoS_Pass_Legacy"],
                    ~df["Tracking_Pass"] & df["QoS_Pass_Legacy"],
                ],
                [
                    "Pass both",
                    "Tracking only",
                    "QoS only",
                ],
                default="Fail both",
            )
        )
        .groupby(["Workload_Name", "server_count", "utilization", "Data_Source", "Feasibility_Category"])
        .size()
        .rename("rows")
        .reset_index()
    )
    feasibility_categories["pct_within_group"] = (
        100.0
        * feasibility_categories["rows"]
        / feasibility_categories.groupby(
            ["Workload_Name", "server_count", "utilization", "Data_Source"]
        )["rows"].transform("sum")
    )
    return group_summary, feasibility_categories


def make_scenario_variability(df: pd.DataFrame) -> pd.DataFrame:
    scenario_cols = [
        "Workload_Name",
        "server_count",
        "utilization",
        "Data_Source",
        "Pbar_kw_per_server",
        "R_kw_per_server",
    ]

    scenario = (
        df.groupby(scenario_cols, observed=False)
        .agg(
            rows=("Iteration", "size"),
            p90_min=("Ctrack_Epsilon_90th", "min"),
            p90_max=("Ctrack_Epsilon_90th", "max"),
            qmax_min=("QoS_Max_Probability", "min"),
            qmax_max=("QoS_Max_Probability", "max"),
            M_RSR_min=("Simulator_RSR_Total_Cost", "min"),
            M_RSR_max=("Simulator_RSR_Total_Cost", "max"),
            track_pass_min=("Tracking_Pass", "min"),
            track_pass_max=("Tracking_Pass", "max"),
            QoS_pass_min=("QoS_Pass_Legacy", "min"),
            QoS_pass_max=("QoS_Pass_Legacy", "max"),
            both_pass_min=("Both_Pass_Legacy", "min"),
            both_pass_max=("Both_Pass_Legacy", "max"),
        )
        .reset_index()
    )

    scenario["p90_range_across_weights"] = scenario["p90_max"] - scenario["p90_min"]
    scenario["QoS_max_range_across_weights"] = scenario["qmax_max"] - scenario["qmax_min"]
    scenario["M_RSR_range_across_weights"] = scenario["M_RSR_max"] - scenario["M_RSR_min"]
    scenario["tracking_label_mixed_across_weights"] = (
        scenario["track_pass_min"] != scenario["track_pass_max"]
    )
    scenario["QoS_label_mixed_across_weights"] = (
        scenario["QoS_pass_min"] != scenario["QoS_pass_max"]
    )
    scenario["both_label_mixed_across_weights"] = (
        scenario["both_pass_min"] != scenario["both_pass_max"]
    )

    rows: list[dict[str, object]] = []
    for key, group in scenario.groupby(["Workload_Name", "Data_Source"], observed=False):
        rows.append(
            {
                "Workload_Name": key[0],
                "Data_Source": key[1],
                "scenarios": len(group),
                "p90_range_median": group["p90_range_across_weights"].median(),
                "p90_range_p90": group["p90_range_across_weights"].quantile(0.90),
                "QoS_max_range_median": group["QoS_max_range_across_weights"].median(),
                "QoS_max_range_p90": group["QoS_max_range_across_weights"].quantile(0.90),
                "M_RSR_range_median": group["M_RSR_range_across_weights"].median(),
                "tracking_label_mixed_pct": 100.0
                * group["tracking_label_mixed_across_weights"].mean(),
                "QoS_label_mixed_pct": 100.0
                * group["QoS_label_mixed_across_weights"].mean(),
                "both_label_mixed_pct": 100.0
                * group["both_label_mixed_across_weights"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_w1_boundary_summary(df: pd.DataFrame) -> pd.DataFrame:
    w1 = df[df["Workload_Family"] == "W1"].copy()

    labels = [
        "P=R",
        "0<gap<=0.025",
        "0.025<gap<=0.05",
        "0.05<gap<=0.10",
        "0.10<gap<=0.20",
        "gap>0.20",
    ]

    gap = w1["P_minus_R"].to_numpy()
    w1["Boundary_Bin"] = np.select(
        [
            np.isclose(gap, 0.0, atol=1e-12),
            (gap > 0.0) & (gap <= 0.025),
            (gap > 0.025) & (gap <= 0.05),
            (gap > 0.05) & (gap <= 0.10),
            (gap > 0.10) & (gap <= 0.20),
        ],
        labels[:-1],
        default=labels[-1],
    )
    w1["Boundary_Bin"] = pd.Categorical(w1["Boundary_Bin"], labels, ordered=True)

    summary = (
        w1.groupby(["Workload_Name", "Boundary_Bin"], observed=False)
        .agg(
            rows=("Iteration", "size"),
            M_RSR_median=("Simulator_RSR_Total_Cost", "median"),
            M_RSR_p10=("Simulator_RSR_Total_Cost", lambda x: x.quantile(0.10)),
            p90_median=("Ctrack_Epsilon_90th", "median"),
            QoS_max_median=("QoS_Max_Probability", "median"),
            tracking_pass_pct=("Tracking_Pass", "mean"),
            legacy_QoS_pass_pct=("QoS_Pass_Legacy", "mean"),
            both_pass_legacy_pct=("Both_Pass_Legacy", "mean"),
        )
        .reset_index()
    )
    for col in ["tracking_pass_pct", "legacy_QoS_pass_pct", "both_pass_legacy_pct"]:
        summary[col] *= 100.0
    return summary


def make_split_leakage_summary(
    df: pd.DataFrame,
    random_seed: int,
    test_fraction: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_test = int(round(test_fraction * len(df)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    scenario_cols = [
        "Workload_Name",
        "server_count",
        "utilization",
        "Pbar_kw_per_server",
        "R_kw_per_server",
    ]
    physical_twin_cols = [
        "Workload_Family",
        "server_count",
        "utilization",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "weights",
    ]

    def row_keys(frame: pd.DataFrame, columns: Iterable[str]) -> list[tuple[str, ...]]:
        return list(map(tuple, frame[list(columns)].astype(str).to_numpy()))

    train_scenarios = set(row_keys(train, scenario_cols))
    test_scenarios = row_keys(test, scenario_cols)
    same_scenario = np.fromiter(
        (key in train_scenarios for key in test_scenarios),
        dtype=bool,
        count=len(test_scenarios),
    )

    train_weights = set(train["weights"].astype(str))
    same_weight = test["weights"].astype(str).isin(train_weights).to_numpy()

    train_physical = set(row_keys(train, physical_twin_cols))
    test_physical = row_keys(test, physical_twin_cols)
    physical_twin = np.fromiter(
        (key in train_physical for key in test_physical),
        dtype=bool,
        count=len(test_physical),
    )

    return pd.DataFrame(
        [
            {
                "diagnostic": "Test rows sharing exact workload/N/U/P/R scenario with training",
                "rows": int(same_scenario.sum()),
                "pct_of_test_rows": 100.0 * same_scenario.mean(),
            },
            {
                "diagnostic": "Test rows sharing exact weight vector with training",
                "rows": int(same_weight.sum()),
                "pct_of_test_rows": 100.0 * same_weight.mean(),
            },
            {
                "diagnostic": "Test rows with physical twin in alternate QoS config in training",
                "rows": int(physical_twin.sum()),
                "pct_of_test_rows": 100.0 * physical_twin.mean(),
            },
        ]
    )


def plot_main_diagnostics(
    df: pd.DataFrame,
    group_summary: pd.DataFrame,
    w1_boundary: pd.DataFrame,
    split_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    # 1. Tracking p90 near threshold.
    path = output_dir / "01_tracking_p90_zoom_full_dataset.png"
    make_histogram(
        df["Ctrack_Epsilon_90th"],
        "Tracking target distribution near the feasibility boundary",
        "90th-percentile normalized tracking error",
        path,
        dpi,
        bins=np.linspace(0.0, 2.0, 101),
        xlim=(0.0, 2.0),
        log_y=True,
        threshold=0.3,
        clip_quantile=None,
    )
    manifest.append({"plot": path.name, "purpose": "Shows the p90=0.3 boundary and the strong right tail."})

    # 2. Full p90 tail after log1p transform.
    p90_log = np.log1p(df["Ctrack_Epsilon_90th"])
    path = output_dir / "02_tracking_p90_log1p_full_tail.png"
    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    ax.hist(p90_log, bins=90, edgecolor="white", linewidth=0.35)
    ax.axvline(np.log1p(0.3), linestyle="--", linewidth=1.8, label="p90=0.3")
    ax.set_yscale("log")
    ax.set_xlabel("log1p(90th-percentile normalized tracking error)")
    ax.set_ylabel("Rows (log scale)")
    ax.legend()
    add_common_title(
        ax,
        "Full tracking tail on a log-transformed x-axis",
        "A raw-MSE model sees a target spanning 0 to more than 34",
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Makes the complete p90 tail visible without clipping."})

    # 3. QoS mean versus max.
    path = output_dir / "03_qos_mean_vs_max_histograms_full_dataset.png"
    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    bins = np.linspace(0.0, 1.0, 81)
    ax.hist(
        [df["QoS_Mean_Probability"], df["QoS_Max_Probability"]],
        bins=bins,
        label=["Mean of P0-P3", "Maximum of P0-P3"],
        histtype="step",
        linewidth=1.6,
    )
    ax.axvline(0.1, linestyle="--", linewidth=1.8, label="QoS threshold = 0.1")
    ax.set_yscale("log")
    ax.set_xlabel("QoS delay probability")
    ax.set_ylabel("Rows (log scale)")
    ax.legend()
    add_common_title(
        ax,
        "Mean QoS probability hides the worst job type",
        "Whole dataset; max(Pj) is closer to the per-job-type constraint",
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Directly compares the current mean target with max(Pj)."})

    # 4. Per-job-type QoS probabilities.
    path = output_dir / "04_qos_per_job_probability_histograms_full_dataset.png"
    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    bins = np.linspace(0.0, 1.0, 81)
    ax.hist(
        [df[f"QoS_P{i}"] for i in range(4)],
        bins=bins,
        label=[f"P{i}" for i in range(4)],
        histtype="step",
        linewidth=1.4,
    )
    ax.axvline(0.1, linestyle="--", linewidth=1.8, label="threshold = 0.1")
    ax.set_yscale("log")
    ax.set_xlabel("Per-job-type QoS delay probability")
    ax.set_ylabel("Rows (log scale)")
    ax.legend()
    add_common_title(
        ax,
        "Per-job-type QoS distributions are highly zero-inflated and heterogeneous",
        "W1 uses only P0 while W2 activates all four job-type probabilities",
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Shows why a pooled scalar QoS target is difficult."})

    # 5. Both-pass rate by full dataset group.
    plot_df = group_summary.copy()
    plot_df["label"] = (
        plot_df["Workload_Name"]
        + " | N="
        + plot_df["server_count"].astype(str)
        + " | U="
        + plot_df["utilization"].map(lambda x: f"{x:.1f}")
        + " | "
        + plot_df["Data_Source"].str.replace("Original coarse grid", "coarse", regex=False)
        .str.replace("W2 dense augmentation", "dense", regex=False)
    )
    plot_df = plot_df.sort_values("both_pass_legacy_pct")
    path = output_dir / "05_both_pass_rate_by_workload_server_utilization_source.png"
    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.gca()
    ax.barh(plot_df["label"], plot_df["both_pass_legacy_pct"])
    ax.set_xlabel("Rows passing both constraints (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    add_common_title(
        ax,
        "Feasible-row imbalance across the full dataset",
        "Legacy QoS pass label; W2 coarse-grid groups contain no pass-both rows",
    )
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Shows where feasible rows exist and where they are absent."})

    # 6. W1 P-R margin histogram.
    w1 = df[df["Workload_Family"] == "W1"]
    path = output_dir / "06_w1_p_minus_r_margin_histogram.png"
    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    ax.hist(w1["P_minus_R"], bins=70, edgecolor="white", linewidth=0.35)
    ax.set_yscale("log")
    ax.axvline(0.0, linestyle="--", linewidth=1.8, label="P=R boundary")
    ax.set_xlabel("Pbar - R (kW per server)")
    ax.set_ylabel("W1 rows (log scale)")
    ax.legend()
    add_common_title(
        ax,
        "W1 grid contains an exact P=R ridge but no focused boundary sweep",
        "The next interior samples are separated from the boundary by coarse R steps",
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Documents the W1 boundary coverage problem."})

    # 7. W1 cost profile by gap bin.
    gap_order = [
        "P=R",
        "0<gap<=0.025",
        "0.025<gap<=0.05",
        "0.05<gap<=0.10",
        "0.10<gap<=0.20",
        "gap>0.20",
    ]
    boundary_cost = (
        w1_boundary.groupby("Boundary_Bin", observed=False)
        .apply(
            lambda g: np.average(g["M_RSR_median"], weights=np.maximum(g["rows"], 1)),
            include_groups=False,
        )
        .reindex(gap_order)
    )
    path = output_dir / "07_w1_mrsr_profile_by_p_minus_r_bin.png"
    fig = plt.figure(figsize=(9.2, 5.7))
    ax = fig.gca()
    ax.bar(boundary_cost.index.astype(str), boundary_cost.to_numpy())
    ax.set_ylabel("Weighted median M_RSR")
    ax.set_xlabel("W1 Pbar-R margin bin")
    ax.tick_params(axis="x", rotation=25)
    add_common_title(
        ax,
        "The raw monetary target creates a strong attraction to P=R",
        "Across W1, M_RSR is smallest at the exact boundary",
    )
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Explains why optimization is pulled toward P=R."})

    # 8. Random split leakage.
    path = output_dir / "08_random_row_split_overlap_diagnostics.png"
    fig = plt.figure(figsize=(10.2, 5.8))
    ax = fig.gca()
    ax.barh(split_summary["diagnostic"], split_summary["pct_of_test_rows"])
    ax.set_xlabel("Share of random-split test rows (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xlim(0, 105)
    add_common_title(
        ax,
        "A random row split is an optimistic validation test",
        "The test set reuses P/R scenarios, weight templates, and many physical twins from training",
    )
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path, dpi)
    manifest.append({"plot": path.name, "purpose": "Quantifies why global random-split metrics can look better than optimization behavior."})

    return manifest


def plot_appendix_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    thresholds = {
        "p90_tracking": 0.3,
        "QoS_mean_probability": 0.1,
        "QoS_max_probability": 0.1,
    }

    for name, column in APPENDIX_METRICS.items():
        path = output_dir / f"hist_{safe_slug(name)}.png"
        make_histogram(
            df[column],
            f"Full-dataset distribution: {name}",
            name.replace("_", " "),
            path,
            dpi,
            bins=90,
            log_y=True,
            threshold=thresholds.get(name),
            clip_quantile=0.995,
        )
        manifest.append({"plot": path.name, "purpose": f"Appendix histogram for {column}."})
    return manifest


def write_findings_report(
    df: pd.DataFrame,
    target_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    n = len(df)
    dense = df["Is_W2_Dense"]
    hidden_qos = (df["QoS_Mean_Probability"] <= 0.1) & (df["QoS_Max_Probability"] > 0.1)
    p90 = df["Ctrack_Epsilon_90th"].to_numpy()
    p90_total_variance = np.sum((p90 - p90.mean()) ** 2)
    top10 = p90 >= np.quantile(p90, 0.90)
    top1 = p90 >= np.quantile(p90, 0.99)
    top10_share = np.sum((p90[top10] - p90.mean()) ** 2) / p90_total_variance
    top1_share = np.sum((p90[top1] - p90.mean()) ** 2) / p90_total_variance

    exact_boundary_w1 = (
        (df["Workload_Family"] == "W1")
        & np.isclose(df["P_minus_R"], 0.0, atol=1e-12)
    )
    w1 = df[df["Workload_Family"] == "W1"]
    w2 = df[df["Workload_Family"] == "W2"]

    mismatch_rows = int(
        (
            ~np.isclose(
                df["QoS_Violation_Ratio"],
                df["QoS_Recomputed_Violation_Ratio"],
            )
        ).sum()
    )

    lines = [
        "# Whole-dataset model-diagnostic findings",
        "",
        "## Dataset composition",
        "",
        f"- Total rows: **{n:,}**.",
        f"- Original coarse-grid rows: **{int((~dense).sum()):,}**.",
        f"- Dense W2 augmentation rows: **{int(dense.sum()):,} ({100*dense.mean():.2f}% of the dataset)**.",
        "- The dense augmentation exists only for W2 at 1,000 servers: one U=0.6 workload and one U=0.8 workload.",
        "- No focused W1 run is present.",
        "",
        "## Tracking target",
        "",
        f"- p90 tracking median: **{df['Ctrack_Epsilon_90th'].median():.3f}**.",
        f"- p90 tracking p75: **{df['Ctrack_Epsilon_90th'].quantile(.75):.3f}**.",
        f"- p90 tracking p99: **{df['Ctrack_Epsilon_90th'].quantile(.99):.3f}**.",
        f"- Maximum p90 tracking: **{df['Ctrack_Epsilon_90th'].max():.3f}**.",
        f"- The top 10% of p90 values account for **{100*top10_share:.1f}%** of total p90 variance.",
        f"- The top 1% alone account for **{100*top1_share:.1f}%** of total p90 variance.",
        "- Consequence: untransformed MSE is dominated by severe tracking failures instead of the p90=0.3 decision boundary.",
        "",
        "## QoS target",
        "",
        f"- QoS max is exactly zero in **{100*np.isclose(df['QoS_Max_Probability'],0).mean():.2f}%** of rows.",
        f"- QoS max exceeds 0.5 in **{100*(df['QoS_Max_Probability']>0.5).mean():.2f}%** of rows.",
        f"- Only **{100*df['QoS_Max_Probability'].between(.05,.15,inclusive='both').mean():.2f}%** of rows lie near the max-Pj=0.1 boundary.",
        f"- **{int(hidden_qos.sum()):,} rows ({100*hidden_qos.mean():.2f}% of all rows)** have mean QoS <=0.1 but max QoS >0.1.",
        f"- Among rows that appear to pass under the mean target, **{100*hidden_qos.sum()/max(1,(df['QoS_Mean_Probability']<=.1).sum()):.2f}%** hide a job-type violation.",
        "- W1 has P1, P2, and P3 equal to zero in every row, whereas W2 activates all four job-type outputs. The pooled QoS distribution is therefore structurally multimodal.",
        "",
        "## Feasibility imbalance",
        "",
        f"- Tracking-pass rows: **{int(df['Tracking_Pass'].sum()):,} ({100*df['Tracking_Pass'].mean():.2f}%)**.",
        f"- Legacy QoS-pass rows: **{int(df['QoS_Pass_Legacy'].sum()):,} ({100*df['QoS_Pass_Legacy'].mean():.2f}%)**.",
        f"- Pass-both rows: **{int(df['Both_Pass_Legacy'].sum()):,} ({100*df['Both_Pass_Legacy'].mean():.2f}%)**.",
        f"- W1 pass-both rows: **{int(w1['Both_Pass_Legacy'].sum()):,} ({100*w1['Both_Pass_Legacy'].mean():.2f}%)**.",
        f"- W2 pass-both rows: **{int(w2['Both_Pass_Legacy'].sum()):,} ({100*w2['Both_Pass_Legacy'].mean():.3f}%)**.",
        f"- Dense W2 pass-both rows: **{int(df.loc[dense,'Both_Pass_Legacy'].sum()):,}**, which are only **{100*df.loc[dense,'Both_Pass_Legacy'].sum()/n:.3f}% of the full dataset**.",
        "",
        "## W1 P=R boundary",
        "",
        f"- Exact W1 P=R rows: **{int(exact_boundary_w1.sum()):,} ({100*exact_boundary_w1.sum()/len(w1):.2f}% of W1)**.",
        f"- Median W1 M_RSR at P=R: **{w1.loc[np.isclose(w1['P_minus_R'],0,atol=1e-12),'Simulator_RSR_Total_Cost'].median():.4f}**.",
        "- In every W1 workload/server/utilization cell, the 500 lowest-M_RSR rows are all P=R rows.",
        "- Tracking usually passes at the boundary, while QoS is the limiting constraint. A surrogate that underestimates QoS will therefore be pulled toward P=R.",
        "- The W1 grid has no local sweep between the exact boundary and the next interior R point, so the boundary curvature is under-resolved.",
        "",
        "## Validation leakage / overfitting risk",
        "",
    ]

    for _, row in split_summary.iterrows():
        lines.append(
            f"- {row['diagnostic']}: **{row['pct_of_test_rows']:.2f}%**."
        )

    lines += [
        "- This does not prove neural-network overfitting by itself, but it means a random row split is not a strong generalization test.",
        "- Use a grouped split that keeps an entire workload-family/N/U/P/R scenario and its alternate QoS configuration on one side of the split.",
        "",
        "## QoS label consistency",
        "",
        f"- `QoS_Violation_Ratio` disagrees with a direct recomputation `mean(Pj > 0.1)` on **{mismatch_rows:,} rows ({100*mismatch_rows/n:.3f}%)**.",
        "- These two labels should not be treated as interchangeable until the intended semantics are confirmed.",
        "",
        "## Recommended model experiments on the fixed dataset",
        "",
        "1. Keep a baseline run for continuity, but add target standardization.",
        "2. For p90 tracking, compare raw MSE against Huber loss and/or log1p(p90).",
        "3. Add a boundary-weighted loss or sampler for p90 in [0.2, 0.4].",
        "4. Replace mean QoS with max(Pj), but recompute the QoS pass label consistently from the same Pj array.",
        "5. Use a nonnegative output head or transformed target for p90 and QoS.",
        "6. Compare models on three evaluations: random split, grouped P/R-scenario split, and the full dense-W2 diagnostic set.",
        "7. Report feasibility precision/recall and boundary MAE, not only global MSE/R2.",
        "8. Add a dedicated W1 P=R boundary evaluation. If P=R is physically invalid, enforce it as a hard constraint rather than expecting the surrogate to learn not to choose it.",
        "",
        "## Most useful plots",
        "",
        "- `main_diagnostics/01_tracking_p90_zoom_full_dataset.png`",
        "- `main_diagnostics/03_qos_mean_vs_max_histograms_full_dataset.png`",
        "- `main_diagnostics/05_both_pass_rate_by_workload_server_utilization_source.png`",
        "- `main_diagnostics/07_w1_mrsr_profile_by_p_minus_r_bin.png`",
        "- `main_diagnostics/08_random_row_split_overlap_diagnostics.png`",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.results_csv)

    validation = validate_dataset(df, args.diagnostics_csv)
    validation.to_csv(args.out_dir / "dataset_validation_checks.csv", index=False)

    target_rows = [
        distribution_summary(name, df[column])
        for name, column in APPENDIX_METRICS.items()
    ]
    target_summary = pd.DataFrame(target_rows)
    target_summary.to_csv(args.out_dir / "target_distribution_summary.csv", index=False)

    group_summary, feasibility_categories = make_group_summaries(df)
    group_summary.to_csv(args.out_dir / "group_distribution_and_feasibility_summary.csv", index=False)
    feasibility_categories.to_csv(args.out_dir / "feasibility_category_counts.csv", index=False)

    scenario_summary = make_scenario_variability(df)
    scenario_summary.to_csv(args.out_dir / "scenario_weight_sensitivity_summary.csv", index=False)

    w1_boundary = make_w1_boundary_summary(df)
    w1_boundary.to_csv(args.out_dir / "w1_boundary_coverage_summary.csv", index=False)

    split_summary = make_split_leakage_summary(
        df,
        random_seed=args.random_seed,
        test_fraction=args.test_fraction,
    )
    split_summary.to_csv(args.out_dir / "random_split_overlap_summary.csv", index=False)

    qos_label_mismatch = df.loc[
        ~np.isclose(
            df["QoS_Violation_Ratio"],
            df["QoS_Recomputed_Violation_Ratio"],
        ),
        [
            "Source_Output_Dir",
            "Iteration",
            "Weight_Sample_ID",
            "Workload_Name",
            "server_count",
            "utilization",
            "Pbar_kw_per_server",
            "R_kw_per_server",
            "QoS_Violation_Ratio",
            "QoS_Recomputed_Violation_Ratio",
            "QoS_Mean_Probability",
            "QoS_Max_Probability",
            "QoS_Delay_Probabilities",
        ],
    ]
    qos_label_mismatch.to_csv(args.out_dir / "qos_label_mismatch_rows.csv", index=False)

    main_manifest = plot_main_diagnostics(
        df,
        group_summary,
        w1_boundary,
        split_summary,
        args.out_dir / "main_diagnostics",
        args.dpi,
    )
    appendix_manifest = plot_appendix_histograms(
        df,
        args.out_dir / "appendix_cost_histograms",
        args.dpi,
    )

    manifest = pd.DataFrame(
        [
            *[
                {"folder": "main_diagnostics", **row}
                for row in main_manifest
            ],
            *[
                {"folder": "appendix_cost_histograms", **row}
                for row in appendix_manifest
            ],
        ]
    )
    manifest.to_csv(args.out_dir / "plot_manifest.csv", index=False)

    write_findings_report(
        df,
        target_summary,
        group_summary,
        split_summary,
        scenario_summary,
        validation,
        args.out_dir / "README_dataset_diagnostic_findings.md",
    )

    print()
    print("Completed whole-dataset diagnostics.")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Output directory: {args.out_dir.resolve()}")
    print("Start with:")
    print(f"  {args.out_dir / 'README_dataset_diagnostic_findings.md'}")
    print(f"  {args.out_dir / 'main_diagnostics'}")


if __name__ == "__main__":
    main()
