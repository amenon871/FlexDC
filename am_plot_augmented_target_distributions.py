#!/usr/bin/env python3
"""
am_plot_augmented_target_distributions.py

Generate histogram/summary diagnostics for FlexDC/CONDOR-on-FlexDC augmented datasets.
Designed for the current FlexDC raw-model debugging path.

Inputs:
  --results-csv      combined grid_search_results.csv
  --diagnostics-csv  combined grid_search_diagnostics.csv
  --out-dir          output folder for PNG/CSV/Markdown summaries

Outputs:
  - target_distribution_summary.csv
  - feasibility_summary.csv
  - histogram PNGs for raw/derived cost and constraint quantities
  - optional W2-local focused histograms when --w2-local-substring is provided

The script is intentionally read-only. It does not modify the input CSVs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "flexdc_M_RSR",
    "Mtrack_Cost",
    "Ctrack_Epsilon_90th",
    "Tracking_Excess",
    "Mean_QoS_Delay_Probability",
    "Max_QoS_Delay_Probability",
    "QoS_Excess_MaxPj",
    "QoS_Violation_Ratio",
    "Paper_Ctrack_Cost",
    "Paper_CQoS_Cost",
    "Paper_FullObjective_Cost",
]


def softplus(x):
    return np.logaddexp(0.0, x)


def parse_probability_vector(value) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return [float(x) for x in parsed]
    except Exception:
        cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        try:
            parsed = ast.literal_eval(cleaned)
            return [float(x) for x in parsed]
        except Exception as exc:
            raise ValueError(f"Could not parse QoS probability vector: {text[:200]}") from exc


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_and_merge(results_csv: Path, diagnostics_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv, low_memory=False)
    diagnostics = pd.read_csv(diagnostics_csv, low_memory=False)

    # Merge on the strongest common key available. The combined files usually have
    # Source_Output_Dir + Iteration and sometimes Weight_Sample_ID.
    preferred = ["Source_Output_Dir", "Iteration", "Weight_Sample_ID"]
    keys = [k for k in preferred if k in results.columns and k in diagnostics.columns]
    if not keys:
        keys = ["Iteration"] if "Iteration" in results.columns and "Iteration" in diagnostics.columns else []
    if not keys:
        raise ValueError("Could not find common merge keys between results and diagnostics.")

    # Keep only diagnostic columns not already present, except merge keys.
    diag_keep = keys + [c for c in diagnostics.columns if c not in results.columns]
    merged = results.merge(diagnostics[diag_keep], on=keys, how="left", validate="one_to_one")
    return merged


def add_derived_columns(
    df: pd.DataFrame,
    tracking_threshold: float,
    qos_threshold: float,
    ctrack_psi: float,
    ctrack_mu: float,
    qos_beta: float,
    qos_rho: float,
) -> pd.DataFrame:
    df = df.copy()

    # Monetary RSR target used by the FlexDC raw model.
    if "flexdc_M_RSR" not in df.columns:
        rsr_col = find_col(df, ["Simulator_RSR_Total_Cost", "Actual_flexdc_M_RSR"])
        if rsr_col:
            df["flexdc_M_RSR"] = pd.to_numeric(df[rsr_col], errors="coerce")

    # p90 tracking.
    eps_col = find_col(df, ["Ctrack_Epsilon_90th", "raw_Ctrack_Epsilon_90th", "Actual_raw_Ctrack_Epsilon_90th"])
    if eps_col and eps_col != "Ctrack_Epsilon_90th":
        df["Ctrack_Epsilon_90th"] = pd.to_numeric(df[eps_col], errors="coerce")
    elif eps_col:
        df["Ctrack_Epsilon_90th"] = pd.to_numeric(df[eps_col], errors="coerce")

    # QoS vectors and derived scalar summaries.
    if "QoS_Delay_Probabilities" in df.columns:
        qos_vecs = df["QoS_Delay_Probabilities"].apply(parse_probability_vector)
        df["Job_Type_Count"] = qos_vecs.apply(len)
        df["Max_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: max(v) if v else np.nan)
        df["Mean_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: float(np.mean(v)) if v else np.nan)
        df["Sum_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: float(np.sum(v)) if v else np.nan)
        # Per-type columns for quick debugging. Up to first 8 to keep sane.
        max_len = min(8, int(df["Job_Type_Count"].max()) if df["Job_Type_Count"].notna().any() else 0)
        for j in range(max_len):
            df[f"QoS_P{j}"] = qos_vecs.apply(lambda v, jj=j: float(v[jj]) if len(v) > jj else np.nan)
        df["Paper_CQoS_Cost"] = qos_vecs.apply(
            lambda v: qos_beta * float(np.sum(softplus(qos_rho * (np.asarray(v, dtype=float) - qos_threshold)))) if v else np.nan
        )
    else:
        mean_col = find_col(df, ["raw_qos_probability_mean", "Actual_raw_qos_probability_mean"])
        if mean_col:
            df["Mean_QoS_Delay_Probability"] = pd.to_numeric(df[mean_col], errors="coerce")
            df["Max_QoS_Delay_Probability"] = np.nan
            df["Paper_CQoS_Cost"] = np.nan

    # Constraint excess variables.
    if "Ctrack_Epsilon_90th" in df.columns:
        df["Tracking_Excess"] = np.maximum(0.0, df["Ctrack_Epsilon_90th"].astype(float) - tracking_threshold)
        df["Paper_Ctrack_Cost"] = ctrack_psi * softplus(ctrack_mu * (df["Ctrack_Epsilon_90th"].astype(float) - tracking_threshold))
    if "Max_QoS_Delay_Probability" in df.columns:
        df["QoS_Excess_MaxPj"] = np.maximum(0.0, df["Max_QoS_Delay_Probability"].astype(float) - qos_threshold)
    if "Mean_QoS_Delay_Probability" in df.columns:
        df["QoS_Excess_MeanPj"] = np.maximum(0.0, df["Mean_QoS_Delay_Probability"].astype(float) - qos_threshold)

    if {"flexdc_M_RSR", "Paper_Ctrack_Cost", "Paper_CQoS_Cost"}.issubset(df.columns):
        df["Paper_FullObjective_Cost"] = df["flexdc_M_RSR"] + df["Paper_Ctrack_Cost"] + df["Paper_CQoS_Cost"]

    # Pass/fail flags.
    if "Ctrack_Epsilon_90th" in df.columns:
        df["Tracking_Pass"] = df["Ctrack_Epsilon_90th"].astype(float) <= tracking_threshold
    if "QoS_Violation_Ratio" in df.columns:
        df["QoS_Pass_CurrentLogic"] = pd.to_numeric(df["QoS_Violation_Ratio"], errors="coerce") <= 0.1
    if "Max_QoS_Delay_Probability" in df.columns:
        df["QoS_Pass_MaxPj"] = df["Max_QoS_Delay_Probability"].astype(float) <= qos_threshold
    if "Tracking_Pass" in df.columns and "QoS_Pass_CurrentLogic" in df.columns:
        df["Both_Pass_CurrentLogic"] = df["Tracking_Pass"] & df["QoS_Pass_CurrentLogic"]
    if "Tracking_Pass" in df.columns and "QoS_Pass_MaxPj" in df.columns:
        df["Both_Pass_MaxPj"] = df["Tracking_Pass"] & df["QoS_Pass_MaxPj"]

    return df


def summary_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "metric": metric,
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "zero_frac": float((s == 0).mean()),
            "negative_frac": float((s < 0).mean()),
        })
    return pd.DataFrame(rows)


def feasibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["Workload_Name", "server_count", "utilization"] if c in df.columns]
    if not group_cols:
        group_cols = ["__all__"]
        df = df.copy()
        df["__all__"] = "all"

    agg = {
        "rows": ("Tracking_Pass", "size") if "Tracking_Pass" in df.columns else (df.columns[0], "size"),
    }
    if "Tracking_Pass" in df.columns:
        agg["tracking_pass"] = ("Tracking_Pass", "sum")
    if "QoS_Pass_CurrentLogic" in df.columns:
        agg["qos_pass_current"] = ("QoS_Pass_CurrentLogic", "sum")
    if "QoS_Pass_MaxPj" in df.columns:
        agg["qos_pass_maxpj"] = ("QoS_Pass_MaxPj", "sum")
    if "Both_Pass_CurrentLogic" in df.columns:
        agg["both_pass_current"] = ("Both_Pass_CurrentLogic", "sum")
    if "Both_Pass_MaxPj" in df.columns:
        agg["both_pass_maxpj"] = ("Both_Pass_MaxPj", "sum")

    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    for c in ["tracking_pass", "qos_pass_current", "qos_pass_maxpj", "both_pass_current", "both_pass_maxpj"]:
        if c in out.columns:
            out[c + "_pct"] = 100.0 * out[c] / out["rows"]
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def plot_histogram(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    title_prefix: str,
    bins: int,
    log_y: bool,
    clip_quantile: Optional[float],
    by_feasible: bool,
) -> Optional[Path]:
    if metric not in df.columns:
        return None
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(s) == 0:
        return None

    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[metric])]

    subtitle = ""
    if clip_quantile is not None and 0 < clip_quantile < 1:
        upper = float(plot_df[metric].quantile(clip_quantile))
        plot_df = plot_df[plot_df[metric] <= upper]
        subtitle = f" (clipped at p{int(clip_quantile*100)})"

    fig = plt.figure(figsize=(9, 5.5))
    ax = plt.gca()

    if by_feasible and "Both_Pass_CurrentLogic" in plot_df.columns:
        fail = plot_df.loc[~plot_df["Both_Pass_CurrentLogic"].fillna(False), metric].dropna().to_numpy()
        passed = plot_df.loc[plot_df["Both_Pass_CurrentLogic"].fillna(False), metric].dropna().to_numpy()
        if len(fail) > 0:
            ax.hist(fail, bins=bins, alpha=0.65, label="Fail both/current")
        if len(passed) > 0:
            ax.hist(passed, bins=bins, alpha=0.65, label="Pass both/current")
        ax.legend()
    else:
        ax.hist(plot_df[metric].dropna().to_numpy(), bins=bins)

    ax.set_title(f"{title_prefix}: {metric}{subtitle}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Row count")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = out_dir / f"hist_{safe_name(title_prefix)}_{safe_name(metric)}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_metric_set(df: pd.DataFrame, metrics: list[str], out_dir: Path, title_prefix: str, bins: int, clip_quantile: Optional[float], by_feasible: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in metrics:
        p = plot_histogram(
            df=df,
            metric=metric,
            out_dir=out_dir,
            title_prefix=title_prefix,
            bins=bins,
            log_y=False,
            clip_quantile=None,
            by_feasible=by_feasible,
        )
        if p:
            paths.append(p)
        p_log = plot_histogram(
            df=df,
            metric=metric,
            out_dir=out_dir,
            title_prefix=title_prefix + "_logY",
            bins=bins,
            log_y=True,
            clip_quantile=None,
            by_feasible=by_feasible,
        )
        if p_log:
            paths.append(p_log)
        if clip_quantile:
            p_clip = plot_histogram(
                df=df,
                metric=metric,
                out_dir=out_dir,
                title_prefix=title_prefix + f"_clip_p{int(clip_quantile*100)}",
                bins=bins,
                log_y=False,
                clip_quantile=clip_quantile,
                by_feasible=by_feasible,
            )
            if p_clip:
                paths.append(p_clip)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Plot augmented FlexDC cost/constraint target distributions.")
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--diagnostics-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--clip-quantile", type=float, default=0.99)
    parser.add_argument("--by-feasible", action="store_true", help="Overlay pass-both vs fail-both distributions when possible.")
    parser.add_argument("--w2-local-substring", default="w2_dense_local_v1", help="Substring in Source_Output_Dir used to isolate local W2 augmentation rows. Empty string disables focused W2 plots.")
    parser.add_argument("--tracking-threshold", type=float, default=0.3)
    parser.add_argument("--qos-threshold", type=float, default=0.1)
    parser.add_argument("--ctrack-psi", type=float, default=1.0)
    parser.add_argument("--ctrack-mu", type=float, default=10.0)
    parser.add_argument("--qos-beta", type=float, default=20.0)
    parser.add_argument("--qos-rho", type=float, default=2.0)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metric columns to plot after derived columns are added.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading and merging CSVs...")
    df = read_and_merge(args.results_csv, args.diagnostics_csv)
    print(f"Rows loaded: {len(df):,}")

    print("Adding derived columns...")
    df = add_derived_columns(
        df,
        tracking_threshold=args.tracking_threshold,
        qos_threshold=args.qos_threshold,
        ctrack_psi=args.ctrack_psi,
        ctrack_mu=args.ctrack_mu,
        qos_beta=args.qos_beta,
        qos_rho=args.qos_rho,
    )

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("Writing summaries...")
    summ = summary_table(df, metrics)
    summ_path = args.out_dir / "target_distribution_summary.csv"
    summ.to_csv(summ_path, index=False)

    feas = feasibility_summary(df)
    feas_path = args.out_dir / "feasibility_summary.csv"
    feas.to_csv(feas_path, index=False)

    derived_path = args.out_dir / "augmented_distribution_rows_with_derived_columns_sample.csv"
    df.head(5000).to_csv(derived_path, index=False)

    print("Plotting full augmented dataset histograms...")
    paths = plot_metric_set(
        df,
        metrics=metrics,
        out_dir=args.out_dir / "histograms_full_augmented",
        title_prefix="full_augmented",
        bins=args.bins,
        clip_quantile=args.clip_quantile,
        by_feasible=args.by_feasible,
    )

    # W2 local subset plots.
    if args.w2_local_substring and "Source_Output_Dir" in df.columns:
        w2 = df[df["Source_Output_Dir"].astype(str).str.contains(args.w2_local_substring, case=False, na=False)].copy()
        if len(w2) > 0:
            print(f"Plotting W2 local subset histograms: {len(w2):,} rows")
            w2_summ = summary_table(w2, metrics)
            w2_summ.to_csv(args.out_dir / "target_distribution_summary_w2_local.csv", index=False)
            feasibility_summary(w2).to_csv(args.out_dir / "feasibility_summary_w2_local.csv", index=False)
            paths.extend(plot_metric_set(
                w2,
                metrics=metrics,
                out_dir=args.out_dir / "histograms_w2_local",
                title_prefix="w2_local",
                bins=args.bins,
                clip_quantile=args.clip_quantile,
                by_feasible=args.by_feasible,
            ))
        else:
            print(f"No W2 local rows found with substring: {args.w2_local_substring!r}")

    # Per-workload histograms for the most important raw constraint quantities.
    per_workload_metrics = [m for m in ["Ctrack_Epsilon_90th", "Max_QoS_Delay_Probability", "Mean_QoS_Delay_Probability", "flexdc_M_RSR"] if m in df.columns]
    if "Workload_Name" in df.columns and per_workload_metrics:
        print("Plotting per-workload focused histograms...")
        for workload, g in df.groupby("Workload_Name"):
            if len(g) == 0:
                continue
            paths.extend(plot_metric_set(
                g,
                metrics=per_workload_metrics,
                out_dir=args.out_dir / "histograms_by_workload" / safe_name(workload),
                title_prefix=f"workload_{safe_name(workload)}",
                bins=args.bins,
                clip_quantile=args.clip_quantile,
                by_feasible=args.by_feasible,
            ))

    # Manifest.
    manifest = pd.DataFrame({"plot_path": [str(p) for p in paths]})
    manifest_path = args.out_dir / "plot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Lightweight markdown README.
    readme = args.out_dir / "README_distribution_plots.md"
    readme.write_text(
        "# FlexDC augmented dataset distribution plots\n\n"
        f"Rows loaded: {len(df):,}\n\n"
        "Key outputs:\n"
        f"- `{summ_path.name}`: quantiles and summary stats for plotted metrics.\n"
        f"- `{feas_path.name}`: tracking/QoS feasibility counts by workload/server/utilization.\n"
        "- `histograms_full_augmented/`: full dataset distributions.\n"
        "- `histograms_w2_local/`: W2 local augmentation distributions, if detected.\n"
        "- `histograms_by_workload/`: focused per-workload histograms.\n\n"
        "Important interpretation notes:\n"
        "- Use `Ctrack_Epsilon_90th` and `Tracking_Excess` to diagnose p90 tracking calibration near the 0.3 threshold.\n"
        "- Use `Max_QoS_Delay_Probability` and `QoS_Excess_MaxPj` instead of only mean QoS, because a mean can hide one violating job type.\n"
        "- Log-y histograms are included because several labels are heavy-tailed.\n",
        encoding="utf-8",
    )

    print("Done.")
    print("Summary:", summ_path)
    print("Feasibility:", feas_path)
    print("Manifest:", manifest_path)
    print("README:", readme)


if __name__ == "__main__":
    main()
