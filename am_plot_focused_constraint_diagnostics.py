#!/usr/bin/env python3
"""
am_plot_focused_constraint_diagnostics.py

Focused plots for the FlexDC raw-model failure analysis.

This intentionally makes only a small number of plots:
  1) W2 local feasibility funnel
  2) W2 local p90 tracking distribution
  3) W2 local max-QoS distribution
  4) W2 local constraint plane: p90 vs max QoS
  5) W2 local mean-QoS vs max-QoS scatter
  6) W2 local QoS violation-ratio counts
  7) Full dataset vs W2-local target quantiles

The goal is not a giant plot dump. The goal is to answer:
  - Are feasible W2 rows a small/rare subset?
  - Is tracking concentrated near the 0.3 threshold?
  - Does mean QoS hide max job-type QoS failures?
  - Which targets should the next model learn?

Read-only: does not modify input CSVs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    preferred = ["Source_Output_Dir", "Iteration", "Weight_Sample_ID"]
    keys = [k for k in preferred if k in results.columns and k in diagnostics.columns]
    if not keys:
        keys = ["Iteration"] if "Iteration" in results.columns and "Iteration" in diagnostics.columns else []
    if not keys:
        raise ValueError("Could not find common merge keys between results and diagnostics.")

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

    if "flexdc_M_RSR" not in df.columns:
        rsr_col = find_col(df, ["Simulator_RSR_Total_Cost", "Actual_flexdc_M_RSR"])
        if rsr_col:
            df["flexdc_M_RSR"] = pd.to_numeric(df[rsr_col], errors="coerce")

    eps_col = find_col(df, ["Ctrack_Epsilon_90th", "raw_Ctrack_Epsilon_90th", "Actual_raw_Ctrack_Epsilon_90th"])
    if eps_col:
        df["Ctrack_Epsilon_90th"] = pd.to_numeric(df[eps_col], errors="coerce")

    if "Mtrack_Cost" in df.columns:
        df["Mtrack_Cost"] = pd.to_numeric(df["Mtrack_Cost"], errors="coerce")

    if "QoS_Delay_Probabilities" in df.columns:
        qos_vecs = df["QoS_Delay_Probabilities"].apply(parse_probability_vector)
        df["Job_Type_Count"] = qos_vecs.apply(len)
        df["Max_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: max(v) if v else np.nan)
        df["Mean_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: float(np.mean(v)) if v else np.nan)
        df["Sum_QoS_Delay_Probability"] = qos_vecs.apply(lambda v: float(np.sum(v)) if v else np.nan)

        max_len = min(8, int(df["Job_Type_Count"].max()) if df["Job_Type_Count"].notna().any() else 0)
        for j in range(max_len):
            df[f"QoS_P{j}"] = qos_vecs.apply(lambda v, jj=j: float(v[jj]) if len(v) > jj else np.nan)

        df["Paper_CQoS_Cost"] = qos_vecs.apply(
            lambda v: qos_beta * float(np.sum(softplus(qos_rho * (np.asarray(v, dtype=float) - qos_threshold)))) if v else np.nan
        )

    if "Ctrack_Epsilon_90th" in df.columns:
        df["Tracking_Excess"] = np.maximum(0.0, df["Ctrack_Epsilon_90th"].astype(float) - tracking_threshold)
        df["Paper_Ctrack_Cost"] = ctrack_psi * softplus(
            ctrack_mu * (df["Ctrack_Epsilon_90th"].astype(float) - tracking_threshold)
        )

    if "Max_QoS_Delay_Probability" in df.columns:
        df["QoS_Excess_MaxPj"] = np.maximum(0.0, df["Max_QoS_Delay_Probability"].astype(float) - qos_threshold)

    if "Mean_QoS_Delay_Probability" in df.columns:
        df["QoS_Excess_MeanPj"] = np.maximum(0.0, df["Mean_QoS_Delay_Probability"].astype(float) - qos_threshold)

    if {"flexdc_M_RSR", "Paper_Ctrack_Cost", "Paper_CQoS_Cost"}.issubset(df.columns):
        df["Paper_FullObjective_Cost"] = (
            df["flexdc_M_RSR"] + df["Paper_Ctrack_Cost"] + df["Paper_CQoS_Cost"]
        )

    if "Ctrack_Epsilon_90th" in df.columns:
        df["Tracking_Pass"] = df["Ctrack_Epsilon_90th"].astype(float) <= tracking_threshold
    if "QoS_Violation_Ratio" in df.columns:
        df["QoS_Violation_Ratio"] = pd.to_numeric(df["QoS_Violation_Ratio"], errors="coerce")
        df["QoS_Pass_CurrentLogic"] = df["QoS_Violation_Ratio"] <= 0.1
    if "Max_QoS_Delay_Probability" in df.columns:
        df["QoS_Pass_MaxPj"] = df["Max_QoS_Delay_Probability"].astype(float) <= qos_threshold

    if "Tracking_Pass" in df.columns and "QoS_Pass_CurrentLogic" in df.columns:
        df["Both_Pass_CurrentLogic"] = df["Tracking_Pass"] & df["QoS_Pass_CurrentLogic"]
    if "Tracking_Pass" in df.columns and "QoS_Pass_MaxPj" in df.columns:
        df["Both_Pass_MaxPj"] = df["Tracking_Pass"] & df["QoS_Pass_MaxPj"]

    return df


def make_w2_local_subset(df: pd.DataFrame, substring: str) -> pd.DataFrame:
    if "Source_Output_Dir" in df.columns:
        mask = df["Source_Output_Dir"].astype(str).str.contains(substring, case=False, na=False)
        if mask.any():
            return df[mask].copy()

    # Fallback for the current dense sweep if Source_Output_Dir naming is unavailable.
    required = {"Workload_Name", "server_count", "utilization", "Pbar_kw_per_server"}
    if required.issubset(df.columns):
        w2 = df["Workload_Name"].astype(str).str.contains("W2-short", case=False, na=False)
        n1000 = pd.to_numeric(df["server_count"], errors="coerce").round().eq(1000)
        util = pd.to_numeric(df["utilization"], errors="coerce")
        pbar = pd.to_numeric(df["Pbar_kw_per_server"], errors="coerce")
        lu_band = util.round(2).eq(0.60) & pbar.between(0.464, 0.480)
        hu_band = util.round(2).eq(0.80) & pbar.between(0.570, 0.593)
        mask = w2 & n1000 & (lu_band | hu_band)
        return df[mask].copy()

    return df.iloc[0:0].copy()


def quantile_summary(df: pd.DataFrame, metrics: list[str], label: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "subset": label,
            "metric": metric,
            "count": int(len(s)),
            "mean": float(s.mean()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "zero_frac": float((s == 0).mean()),
        })
    return pd.DataFrame(rows)


def feasibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["Workload_Name", "server_count", "utilization"] if c in df.columns]
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [("all", df)]
    for key, g in grouped:
        row = {}
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            row.update({c: v for c, v in zip(group_cols, key)})
        else:
            row["subset"] = "all"

        row["rows"] = int(len(g))
        for col, out_name in [
            ("Tracking_Pass", "tracking_pass"),
            ("QoS_Pass_CurrentLogic", "qos_pass_current"),
            ("QoS_Pass_MaxPj", "qos_pass_maxpj"),
            ("Both_Pass_CurrentLogic", "both_pass_current"),
            ("Both_Pass_MaxPj", "both_pass_maxpj"),
        ]:
            if col in g.columns:
                row[out_name] = int(g[col].fillna(False).sum())
                row[f"{out_name}_pct"] = 100.0 * row[out_name] / len(g) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_feasibility_funnel(w2: pd.DataFrame, out_dir: Path):
    if w2.empty:
        return None
    summary = feasibility_summary(w2)
    if not {"tracking_pass", "qos_pass_current", "both_pass_current"}.issubset(summary.columns):
        return None

    labels = []
    vals = []
    for _, row in summary.iterrows():
        wl = str(row.get("Workload_Name", "W2"))
        util = row.get("utilization", "")
        prefix = f"{wl}\nU={float(util):.1f}" if str(util) != "" else wl
        rows = float(row["rows"])
        labels += [f"{prefix}\ntracking", f"{prefix}\nQoS", f"{prefix}\nboth"]
        vals += [
            100.0 * float(row["tracking_pass"]) / rows,
            100.0 * float(row["qos_pass_current"]) / rows,
            100.0 * float(row["both_pass_current"]) / rows,
        ]

    plt.figure(figsize=(11, 5.5))
    plt.bar(labels, vals)
    plt.ylabel("Rows passing (%)")
    plt.title("W2 local feasibility funnel")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(vals) * 1.25 if vals else 1)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    path = out_dir / "01_w2_local_feasibility_funnel.png"
    savefig(path)
    return path


def plot_hist(df: pd.DataFrame, metric: str, title: str, xlabel: str, threshold: Optional[float], out_path: Path, xmax: Optional[float] = None):
    if df.empty or metric not in df.columns:
        return None
    s_all = pd.to_numeric(df[metric], errors="coerce").dropna()
    if xmax is not None:
        s_all = s_all[s_all <= xmax]
    if len(s_all) == 0:
        return None

    plt.figure(figsize=(8.5, 5.2))
    bins = 60
    plt.hist(s_all, bins=bins, alpha=0.55, density=True, label="All W2 local rows")

    if "Both_Pass_CurrentLogic" in df.columns:
        pass_s = pd.to_numeric(df.loc[df["Both_Pass_CurrentLogic"].fillna(False), metric], errors="coerce").dropna()
        if xmax is not None:
            pass_s = pass_s[pass_s <= xmax]
        if len(pass_s) > 0:
            plt.hist(pass_s, bins=bins, alpha=0.55, density=True, label="Pass both constraints")

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:g}")

    if xmax is not None:
        plt.xlim(0, xmax)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    savefig(out_path)
    return out_path


def plot_constraint_plane(w2: pd.DataFrame, out_dir: Path, tracking_threshold: float, qos_threshold: float, sample_n: int):
    needed = {"Ctrack_Epsilon_90th", "Max_QoS_Delay_Probability"}
    if w2.empty or not needed.issubset(w2.columns):
        return None

    plot_df = w2.copy()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=7)

    plt.figure(figsize=(7.6, 6.2))
    if "Both_Pass_CurrentLogic" in plot_df.columns:
        fail = plot_df[~plot_df["Both_Pass_CurrentLogic"].fillna(False)]
        passed = plot_df[plot_df["Both_Pass_CurrentLogic"].fillna(False)]
        plt.scatter(
            pd.to_numeric(fail["Ctrack_Epsilon_90th"], errors="coerce"),
            pd.to_numeric(fail["Max_QoS_Delay_Probability"], errors="coerce"),
            s=12,
            alpha=0.25,
            label="Fail at least one constraint",
        )
        plt.scatter(
            pd.to_numeric(passed["Ctrack_Epsilon_90th"], errors="coerce"),
            pd.to_numeric(passed["Max_QoS_Delay_Probability"], errors="coerce"),
            s=20,
            alpha=0.85,
            label="Pass both constraints",
        )
    else:
        plt.scatter(
            pd.to_numeric(plot_df["Ctrack_Epsilon_90th"], errors="coerce"),
            pd.to_numeric(plot_df["Max_QoS_Delay_Probability"], errors="coerce"),
            s=12,
            alpha=0.4,
        )

    plt.axvline(tracking_threshold, linestyle="--", linewidth=2, label="Tracking threshold")
    plt.axhline(qos_threshold, linestyle=":", linewidth=2, label="QoS threshold")
    plt.xlim(0, min(1.2, float(pd.to_numeric(w2["Ctrack_Epsilon_90th"], errors="coerce").quantile(0.995)) * 1.05))
    plt.ylim(0, 1.0)
    plt.xlabel("p90 normalized tracking error")
    plt.ylabel("Max QoS violation probability")
    plt.title("W2 local constraint plane")
    plt.legend()
    path = out_dir / "04_w2_local_constraint_plane_p90_vs_max_qos.png"
    savefig(path)
    return path


def plot_mean_vs_max_qos(w2: pd.DataFrame, out_dir: Path, qos_threshold: float, sample_n: int):
    needed = {"Mean_QoS_Delay_Probability", "Max_QoS_Delay_Probability"}
    if w2.empty or not needed.issubset(w2.columns):
        return None

    plot_df = w2.copy()
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=11)

    plt.figure(figsize=(7.4, 6.2))
    if "QoS_Pass_CurrentLogic" in plot_df.columns:
        fail = plot_df[~plot_df["QoS_Pass_CurrentLogic"].fillna(False)]
        passed = plot_df[plot_df["QoS_Pass_CurrentLogic"].fillna(False)]
        plt.scatter(
            pd.to_numeric(fail["Mean_QoS_Delay_Probability"], errors="coerce"),
            pd.to_numeric(fail["Max_QoS_Delay_Probability"], errors="coerce"),
            s=12,
            alpha=0.25,
            label="QoS fail",
        )
        plt.scatter(
            pd.to_numeric(passed["Mean_QoS_Delay_Probability"], errors="coerce"),
            pd.to_numeric(passed["Max_QoS_Delay_Probability"], errors="coerce"),
            s=18,
            alpha=0.75,
            label="QoS pass",
        )
    else:
        plt.scatter(
            pd.to_numeric(plot_df["Mean_QoS_Delay_Probability"], errors="coerce"),
            pd.to_numeric(plot_df["Max_QoS_Delay_Probability"], errors="coerce"),
            s=12,
            alpha=0.35,
        )

    plt.axvline(qos_threshold, linestyle="--", linewidth=2, label="Mean threshold")
    plt.axhline(qos_threshold, linestyle=":", linewidth=2, label="Max threshold")
    plt.xlabel("Mean QoS violation probability")
    plt.ylabel("Max QoS violation probability")
    plt.title("Mean QoS can hide max job-type QoS failures")
    plt.xlim(0, min(0.75, float(pd.to_numeric(w2["Mean_QoS_Delay_Probability"], errors="coerce").quantile(0.995)) * 1.1))
    plt.ylim(0, 1.0)
    plt.legend()
    path = out_dir / "05_w2_local_mean_vs_max_qos.png"
    savefig(path)
    return path


def plot_qos_violation_counts(w2: pd.DataFrame, out_dir: Path):
    if w2.empty or "QoS_Violation_Ratio" not in w2.columns:
        return None

    counts = (
        w2.assign(QoS_Violation_Ratio=pd.to_numeric(w2["QoS_Violation_Ratio"], errors="coerce").round(2))
        .groupby(["Workload_Name", "QoS_Violation_Ratio"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    counts.to_csv(out_dir / "qos_violation_ratio_counts_w2_local.csv", index=False)

    labels = []
    vals = []
    for _, row in counts.iterrows():
        labels.append(f"{row['Workload_Name']}\n{row['QoS_Violation_Ratio']:g}")
        vals.append(int(row["rows"]))

    plt.figure(figsize=(12, 5.6))
    plt.bar(labels, vals)
    plt.ylabel("Rows")
    plt.xlabel("Workload and QoS violation ratio")
    plt.title("W2 local QoS violation-ratio counts")
    plt.xticks(rotation=30, ha="right")
    path = out_dir / "06_w2_local_qos_violation_ratio_counts.png"
    savefig(path)
    return path


def plot_quantile_comparison(summary: pd.DataFrame, out_dir: Path):
    keep_metrics = ["Ctrack_Epsilon_90th", "Max_QoS_Delay_Probability", "Mean_QoS_Delay_Probability", "flexdc_M_RSR"]
    q = summary[summary["metric"].isin(keep_metrics)].copy()
    if q.empty:
        return None

    # One compact plot comparing p50/p75/p95 for p90 and max-QoS only.
    q = q[q["metric"].isin(["Ctrack_Epsilon_90th", "Max_QoS_Delay_Probability"])]
    labels = []
    vals = []
    for _, row in q.iterrows():
        labels.append(f"{row['subset']}\n{row['metric']}\np50")
        vals.append(float(row["p50"]))
        labels.append(f"{row['subset']}\n{row['metric']}\np75")
        vals.append(float(row["p75"]))
        labels.append(f"{row['subset']}\n{row['metric']}\np95")
        vals.append(float(row["p95"]))

    plt.figure(figsize=(13, 5.8))
    plt.bar(labels, vals)
    plt.ylabel("Value")
    plt.title("Full dataset vs W2 local: key target quantiles")
    plt.xticks(rotation=35, ha="right")
    path = out_dir / "07_full_vs_w2_key_target_quantiles.png"
    savefig(path)
    return path


def write_readme(out_dir: Path, full: pd.DataFrame, w2: pd.DataFrame, tracking_threshold: float, qos_threshold: float):
    lines = []
    lines.append("# Focused FlexDC constraint diagnostics\n")
    lines.append(f"Full rows: {len(full):,}\n")
    lines.append(f"W2 local rows: {len(w2):,}\n")
    lines.append("\n## Generated plots\n")
    for p in sorted(out_dir.glob("*.png")):
        lines.append(f"- `{p.name}`\n")

    lines.append("\n## Interpretation guide\n")
    lines.append(f"- Tracking pass threshold: `Ctrack_Epsilon_90th <= {tracking_threshold}`.\n")
    lines.append(f"- QoS pass threshold for max-Pj view: `Max_QoS_Delay_Probability <= {qos_threshold}`.\n")
    lines.append("- For the current model failure, the most important plots are `02`, `04`, and `05`.\n")
    lines.append("- If actual feasible rows exist but cluster in a thin corner of the constraint plane, global MSE is not enough.\n")
    lines.append("- If mean QoS is low while max QoS is high, mean QoS is a weak training target.\n")
    (out_dir / "README_focused_constraint_diagnostics.md").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", type=Path)
    ap.add_argument("--diagnostics-csv", type=Path)
    ap.add_argument("--premerged-csv", type=Path, help="Optional already-merged/derived CSV for quick testing.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--w2-local-substring", default="w2_dense")
    ap.add_argument("--tracking-threshold", type=float, default=0.3)
    ap.add_argument("--qos-threshold", type=float, default=0.1)
    ap.add_argument("--ctrack-psi", type=float, default=1.0)
    ap.add_argument("--ctrack-mu", type=float, default=10.0)
    ap.add_argument("--qos-beta", type=float, default=20.0)
    ap.add_argument("--qos-rho", type=float, default=2.0)
    ap.add_argument("--sample-n", type=int, default=20000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.premerged_csv:
        df = pd.read_csv(args.premerged_csv, low_memory=False)
    else:
        if not args.results_csv or not args.diagnostics_csv:
            raise ValueError("Provide --results-csv and --diagnostics-csv, or provide --premerged-csv.")
        df = read_and_merge(args.results_csv, args.diagnostics_csv)

    df = add_derived_columns(
        df,
        tracking_threshold=args.tracking_threshold,
        qos_threshold=args.qos_threshold,
        ctrack_psi=args.ctrack_psi,
        ctrack_mu=args.ctrack_mu,
        qos_beta=args.qos_beta,
        qos_rho=args.qos_rho,
    )

    w2 = make_w2_local_subset(df, args.w2_local_substring)

    metrics = [
        "flexdc_M_RSR",
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

    summary = pd.concat(
        [
            quantile_summary(df, metrics, "full_augmented"),
            quantile_summary(w2, metrics, "w2_local"),
        ],
        ignore_index=True,
    )
    summary.to_csv(args.out_dir / "focused_target_quantiles.csv", index=False)

    feasibility_summary(df).to_csv(args.out_dir / "focused_feasibility_summary_full.csv", index=False)
    feasibility_summary(w2).to_csv(args.out_dir / "focused_feasibility_summary_w2_local.csv", index=False)

    # A smaller data extract for manual inspection in Excel.
    useful_cols = [
        "Source_Output_Dir", "Workload_Name", "server_count", "utilization",
        "Pbar_kw_per_server", "R_kw_per_server", "weights",
        "flexdc_M_RSR", "Ctrack_Epsilon_90th", "Tracking_Excess",
        "Mean_QoS_Delay_Probability", "Max_QoS_Delay_Probability", "QoS_Excess_MaxPj",
        "QoS_Violation_Ratio", "Paper_Ctrack_Cost", "Paper_CQoS_Cost", "Paper_FullObjective_Cost",
        "Tracking_Pass", "QoS_Pass_CurrentLogic", "QoS_Pass_MaxPj",
        "Both_Pass_CurrentLogic", "Both_Pass_MaxPj",
    ]
    useful_cols = [c for c in useful_cols if c in w2.columns]
    w2[useful_cols].to_csv(args.out_dir / "focused_w2_local_rows_key_columns.csv", index=False)

    generated = []
    generated.append(plot_feasibility_funnel(w2, args.out_dir))
    generated.append(plot_hist(
        w2,
        "Ctrack_Epsilon_90th",
        "W2 local p90 tracking distribution",
        "p90 normalized tracking error",
        args.tracking_threshold,
        args.out_dir / "02_w2_local_p90_tracking_distribution_zoom.png",
        xmax=1.0,
    ))
    generated.append(plot_hist(
        w2,
        "Max_QoS_Delay_Probability",
        "W2 local max-QoS distribution",
        "Max job-type QoS violation probability",
        args.qos_threshold,
        args.out_dir / "03_w2_local_max_qos_distribution.png",
        xmax=1.0,
    ))
    generated.append(plot_constraint_plane(w2, args.out_dir, args.tracking_threshold, args.qos_threshold, args.sample_n))
    generated.append(plot_mean_vs_max_qos(w2, args.out_dir, args.qos_threshold, args.sample_n))
    generated.append(plot_qos_violation_counts(w2, args.out_dir))
    generated.append(plot_quantile_comparison(summary, args.out_dir))

    write_readme(args.out_dir, df, w2, args.tracking_threshold, args.qos_threshold)

    manifest = pd.DataFrame(
        [{"plot_path": str(p)} for p in generated if p is not None]
    )
    manifest.to_csv(args.out_dir / "focused_plot_manifest.csv", index=False)

    print(f"Loaded rows: {len(df):,}")
    print(f"W2 local rows: {len(w2):,}")
    print(f"Output directory: {args.out_dir}")
    print("Generated:")
    for p in generated:
        if p is not None:
            print(" ", p.name)


if __name__ == "__main__":
    main()
