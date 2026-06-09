r"""
Tracking/R diagnostic script for the rangefind64 FlexDC subset.

Run from FlexDC repo root, for example:

python -u .\analyze_tracking_r_effect_for_meeting.py `
  --raw-csv .\combined_outputs\rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_raw.csv `
  --diagnostics-csv .\combined_outputs\rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_diagnostics.csv `
  --out-dir .\figures_tracking_r_effect_meeting

This script intentionally avoids SoftPlus/old residual plots. It focuses only on:
- Simulator_Tracking_Cost
- implied mean absolute tracking error in watts
- Normalized_Tracking_Error_90th from diagnostics
- QoS_Delay_Probability_Sum / QoS_Violation_Ratio
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resolve_path(path_text, fallback_name=None):
    p = Path(path_text)
    if p.exists():
        return p
    if fallback_name:
        p2 = Path(fallback_name)
        if p2.exists():
            return p2
    raise FileNotFoundError(f"Could not find file: {path_text}")


def normalize_cols(df):
    # make sure numeric columns are numeric even if pandas read a mixed string
    numeric = [
        "Pbar_kw_per_server", "R_kw_per_server", "P_actual_watts", "R_actual_watts",
        "Pbar_ratio", "R_ratio", "server_count", "utilization",
        "Simulator_RSR_Total_Cost", "Simulator_Power_Cost", "Simulator_Tracking_Cost",
        "QoS_Delay_Probability_Sum", "QoS_Violation_Ratio",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_raw_diag(raw, diag):
    raw = raw.copy().reset_index(drop=True)
    if diag is None:
        return raw
    diag = diag.copy().reset_index(drop=True)
    cols_to_add = [
        "Normalized_Tracking_Error_90th",
        "Normalized_Tracking_Error_90th_Residual",
        "QoS_Delay_Probability_Residual_Sum",
    ]
    for c in cols_to_add:
        if c in diag.columns and c not in raw.columns:
            raw[c] = pd.to_numeric(diag[c], errors="coerce")
    return raw


def agg_df(df, group_cols, metric, agg):
    return df.groupby(group_cols, dropna=False)[metric].agg(agg).reset_index()


def plot_tracking_vs_r_by_pbar_per_workload(df, out_dir, agg="median"):
    metric = "Simulator_Tracking_Cost"
    workloads = sorted(df["Source_Workload_Name"].unique())
    pvals = sorted(df["Pbar_kw_per_server"].unique())

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, wl in zip(axes, workloads):
        sub = df[df["Source_Workload_Name"] == wl]
        g = agg_df(sub, ["Pbar_kw_per_server", "R_kw_per_server"], metric, agg)
        for p in pvals:
            gp = g[np.isclose(g["Pbar_kw_per_server"], p)].sort_values("R_kw_per_server")
            if len(gp):
                ax.plot(gp["R_kw_per_server"], gp[metric], marker="o", label=f"P={p:.3f}")
        ax.set_title(wl)
        ax.set_xlabel("R kW/server")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Pbar kW/server")
    fig.suptitle(f"Simulator tracking cost vs R by workload ({agg} across utilization/weights)")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(out_dir / "01_tracking_cost_vs_r_controlled_by_pbar_workload.png", dpi=200)
    plt.close(fig)


def plot_tracking_range_by_pbar(df, out_dir, agg="median"):
    metric = "Simulator_Tracking_Cost"
    g = agg_df(df, ["Source_Workload_Name", "Pbar_kw_per_server", "R_kw_per_server"], metric, agg)
    rows = []
    for (wl, p), sub in g.groupby(["Source_Workload_Name", "Pbar_kw_per_server"]):
        rows.append({
            "Source_Workload_Name": wl,
            "Pbar_kw_per_server": p,
            "tracking_min_over_R": sub[metric].min(),
            "tracking_max_over_R": sub[metric].max(),
            "tracking_range_over_R": sub[metric].max() - sub[metric].min(),
            "tracking_first_R": sub.sort_values("R_kw_per_server")[metric].iloc[0],
            "tracking_last_R": sub.sort_values("R_kw_per_server")[metric].iloc[-1],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "tracking_r_effect_by_pbar_workload.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    for wl, sub in summary.groupby("Source_Workload_Name"):
        sub = sub.sort_values("Pbar_kw_per_server")
        ax.plot(sub["Pbar_kw_per_server"], sub["tracking_range_over_R"], marker="o", label=wl)
    ax.set_title("How much simulator tracking cost changes over R at each Pbar")
    ax.set_xlabel("Pbar kW/server")
    ax.set_ylabel("max_R tracking cost - min_R tracking cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_tracking_cost_r_sensitivity_by_pbar.png", dpi=200)
    plt.close(fig)
    return summary


def plot_normalized_tracking_vs_r(df, out_dir, agg="median"):
    if "Normalized_Tracking_Error_90th" not in df.columns:
        return
    metric = "Normalized_Tracking_Error_90th"
    g = agg_df(df, ["Pbar_kw_per_server", "R_kw_per_server"], metric, agg)

    fig, ax = plt.subplots(figsize=(11, 7))
    for p, sub in g.groupby("Pbar_kw_per_server"):
        sub = sub.sort_values("R_kw_per_server")
        ax.plot(sub["R_kw_per_server"], sub[metric], marker="o", label=f"P={p:.3f}")
    ax.axhline(0.3, linestyle="--", linewidth=1.5, label="tracking constraint gamma=0.3")
    ax.set_title(f"Normalized tracking error 90th vs R ({agg}, all workloads/util/weights)")
    ax.set_xlabel("R kW/server")
    ax.set_ylabel("Normalized tracking error at 90th percentile")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "03_normalized_tracking_error90_vs_r_by_pbar.png", dpi=200)
    plt.close(fig)


def plot_qos_vs_r(df, out_dir, agg="median"):
    metrics = ["QoS_Delay_Probability_Sum", "QoS_Violation_Ratio"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        g = agg_df(df, ["Pbar_kw_per_server", "R_kw_per_server"], metric, agg)
        fig, ax = plt.subplots(figsize=(11, 7))
        for p, sub in g.groupby("Pbar_kw_per_server"):
            sub = sub.sort_values("R_kw_per_server")
            ax.plot(sub["R_kw_per_server"], sub[metric], marker="o", label=f"P={p:.3f}")
        ax.set_title(f"{metric} vs R ({agg}, all workloads/util/weights)")
        ax.set_xlabel("R kW/server")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        safe_metric = metric.lower()
        fig.savefig(out_dir / f"04_{safe_metric}_vs_r_by_pbar.png", dpi=200)
        plt.close(fig)


def write_report(df, sensitivity, out_dir):
    report = []
    report.append("TRACKING / R EFFECT VALIDATION REPORT")
    report.append("====================================")
    report.append("")
    report.append(f"Rows: {len(df)}")
    report.append(f"Workloads: {sorted(df['Source_Workload_Name'].unique().tolist())}")
    report.append(f"Pbar values: {sorted(df['Pbar_kw_per_server'].unique().tolist())}")
    report.append(f"R values: {sorted(df['R_kw_per_server'].unique().tolist())}")
    report.append("")

    corr_cols = ["Pbar_kw_per_server", "R_kw_per_server", "Simulator_Tracking_Cost", "QoS_Delay_Probability_Sum", "QoS_Violation_Ratio"]
    if "Normalized_Tracking_Error_90th" in df.columns:
        corr_cols.append("Normalized_Tracking_Error_90th")
    corr = df[corr_cols].corr(numeric_only=True)
    report.append("Correlation matrix:")
    report.append(corr.to_string())
    report.append("")

    report.append("Important interpretation:")
    report.append("- Do not validate the R effect using an all-Pbar aggregate heatmap alone; Pbar dominates the tracking-cost scale.")
    report.append("- Use controlled plots: fixed workload, fixed Pbar, x-axis=R.")
    report.append("- Simulator_Tracking_Cost is the simulator monetary tracking term, approximately proportional to mean absolute power tracking error.")
    report.append("- Normalized_Tracking_Error_90th is divided by R and can decrease as R increases; it is a constraint diagnostic, not the same as monetary tracking cost.")
    report.append("")
    report.append("Tracking-cost sensitivity over R by workload/Pbar:")
    report.append(sensitivity.to_string(index=False))
    (out_dir / "00_tracking_r_effect_report.txt").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", default="combined_outputs/rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_raw.csv")
    parser.add_argument("--diagnostics-csv", default="combined_outputs/rangefind64_p020_084_r001_020_AQA_exp_servers_64_combined_diagnostics.csv")
    parser.add_argument("--out-dir", default="figures_tracking_r_effect_meeting")
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        raw_path = Path(raw_path.name)
    diag_path = Path(args.diagnostics_csv)
    if not diag_path.exists():
        diag_path = Path(diag_path.name)

    raw = normalize_cols(pd.read_csv(raw_path))
    diag = pd.read_csv(diag_path) if diag_path.exists() else None
    df = merge_raw_diag(raw, diag)

    # Useful physical quantity. Since tracking_cost = 0.0001 * mean_abs_error_watts for 1h with piE=0.1/kWh.
    df["Implied_Mean_Absolute_Tracking_Error_Watts"] = df["Simulator_Tracking_Cost"] * 10000.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tracking_vs_r_by_pbar_per_workload(df, out_dir, args.agg)
    sensitivity = plot_tracking_range_by_pbar(df, out_dir, args.agg)
    plot_normalized_tracking_vs_r(df, out_dir, args.agg)
    plot_qos_vs_r(df, out_dir, args.agg)
    write_report(df, sensitivity, out_dir)

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Read report: {out_dir / '00_tracking_r_effect_report.txt'}")
    print(f"Key plot: {out_dir / '01_tracking_cost_vs_r_controlled_by_pbar_workload.png'}")


if __name__ == "__main__":
    main()
