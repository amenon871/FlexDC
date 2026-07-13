#!/usr/bin/env python3
"""Combine, audit, and prepare a FlexDC exact-plan dataset.

This script is the post-generation gate. It verifies that every planned row was
produced exactly once in both result files, validates physical inputs and raw
labels, checks cost lineage, summarizes feasibility/boundary coverage, measures
seed-replicate variability, and writes one training-ready CSV.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

TRACKING_THRESHOLD = 0.3
QOS_THRESHOLD = 0.1
TOL = 1e-8


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
        parsed = ast.literal_eval(text)
    return [float(x) for x in parsed]


def collapse_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "weights" in df.columns:
        return df
    cols = sorted(
        [c for c in df.columns if re.fullmatch(r"Weight_\d+", str(c))],
        key=lambda c: int(c.split("_")[1]),
    )
    if not cols:
        return df
    insert_at = df.columns.get_loc(cols[0])
    weights = df[cols].apply(
        lambda row: json.dumps([float(x) for x in row if not pd.isna(x)]), axis=1
    )
    df = df.drop(columns=cols)
    df.insert(insert_at, "weights", weights)
    return df


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def collect_from_manifest(manifest_path: Path) -> tuple[list[Path], list[Path]]:
    manifest = pd.read_csv(manifest_path)
    if "status" in manifest.columns:
        ok = manifest["status"].astype(str).isin(["completed", "skipped_complete"])
        bad = manifest[~ok]
        if len(bad):
            raise ValueError(
                f"Launch manifest contains {len(bad)} non-complete rows. "
                f"Statuses: {bad['status'].value_counts().to_dict()}"
            )
    results = [Path(x) for x in manifest["results_csv"].dropna().astype(str) if x]
    diagnostics = [Path(x) for x in manifest["diagnostics_csv"].dropna().astype(str) if x]
    if not results or not diagnostics:
        raise ValueError("Manifest does not contain completed results_csv/diagnostics_csv paths.")
    return results, diagnostics


def collect_from_scan(optimization_dir: Path, prefix: str) -> tuple[list[Path], list[Path]]:
    results = sorted(optimization_dir.glob(f"{prefix}*/grid_search_results.csv"))
    diagnostics = sorted(optimization_dir.glob(f"{prefix}*/grid_search_diagnostics.csv"))
    if not results or not diagnostics:
        raise FileNotFoundError(
            f"No outputs found under {optimization_dir} for prefix {prefix}"
        )
    return results, diagnostics


def read_and_combine(paths: Iterable[Path], source_kind: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "Source_Output_Dir", path.parent.name)
        frame.insert(1, "Source_CSV_Path", str(path))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "Plan_Row_ID" not in combined.columns:
        raise ValueError(f"Combined {source_kind} output lacks Plan_Row_ID.")
    if combined["Plan_Row_ID"].astype(str).duplicated().any():
        dupes = combined.loc[
            combined["Plan_Row_ID"].astype(str).duplicated(keep=False),
            ["Plan_Row_ID", "Source_Output_Dir"],
        ]
        raise ValueError(
            f"Duplicate Plan_Row_ID values in {source_kind}; examples:\n{dupes.head(20)}"
        )
    return combined


def safe_max(values: pd.Series) -> float:
    return float(np.nanmax(values.to_numpy(dtype=float))) if len(values) else float("nan")


def safe_mean(values: pd.Series) -> float:
    return float(np.nanmean(values.to_numpy(dtype=float))) if len(values) else float("nan")


def audit_and_prepare(plan: pd.DataFrame, results: pd.DataFrame, diagnostics: pd.DataFrame):
    hard_errors: list[str] = []
    warnings: list[str] = []

    plan = plan.copy()
    results = collapse_weight_columns(results.copy())
    diagnostics = collapse_weight_columns(diagnostics.copy())
    for frame_name, frame in [("plan", plan), ("results", results), ("diagnostics", diagnostics)]:
        frame["Plan_Row_ID"] = frame["Plan_Row_ID"].astype(str)
        if frame["Plan_Row_ID"].duplicated().any():
            hard_errors.append(f"{frame_name} has duplicate Plan_Row_ID values")

    planned_ids = set(plan["Plan_Row_ID"])
    result_ids = set(results["Plan_Row_ID"])
    diagnostic_ids = set(diagnostics["Plan_Row_ID"])
    for name, ids in [("results", result_ids), ("diagnostics", diagnostic_ids)]:
        missing = planned_ids - ids
        extra = ids - planned_ids
        if missing:
            hard_errors.append(f"{name} missing {len(missing)} planned rows; examples={sorted(missing)[:10]}")
        if extra:
            hard_errors.append(f"{name} has {len(extra)} unplanned rows; examples={sorted(extra)[:10]}")

    # Diagnostics contains identity columns also present in results. Keep only diagnostics-only
    # columns plus Plan_Row_ID to create a clean training table.
    diag_unique_cols = ["Plan_Row_ID"] + [
        c for c in diagnostics.columns
        if c != "Plan_Row_ID" and c not in results.columns
    ]
    merged = results.merge(
        diagnostics[diag_unique_cols], on="Plan_Row_ID", how="outer", validate="one_to_one"
    )
    # Plan is authoritative for provenance. Add plan-only fields; compare shared input columns.
    shared_compare = [
        "Pbar_kw_per_server", "R_kw_per_server", "server_count", "utilization",
        "Simulation_Seed", "weights",
    ]
    for col in shared_compare:
        plan_col = {
            "Simulation_Seed": "simulation_seed",
        }.get(col, col)
        if col in merged.columns and plan_col in plan.columns:
            left = merged.set_index("Plan_Row_ID")[col]
            right = plan.set_index("Plan_Row_ID")[plan_col]
            if col == "weights":
                bad = []
                for key in planned_ids & set(left.index):
                    if not np.allclose(parse_vector(left.loc[key]), parse_vector(right.loc[key]), atol=1e-8):
                        bad.append(key)
                if bad:
                    hard_errors.append(f"Output weights differ from plan for {len(bad)} rows; examples={bad[:10]}")
            else:
                lnum = pd.to_numeric(left, errors="coerce")
                rnum = pd.to_numeric(right, errors="coerce")
                bad = ~np.isclose(lnum, rnum, rtol=0, atol=1e-8, equal_nan=True)
                if bad.any():
                    hard_errors.append(f"Output {col} differs from plan for {int(bad.sum())} rows")

    plan_only = [c for c in plan.columns if c not in merged.columns]
    merged = merged.merge(plan[["Plan_Row_ID"] + plan_only], on="Plan_Row_ID", how="left", validate="one_to_one")

    # Basic physical and label validation.
    weights_parsed = merged["weights"].map(parse_vector)
    probs_parsed = merged["QoS_Delay_Probabilities"].map(parse_vector)
    merged["actual_max_pj"] = probs_parsed.map(lambda x: max(x) if x else np.nan)
    merged["actual_mean_pj"] = probs_parsed.map(lambda x: float(np.mean(x)) if x else np.nan)
    merged["Actual_Tracking_Pass"] = merged["Ctrack_Epsilon_90th"].astype(float) <= TRACKING_THRESHOLD
    merged["Actual_QoS_Pass_CurrentLogic"] = merged["QoS_Violation_Ratio"].astype(float) <= QOS_THRESHOLD
    merged["Actual_QoS_Pass_MaxPj"] = merged["actual_max_pj"].astype(float) <= QOS_THRESHOLD
    merged["Actual_Both_Pass_CurrentLogic"] = (
        merged["Actual_Tracking_Pass"] & merged["Actual_QoS_Pass_CurrentLogic"]
    )
    merged["Actual_Both_Pass_MaxPj"] = (
        merged["Actual_Tracking_Pass"] & merged["Actual_QoS_Pass_MaxPj"]
    )

    for idx, (weights, probs, j, lower, upper) in enumerate(zip(
        weights_parsed,
        probs_parsed,
        merged["workload_mix_size"].astype(int),
        merged["Weight_Final_Lower_Bound"].astype(float),
        merged["Weight_Final_Upper_Bound"].astype(float),
    )):
        if len(weights) != j:
            hard_errors.append(f"row {idx}: weight length {len(weights)} != J {j}")
        if len(probs) != j:
            hard_errors.append(f"row {idx}: QoS probability length {len(probs)} != J {j}")
        if not np.isclose(sum(weights), 1.0, atol=1e-7):
            hard_errors.append(f"row {idx}: weight sum={sum(weights)}")
        if weights and (min(weights) < lower - 1e-7 or max(weights) > upper + 1e-7):
            hard_errors.append(f"row {idx}: weights violate bounds [{lower},{upper}]")
        if probs and (min(probs) < -1e-12 or max(probs) > 1 + 1e-12):
            hard_errors.append(f"row {idx}: QoS probability outside [0,1]")
        if len(hard_errors) > 100:
            break

    p = merged["Pbar_kw_per_server"].astype(float)
    r = merged["R_kw_per_server"].astype(float)
    p_low = merged["Pbar_lower_bound_kw_per_server"].astype(float)
    p_high = merged["Pbar_upper_bound_kw_per_server"].astype(float)
    pr_high = merged["PR_upper_bound_kw_per_server"].astype(float)
    r_low = merged["R_lower_bound_kw_per_server"].astype(float)
    physical_bad = (
        (p < p_low - 1e-8) | (p > p_high + 1e-8)
        | (r < r_low - 1e-8) | (r > p + 1e-8)
        | (p + r > pr_high + 1e-8)
    )
    if physical_bad.any():
        hard_errors.append(f"{int(physical_bad.sum())} rows violate physical P/R bounds")

    for col in ["Mtrack_Error_MeanAbs_Normalized", "Mtrack_Error_MeanAbs_Watts", "Ctrack_Epsilon_90th"]:
        values = pd.to_numeric(merged[col], errors="coerce")
        bad = ~np.isfinite(values) | (values < 0)
        if bad.any():
            hard_errors.append(f"{col} has {int(bad.sum())} non-finite/negative rows")

    # Cost lineage checks.
    rsr_recon = merged["Simulator_Power_Cost"].astype(float) + merged["Mtrack_Cost"].astype(float)
    rsr_err = (merged["Simulator_RSR_Total_Cost"].astype(float) - rsr_recon).abs()
    mtrack_recon = (
        merged["Mtrack_Price_Coefficient_piE"].astype(float)
        * merged["Mtrack_Error_MeanAbs_Watts"].astype(float)
        * merged["Mtrack_Hour_Seconds"].astype(float)
    )
    mtrack_err = (merged["Mtrack_Cost"].astype(float) - mtrack_recon).abs()
    full_recon = (
        merged["Simulator_RSR_Total_Cost"].astype(float)
        + merged["Ctrack_Weighted_Cost"].astype(float)
        + merged["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].astype(float)
    )
    full_err = (merged["Diagnostic_FullPaperObjective_Cost"].astype(float) - full_recon).abs()
    lineage = {
        "max_rsr_identity_error": float(rsr_err.max()),
        "max_mtrack_identity_error": float(mtrack_err.max()),
        "max_full_objective_identity_error": float(full_err.max()),
    }
    if lineage["max_rsr_identity_error"] > 1e-6:
        hard_errors.append(f"RSR lineage max error {lineage['max_rsr_identity_error']}")
    if lineage["max_mtrack_identity_error"] > 1e-6:
        hard_errors.append(f"Mtrack lineage max error {lineage['max_mtrack_identity_error']}")
    if lineage["max_full_objective_identity_error"] > 1e-6:
        hard_errors.append(f"Full objective lineage max error {lineage['max_full_objective_identity_error']}")

    # Exact duplicate input report. Replicates intentionally duplicate inputs except seed.
    input_cols = [
        "context_id", "Pbar_kw_per_server", "R_kw_per_server", "weights", "Simulation_Seed"
    ]
    duplicate_mask = merged.duplicated(input_cols, keep=False)
    duplicate_report = merged.loc[duplicate_mask, ["Plan_Row_ID"] + input_cols].sort_values(input_cols)
    accidental_duplicates = int(duplicate_mask.sum())
    if accidental_duplicates:
        hard_errors.append(f"{accidental_duplicates} rows duplicate the full input identity including seed")

    # Context-level coverage and feasibility.
    summary_rows = []
    for context_id, g in merged.groupby("context_id", sort=True):
        feasible = g["Actual_Both_Pass_CurrentLogic"]
        summary_rows.append({
            "context_id": context_id,
            "workload_name": g["Workload_Name"].iloc[0],
            "server_count": int(g["server_count"].iloc[0]),
            "utilization": float(g["utilization"].iloc[0]),
            "rows": int(len(g)),
            "base_rows": int((~bool_series(g["Is_Replicate"])).sum()),
            "replicate_rows": int(bool_series(g["Is_Replicate"]).sum()),
            "unique_p": int(g["Pbar_kw_per_server"].nunique()),
            "unique_pr": int(g[["Pbar_kw_per_server", "R_kw_per_server"]].drop_duplicates().shape[0]),
            "weight_min": float(min(min(x) for x in g["weights"].map(parse_vector))),
            "weight_max": float(max(max(x) for x in g["weights"].map(parse_vector))),
            "tracking_pass": int(g["Actual_Tracking_Pass"].sum()),
            "qos_pass_current": int(g["Actual_QoS_Pass_CurrentLogic"].sum()),
            "both_pass_current": int(feasible.sum()),
            "tracking_only_fail": int((~g["Actual_Tracking_Pass"] & g["Actual_QoS_Pass_CurrentLogic"]).sum()),
            "qos_only_fail": int((g["Actual_Tracking_Pass"] & ~g["Actual_QoS_Pass_CurrentLogic"]).sum()),
            "both_fail": int((~g["Actual_Tracking_Pass"] & ~g["Actual_QoS_Pass_CurrentLogic"]).sum()),
            "tracking_decision_band_0.2_0.4": int(g["Ctrack_Epsilon_90th"].between(0.2, 0.4).sum()),
            "qos_decision_band_0.05_0.15": int(g["actual_max_pj"].between(0.05, 0.15).sum()),
            "best_feasible_objective": float(g.loc[feasible, "Diagnostic_FullPaperObjective_Cost"].min()) if feasible.any() else np.nan,
        })
    context_summary = pd.DataFrame(summary_rows)

    # Known-feasible pilot anchors are exact row reproductions (same P/R/weight/base seed).
    pilot_anchor_mask = (
        merged["weight_stratum"].astype(str).eq("pilot_anchor")
        & merged["pilot_anchor_actual_feasible"].astype(str).str.lower().eq("true")
        & ~bool_series(merged["Is_Replicate"])
    )
    pilot_anchor_rows = merged[pilot_anchor_mask].copy()
    pilot_anchor_reproduction = {
        "planned_known_feasible_anchor_rows": int(len(pilot_anchor_rows)),
        "reproduced_feasible_current_logic": int(pilot_anchor_rows["Actual_Both_Pass_CurrentLogic"].sum()),
        "failed_reproduction_count": int((~pilot_anchor_rows["Actual_Both_Pass_CurrentLogic"]).sum()),
    }
    if pilot_anchor_reproduction["failed_reproduction_count"]:
        warnings.append(
            f"{pilot_anchor_reproduction['failed_reproduction_count']} historical pilot anchors did not match current feasibility. "
            "Historical wizard runs created job-arrival tables before resetting the experiment random seed, so exact standalone "
            "row reproduction is not guaranteed for legacy data. Treat these anchors as pilot-guided regions; the new wizard "
            "re-seeds before job-table construction so all new plan rows are reproducible and order-independent."
        )

    # Replicate variability. base_plan_row_id groups seed-20 base + seed-21/22 repeats.
    replicate_rows = []
    label_flip_groups = 0
    for base_id, g in merged.groupby("Base_Plan_Row_ID", sort=False):
        if len(g) < 2:
            continue
        p90 = g["Ctrack_Epsilon_90th"].astype(float)
        maxpj = g["actual_max_pj"].astype(float)
        labels = g["Actual_Both_Pass_CurrentLogic"].astype(bool)
        flip = labels.nunique() > 1
        label_flip_groups += int(flip)
        replicate_rows.append({
            "base_plan_row_id": base_id,
            "runs": int(len(g)),
            "seeds": json.dumps(sorted(int(x) for x in g["Simulation_Seed"].astype(int).unique())),
            "p90_min": float(p90.min()),
            "p90_max": float(p90.max()),
            "p90_range": float(p90.max() - p90.min()),
            "max_pj_min": float(maxpj.min()),
            "max_pj_max": float(maxpj.max()),
            "max_pj_range": float(maxpj.max() - maxpj.min()),
            "feasibility_label_flip": bool(flip),
        })
    replicate_summary = pd.DataFrame(replicate_rows)

    # W2 expectation is context-aware: only contexts with known-feasible pilot anchors are
    # required to contain feasible rows. Other W2 contexts may truly have no feasible region.
    known_feasible_contexts = set(
        merged.loc[pilot_anchor_mask, "context_id"].astype(str).unique()
    )
    for context_id in sorted(known_feasible_contexts):
        row = context_summary[context_summary["context_id"] == context_id]
        if len(row) and int(row["both_pass_current"].iloc[0]) == 0:
            warnings.append(
                f"Pilot-guided context {context_id} contains zero feasible generated rows. "
                "Review simulator/config lineage and revise the sweep before training."
            )

    audit = {
        "status": "FAIL" if hard_errors else "PASS",
        "plan_rows": int(len(plan)),
        "results_rows": int(len(results)),
        "diagnostics_rows": int(len(diagnostics)),
        "training_ready_rows": int(len(merged)),
        "contexts": int(merged["context_id"].nunique()),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "lineage": lineage,
        "pilot_anchor_reproduction": pilot_anchor_reproduction,
        "replicate_groups": int(len(replicate_summary)),
        "replicate_label_flip_groups": int(label_flip_groups),
        "replicate_label_flip_rate": float(label_flip_groups / len(replicate_summary)) if len(replicate_summary) else np.nan,
        "overall_feasibility": {
            "tracking_pass": int(merged["Actual_Tracking_Pass"].sum()),
            "qos_pass_current": int(merged["Actual_QoS_Pass_CurrentLogic"].sum()),
            "both_pass_current": int(merged["Actual_Both_Pass_CurrentLogic"].sum()),
            "both_pass_max_pj": int(merged["Actual_Both_Pass_MaxPj"].sum()),
        },
        "pr_strata": {str(k): int(v) for k, v in merged["PR_Stratum"].value_counts().to_dict().items()},
        "weight_strata": {str(k): int(v) for k, v in merged["Weight_Stratum"].value_counts().to_dict().items()},
    }
    return merged, context_summary, duplicate_report, replicate_summary, pilot_anchor_rows, audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-file", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--launch-manifest", default=None)
    source.add_argument("--optimization-dir", default=None)
    source.add_argument("--results-csv", default=None)
    p.add_argument("--diagnostics-csv", default=None, help="Required when --results-csv is used.")
    p.add_argument("--prefix", default=None, help="Required with --optimization-dir.")
    p.add_argument("--out-dir", default="flexdc_sweep_v2_vetted")
    p.add_argument("--output-prefix", default="flexdc_sweep_v2")
    return p


def main() -> None:
    args = build_parser().parse_args()
    plan_path = Path(args.plan_file).resolve()
    plan = pd.read_csv(plan_path, low_memory=False)
    if "plan_row_id" not in plan.columns:
        raise ValueError("Plan file lacks plan_row_id.")
    plan = plan.rename(columns={"plan_row_id": "Plan_Row_ID"})

    if args.launch_manifest:
        results_paths, diagnostics_paths = collect_from_manifest(Path(args.launch_manifest).resolve())
    elif args.optimization_dir:
        if not args.prefix:
            raise ValueError("--prefix is required with --optimization-dir")
        results_paths, diagnostics_paths = collect_from_scan(Path(args.optimization_dir).resolve(), args.prefix)
    else:
        if not args.diagnostics_csv:
            raise ValueError("--diagnostics-csv is required with --results-csv")
        results_paths = [Path(args.results_csv).resolve()]
        diagnostics_paths = [Path(args.diagnostics_csv).resolve()]

    results = read_and_combine(results_paths, "results")
    diagnostics = read_and_combine(diagnostics_paths, "diagnostics")
    merged, context_summary, duplicates, replicate_summary, pilot_anchors, audit = audit_and_prepare(
        plan, results, diagnostics
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    training_path = out_dir / f"{prefix}_training_ready.csv"
    context_path = out_dir / f"{prefix}_context_summary.csv"
    duplicate_path = out_dir / f"{prefix}_duplicate_report.csv"
    replicate_path = out_dir / f"{prefix}_replicate_summary.csv"
    anchor_path = out_dir / f"{prefix}_known_feasible_anchor_reproduction.csv"
    audit_path = out_dir / f"{prefix}_final_audit.json"
    combined_results_path = out_dir / f"{prefix}_combined_grid_search_results.csv"
    combined_diag_path = out_dir / f"{prefix}_combined_grid_search_diagnostics.csv"

    merged.to_csv(training_path, index=False)
    context_summary.to_csv(context_path, index=False)
    duplicates.to_csv(duplicate_path, index=False)
    replicate_summary.to_csv(replicate_path, index=False)
    pilot_anchors.to_csv(anchor_path, index=False)
    results.to_csv(combined_results_path, index=False)
    diagnostics.to_csv(combined_diag_path, index=False)
    audit.update({
        "plan_file": str(plan_path),
        "source_results_files": [str(x) for x in results_paths],
        "source_diagnostics_files": [str(x) for x in diagnostics_paths],
        "outputs": {
            "training_ready": str(training_path),
            "context_summary": str(context_path),
            "duplicate_report": str(duplicate_path),
            "replicate_summary": str(replicate_path),
            "known_feasible_anchor_reproduction": str(anchor_path),
            "combined_results": str(combined_results_path),
            "combined_diagnostics": str(combined_diag_path),
        },
    })
    audit_path.write_text(json.dumps(audit, indent=2))

    print("FlexDC sweep dataset audit")
    print("---------------------------")
    print(f"Status: {audit['status']}")
    print(f"Rows: {len(merged):,}")
    print(f"Contexts: {merged['context_id'].nunique()}")
    print(f"Training-ready CSV: {training_path}")
    print(f"Audit JSON: {audit_path}")
    print()
    print(context_summary.to_string(index=False))
    if audit["warnings"]:
        print("\nWARNINGS:")
        for warning in audit["warnings"]:
            print(" -", warning)
    if audit["hard_errors"]:
        print("\nHARD ERRORS:")
        for error in audit["hard_errors"]:
            print(" -", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
