#!/usr/bin/env python3
"""Preflight and post-generation audit for the FlexDC Sweep V3 dataset.

The script validates the immutable plan before simulation and, after the run,
combines all exact-plan worker outputs into one training-ready dataset.  It
checks:

* exact Plan_Row_ID coverage in results and diagnostics;
* V3 P/R and weight constraints;
* 288 unique P/R points/context and 40 weights at every P/R point;
* structured/Sobol/boundary design counts;
* grouped 70/20/10 split assignments;
* seed-repeat grouping and label variability;
* raw-label ranges and cost-equation lineage;
* per-context tracking, per-job QoS, and feasibility coverage.

Use --plan-only before launching the sweep.  After generation, provide one of
--launch-manifest, --optimization-dir/--prefix, or --results-csv plus
--diagnostics-csv.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
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
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a vector, got: {value}")
    return [float(x) for x in parsed]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y", "t"]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "weights" in df.columns:
        return df
    columns = sorted(
        [column for column in df.columns if re.fullmatch(r"Weight_\d+", str(column))],
        key=lambda column: int(column.split("_")[1]),
    )
    if not columns:
        return df
    insert_at = df.columns.get_loc(columns[0])
    weights = df[columns].apply(
        lambda row: json.dumps([float(value) for value in row if not pd.isna(value)]),
        axis=1,
    )
    output = df.drop(columns=columns)
    output.insert(insert_at, "weights", weights)
    return output


def collect_from_manifest(manifest_path: Path) -> tuple[list[Path], list[Path]]:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if "status" in manifest.columns:
        allowed = {"completed", "skipped_complete"}
        good = manifest["status"].astype(str).isin(allowed)
        if (~good).any():
            statuses = manifest.loc[~good, "status"].astype(str).value_counts().to_dict()
            raise ValueError(
                f"Launch manifest contains {int((~good).sum())} incomplete rows: {statuses}"
            )
        manifest = manifest[good].copy()
    results = [Path(value) for value in manifest["results_csv"].dropna().astype(str) if value]
    diagnostics = [
        Path(value) for value in manifest["diagnostics_csv"].dropna().astype(str) if value
    ]
    if not results or not diagnostics:
        raise ValueError("Manifest lacks completed results_csv/diagnostics_csv paths.")
    return results, diagnostics


def collect_from_scan(optimization_dir: Path, prefix: str) -> tuple[list[Path], list[Path]]:
    results = sorted(optimization_dir.glob(f"{prefix}*/grid_search_results.csv"))
    diagnostics = sorted(optimization_dir.glob(f"{prefix}*/grid_search_diagnostics.csv"))
    if not results or not diagnostics:
        raise FileNotFoundError(
            f"No worker outputs found under {optimization_dir} for prefix {prefix}"
        )
    return results, diagnostics


def read_and_combine(paths: Iterable[Path], source_kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "Source_Output_Dir", path.parent.name)
        frame.insert(1, "Source_CSV_Path", str(path))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "Plan_Row_ID" not in combined.columns:
        raise ValueError(f"Combined {source_kind} output lacks Plan_Row_ID.")
    combined["Plan_Row_ID"] = combined["Plan_Row_ID"].astype(str)
    if combined["Plan_Row_ID"].duplicated().any():
        duplicates = combined.loc[
            combined["Plan_Row_ID"].duplicated(keep=False),
            ["Plan_Row_ID", "Source_Output_Dir"],
        ]
        raise ValueError(
            f"Duplicate Plan_Row_ID values in {source_kind}; examples:\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )
    return combined


def expected_split_counts(total: int) -> dict[str, int]:
    return {
        "train": int(round(total * 0.70)),
        "validation": int(round(total * 0.20)),
        "test": total - int(round(total * 0.70)) - int(round(total * 0.20)),
    }


def validate_plan(
    plan: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hard_errors: list[str] = []
    warnings: list[str] = []

    required = [
        "plan_row_id",
        "base_plan_row_id",
        "design_version",
        "context_id",
        "workload_config",
        "experiment_config",
        "server_count",
        "utilization",
        "job_type_count",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "configured_r_over_p_max",
        "pr_id",
        "pr_stratum",
        "weight_id",
        "weight_family",
        "configured_weight_lower",
        "configured_weight_upper",
        "weights",
        "simulation_seed",
        "is_replicate",
        "data_split",
    ]
    missing = [column for column in required if column not in plan.columns]
    if missing:
        raise ValueError(f"Plan is missing required V3 columns: {missing}")

    plan = plan.copy()
    plan["plan_row_id"] = plan["plan_row_id"].astype(str)
    plan["base_plan_row_id"] = plan["base_plan_row_id"].astype(str)
    if plan["plan_row_id"].duplicated().any():
        hard_errors.append("Plan contains duplicate plan_row_id values.")

    plan["is_replicate_bool"] = bool_series(plan["is_replicate"])
    unique_weight_texts = plan["weights"].astype(str).unique()
    weight_parse_cache = {text: parse_vector(text) for text in unique_weight_texts}
    plan["weights_parsed"] = plan["weights"].astype(str).map(weight_parse_cache)
    plan["computed_r_over_p"] = (
        plan["R_kw_per_server"].astype(float)
        / plan["Pbar_kw_per_server"].astype(float)
    )

    # Row-level physical and weight validation.
    pbar = plan["Pbar_kw_per_server"].astype(float)
    reserve = plan["R_kw_per_server"].astype(float)
    r_floor = plan["r_lower_bound_kw_per_server"].astype(float)
    pr_upper = plan["pr_upper_bound_kw_per_server"].astype(float)
    r_over_p_limit = plan["configured_r_over_p_max"].astype(float)

    pr_bad = (
        (reserve < r_floor - TOL)
        | (reserve > r_over_p_limit * pbar + TOL)
        | (pbar + reserve > pr_upper + TOL)
        | (pbar - reserve <= -TOL)
    )
    if pr_bad.any():
        hard_errors.append(f"{int(pr_bad.sum())} plan rows violate P/R constraints.")

    if not np.allclose(
        plan["computed_r_over_p"],
        plan.get("R_over_P", plan["computed_r_over_p"]).astype(float),
        atol=1e-9,
    ):
        hard_errors.append("Stored R_over_P values differ from R/P recomputation.")

    unique_weight_rows = plan[[
        "weights", "job_type_count", "configured_weight_lower", "configured_weight_upper"
    ]].drop_duplicates()
    for row_index, row in unique_weight_rows.iterrows():
        weights = weight_parse_cache[str(row["weights"])]
        job_type_count = int(row["job_type_count"])
        lower = float(row["configured_weight_lower"])
        upper = float(row["configured_weight_upper"])
        if len(weights) != job_type_count:
            hard_errors.append(
                f"unique weight row {row_index}: length {len(weights)} != J={job_type_count}"
            )
        if not np.isclose(sum(weights), 1.0, atol=1e-8):
            hard_errors.append(
                f"unique weight row {row_index}: sum={sum(weights)}"
            )
        if weights and (min(weights) < lower - TOL or max(weights) > upper + TOL):
            hard_errors.append(
                f"unique weight row {row_index}: weights violate [{lower},{upper}]"
            )

    base = plan[~plan["is_replicate_bool"]].copy()
    repeats = plan[plan["is_replicate_bool"]].copy()

    if args.expected_total_rows is not None and len(plan) != args.expected_total_rows:
        hard_errors.append(
            f"Total plan rows={len(plan):,}, expected={args.expected_total_rows:,}."
        )
    if args.expected_contexts is not None and plan["context_id"].nunique() != args.expected_contexts:
        hard_errors.append(
            f"Contexts={plan['context_id'].nunique()}, expected={args.expected_contexts}."
        )

    context_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    replicate_design_rows: list[dict[str, Any]] = []

    for context_id, group in plan.groupby("context_id", sort=True):
        base_group = group[~group["is_replicate_bool"]].copy()
        repeat_group = group[group["is_replicate_bool"]].copy()
        unique_pr = int(base_group["pr_id"].nunique())
        unique_weights = int(base_group["weight_id"].nunique())
        cross_counts = base_group.groupby("pr_id")["weight_id"].nunique()

        expected_base = args.expected_pr_per_context * args.expected_weights_per_pr
        expected_repeats = args.expected_replicate_rows_per_context
        if len(base_group) != expected_base:
            hard_errors.append(
                f"{context_id}: base rows={len(base_group)}, expected={expected_base}."
            )
        if len(repeat_group) != expected_repeats:
            hard_errors.append(
                f"{context_id}: replicate rows={len(repeat_group)}, expected={expected_repeats}."
            )
        if unique_pr != args.expected_pr_per_context:
            hard_errors.append(
                f"{context_id}: P/R count={unique_pr}, expected={args.expected_pr_per_context}."
            )
        if unique_weights != args.expected_weights_per_pr:
            hard_errors.append(
                f"{context_id}: weight count={unique_weights}, expected={args.expected_weights_per_pr}."
            )
        if not (cross_counts == args.expected_weights_per_pr).all():
            hard_errors.append(f"{context_id}: at least one P/R point lacks all weights.")

        pr_counts = base_group.groupby("pr_stratum")["pr_id"].nunique().to_dict()
        expected_pr_counts = {
            "structured_grid": args.expected_structured_pr,
            "sobol_interior": args.expected_sobol_pr,
            "physical_boundary": args.expected_boundary_pr,
        }
        for stratum, expected_count in expected_pr_counts.items():
            if int(pr_counts.get(stratum, 0)) != expected_count:
                hard_errors.append(
                    f"{context_id}: {stratum} P/R count={pr_counts.get(stratum,0)}, "
                    f"expected={expected_count}."
                )

        split_counts = base_group["data_split"].value_counts().to_dict()
        expected_splits = expected_split_counts(expected_base)
        for split_name, expected_count in expected_splits.items():
            if int(split_counts.get(split_name, 0)) != expected_count:
                hard_errors.append(
                    f"{context_id}: base {split_name} rows={split_counts.get(split_name,0)}, "
                    f"expected={expected_count}."
                )

        # Seed groups must be complete and stay in one split.
        expected_seed_set = {args.base_seed, *args.replicate_seeds}
        replicate_base_ids = repeat_group["base_plan_row_id"].astype(str).unique()
        if len(replicate_base_ids):
            replicate_groups = group[
                group["base_plan_row_id"].astype(str).isin(replicate_base_ids)
            ].groupby("base_plan_row_id", sort=False).agg(
                seed_count=("simulation_seed", "nunique"),
                seed_min=("simulation_seed", "min"),
                seed_max=("simulation_seed", "max"),
                split_count=("data_split", "nunique"),
            )
            expected_seed_count = len(expected_seed_set)
            bad_count = replicate_groups["seed_count"] != expected_seed_count
            if bad_count.any():
                hard_errors.append(
                    f"{context_id}: {int(bad_count.sum())} replicate groups have wrong seed counts."
                )
            bad_split = replicate_groups["split_count"] != 1
            if bad_split.any():
                hard_errors.append(
                    f"{context_id}: {int(bad_split.sum())} replicate groups cross splits."
                )
            actual_seed_values = set(
                group[
                    group["base_plan_row_id"].astype(str).isin(replicate_base_ids)
                ]["simulation_seed"].astype(int).unique()
            )
            if actual_seed_values != expected_seed_set:
                hard_errors.append(
                    f"{context_id}: replicate seed set={sorted(actual_seed_values)}, "
                    f"expected={sorted(expected_seed_set)}."
                )

        context_rows.append({
            "context_id": context_id,
            "workload_name": str(base_group["workload_name"].iloc[0]),
            "server_count": int(base_group["server_count"].iloc[0]),
            "utilization": float(base_group["utilization"].iloc[0]),
            "base_rows": int(len(base_group)),
            "replicate_rows": int(len(repeat_group)),
            "total_rows": int(len(group)),
            "unique_pr_points": unique_pr,
            "unique_weights": unique_weights,
            "structured_pr_points": int(pr_counts.get("structured_grid", 0)),
            "sobol_pr_points": int(pr_counts.get("sobol_interior", 0)),
            "boundary_pr_points": int(pr_counts.get("physical_boundary", 0)),
            "train_base_rows": int(split_counts.get("train", 0)),
            "validation_base_rows": int(split_counts.get("validation", 0)),
            "test_base_rows": int(split_counts.get("test", 0)),
            "weight_min": float(min(min(value) for value in base_group["weights_parsed"])),
            "weight_max": float(max(max(value) for value in base_group["weights_parsed"])),
            "r_over_p_min": float(base_group["computed_r_over_p"].min()),
            "r_over_p_max": float(base_group["computed_r_over_p"].max()),
        })

        for (stratum, family), cell in base_group.groupby(
            ["pr_stratum", "weight_family"], sort=True
        ):
            design_rows.append({
                "context_id": context_id,
                "pr_stratum": stratum,
                "weight_family": family,
                "base_rows": int(len(cell)),
                "unique_pr_points": int(cell["pr_id"].nunique()),
                "unique_weights": int(cell["weight_id"].nunique()),
            })
        for (split_name, stratum, family), cell in base_group.groupby(
            ["data_split", "pr_stratum", "weight_family"], sort=True
        ):
            split_rows.append({
                "context_id": context_id,
                "data_split": split_name,
                "pr_stratum": stratum,
                "weight_family": family,
                "base_rows": int(len(cell)),
            })
        for (stratum, family), cell in repeat_group.groupby(
            ["pr_stratum", "weight_family"], sort=True
        ):
            replicate_design_rows.append({
                "context_id": context_id,
                "pr_stratum": stratum,
                "weight_family": family,
                "replicate_rows": int(len(cell)),
                "replicated_base_groups": int(cell["base_plan_row_id"].nunique()),
            })

    # Same weight bank should be reused in all contexts with the same J/bounds.
    base_bank = base[[
        "job_type_count",
        "configured_weight_lower",
        "configured_weight_upper",
        "weight_id",
        "weight_family",
        "weight_detail",
        "weights",
    ]].drop_duplicates()
    bank_duplicates = base_bank.groupby(
        ["job_type_count", "configured_weight_lower", "configured_weight_upper", "weight_id"]
    )["weights"].nunique()
    if (bank_duplicates > 1).any():
        hard_errors.append("A weight_id maps to different vectors across contexts.")

    audit = {
        "status": "PASS" if not hard_errors else "FAIL",
        "plan_rows": int(len(plan)),
        "base_rows": int(len(base)),
        "replicate_rows": int(len(repeats)),
        "contexts": int(plan["context_id"].nunique()),
        "plan_sha256": "",
        "hard_errors": hard_errors,
        "warnings": warnings,
        "split_counts_base": {
            str(key): int(value)
            for key, value in base["data_split"].value_counts().to_dict().items()
        },
        "split_counts_all_rows": {
            str(key): int(value)
            for key, value in plan["data_split"].value_counts().to_dict().items()
        },
        "pr_strata_base": {
            str(key): int(value)
            for key, value in base["pr_stratum"].value_counts().to_dict().items()
        },
        "weight_families_base": {
            str(key): int(value)
            for key, value in base["weight_family"].value_counts().to_dict().items()
        },
    }

    return (
        plan,
        pd.DataFrame(context_rows),
        pd.DataFrame(design_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(replicate_design_rows),
        base_bank.sort_values(["job_type_count", "weight_id"]),
        audit,
    )


def compare_output_to_plan(
    plan: pd.DataFrame,
    output: pd.DataFrame,
    hard_errors: list[str],
    label: str,
) -> None:
    plan_indexed = plan.set_index("plan_row_id")
    output_indexed = output.set_index("Plan_Row_ID")
    common_ids = sorted(set(plan_indexed.index) & set(output_indexed.index))

    numeric_pairs = [
        ("Pbar_kw_per_server", "Pbar_kw_per_server"),
        ("R_kw_per_server", "R_kw_per_server"),
        ("server_count", "server_count"),
        ("utilization", "utilization"),
        ("simulation_seed", "Simulation_Seed"),
    ]
    for plan_column, output_column in numeric_pairs:
        if output_column not in output_indexed.columns:
            hard_errors.append(f"{label} lacks {output_column}.")
            continue
        left = pd.to_numeric(plan_indexed.loc[common_ids, plan_column], errors="coerce")
        right = pd.to_numeric(output_indexed.loc[common_ids, output_column], errors="coerce")
        bad = ~np.isclose(left, right, rtol=0, atol=1e-8, equal_nan=True)
        if bad.any():
            hard_errors.append(
                f"{label} {output_column} differs from plan for {int(bad.sum())} rows."
            )

    # Raw results contain dynamic Weight_0..Weight_(J-1) columns, which are
    # collapsed into `weights`. Diagnostics intentionally record only weight
    # bounds/min/max, so absence of the full vector there is expected.
    if "weights" in output_indexed.columns:
        bad_weights: list[str] = []
        for plan_id in common_ids:
            if not np.allclose(
                parse_vector(plan_indexed.at[plan_id, "weights"]),
                parse_vector(output_indexed.at[plan_id, "weights"]),
                atol=1e-8,
            ):
                bad_weights.append(plan_id)
                if len(bad_weights) >= 10:
                    break
        if bad_weights:
            hard_errors.append(
                f"{label} weights differ from plan; examples={bad_weights}"
            )
    elif label == "results":
        hard_errors.append("results lacks collapsed weights.")


def combine_and_audit_outputs(
    plan: pd.DataFrame,
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    plan_audit: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    hard_errors = list(plan_audit["hard_errors"])
    warnings = list(plan_audit["warnings"])

    results = collapse_weight_columns(results.copy())
    diagnostics = collapse_weight_columns(diagnostics.copy())
    results["Plan_Row_ID"] = results["Plan_Row_ID"].astype(str)
    diagnostics["Plan_Row_ID"] = diagnostics["Plan_Row_ID"].astype(str)

    planned_ids = set(plan["plan_row_id"].astype(str))
    for label, frame in (("results", results), ("diagnostics", diagnostics)):
        ids = set(frame["Plan_Row_ID"])
        missing = planned_ids - ids
        extra = ids - planned_ids
        if missing:
            hard_errors.append(
                f"{label} missing {len(missing)} planned rows; examples={sorted(missing)[:10]}"
            )
        if extra:
            hard_errors.append(
                f"{label} has {len(extra)} unplanned rows; examples={sorted(extra)[:10]}"
            )

    compare_output_to_plan(plan, results, hard_errors, "results")
    compare_output_to_plan(plan, diagnostics, hard_errors, "diagnostics")

    diagnostic_unique = ["Plan_Row_ID"] + [
        column
        for column in diagnostics.columns
        if column != "Plan_Row_ID" and column not in results.columns
    ]
    merged = results.merge(
        diagnostics[diagnostic_unique],
        on="Plan_Row_ID",
        how="outer",
        validate="one_to_one",
    )

    # Add authoritative lowercase plan columns that are not already present.
    plan_for_merge = plan.rename(columns={"plan_row_id": "Plan_Row_ID"})
    for column in plan_for_merge.columns:
        if column == "Plan_Row_ID" or column in merged.columns:
            continue
        merged = merged.merge(
            plan_for_merge[["Plan_Row_ID", column]],
            on="Plan_Row_ID",
            how="left",
            validate="one_to_one",
        )

    merged["weights"] = merged["weights"].map(
        lambda value: json.dumps(parse_vector(value))
    )
    probabilities = merged["QoS_Delay_Probabilities"].map(parse_vector)
    merged["actual_max_pj"] = probabilities.map(lambda values: max(values) if values else np.nan)
    merged["actual_mean_pj"] = probabilities.map(
        lambda values: float(np.mean(values)) if values else np.nan
    )
    merged["Actual_Tracking_Pass"] = (
        merged["Ctrack_Epsilon_90th"].astype(float) <= TRACKING_THRESHOLD
    )
    merged["Actual_QoS_Pass_MaxPj"] = merged["actual_max_pj"] <= QOS_THRESHOLD
    merged["Actual_QoS_Pass_ViolationRatio"] = (
        merged["QoS_Violation_Ratio"].astype(float) <= QOS_THRESHOLD
    )
    merged["Actual_Both_Pass_MaxPj"] = (
        merged["Actual_Tracking_Pass"] & merged["Actual_QoS_Pass_MaxPj"]
    )
    merged["Actual_Both_Pass_ViolationRatio"] = (
        merged["Actual_Tracking_Pass"]
        & merged["Actual_QoS_Pass_ViolationRatio"]
    )

    disagreement = (
        merged["Actual_QoS_Pass_MaxPj"]
        != merged["Actual_QoS_Pass_ViolationRatio"]
    )
    if disagreement.any():
        warnings.append(
            f"{int(disagreement.sum())} rows differ between max-Pj and QoS_Violation_Ratio pass logic. "
            "The training-ready primary label uses max Pj <= 0.1."
        )

    # Cost lineage.
    reconstructed = (
        merged["Simulator_RSR_Total_Cost"].astype(float)
        + merged["Ctrack_Weighted_Cost"].astype(float)
        + merged["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].astype(float)
    )
    reported = merged["Diagnostic_FullPaperObjective_Cost"].astype(float)
    objective_error = (reconstructed - reported).abs()
    if float(objective_error.max()) > 1e-7:
        hard_errors.append(
            f"Full objective reconstruction max error={float(objective_error.max())}."
        )
    merged["Recomputed_FullPaperObjective_Cost"] = reconstructed
    merged["FullObjective_Reconstruction_Error"] = reconstructed - reported

    for column in [
        "Mtrack_Error_MeanAbs_Normalized",
        "Mtrack_Error_MeanAbs_Watts",
        "Mtrack_Cost",
        "Ctrack_Epsilon_90th",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    ]:
        values = pd.to_numeric(merged[column], errors="coerce")
        if not np.isfinite(values).all():
            hard_errors.append(f"Non-finite values found in {column}.")

    # Preserve equal total contribution per base configuration despite repeats.
    group_sizes = merged.groupby("Base_Plan_Row_ID")["Plan_Row_ID"].transform("size")
    merged["Base_Group_Row_Count"] = group_sizes.astype(int)
    merged["Base_Group_Normalization_Weight"] = 1.0 / group_sizes.astype(float)

    context_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for context_id, group in merged.groupby("Context_ID", sort=True):
        feasible = group["Actual_Both_Pass_MaxPj"]
        context_rows.append({
            "context_id": context_id,
            "workload_name": str(group["Workload_Name"].iloc[0]),
            "server_count": int(group["server_count"].iloc[0]),
            "utilization": float(group["utilization"].iloc[0]),
            "rows": int(len(group)),
            "base_groups": int(group["Base_Plan_Row_ID"].nunique()),
            "tracking_pass": int(group["Actual_Tracking_Pass"].sum()),
            "qos_pass_max_pj": int(group["Actual_QoS_Pass_MaxPj"].sum()),
            "both_pass_max_pj": int(feasible.sum()),
            "tracking_only_fail": int((~group["Actual_Tracking_Pass"] & group["Actual_QoS_Pass_MaxPj"]).sum()),
            "qos_only_fail": int((group["Actual_Tracking_Pass"] & ~group["Actual_QoS_Pass_MaxPj"]).sum()),
            "both_fail": int((~group["Actual_Tracking_Pass"] & ~group["Actual_QoS_Pass_MaxPj"]).sum()),
            "tracking_decision_band_0.2_0.4": int(group["Ctrack_Epsilon_90th"].between(0.2, 0.4).sum()),
            "qos_decision_band_0.05_0.15": int(group["actual_max_pj"].between(0.05, 0.15).sum()),
            "best_feasible_objective": (
                float(group.loc[feasible, "Diagnostic_FullPaperObjective_Cost"].min())
                if feasible.any()
                else np.nan
            ),
        })
        for split_name, split_group in group.groupby("Data_Split", sort=True):
            split_rows.append({
                "context_id": context_id,
                "data_split": split_name,
                "rows": int(len(split_group)),
                "base_groups": int(split_group["Base_Plan_Row_ID"].nunique()),
                "tracking_pass": int(split_group["Actual_Tracking_Pass"].sum()),
                "qos_pass_max_pj": int(split_group["Actual_QoS_Pass_MaxPj"].sum()),
                "both_pass_max_pj": int(split_group["Actual_Both_Pass_MaxPj"].sum()),
            })

    # Replicate variability.
    replicate_rows: list[dict[str, Any]] = []
    flip_count = 0
    for base_id, group in merged.groupby("Base_Plan_Row_ID", sort=False):
        if len(group) < 2:
            continue
        p90 = group["Ctrack_Epsilon_90th"].astype(float)
        max_pj = group["actual_max_pj"].astype(float)
        labels = group["Actual_Both_Pass_MaxPj"].astype(bool)
        flip = labels.nunique() > 1
        flip_count += int(flip)
        replicate_rows.append({
            "base_plan_row_id": base_id,
            "context_id": str(group["Context_ID"].iloc[0]),
            "data_split": str(group["Data_Split"].iloc[0]),
            "runs": int(len(group)),
            "seeds": json.dumps(sorted(int(value) for value in group["Simulation_Seed"].unique())),
            "p90_min": float(p90.min()),
            "p90_max": float(p90.max()),
            "p90_range": float(p90.max() - p90.min()),
            "max_pj_min": float(max_pj.min()),
            "max_pj_max": float(max_pj.max()),
            "max_pj_range": float(max_pj.max() - max_pj.min()),
            "feasibility_label_flip": bool(flip),
        })
    replicate_summary = pd.DataFrame(replicate_rows)

    # Weight coverage by job and family.
    weight_rows: list[dict[str, Any]] = []
    base_plan = plan[~plan["is_replicate_bool"]].drop_duplicates("weight_id")
    for _, row in base_plan.sort_values("weight_id").iterrows():
        weights = row["weights_parsed"]
        record = {
            "weight_id": row["weight_id"],
            "weight_family": row["weight_family"],
            "weight_detail": row.get("weight_detail", ""),
            "weights": json.dumps(weights),
            "weight_min": min(weights),
            "weight_max": max(weights),
        }
        for index, value in enumerate(weights):
            record[f"w{index}"] = value
        weight_rows.append(record)
    weight_coverage = pd.DataFrame(weight_rows)

    pr_rows: list[dict[str, Any]] = []
    base_plan = plan[~plan["is_replicate_bool"]]
    for (context_id, stratum), group in base_plan.groupby(
        ["context_id", "pr_stratum"], sort=True
    ):
        unique = group.drop_duplicates("pr_id")
        pr_rows.append({
            "context_id": context_id,
            "pr_stratum": stratum,
            "unique_pr_points": int(len(unique)),
            "pbar_min": float(unique["Pbar_kw_per_server"].min()),
            "pbar_max": float(unique["Pbar_kw_per_server"].max()),
            "r_min": float(unique["R_kw_per_server"].min()),
            "r_max": float(unique["R_kw_per_server"].max()),
            "r_over_p_min": float(unique["computed_r_over_p"].min()),
            "r_over_p_max": float(unique["computed_r_over_p"].max()),
        })
    pr_coverage = pd.DataFrame(pr_rows)

    final_audit = dict(plan_audit)
    final_audit.update({
        "status": "PASS" if not hard_errors else "FAIL",
        "results_rows": int(len(results)),
        "diagnostics_rows": int(len(diagnostics)),
        "training_ready_rows": int(len(merged)),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "objective_lineage_max_abs_error": float(objective_error.max()),
        "qos_logic_disagreement_rows": int(disagreement.sum()),
        "overall_feasibility": {
            "tracking_pass": int(merged["Actual_Tracking_Pass"].sum()),
            "qos_pass_max_pj": int(merged["Actual_QoS_Pass_MaxPj"].sum()),
            "both_pass_max_pj": int(merged["Actual_Both_Pass_MaxPj"].sum()),
        },
        "replicate_groups": int(len(replicate_summary)),
        "replicate_label_flip_groups": int(flip_count),
        "replicate_label_flip_rate": (
            float(flip_count / len(replicate_summary))
            if len(replicate_summary)
            else np.nan
        ),
    })

    return (
        merged,
        pd.DataFrame(context_rows),
        pd.DataFrame(split_rows),
        replicate_summary,
        weight_coverage,
        pr_coverage,
        final_audit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--plan-only", action="store_true")

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--launch-manifest")
    source.add_argument("--optimization-dir")
    source.add_argument("--results-csv")
    parser.add_argument("--diagnostics-csv")
    parser.add_argument("--prefix")

    parser.add_argument("--out-dir", default="flexdc_sweep_v3_vetted")
    parser.add_argument("--output-prefix", default="flexdc_sweep_v3")

    parser.add_argument("--expected-contexts", type=int, default=16)
    parser.add_argument("--expected-pr-per-context", type=int, default=288)
    parser.add_argument("--expected-weights-per-pr", type=int, default=40)
    parser.add_argument("--expected-structured-pr", type=int, default=160)
    parser.add_argument("--expected-sobol-pr", type=int, default=96)
    parser.add_argument("--expected-boundary-pr", type=int, default=32)
    parser.add_argument("--expected-replicate-rows-per-context", type=int, default=1152)
    parser.add_argument("--expected-total-rows", type=int, default=202752)
    parser.add_argument("--base-seed", type=int, default=20)
    parser.add_argument("--replicate-seeds", default="21,22")
    parser.add_argument("--expected-r-over-p-max", type=float, default=0.6)
    parser.add_argument("--expected-weight-lower", type=float, default=0.15)
    parser.add_argument("--expected-weight-upper", type=float, default=0.45)
    return parser


def write_plan_audit_outputs(
    out_dir: Path,
    prefix: str,
    plan_path: Path,
    plan: pd.DataFrame,
    context_summary: pd.DataFrame,
    design_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    replicate_design: pd.DataFrame,
    weight_bank: pd.DataFrame,
    audit: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    context_path = out_dir / f"{prefix}_plan_context_summary.csv"
    design_path = out_dir / f"{prefix}_plan_design_summary.csv"
    split_path = out_dir / f"{prefix}_plan_split_summary.csv"
    replicate_design_path = out_dir / f"{prefix}_plan_replicate_design.csv"
    weight_bank_path = out_dir / f"{prefix}_plan_weight_bank.csv"
    audit_path = out_dir / f"{prefix}_plan_audit.json"

    context_summary.to_csv(context_path, index=False)
    design_summary.to_csv(design_path, index=False)
    split_summary.to_csv(split_path, index=False)
    replicate_design.to_csv(replicate_design_path, index=False)
    weight_bank.to_csv(weight_bank_path, index=False)

    audit = dict(audit)
    audit["plan_file"] = str(plan_path)
    audit["plan_sha256"] = sha256_file(plan_path)
    audit["outputs"] = {
        "context_summary": str(context_path),
        "design_summary": str(design_path),
        "split_summary": str(split_path),
        "replicate_design": str(replicate_design_path),
        "weight_bank": str(weight_bank_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path


def main() -> None:
    args = build_parser().parse_args()
    args.replicate_seeds = [
        int(value.strip())
        for value in str(args.replicate_seeds).split(",")
        if value.strip()
    ]

    plan_path = Path(args.plan_file).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    plan = pd.read_csv(plan_path, low_memory=False)

    # Confirm the top-level configured values match this V3 audit invocation.
    for column, expected in [
        ("configured_r_over_p_max", args.expected_r_over_p_max),
        ("configured_weight_lower", args.expected_weight_lower),
        ("configured_weight_upper", args.expected_weight_upper),
    ]:
        if column not in plan.columns:
            raise ValueError(f"Plan lacks {column}.")
        values = pd.to_numeric(plan[column], errors="coerce")
        if not np.allclose(values, expected, atol=1e-9):
            raise ValueError(
                f"Plan {column} does not uniformly match expected value {expected}."
            )

    (
        plan,
        plan_context_summary,
        plan_design_summary,
        plan_split_summary,
        plan_replicate_design,
        plan_weight_bank,
        plan_audit,
    ) = validate_plan(plan, args)

    out_dir = Path(args.out_dir).expanduser().resolve()
    prefix = args.output_prefix
    plan_audit_path = write_plan_audit_outputs(
        out_dir,
        prefix,
        plan_path,
        plan,
        plan_context_summary,
        plan_design_summary,
        plan_split_summary,
        plan_replicate_design,
        plan_weight_bank,
        plan_audit,
    )

    print("FlexDC Sweep V3 plan audit")
    print("--------------------------")
    print(f"Status: {plan_audit['status']}")
    print(f"Rows: {len(plan):,}")
    print(f"Contexts: {plan['context_id'].nunique()}")
    print(f"Plan audit: {plan_audit_path}")
    print()
    print(plan_context_summary.to_string(index=False))

    if plan_audit["hard_errors"]:
        print("\nHARD ERRORS:")
        for error in plan_audit["hard_errors"]:
            print(" -", error)
        raise SystemExit(1)

    if args.plan_only:
        return

    if not any([args.launch_manifest, args.optimization_dir, args.results_csv]):
        raise ValueError(
            "Post-generation audit requires --launch-manifest, --optimization-dir, "
            "or --results-csv, unless --plan-only is used."
        )

    if args.launch_manifest:
        results_paths, diagnostics_paths = collect_from_manifest(
            Path(args.launch_manifest).expanduser().resolve()
        )
    elif args.optimization_dir:
        if not args.prefix:
            raise ValueError("--prefix is required with --optimization-dir.")
        results_paths, diagnostics_paths = collect_from_scan(
            Path(args.optimization_dir).expanduser().resolve(),
            args.prefix,
        )
    else:
        if not args.diagnostics_csv:
            raise ValueError("--diagnostics-csv is required with --results-csv.")
        results_paths = [Path(args.results_csv).expanduser().resolve()]
        diagnostics_paths = [Path(args.diagnostics_csv).expanduser().resolve()]

    results = read_and_combine(results_paths, "results")
    diagnostics = read_and_combine(diagnostics_paths, "diagnostics")

    (
        training_ready,
        context_summary,
        split_summary,
        replicate_summary,
        weight_coverage,
        pr_coverage,
        final_audit,
    ) = combine_and_audit_outputs(plan, results, diagnostics, plan_audit)

    training_path = out_dir / f"{prefix}_training_ready.csv"
    train_path = out_dir / f"{prefix}_train.csv"
    validation_path = out_dir / f"{prefix}_validation.csv"
    test_path = out_dir / f"{prefix}_test.csv"
    context_path = out_dir / f"{prefix}_context_summary.csv"
    split_path = out_dir / f"{prefix}_split_summary.csv"
    replicate_path = out_dir / f"{prefix}_replicate_noise_summary.csv"
    weight_path = out_dir / f"{prefix}_weight_coverage.csv"
    pr_path = out_dir / f"{prefix}_pr_coverage.csv"
    combined_results_path = out_dir / f"{prefix}_combined_grid_search_results.csv"
    combined_diagnostics_path = out_dir / f"{prefix}_combined_grid_search_diagnostics.csv"
    final_audit_path = out_dir / f"{prefix}_final_audit.json"

    training_ready.to_csv(training_path, index=False)
    for split_name, path in [
        ("train", train_path),
        ("validation", validation_path),
        ("test", test_path),
    ]:
        training_ready[
            training_ready["Data_Split"].astype(str) == split_name
        ].to_csv(path, index=False)
    context_summary.to_csv(context_path, index=False)
    split_summary.to_csv(split_path, index=False)
    replicate_summary.to_csv(replicate_path, index=False)
    weight_coverage.to_csv(weight_path, index=False)
    pr_coverage.to_csv(pr_path, index=False)
    results.to_csv(combined_results_path, index=False)
    diagnostics.to_csv(combined_diagnostics_path, index=False)

    final_audit.update({
        "plan_file": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "source_results_files": [str(path) for path in results_paths],
        "source_diagnostics_files": [str(path) for path in diagnostics_paths],
        "outputs": {
            "training_ready": str(training_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "context_summary": str(context_path),
            "split_summary": str(split_path),
            "replicate_noise_summary": str(replicate_path),
            "weight_coverage": str(weight_path),
            "pr_coverage": str(pr_path),
            "combined_results": str(combined_results_path),
            "combined_diagnostics": str(combined_diagnostics_path),
        },
    })
    final_audit_path.write_text(json.dumps(final_audit, indent=2), encoding="utf-8")

    print("\nFlexDC Sweep V3 post-generation audit")
    print("---------------------------------------")
    print(f"Status: {final_audit['status']}")
    print(f"Training-ready rows: {len(training_ready):,}")
    print(f"Training-ready CSV: {training_path}")
    print(f"Final audit: {final_audit_path}")
    print()
    print(context_summary.to_string(index=False))

    if final_audit["warnings"]:
        print("\nWARNINGS:")
        for warning in final_audit["warnings"]:
            print(" -", warning)
    if final_audit["hard_errors"]:
        print("\nHARD ERRORS:")
        for error in final_audit["hard_errors"]:
            print(" -", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
