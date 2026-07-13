#!/usr/bin/env python
"""
Recompute FlexDC objective columns with the simulated-annealing cost constants,
then write training/evaluation-compatible CSVs that keep the original column names.

Purpose
-------
This is a one-off compatibility script for the existing FlexDC dataset. It does
not rerun the simulator. It uses raw quantities already logged by the data wizard:

  - Simulator_RSR_Total_Cost           (from grid_search_results.csv)
  - Ctrack_Epsilon_90th                (from grid_search_diagnostics.csv)
  - QoS_Delay_Probabilities            (from grid_search_results.csv)

It recomputes the FlexDC paper-style tracking/QoS penalties using the constants
from configs/optimization/simulated_annealing/SA_train_low_util_RSR_1000server.ini:

  Ctrack = psi1 * SoftPlus(psi2 * (epsilon_90 - tracking_error_constraint))
  CQoS   = beta * sum_j SoftPlus(rho * (QoS_probability_j - qos_constraint))
  Cfull  = Simulator_RSR_Total_Cost + Ctrack + CQoS

Default constants:
  psi1=1, psi2=10, tracking_error_constraint=0.3, beta=20, rho=2, qos_constraint=0.1

What it writes
--------------
By default, the output directory contains CSV files with the SAME BASENAMES as
input files, but with the original FlexDC normal objective columns replaced by
SA-constant values:

  <out-dir>/<original_results_basename>.csv
  <out-dir>/<original_diagnostics_basename>.csv

That means downstream training/inference code can keep using the original column
names without modification, as long as it points to the recomputed output folder.

It also writes audit files:
  flexdc_sa_recompute_summary.csv
  flexdc_sa_recompute_metadata.json

Safety
------
Default behavior does NOT overwrite the original input files. Use --in-place only
if you intentionally want to overwrite the provided CSVs after a .bak backup is
created.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SA_PSI1 = 1.0
DEFAULT_SA_PSI2 = 10.0
DEFAULT_TRACKING_CONSTRAINT = 0.3
DEFAULT_SA_BETA = 20.0
DEFAULT_SA_RHO = 2.0
DEFAULT_SA_QOS_CONSTRAINT = 0.1

# These validate the old gradient_descent.ini-derived columns before replacement.
# They should match the current dataset columns before this script is applied.
DEFAULT_CURRENT_QOS_BETA = 0.5
DEFAULT_CURRENT_QOS_RHO = 2.0
DEFAULT_CURRENT_QOS_THRESHOLD = 0.1

PREFERRED_MERGE_KEYS = ["Source_Output_Dir", "Iteration"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute FlexDC normal-objective columns using simulated-annealing "
            "constants and write compatibility CSVs with original column names."
        )
    )
    parser.add_argument("--results-csv", required=True, help="Combined grid_search_results.csv")
    parser.add_argument("--diagnostics-csv", required=True, help="Combined grid_search_diagnostics.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Required unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSVs in place after writing .bak backups. Use carefully.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".before_sa_recompute.bak",
        help="Backup suffix used with --in-place.",
    )
    parser.add_argument(
        "--keep-original-columns",
        action="store_true",
        help="Keep old values as Original_* columns before replacing them.",
    )

    # SA constants.
    parser.add_argument("--psi1", type=float, default=DEFAULT_SA_PSI1)
    parser.add_argument("--psi2", type=float, default=DEFAULT_SA_PSI2)
    parser.add_argument("--tracking-error-constraint", type=float, default=DEFAULT_TRACKING_CONSTRAINT)
    parser.add_argument("--beta", type=float, default=DEFAULT_SA_BETA)
    parser.add_argument("--rho", type=float, default=DEFAULT_SA_RHO)
    parser.add_argument("--qos-constraint", type=float, default=DEFAULT_SA_QOS_CONSTRAINT)

    # Existing-column validation constants.
    parser.add_argument("--current-qos-beta", type=float, default=DEFAULT_CURRENT_QOS_BETA)
    parser.add_argument("--current-qos-rho", type=float, default=DEFAULT_CURRENT_QOS_RHO)
    parser.add_argument("--current-qos-threshold", type=float, default=DEFAULT_CURRENT_QOS_THRESHOLD)
    parser.add_argument("--validation-atol", type=float, default=1e-8)
    parser.add_argument("--validation-rtol", type=float, default=1e-6)
    parser.add_argument(
        "--no-fail-on-current-validation",
        action="store_true",
        help=(
            "Warn instead of failing if old logged columns do not match the "
            "gradient_descent.ini validation constants. Useful if input was already recomputed."
        ),
    )
    return parser.parse_args()


def stable_softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def parse_probability_vector(value) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]

    text = str(value).strip()
    if not text:
        raise ValueError("Empty QoS_Delay_Probabilities entry")

    try:
        parsed = json.loads(text)
        return [float(x) for x in parsed]
    except Exception:
        pass

    # Robust fallback for text like [np.float64(0.1), ...].
    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
    try:
        parsed = ast.literal_eval(cleaned)
        return [float(x) for x in parsed]
    except Exception as exc:
        raise ValueError(f"Could not parse QoS_Delay_Probabilities value: {text[:200]}") from exc


def parse_probability_matrix(series: pd.Series) -> np.ndarray:
    rows = [parse_probability_vector(value) for value in series]
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"QoS probability vector lengths are inconsistent: {sorted(lengths)}")
    return np.asarray(rows, dtype=float)


def validate_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def choose_merge_keys(results: pd.DataFrame, diagnostics: pd.DataFrame) -> list[str]:
    if all(col in results.columns and col in diagnostics.columns for col in PREFERRED_MERGE_KEYS):
        return PREFERRED_MERGE_KEYS

    fallback = [
        "Iteration",
        "Weight_Sample_ID",
        "Workload_Name",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "server_count",
        "utilization",
    ]
    keys = [col for col in fallback if col in results.columns and col in diagnostics.columns]
    if not keys:
        raise ValueError("Could not determine merge keys shared by results and diagnostics.")
    return keys


def assert_or_warn(condition: bool, message: str, fail: bool) -> None:
    if condition:
        print("PASS:", message)
    elif fail:
        raise ValueError("FAILED: " + message)
    else:
        print("WARNING:", message)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def maybe_keep_original(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            backup_col = f"Original_{col}"
            if backup_col not in df.columns:
                df[backup_col] = df[col]


def write_in_place_with_backup(path: Path, df: pd.DataFrame, backup_suffix: str) -> None:
    backup_path = path.with_name(path.name + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print("Backup written:", backup_path)
    else:
        print("Backup already exists, leaving untouched:", backup_path)
    df.to_csv(path, index=False)
    print("Overwrote:", path)


def main() -> None:
    args = parse_args()

    results_path = Path(args.results_csv)
    diagnostics_path = Path(args.diagnostics_csv)

    if not args.in_place and args.out_dir is None:
        raise ValueError("Provide --out-dir, or use --in-place intentionally.")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = results_path.parent

    print("Loading results:", results_path)
    results = pd.read_csv(results_path)
    print("Loading diagnostics:", diagnostics_path)
    diagnostics = pd.read_csv(diagnostics_path)

    merge_keys = choose_merge_keys(results, diagnostics)
    print("Merge keys:", merge_keys)

    validate_required_columns(
        results,
        merge_keys + [
            "Simulator_RSR_Total_Cost",
            "QoS_Delay_Probabilities",
            "QoS_Delay_Probability_Sum",
        ],
        "results CSV",
    )
    validate_required_columns(
        diagnostics,
        merge_keys + [
            "Ctrack_Epsilon_90th",
            "Ctrack_Gamma",
            "Ctrack_Mu",
            "Ctrack_Psi",
            "Ctrack_Weighted_Cost",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            "Diagnostic_FullPaperObjective_Cost",
        ],
        "diagnostics CSV",
    )

    if len(results) != len(diagnostics):
        raise ValueError(f"Row-count mismatch: results={len(results)}, diagnostics={len(diagnostics)}")

    result_calc_cols = list(dict.fromkeys(merge_keys + [
        "Simulator_RSR_Total_Cost",
        "QoS_Delay_Probabilities",
        "QoS_Delay_Probability_Sum",
    ]))
    calc = diagnostics.merge(
        results[result_calc_cols],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_from_results"),
    )
    if len(calc) != len(diagnostics):
        raise ValueError(f"Merge did not preserve all rows: merged={len(calc)}, diagnostics={len(diagnostics)}")

    print("Rows:", len(calc))

    epsilon_90 = calc["Ctrack_Epsilon_90th"].to_numpy(dtype=float)
    probabilities = parse_probability_matrix(calc["QoS_Delay_Probabilities"])
    simulator_rsr = calc["Simulator_RSR_Total_Cost"].to_numpy(dtype=float)

    # Validate that QoS_Delay_Probability_Sum matches the probability vector.
    prob_sum = probabilities.sum(axis=1)
    logged_prob_sum = calc["QoS_Delay_Probability_Sum"].to_numpy(dtype=float)
    prob_sum_ok = np.allclose(prob_sum, logged_prob_sum, rtol=args.validation_rtol, atol=args.validation_atol)
    assert_or_warn(
        prob_sum_ok,
        f"QoS_Delay_Probability_Sum matches parsed probability vectors; max_abs_diff={max_abs_diff(prob_sum, logged_prob_sum):.3e}",
        fail=True,
    )

    # Validate old logged Ctrack/QoS/full from current columns before replacement.
    old_gamma = calc["Ctrack_Gamma"].to_numpy(dtype=float)
    old_mu = calc["Ctrack_Mu"].to_numpy(dtype=float)
    old_psi = calc["Ctrack_Psi"].to_numpy(dtype=float)
    old_ctrack = old_psi * stable_softplus(old_mu * (epsilon_90 - old_gamma))
    old_qos = args.current_qos_beta * stable_softplus(
        args.current_qos_rho * (probabilities - args.current_qos_threshold)
    ).sum(axis=1)
    old_full = simulator_rsr + old_ctrack + old_qos

    old_ctrack_logged = calc["Ctrack_Weighted_Cost"].to_numpy(dtype=float)
    old_qos_logged = calc["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].to_numpy(dtype=float)
    old_full_logged = calc["Diagnostic_FullPaperObjective_Cost"].to_numpy(dtype=float)

    fail_current = not args.no_fail_on_current_validation
    ctrack_ok = np.allclose(old_ctrack, old_ctrack_logged, rtol=args.validation_rtol, atol=args.validation_atol)
    qos_ok = np.allclose(old_qos, old_qos_logged, rtol=args.validation_rtol, atol=args.validation_atol)
    full_ok = np.allclose(old_full, old_full_logged, rtol=args.validation_rtol, atol=args.validation_atol)

    assert_or_warn(
        ctrack_ok,
        f"current Ctrack reconstructs logged Ctrack; max_abs_diff={max_abs_diff(old_ctrack, old_ctrack_logged):.3e}",
        fail_current,
    )
    assert_or_warn(
        qos_ok,
        f"current QoS reconstructs logged QoS; max_abs_diff={max_abs_diff(old_qos, old_qos_logged):.3e}",
        fail_current,
    )
    assert_or_warn(
        full_ok,
        f"current full objective reconstructs logged full objective; max_abs_diff={max_abs_diff(old_full, old_full_logged):.3e}",
        fail_current,
    )

    # Recompute using simulated-annealing cost-function constants.
    new_tracking_residual = epsilon_90 - args.tracking_error_constraint
    new_tracking_scaled = args.psi2 * new_tracking_residual
    new_ctrack_softplus = stable_softplus(new_tracking_scaled)
    new_ctrack = args.psi1 * new_ctrack_softplus

    new_qos_residuals = probabilities - args.qos_constraint
    new_qos_scaled = args.rho * new_qos_residuals
    new_qos_softplus_per_job = stable_softplus(new_qos_scaled)
    new_qos = args.beta * new_qos_softplus_per_job.sum(axis=1)
    new_qos_residual_sum = new_qos_residuals.sum(axis=1)

    new_full = simulator_rsr + new_ctrack + new_qos

    # Post-compute validation.
    full_reconstruct_diff = max_abs_diff(new_full, simulator_rsr + new_ctrack + new_qos)
    assert_or_warn(
        full_reconstruct_diff <= args.validation_atol,
        f"new full objective equals M_RSR + Ctrack + CQoS; max_abs_diff={full_reconstruct_diff:.3e}",
        fail=True,
    )
    finite_ok = np.isfinite(new_ctrack).all() and np.isfinite(new_qos).all() and np.isfinite(new_full).all()
    assert_or_warn(finite_ok, "new recomputed columns are finite", fail=True)

    diag_out_df = diagnostics.copy()
    result_out_df = results.copy()

    if args.keep_original_columns:
        maybe_keep_original(
            diag_out_df,
            [
                "Ctrack_Gamma",
                "Ctrack_Psi",
                "Ctrack_Mu",
                "Ctrack_Residual",
                "Ctrack_MuScaled_Residual",
                "Ctrack_SoftPlus_Value",
                "Ctrack_Weighted_Cost",
                "QoS_Delay_Probability_Residual_Sum",
                "QoS_Delay_Probability_Residuals",
                "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
                "Diagnostic_FullPaperObjective_Cost",
            ],
        )
        maybe_keep_original(
            result_out_df,
            [
                "Ctrack_Weighted_Cost",
                "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
                "Diagnostic_FullPaperObjective_Cost",
            ],
        )

    # Replace diagnostics columns with recomputed SA-constant values under original names.
    diag_out_df["Ctrack_Gamma"] = float(args.tracking_error_constraint)
    diag_out_df["Ctrack_Psi"] = float(args.psi1)
    diag_out_df["Ctrack_Mu"] = float(args.psi2)
    diag_out_df["Ctrack_Residual"] = new_tracking_residual
    diag_out_df["Ctrack_MuScaled_Residual"] = new_tracking_scaled
    diag_out_df["Ctrack_SoftPlus_Value"] = new_ctrack_softplus
    diag_out_df["Ctrack_Weighted_Cost"] = new_ctrack
    diag_out_df["QoS_Delay_Probability_Residual_Sum"] = new_qos_residual_sum
    diag_out_df["QoS_Delay_Probability_Residuals"] = [
        json.dumps([float(x) for x in row]) for row in new_qos_residuals
    ]
    diag_out_df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"] = new_qos
    diag_out_df["Diagnostic_FullPaperObjective_Cost"] = new_full

    # Add/replace the most important FlexDC-normal target columns in results too.
    result_out_df = result_out_df.merge(
        diag_out_df[merge_keys + [
            "Ctrack_Epsilon_90th",
            "Ctrack_Gamma",
            "Ctrack_Psi",
            "Ctrack_Mu",
            "Ctrack_Residual",
            "Ctrack_MuScaled_Residual",
            "Ctrack_SoftPlus_Value",
            "Ctrack_Weighted_Cost",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
            "Diagnostic_FullPaperObjective_Cost",
        ]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "__new"),
    )
    for col in [
        "Ctrack_Epsilon_90th",
        "Ctrack_Gamma",
        "Ctrack_Psi",
        "Ctrack_Mu",
        "Ctrack_Residual",
        "Ctrack_MuScaled_Residual",
        "Ctrack_SoftPlus_Value",
        "Ctrack_Weighted_Cost",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    ]:
        new_col = f"{col}__new"
        if new_col in result_out_df.columns:
            result_out_df[col] = result_out_df[new_col]
            result_out_df.drop(columns=[new_col], inplace=True)

    # Validate output dataframes contain the recomputed values under original names.
    check_diag_ctrack = diag_out_df["Ctrack_Weighted_Cost"].to_numpy(dtype=float)
    check_diag_qos = diag_out_df["Diagnostic_FlexDC_SoftPlus_QoS_Cost"].to_numpy(dtype=float)
    check_diag_full = diag_out_df["Diagnostic_FullPaperObjective_Cost"].to_numpy(dtype=float)

    assert_or_warn(
        np.allclose(check_diag_ctrack, new_ctrack, rtol=args.validation_rtol, atol=args.validation_atol),
        f"diagnostics Ctrack_Weighted_Cost replaced correctly; max_abs_diff={max_abs_diff(check_diag_ctrack, new_ctrack):.3e}",
        fail=True,
    )
    assert_or_warn(
        np.allclose(check_diag_qos, new_qos, rtol=args.validation_rtol, atol=args.validation_atol),
        f"diagnostics Diagnostic_FlexDC_SoftPlus_QoS_Cost replaced correctly; max_abs_diff={max_abs_diff(check_diag_qos, new_qos):.3e}",
        fail=True,
    )
    assert_or_warn(
        np.allclose(check_diag_full, new_full, rtol=args.validation_rtol, atol=args.validation_atol),
        f"diagnostics Diagnostic_FullPaperObjective_Cost replaced correctly; max_abs_diff={max_abs_diff(check_diag_full, new_full):.3e}",
        fail=True,
    )

    if "Diagnostic_FullPaperObjective_Cost" in result_out_df.columns:
        result_full = result_out_df["Diagnostic_FullPaperObjective_Cost"].to_numpy(dtype=float)
        assert_or_warn(
            np.allclose(result_full, new_full, rtol=args.validation_rtol, atol=args.validation_atol),
            f"results Diagnostic_FullPaperObjective_Cost matches diagnostics; max_abs_diff={max_abs_diff(result_full, new_full):.3e}",
            fail=True,
        )

    results_out = out_dir / results_path.name
    diagnostics_out = out_dir / diagnostics_path.name
    summary_out = out_dir / "flexdc_sa_recompute_summary.csv"
    metadata_out = out_dir / "flexdc_sa_recompute_metadata.json"

    summary = pd.DataFrame([
        {
            "Column": "Ctrack_Weighted_Cost_old",
            "Mean": float(old_ctrack_logged.mean()),
            "Std": float(old_ctrack_logged.std()),
            "Min": float(old_ctrack_logged.min()),
            "Max": float(old_ctrack_logged.max()),
        },
        {
            "Column": "Ctrack_Weighted_Cost_recomputed",
            "Mean": float(new_ctrack.mean()),
            "Std": float(new_ctrack.std()),
            "Min": float(new_ctrack.min()),
            "Max": float(new_ctrack.max()),
        },
        {
            "Column": "Diagnostic_FlexDC_SoftPlus_QoS_Cost_old",
            "Mean": float(old_qos_logged.mean()),
            "Std": float(old_qos_logged.std()),
            "Min": float(old_qos_logged.min()),
            "Max": float(old_qos_logged.max()),
        },
        {
            "Column": "Diagnostic_FlexDC_SoftPlus_QoS_Cost_recomputed",
            "Mean": float(new_qos.mean()),
            "Std": float(new_qos.std()),
            "Min": float(new_qos.min()),
            "Max": float(new_qos.max()),
        },
        {
            "Column": "Diagnostic_FullPaperObjective_Cost_old",
            "Mean": float(old_full_logged.mean()),
            "Std": float(old_full_logged.std()),
            "Min": float(old_full_logged.min()),
            "Max": float(old_full_logged.max()),
        },
        {
            "Column": "Diagnostic_FullPaperObjective_Cost_recomputed",
            "Mean": float(new_full.mean()),
            "Std": float(new_full.std()),
            "Min": float(new_full.min()),
            "Max": float(new_full.max()),
        },
    ])

    metadata = {
        "results_csv": str(results_path),
        "diagnostics_csv": str(diagnostics_path),
        "rows": int(len(calc)),
        "merge_keys": merge_keys,
        "mode": "replace_original_columns_with_sa_recomputed_values",
        "sa_constants": {
            "psi1": args.psi1,
            "psi2": args.psi2,
            "tracking_error_constraint": args.tracking_error_constraint,
            "beta": args.beta,
            "rho": args.rho,
            "qos_constraint": args.qos_constraint,
        },
        "formula": {
            "Ctrack_Weighted_Cost": "psi1 * softplus(psi2 * (Ctrack_Epsilon_90th - tracking_error_constraint))",
            "Diagnostic_FlexDC_SoftPlus_QoS_Cost": "beta * sum_j softplus(rho * (QoS_Delay_Probability_j - qos_constraint))",
            "Diagnostic_FullPaperObjective_Cost": "Simulator_RSR_Total_Cost + Ctrack_Weighted_Cost + Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        },
        "validation": {
            "input_current_ctrack_max_abs_diff": max_abs_diff(old_ctrack, old_ctrack_logged),
            "input_current_qos_max_abs_diff": max_abs_diff(old_qos, old_qos_logged),
            "input_current_full_max_abs_diff": max_abs_diff(old_full, old_full_logged),
            "output_full_reconstruction_max_abs_diff": full_reconstruct_diff,
            "qos_probability_sum_max_abs_diff": max_abs_diff(prob_sum, logged_prob_sum),
        },
        "outputs": {
            "results_csv": str(results_out if not args.in_place else results_path),
            "diagnostics_csv": str(diagnostics_out if not args.in_place else diagnostics_path),
            "summary": str(summary_out),
            "metadata": str(metadata_out),
        },
    }

    if args.in_place:
        write_in_place_with_backup(results_path, result_out_df, args.backup_suffix)
        write_in_place_with_backup(diagnostics_path, diag_out_df, args.backup_suffix)
    else:
        result_out_df.to_csv(results_out, index=False)
        diag_out_df.to_csv(diagnostics_out, index=False)
        print("Wrote compatibility results CSV:", results_out)
        print("Wrote compatibility diagnostics CSV:", diagnostics_out)

    summary.to_csv(summary_out, index=False)
    metadata_out.write_text(json.dumps(metadata, indent=2))

    print("\nRecomputed constants used:")
    print(json.dumps(metadata["sa_constants"], indent=2))

    print("\nSummary:")
    print(summary.round(6).to_string(index=False))

    print("\nAudit files:")
    print(" ", summary_out)
    print(" ", metadata_out)


if __name__ == "__main__":
    main()
