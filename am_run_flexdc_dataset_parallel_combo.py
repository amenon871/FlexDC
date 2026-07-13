import argparse
import csv
import json
import re
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
PEACSIM_DIR = SRC_DIR / "peacsim"

DATA_GENERATOR = "am_data_extraction_wizard.py"

DEFAULT_GRADIENT_CONFIG = "../../configs/gradient_descent/gradient_descent.ini"
DEFAULT_CLUSTER_CONFIG = "../../configs/cluster/cluster.ini"
DEFAULT_POLICY_NAME = "AQA"
DEFAULT_PR_NUM_CHUNKS = 10
DEFAULT_MAX_WORKERS = 24

SUPPORTED_RUNTIME_POLICIES = [
    "AQA",
    "flex-resource",
    "priority-cap",
    "proportional-cap",
]

# These remain fully overrideable with --workload-configs.
DEFAULT_WORKLOAD_CONFIGS = [
    "../../configs/workload/W9.ini",
    "../../configs/workload/W10.ini",
    "../../configs/workload/W-train-2app.ini",
    "../../configs/workload/W-inference-2app.ini",
]

# This remains fully overrideable with --experiment-configs.
DEFAULT_EXPERIMENT_CONFIGS = [
    "../../configs/experiment/generated_server_counts/exp_servers_64.ini",
]


def safe_name(path_str: str) -> str:
    return Path(path_str).stem.replace(".", "_").replace("-", "_")


def format_minutes(seconds: float) -> str:
    return f"{seconds / 60:.2f} min"


def parse_csv_list(value: Optional[str], default: list[str]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)

    items = [item.strip() for item in str(value).split(",") if item.strip()]

    if not items:
        raise ValueError(f"Expected at least one item, got: {value}")

    return items


def validate_existing_file(path_str: str, base_dir: Path, label: str) -> None:
    path = Path(path_str)

    if not path.is_absolute():
        path = base_dir / path

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path_str} -> resolved as {path}")


def build_optional_generator_args(args) -> list[str]:
    """
    Pass optional modular arguments through to the current data extraction wizard.
    Important: these are physical kW/server P/R flags, not old FlexDC ratio flags.
    """
    optional_args = []

    passthrough_pairs = [
        ("--node-count-control", args.node_count_control),
        ("--pbar-kw-per-server-range", args.pbar_kw_per_server_range),
        ("--r-kw-per-server-range", args.r_kw_per_server_range),
        ("--pbar-kw-per-server-values", args.pbar_kw_per_server_values),
        ("--r-kw-per-server-values", args.r_kw_per_server_values),
        ("--utilization-values", args.utilization_values),
        ("--auto-workload-pr-sweep", args.auto_workload_pr_sweep),
        ("--pbar-points", args.pbar_points),
        ("--r-points-per-pbar", args.r_points_per_pbar),
        ("--r-lower-kw-per-server", args.r_lower_kw_per_server),
        ("--pbar-lower-factor", args.pbar_lower_factor),
        ("--pbar-upper-factor", args.pbar_upper_factor),
        ("--pr-upper-factor", args.pr_upper_factor),
        ("--weight-vectors", args.weight_vectors),
        ("--weight-file", args.weight_file),
        ("--power-cap-percentage", args.power_cap_percentage),
        ("--probabilities", args.probabilities),
    ]

    for flag, value in passthrough_pairs:
        if value is not None:
            optional_args.extend([flag, str(value)])

    return optional_args


def preflight_validate(args, workload_configs: list[str], experiment_configs: list[str]) -> None:
    data_generator_path = PEACSIM_DIR / DATA_GENERATOR
    if not data_generator_path.exists():
        raise FileNotFoundError(f"Data generator not found: {data_generator_path}")

    validate_existing_file(args.gradient_config, PEACSIM_DIR, "--gradient-config")
    validate_existing_file(args.cluster_config, PEACSIM_DIR, "--cluster-config")

    for workload_config in workload_configs:
        validate_existing_file(workload_config, PEACSIM_DIR, "--workload-configs item")

    for experiment_config in experiment_configs:
        validate_existing_file(experiment_config, PEACSIM_DIR, "--experiment-configs item")

    if args.weight_file is not None:
        validate_existing_file(args.weight_file, PEACSIM_DIR, "--weight-file")

    if args.policy_name not in SUPPORTED_RUNTIME_POLICIES:
        raise ValueError(
            f"Unsupported policy {args.policy_name}. "
            f"Expected one of: {SUPPORTED_RUNTIME_POLICIES}"
        )

    if args.node_count_control is not None:
        normalized = str(args.node_count_control).strip().lower()
        if normalized not in ["true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"]:
            raise ValueError("--node-count-control must be true or false.")

    if args.auto_workload_pr_sweep is not None:
        normalized = str(args.auto_workload_pr_sweep).strip().lower()
        if normalized not in ["true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"]:
            raise ValueError("--auto-workload-pr-sweep must be true or false.")

    if args.pr_num_chunks <= 0:
        raise ValueError(f"--pr-num-chunks must be positive. Got: {args.pr_num_chunks}")

    if args.max_workers <= 0:
        raise ValueError(f"--max-workers must be positive. Got: {args.max_workers}")

    if args.worker_pr_chunk_index is not None:
        if args.worker_pr_chunk_index < 0 or args.worker_pr_chunk_index >= args.pr_num_chunks:
            raise ValueError(
                f"--worker-pr-chunk-index must be in [0, {args.pr_num_chunks - 1}]. "
                f"Got: {args.worker_pr_chunk_index}"
            )

    if args.pbar_kw_per_server_range is not None and args.pbar_kw_per_server_values is not None:
        raise ValueError("Use either --pbar-kw-per-server-range or --pbar-kw-per-server-values, not both.")

    if args.r_kw_per_server_range is not None and args.r_kw_per_server_values is not None:
        raise ValueError("Use either --r-kw-per-server-range or --r-kw-per-server-values, not both.")

    if args.weight_vectors is not None and args.weight_file is not None:
        raise ValueError("Use either --weight-vectors or --weight-file, not both.")


def validate_grid_search_output(output_dir_label: str) -> tuple[str, int, str, int]:
    """
    Validate that the generator produced both:
    - grid_search_results.csv: raw simulator/data collection outputs only
    - grid_search_diagnostics.csv: residuals, percentiles, and diagnostic activation terms
    """
    optimization_dir = PEACSIM_DIR / "output" / "optimization"
    matches = sorted(
        optimization_dir.glob(f"{output_dir_label}_*/grid_search_results.csv"),
        key=lambda p: p.stat().st_mtime
    )

    if not matches:
        raise FileNotFoundError(
            f"No grid_search_results.csv found for output label {output_dir_label} "
            f"under {optimization_dir}"
        )

    results_csv_path = matches[-1]
    diagnostics_csv_path = results_csv_path.parent / "grid_search_diagnostics.csv"

    required_results_columns = [
        "Iteration",
        "Weight_Sample_ID",
        "Policy_Name",
        "Workload_Name",
        "Workload_Config",
        "Node_Count_Control",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "P_actual_watts",
        "R_actual_watts",
        "Pbar_ratio",
        "R_ratio",
        "Pbar_denominator_watts",
        "R_denominator_watts",
        "server_count",
        "utilization",
        "workload_mix_size",
        "workload_mix",
        "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost",
        "Mtrack_Error_MeanAbs_Normalized",
        "Mtrack_Error_MeanAbs_Watts",
        "Mtrack_Price_Coefficient_piE",
        "Mtrack_Hour_Seconds",
        "Mtrack_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "QoS_Delay_Probabilities",
    ]

    first_results_row = None
    with results_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing_columns = [col for col in required_results_columns if col not in headers]
        if missing_columns:
            raise ValueError(f"{results_csv_path} is missing required columns: {missing_columns}")

        results_row_count = 0
        for row in reader:
            results_row_count += 1
            if first_results_row is None:
                first_results_row = row

    if results_row_count <= 0:
        raise ValueError(f"{results_csv_path} has no data rows.")

    if first_results_row is not None:
        workload_mix_size = int(float(first_results_row["workload_mix_size"]))
        missing_weight_columns = [f"Weight_{i}" for i in range(workload_mix_size) if f"Weight_{i}" not in headers]
        if missing_weight_columns:
            raise ValueError(f"{results_csv_path} is missing dynamic weight columns: {missing_weight_columns}")

        workload_mix = json.loads(first_results_row["workload_mix"])
        if len(workload_mix) != workload_mix_size:
            raise ValueError(
                f"workload_mix length mismatch in {results_csv_path}: "
                f"expected {workload_mix_size}, got {len(workload_mix)}"
            )

        bad_workload_rows = [idx for idx, item in enumerate(workload_mix) if len(item) != 6]
        if bad_workload_rows:
            raise ValueError(
                f"workload_mix should contain 6 job-feature values per job type and no embedded weights. "
                f"Rows with unexpected length: {bad_workload_rows}"
            )

    if not diagnostics_csv_path.exists():
        raise FileNotFoundError(f"Expected diagnostics CSV not found: {diagnostics_csv_path}")

    required_diagnostic_columns = [
        "Iteration",
        "Weight_Sample_ID",
        "Workload_Name",
        "Workload_Config",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "PR_Sweep_Mode",
        "Pmin_kw_per_server",
        "Pmax_kw_per_server",
        "Pbar_lower_bound_kw_per_server",
        "Pbar_upper_bound_kw_per_server",
        "PR_upper_bound_kw_per_server",
        "R_lower_bound_kw_per_server",
        "R_max_for_this_Pbar_kw_per_server",
        "P_plus_R_kw_per_server",
        "P_minus_R_kw_per_server",
        "Pbar_points",
        "R_points_per_Pbar",
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
        "Ctrack_Epsilon_90th",
        "Ctrack_Gamma",
        "Ctrack_Residual",
        "Ctrack_Mu",
        "Ctrack_MuScaled_Residual",
        "Ctrack_SoftPlus_Value",
        "Ctrack_Psi",
        "Ctrack_Weighted_Cost",
        "QoS_Delay_Probability_Residual_Sum",
        "QoS_Delay_Probability_Residuals",
        "Diagnostic_FlexDC_SoftPlus_QoS_Cost",
        "Diagnostic_FullPaperObjective_Cost",
    ]

    with diagnostics_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing_columns = [col for col in required_diagnostic_columns if col not in headers]
        if missing_columns:
            raise ValueError(f"{diagnostics_csv_path} is missing required columns: {missing_columns}")

        diagnostics_row_count = 0
        for row in reader:
            diagnostics_row_count += 1
            lower = float(row["Weight_Final_Lower_Bound"])
            upper = float(row["Weight_Final_Upper_Bound"])
            weight_min = float(row["Weight_Min"])
            weight_max = float(row["Weight_Max"])

            if weight_min < lower - 1e-6:
                raise ValueError(
                    f"Weight_Min violates Weight_Final_Lower_Bound in {diagnostics_csv_path}: "
                    f"Weight_Min={weight_min}, lower={lower}"
                )

            if weight_max > upper + 1e-6:
                raise ValueError(
                    f"Weight_Max violates Weight_Final_Upper_Bound in {diagnostics_csv_path}: "
                    f"Weight_Max={weight_max}, upper={upper}"
                )

    if diagnostics_row_count != results_row_count:
        raise ValueError(
            f"Row count mismatch: {results_csv_path} has {results_row_count} rows, "
            f"but {diagnostics_csv_path} has {diagnostics_row_count} rows."
        )

    return str(results_csv_path), results_row_count, str(diagnostics_csv_path), diagnostics_row_count


def build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = str(SRC_DIR)

    return env


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gradient-config", type=str, default=DEFAULT_GRADIENT_CONFIG,
                        help=f"Gradient config path. Default: {DEFAULT_GRADIENT_CONFIG}")

    parser.add_argument("--cluster-config", type=str, default=DEFAULT_CLUSTER_CONFIG,
                        help=f"Cluster config path. Default: {DEFAULT_CLUSTER_CONFIG}")

    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY_NAME,
                        choices=SUPPORTED_RUNTIME_POLICIES,
                        help=f"Runtime policy. Default: {DEFAULT_POLICY_NAME}")

    parser.add_argument("--workload-configs", type=str, default=None,
                        help="Comma-separated workload config paths. Default: W9, W10, W-train-2app, W-inference-2app.")

    parser.add_argument("--experiment-configs", type=str, default=None,
                        help="Comma-separated experiment config paths. Default: exp_servers_64 only.")

    parser.add_argument("--output-prefix", type=str, default="rangefind64",
                        help="Prefix for output folder labels. Default: rangefind64")

    parser.add_argument("--pr-num-chunks", type=int, default=DEFAULT_PR_NUM_CHUNKS,
                        help=f"Split the generated P/R list into this many deterministic chunks. "
                             f"Default: {DEFAULT_PR_NUM_CHUNKS}")

    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Maximum concurrent local workers on Linux/macOS. "
                             f"Ignored by the Windows visible-window launcher. Default: {DEFAULT_MAX_WORKERS}")

    # Optional generator arguments. P/R are physical kW/server values.
    parser.add_argument("--node-count-control", type=str, default=None,
                        help="Optional override: true or false.")

    parser.add_argument("--pbar-kw-per-server-range", type=str, default=None,
                        help="Optional physical Pbar range in kW/server: 'min,max,num_points'.")

    parser.add_argument("--r-kw-per-server-range", type=str, default=None,
                        help="Optional physical R range in kW/server: 'min,max,num_points'.")

    parser.add_argument("--pbar-kw-per-server-values", type=str, default=None,
                        help="Optional exact comma-separated physical Pbar values in kW/server.")

    parser.add_argument("--r-kw-per-server-values", type=str, default=None,
                        help="Optional exact comma-separated physical R values in kW/server.")

    parser.add_argument("--utilization-values", type=str, default=None,
                        help="Optional comma-separated utilization values. Omit for generator default full set.")

    parser.add_argument("--auto-workload-pr-sweep", type=str, default=None,
                        help="If true, generator derives Pbar/R sweep from each workload config's Pmin/Pmax.")

    parser.add_argument("--pbar-points", type=int, default=None,
                        help="Number of Pbar points for auto workload P/R sweep.")

    parser.add_argument("--r-points-per-pbar", type=int, default=None,
                        help="Number of R points generated inside each Pbar's valid interval.")

    parser.add_argument("--r-lower-kw-per-server", type=float, default=None,
                        help="Lower R value for auto workload P/R sweep in kW/server.")

    parser.add_argument("--pbar-lower-factor", type=float, default=None,
                        help="Lower Pbar factor applied to workload Pmin.")

    parser.add_argument("--pbar-upper-factor", type=float, default=None,
                        help="Upper Pbar factor applied to workload Pmax.")

    parser.add_argument("--pr-upper-factor", type=float, default=None,
                        help="Upper Pbar+R factor applied to workload Pmax.")

    parser.add_argument("--weight-vectors", type=str, default=None,
                        help="Optional inline custom workload weight vectors.")

    parser.add_argument("--weight-file", type=str, default=None,
                        help="Optional CSV or JSON weight file.")

    parser.add_argument("--power-cap-percentage", type=float, default=None,
                        help="PriorityCap alpha value in [0,1]. Only used for priority-cap.")

    parser.add_argument("--probabilities", type=str, default=None,
                        help="FlexResource probabilities, one value per job type.")


    # Exact sweep-plan mode. Legacy Cartesian/auto-grid options remain available when this is omitted.
    parser.add_argument("--sweep-plan-file", type=str, default=None,
                        help="Master exact sweep plan CSV generated by am_generate_flexdc_sweep_plan_v2.py.")
    parser.add_argument("--plan-num-chunks-per-context", type=int, default=4,
                        help="Split each workload/server/utilization context into this many local workers. Default: 4.")
    parser.add_argument("--plan-max-workers", type=int, default=2,
                        help="Maximum concurrent exact-plan workers on this PC. Default: 2.")
    parser.add_argument("--plan-run-id", type=str, default=None,
                        help="Stable run ID used for chunk files and resume. Default: timestamp.")
    parser.add_argument("--plan-resume", action="store_true",
                        help="Skip a plan chunk when a complete validated output already exists.")
    parser.add_argument("--plan-dry-run", action="store_true",
                        help="Write plan chunks and commands but do not launch FlexDC.")



def build_generator_command(args, experiment_config: str, workload_config: str, output_dir: str, pr_chunk_index: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        DATA_GENERATOR,
        "--gradient-config", args.gradient_config,
        "--experiment-config", experiment_config,
        "--cluster-config", args.cluster_config,
        "--policy-name", args.policy_name,
        "--job-config", workload_config,
        "--output-dir", output_dir,
        "--pr-chunk-index", str(pr_chunk_index),
        "--pr-num-chunks", str(args.pr_num_chunks),
    ] + build_optional_generator_args(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch 1 FlexDC worker per workload/experiment combo.")
    parser.add_argument(
        "--worker-experiment",
        type=str,
        default=None,
        help="Internal worker mode: one experiment config."
    )

    parser.add_argument(
        "--worker-workload",
        type=str,
        default=None,
        help="Internal worker mode: one workload config."
    )

    parser.add_argument(
        "--worker-pr-chunk-index",
        type=int,
        default=None,
        help="Internal worker mode: zero-based P/R chunk index."
    )

    add_common_arguments(parser)
    return parser


def get_common_launch_args(args) -> list[str]:
    common_args = [
        "--gradient-config", args.gradient_config,
        "--cluster-config", args.cluster_config,
        "--policy-name", args.policy_name,
        "--output-prefix", args.output_prefix,
        "--pr-num-chunks", str(args.pr_num_chunks),
    ]

    common_args.extend(build_optional_generator_args(args))
    return common_args


def run_one_combo_worker(args) -> int:
    experiment_config = args.worker_experiment
    workload_config = args.worker_workload
    pr_chunk_index = 0 if args.worker_pr_chunk_index is None else int(args.worker_pr_chunk_index)

    if not experiment_config or not workload_config:
        raise ValueError("Worker mode requires both --worker-experiment and --worker-workload.")

    if pr_chunk_index < 0 or pr_chunk_index >= args.pr_num_chunks:
        raise ValueError(
            f"Worker P/R chunk index must be in [0, {args.pr_num_chunks - 1}]. Got: {pr_chunk_index}"
        )

    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(
        args=args,
        workload_configs=[workload_config],
        experiment_configs=[experiment_config],
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = safe_name(experiment_config)
    workload_name = safe_name(workload_config)

    chunk_label = f"prchunk{pr_chunk_index + 1:02d}of{args.pr_num_chunks:02d}"
    output_dir = f"{args.output_prefix}_{args.policy_name}_{workload_name}_{exp_name}_{chunk_label}"
    manifest_path = REPO_ROOT / (
        f"dataset_worker_manifest_{args.policy_name}_{workload_name}_{exp_name}_{chunk_label}_{timestamp}.csv"
    )

    cmd = build_generator_command(args, experiment_config, workload_config, output_dir, pr_chunk_index)

    print("=" * 100)
    print(f"Worker combo: {workload_config} × {experiment_config}")
    print(f"Policy: {args.policy_name}")
    print(f"P/R chunk: {pr_chunk_index + 1}/{args.pr_num_chunks} (zero-based index {pr_chunk_index})")
    print(f"Output dir label: {output_dir}")
    print(f"P/R chunks per workload/server combo: {args.pr_num_chunks}")
    print(f"Optional generator args: {optional_generator_args if optional_generator_args else '[generator defaults]'}")
    print(f"Working directory: {PEACSIM_DIR}")
    print(f"PYTHONPATH includes: {SRC_DIR}")
    print(f"Manifest: {manifest_path}")
    print("=" * 100)

    env = build_env()
    start = time.time()

    status = "running"
    grid_search_results_path = ""
    grid_search_rows = ""
    diagnostics_path = ""
    diagnostics_rows = ""
    final_return_code = 0

    try:
        subprocess.run(
            cmd,
            cwd=PEACSIM_DIR,
            check=True,
            env=env,
        )

        grid_search_results_path, grid_search_rows, diagnostics_path, diagnostics_rows = validate_grid_search_output(output_dir)
        status = "completed"

        print(f"Validated raw output CSV: {grid_search_results_path}")
        print(f"Validated raw output rows: {grid_search_rows}")
        print(f"Validated diagnostics CSV: {diagnostics_path}")
        print(f"Validated diagnostics rows: {diagnostics_rows}")

    except subprocess.CalledProcessError as e:
        status = f"failed_returncode_{e.returncode}"
        final_return_code = e.returncode
        print(f"\nRun failed with return code {e.returncode}.")

    except KeyboardInterrupt:
        status = "interrupted_by_user"
        final_return_code = 130
        print("\nInterrupted by user.")

    except Exception as e:
        status = f"failed_validation_or_runtime_error_{type(e).__name__}"
        final_return_code = 1
        print("\nRun failed during validation or runtime checks.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")

    finally:
        elapsed = time.time() - start

        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "policy_name",
                    "experiment_config",
                    "workload_config",
                    "output_dir",
                    "pr_chunk_index",
                    "pr_num_chunks",
                    "grid_search_results",
                    "grid_search_rows",
                    "grid_search_diagnostics",
                    "grid_search_diagnostic_rows",
                    "status",
                    "elapsed_minutes",
                    "command",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "policy_name": args.policy_name,
                "experiment_config": experiment_config,
                "workload_config": workload_config,
                "output_dir": output_dir,
                "pr_chunk_index": pr_chunk_index,
                "pr_num_chunks": args.pr_num_chunks,
                "grid_search_results": grid_search_results_path,
                "grid_search_rows": grid_search_rows,
                "grid_search_diagnostics": diagnostics_path,
                "grid_search_diagnostic_rows": diagnostics_rows,
                "status": status,
                "elapsed_minutes": f"{elapsed / 60:.2f}",
                "command": " ".join(cmd),
            })

        print("\n" + "=" * 100)
        print(f"Worker combo done: {workload_name} × {exp_name} × pr chunk {pr_chunk_index + 1}/{args.pr_num_chunks}")
        print(f"Status: {status}")
        print(f"Elapsed: {format_minutes(elapsed)}")
        print(f"Manifest saved to: {manifest_path}")
        print("=" * 100)

    return final_return_code


def write_worker_bat_files(args, workload_configs: list[str], experiment_configs: list[str]) -> list[tuple[str, str, int, Path]]:
    batch_dir = REPO_ROOT / ".flexdc_worker_bats"
    batch_dir.mkdir(exist_ok=True)

    script_path = Path(__file__).resolve()
    bat_files = []
    common_args = get_common_launch_args(args)

    for experiment_config in experiment_configs:
        exp_name = safe_name(experiment_config)

        for workload_config in workload_configs:
            workload_name = safe_name(workload_config)

            for pr_chunk_index in range(args.pr_num_chunks):
                chunk_label = f"prchunk{pr_chunk_index + 1:02d}of{args.pr_num_chunks:02d}"
                bat_path = batch_dir / f"run_worker_{args.policy_name}_{workload_name}_{exp_name}_{chunk_label}.bat"

                worker_cmd = [
                    sys.executable,
                    "-u",
                    str(script_path),
                    "--worker-experiment",
                    experiment_config,
                    "--worker-workload",
                    workload_config,
                    "--worker-pr-chunk-index",
                    str(pr_chunk_index),
                ] + common_args

                worker_cmd_text = subprocess.list2cmdline(worker_cmd)

                bat_text = f"""@echo off
title FlexDC {args.policy_name} {workload_name} {exp_name} {chunk_label}
cd /d "{REPO_ROOT}"
set PYTHONUNBUFFERED=1
set PYTHONPATH={SRC_DIR};%PYTHONPATH%
echo Starting FlexDC worker: {args.policy_name} {workload_name} {exp_name} {chunk_label}
echo Repo root: {REPO_ROOT}
echo Command:
echo {worker_cmd_text}
echo.
{worker_cmd_text}
echo.
echo Worker finished: {args.policy_name} {workload_name} {exp_name} {chunk_label}
echo Press any key to close this window.
pause > nul
"""

                bat_path.write_text(bat_text)
                bat_files.append((workload_name, exp_name, pr_chunk_index, bat_path))

    return bat_files

def build_worker_invocations(args, workload_configs: list[str], experiment_configs: list[str]) -> list[tuple[str, str, int, list[str]]]:
    """
    Build one runner invocation per workload/server/P-R-chunk combination.
    Used by the Linux/macOS local launcher and mirrors the Windows .bat launcher.
    """
    script_path = Path(__file__).resolve()
    common_args = get_common_launch_args(args)
    invocations = []

    for experiment_config in experiment_configs:
        for workload_config in workload_configs:
            for pr_chunk_index in range(args.pr_num_chunks):
                cmd = [
                    sys.executable,
                    "-u",
                    str(script_path),
                    "--worker-experiment",
                    experiment_config,
                    "--worker-workload",
                    workload_config,
                    "--worker-pr-chunk-index",
                    str(pr_chunk_index),
                ] + common_args
                invocations.append((workload_config, experiment_config, pr_chunk_index, cmd))

    return invocations


def launch_workers_locally(args, workload_configs: list[str], experiment_configs: list[str], launch_manifest_path: Path) -> None:
    """
    Linux/macOS launcher. It runs worker processes locally with a concurrency cap instead
    of opening Windows Command Prompt windows.
    """
    invocations = build_worker_invocations(args, workload_configs, experiment_configs)
    max_workers = min(args.max_workers, len(invocations))
    log_dir = REPO_ROOT / ".flexdc_worker_logs"
    log_dir.mkdir(exist_ok=True)
    env = build_env()

    print(f"Running {len(invocations)} worker process(es) locally with max concurrency {max_workers}.")
    print(f"Worker logs: {log_dir}")
    print()

    running = []
    next_index = 0
    completed = 0
    failures = []

    with launch_manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "launch_index",
                "policy_name",
                "workload_name",
                "experiment_name",
                "pr_chunk_index",
                "pr_num_chunks",
                "log_file",
                "status",
                "return_code",
                "command",
            ],
        )
        writer.writeheader()

        def start_one(i: int) -> None:
            workload_config, experiment_config, pr_chunk_index, cmd = invocations[i]
            workload_name = safe_name(workload_config)
            exp_name = safe_name(experiment_config)
            chunk_label = f"prchunk{pr_chunk_index + 1:02d}of{args.pr_num_chunks:02d}"
            log_path = log_dir / f"worker_{i + 1:04d}_{args.policy_name}_{workload_name}_{exp_name}_{chunk_label}.log"
            log_f = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append({
                "launch_index": i + 1,
                "workload_name": workload_name,
                "experiment_name": exp_name,
                "pr_chunk_index": pr_chunk_index,
                "cmd": cmd,
                "log_path": log_path,
                "log_file": log_f,
                "proc": proc,
            })
            writer.writerow({
                "launch_index": i + 1,
                "policy_name": args.policy_name,
                "workload_name": workload_name,
                "experiment_name": exp_name,
                "pr_chunk_index": pr_chunk_index,
                "pr_num_chunks": args.pr_num_chunks,
                "log_file": str(log_path),
                "status": "launched",
                "return_code": "",
                "command": " ".join(cmd),
            })
            f.flush()
            print(
                f"Launched {i + 1}/{len(invocations)}: "
                f"{args.policy_name} {workload_name} {exp_name} "
                f"P/R chunk {pr_chunk_index + 1}/{args.pr_num_chunks}"
            )

        while next_index < len(invocations) and len(running) < max_workers:
            start_one(next_index)
            next_index += 1

        while running:
            time.sleep(2)
            still_running = []

            for item in running:
                return_code = item["proc"].poll()
                if return_code is None:
                    still_running.append(item)
                    continue

                item["log_file"].close()
                completed += 1
                status = "completed" if return_code == 0 else "failed"
                if return_code != 0:
                    failures.append(item)

                writer.writerow({
                    "launch_index": item["launch_index"],
                    "policy_name": args.policy_name,
                    "workload_name": item["workload_name"],
                    "experiment_name": item["experiment_name"],
                    "pr_chunk_index": item["pr_chunk_index"],
                    "pr_num_chunks": args.pr_num_chunks,
                    "log_file": str(item["log_path"]),
                    "status": status,
                    "return_code": return_code,
                    "command": " ".join(item["cmd"]),
                })
                f.flush()

                print(
                    f"Finished {completed}/{len(invocations)}: "
                    f"{item['workload_name']} {item['experiment_name']} "
                    f"P/R chunk {item['pr_chunk_index'] + 1}/{args.pr_num_chunks} "
                    f"status={status} rc={return_code}"
                )

                while next_index < len(invocations) and len(still_running) < max_workers:
                    start_one(next_index)
                    next_index += 1

            running = still_running

    print()
    print(f"All local workers finished. Launch manifest saved to: {launch_manifest_path}")
    if failures:
        print(f"Failures: {len(failures)}")
        print("Inspect these log files first:")
        for item in failures[:10]:
            print(f"  {item['log_path']}")
        raise SystemExit(1)

def launch_workers(args) -> None:
    workload_configs = parse_csv_list(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_configs = parse_csv_list(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(args, workload_configs, experiment_configs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_manifest_path = REPO_ROOT / f"dataset_parallel_combo_launch_manifest_{args.policy_name}_{timestamp}.csv"
    total_workers = len(workload_configs) * len(experiment_configs) * args.pr_num_chunks

    print(f"Launching {total_workers} visible Command Prompt worker window(s)...")
    print("Each worker runs exactly one workload config, one experiment/server-count config, and one P/R chunk.")
    print()
    print(f"Policy: {args.policy_name}")
    print(f"Workload configs: {workload_configs}")
    print(f"Experiment configs: {experiment_configs}")
    print(f"Optional generator args: {optional_generator_args if optional_generator_args else '[generator defaults]'}")
    print(f"Working directory for workers: {PEACSIM_DIR}")
    print(f"PYTHONPATH includes: {SRC_DIR}")
    print(f"Launch manifest: {launch_manifest_path}")
    print()
    print("Data quality note:")
    print("- This launcher changes only parallelism, not simulator logic.")
    print("- It launches one worker for each workload/server/P-R-chunk combination.")
    print(f"- This launch uses {len(workload_configs)} workloads × {len(experiment_configs)} server configs × {args.pr_num_chunks} P/R chunks = {total_workers} workers.")
    print()

    if os.name != "nt":
        launch_workers_locally(args, workload_configs, experiment_configs, launch_manifest_path)
        return

    bat_files = write_worker_bat_files(args, workload_configs, experiment_configs)

    with launch_manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "launch_index",
                "policy_name",
                "workload_name",
                "experiment_name",
                "pr_chunk_index",
                "pr_num_chunks",
                "batch_file",
                "status",
            ],
        )
        writer.writeheader()

        for launch_index, (workload_name, exp_name, pr_chunk_index, bat_path) in enumerate(bat_files, start=1):
            subprocess.Popen(
                [
                    "cmd.exe",
                    "/c",
                    "start",
                    f"FlexDC {args.policy_name} {workload_name} {exp_name} prchunk{pr_chunk_index + 1:02d}of{args.pr_num_chunks:02d}",
                    "cmd.exe",
                    "/k",
                    str(bat_path),
                ],
                cwd=REPO_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

            writer.writerow({
                "launch_index": launch_index,
                "policy_name": args.policy_name,
                "workload_name": workload_name,
                "experiment_name": exp_name,
                "pr_chunk_index": pr_chunk_index,
                "pr_num_chunks": args.pr_num_chunks,
                "batch_file": str(bat_path),
                "status": "launched",
            })
            f.flush()

            print(
                f"Launched {launch_index}/{len(bat_files)}: "
                f"{args.policy_name} {workload_name} {exp_name} "
                f"pr chunk {pr_chunk_index + 1}/{args.pr_num_chunks}"
            )
            print(f"  Batch file: {bat_path}")

    print()
    print("All workers launched.")
    print("Check the Windows taskbar for Command Prompt windows.")
    print("Each window should eventually print: Status: completed")
    print(f"Launch manifest saved to: {launch_manifest_path}")




def resolve_from_repo_or_peacsim(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (PEACSIM_DIR / path).resolve()


def latest_plan_output(output_label: str) -> Optional[Path]:
    optimization_dir = PEACSIM_DIR / "output" / "optimization"
    matches = sorted(
        optimization_dir.glob(f"{output_label}_*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def validate_plan_worker_output(output_label: str, expected_plan_ids: list[str]) -> dict:
    output_dir = latest_plan_output(output_label)
    if output_dir is None:
        raise FileNotFoundError(f"No output directory found for {output_label}")
    results_path = output_dir / "grid_search_results.csv"
    diagnostics_path = output_dir / "grid_search_diagnostics.csv"
    if not results_path.exists() or not diagnostics_path.exists():
        raise FileNotFoundError(f"Incomplete output in {output_dir}")
    results = pd.read_csv(results_path)
    diagnostics = pd.read_csv(diagnostics_path)
    for frame_name, frame in [("results", results), ("diagnostics", diagnostics)]:
        if "Plan_Row_ID" not in frame.columns:
            raise ValueError(f"{frame_name} output lacks Plan_Row_ID: {output_dir}")
        if frame["Plan_Row_ID"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate Plan_Row_ID in {frame_name}: {output_dir}")
    actual_ids = set(results["Plan_Row_ID"].astype(str))
    diagnostic_ids = set(diagnostics["Plan_Row_ID"].astype(str))
    expected_ids = set(str(x) for x in expected_plan_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(f"Plan result IDs mismatch. Missing={missing}, extra={extra}")
    if diagnostic_ids != expected_ids:
        missing = sorted(expected_ids - diagnostic_ids)[:10]
        extra = sorted(diagnostic_ids - expected_ids)[:10]
        raise ValueError(f"Plan diagnostic IDs mismatch. Missing={missing}, extra={extra}")
    if len(results) != len(diagnostics) or len(results) != len(expected_plan_ids):
        raise ValueError(
            f"Plan row-count mismatch: expected={len(expected_plan_ids)}, "
            f"results={len(results)}, diagnostics={len(diagnostics)}"
        )
    return {
        "output_dir": str(output_dir),
        "results_csv": str(results_path),
        "diagnostics_csv": str(diagnostics_path),
        "rows": int(len(results)),
    }


def prepare_plan_invocations(args) -> tuple[list[dict], Path, Path]:
    plan_path = resolve_from_repo_or_peacsim(args.sweep_plan_file)
    if not plan_path.exists():
        raise FileNotFoundError(f"Sweep plan not found: {plan_path}")
    plan = pd.read_csv(plan_path)
    required = [
        "plan_row_id", "context_id", "workload_config", "experiment_config",
        "server_count", "utilization", "Pbar_kw_per_server", "R_kw_per_server",
        "weights", "simulation_seed",
    ]
    missing = [c for c in required if c not in plan.columns]
    if missing:
        raise ValueError(f"Sweep plan missing required columns: {missing}")
    if plan["plan_row_id"].astype(str).duplicated().any():
        raise ValueError("Sweep plan has duplicate plan_row_id values.")
    if args.plan_num_chunks_per_context <= 0:
        raise ValueError("--plan-num-chunks-per-context must be positive.")
    if args.plan_max_workers <= 0:
        raise ValueError("--plan-max-workers must be positive.")

    run_id = args.plan_run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_root = REPO_ROOT / ".flexdc_plan_chunks" / run_id
    log_root = REPO_ROOT / ".flexdc_worker_logs" / f"plan_{run_id}"
    chunk_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    invocations: list[dict] = []
    group_cols = ["context_id", "workload_config", "experiment_config"]
    for (context_id, workload_config, experiment_config), group in plan.groupby(group_cols, sort=True):
        group = group.sort_values("plan_row_id").reset_index(drop=True)
        n_chunks = min(args.plan_num_chunks_per_context, len(group))
        split_indices = np.array_split(np.arange(len(group)), n_chunks)
        chunks = [group.iloc[idx].copy() for idx in split_indices]
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            chunk_label = f"prchunk{chunk_index + 1:02d}of{n_chunks:02d}"
            safe_context = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(context_id))
            chunk_path = chunk_root / f"{safe_context}_{chunk_label}.csv"
            chunk.to_csv(chunk_path, index=False)
            output_label = f"{args.output_prefix}_{args.policy_name}_{safe_context}_{chunk_label}"
            log_path = log_root / f"{output_label}.log"
            workload_path = resolve_from_repo_or_peacsim(str(workload_config))
            experiment_path = resolve_from_repo_or_peacsim(str(experiment_config))
            if not workload_path.exists():
                raise FileNotFoundError(workload_path)
            if not experiment_path.exists():
                raise FileNotFoundError(experiment_path)

            gradient_path = resolve_from_repo_or_peacsim(args.gradient_config)
            cluster_path = resolve_from_repo_or_peacsim(args.cluster_config)
            cmd = [
                sys.executable, "-u", DATA_GENERATOR,
                "--gradient-config", str(gradient_path),
                "--experiment-config", str(experiment_path),
                "--cluster-config", str(cluster_path),
                "--policy-name", args.policy_name,
                "--job-config", str(workload_path),
                "--output-dir", output_label,
                "--sweep-plan-file", str(chunk_path),
                "--plan-chunk-index", "0",
                "--plan-num-chunks", "1",
                "--node-count-control", str(args.node_count_control or "true"),
                "--pbar-lower-factor", str(args.pbar_lower_factor if args.pbar_lower_factor is not None else 0.9),
                "--pbar-upper-factor", str(args.pbar_upper_factor if args.pbar_upper_factor is not None else 1.0),
                "--pr-upper-factor", str(args.pr_upper_factor if args.pr_upper_factor is not None else 1.2),
                "--r-lower-kw-per-server", str(args.r_lower_kw_per_server if args.r_lower_kw_per_server is not None else 0.01),
            ]
            if args.power_cap_percentage is not None:
                cmd.extend(["--power-cap-percentage", str(args.power_cap_percentage)])
            if args.probabilities is not None:
                cmd.extend(["--probabilities", str(args.probabilities)])
            invocations.append({
                "context_id": str(context_id),
                "chunk_index": chunk_index,
                "chunk_count": n_chunks,
                "chunk_file": str(chunk_path),
                "output_label": output_label,
                "expected_plan_ids": chunk["plan_row_id"].astype(str).tolist(),
                "expected_rows": int(len(chunk)),
                "log_path": str(log_path),
                "command": cmd,
            })
    return invocations, chunk_root, log_root


def run_exact_plan(args) -> None:
    invocations, chunk_root, log_root = prepare_plan_invocations(args)
    run_id = chunk_root.name
    manifest_path = REPO_ROOT / f"dataset_plan_launch_manifest_{args.policy_name}_{run_id}.csv"
    env = build_env()
    rows: list[dict] = []
    pending: list[dict] = []

    for i, item in enumerate(invocations, start=1):
        base = {
            "launch_index": i,
            "context_id": item["context_id"],
            "chunk_index": item["chunk_index"],
            "chunk_count": item["chunk_count"],
            "chunk_file": item["chunk_file"],
            "expected_rows": item["expected_rows"],
            "output_label": item["output_label"],
            "log_file": item["log_path"],
            "command_text": subprocess.list2cmdline(item["command"]),
        }
        if args.plan_resume:
            try:
                validation = validate_plan_worker_output(item["output_label"], item["expected_plan_ids"])
                rows.append({**base, **validation, "status": "skipped_complete", "return_code": 0})
                print(f"Resume: already complete {item['output_label']}")
                continue
            except Exception:
                pass
        if args.plan_dry_run:
            rows.append({**base, "status": "dry_run", "return_code": ""})
        else:
            pending.append({**item, **base})

    def write_manifest() -> None:
        manifest_rows = []
        for row in rows:
            row = dict(row)
            if "command_text" in row and "command" not in row:
                row["command"] = row.pop("command_text")
            manifest_rows.append(row)
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    if args.plan_dry_run:
        write_manifest()
        print(f"Dry run prepared {len(invocations)} plan worker commands.")
        print(f"Chunk files: {chunk_root}")
        print(f"Manifest: {manifest_path}")
        return

    max_workers = min(args.plan_max_workers, max(1, len(pending)))
    running: list[dict] = []
    next_index = 0
    completed = 0
    failures = 0
    print(f"Launching {len(pending)} exact-plan workers with max concurrency {max_workers}.")
    print(f"Logs: {log_root}")

    def start_one(item: dict) -> None:
        log_f = Path(item["log_path"]).open("w")
        proc = subprocess.Popen(
            item["command"], cwd=PEACSIM_DIR, env=env,
            stdout=log_f, stderr=subprocess.STDOUT, text=True,
        )
        running.append({**item, "proc": proc, "log_handle": log_f, "started": time.time()})
        print(f"Launched {item['output_label']} ({item['expected_rows']} rows)")

    while next_index < len(pending) and len(running) < max_workers:
        start_one(pending[next_index]); next_index += 1

    while running:
        time.sleep(2)
        still_running: list[dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            elapsed = time.time() - item["started"]
            completed += 1
            record = {k: item[k] for k in [
                "launch_index", "context_id", "chunk_index", "chunk_count", "chunk_file",
                "expected_rows", "output_label", "log_file", "command_text"
            ]}
            record["command"] = record.pop("command_text")
            record["return_code"] = int(rc)
            record["elapsed_minutes"] = elapsed / 60.0
            if rc == 0:
                try:
                    validation = validate_plan_worker_output(item["output_label"], item["expected_plan_ids"])
                    record.update(validation)
                    record["status"] = "completed"
                except Exception as exc:
                    record["status"] = f"failed_output_validation_{type(exc).__name__}"
                    record["error"] = str(exc)
                    failures += 1
            else:
                record["status"] = "failed_process"
                failures += 1
            rows.append(record)
            write_manifest()
            print(f"Finished {completed}/{len(pending)} {item['output_label']} status={record['status']}")
            while next_index < len(pending) and len(still_running) < max_workers:
                start_one(pending[next_index]); next_index += 1
        running = still_running

    write_manifest()
    print(f"Plan launch manifest: {manifest_path}")
    if failures:
        raise SystemExit(f"{failures} exact-plan worker(s) failed. Inspect {log_root} and the manifest.")
    print("All exact-plan workers completed and passed row/Plan_Row_ID validation.")



def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.sweep_plan_file:
        if args.worker_experiment or args.worker_workload:
            raise ValueError("Internal legacy worker flags cannot be combined with --sweep-plan-file.")
        run_exact_plan(args)
        return

    if args.worker_experiment or args.worker_workload:
        raise SystemExit(run_one_combo_worker(args))

    launch_workers(args)


if __name__ == "__main__":
    main()
