import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
PEACSIM_DIR = SRC_DIR / "peacsim"

DATA_GENERATOR = "am_data_extraction_wizard.py"

DEFAULT_GRADIENT_CONFIG = "../../configs/gradient_descent/gradient_descent.ini"
DEFAULT_CLUSTER_CONFIG = "../../configs/cluster/cluster.ini"
DEFAULT_POLICY_NAME = "AQA"

SUPPORTED_RUNTIME_POLICIES = [
    "AQA",
    "flex-resource",
    "priority-cap",
    "proportional-cap",
]

# Tomorrow's range-finding subset: 2 CPU/SCC-style workloads and 2 GPU/AI workloads.
# These remain fully overrideable with --workload-configs.
DEFAULT_WORKLOAD_CONFIGS = [
    "../../configs/workload/W9.ini",
    "../../configs/workload/W10.ini",
    "../../configs/workload/W-train-2app.ini",
    "../../configs/workload/W-inference-2app.ini",
]

# Tomorrow's range-finding subset is 64 servers only.
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
    Pass optional modular arguments through to am_data_extraction_wizard.py.
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

    with results_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing_columns = [col for col in required_results_columns if col not in headers]
        if missing_columns:
            raise ValueError(f"{results_csv_path} is missing required columns: {missing_columns}")
        results_row_count = sum(1 for _ in reader)

    if results_row_count <= 0:
        raise ValueError(f"{results_csv_path} has no data rows.")

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
        diagnostics_row_count = sum(1 for _ in reader)

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


def build_generator_command(args, experiment_config: str, workload_config: str, output_dir: str) -> list[str]:
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

    add_common_arguments(parser)
    return parser


def get_common_launch_args(args) -> list[str]:
    common_args = [
        "--gradient-config", args.gradient_config,
        "--cluster-config", args.cluster_config,
        "--policy-name", args.policy_name,
        "--output-prefix", args.output_prefix,
    ]

    common_args.extend(build_optional_generator_args(args))
    return common_args


def run_one_combo_worker(args) -> int:
    experiment_config = args.worker_experiment
    workload_config = args.worker_workload

    if not experiment_config or not workload_config:
        raise ValueError("Worker mode requires both --worker-experiment and --worker-workload.")

    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(
        args=args,
        workload_configs=[workload_config],
        experiment_configs=[experiment_config],
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = safe_name(experiment_config)
    workload_name = safe_name(workload_config)

    output_dir = f"{args.output_prefix}_{args.policy_name}_{workload_name}_{exp_name}"
    manifest_path = REPO_ROOT / (
        f"dataset_worker_manifest_{args.policy_name}_{workload_name}_{exp_name}_{timestamp}.csv"
    )

    cmd = build_generator_command(args, experiment_config, workload_config, output_dir)

    print("=" * 100)
    print(f"Worker combo: {workload_config} × {experiment_config}")
    print(f"Policy: {args.policy_name}")
    print(f"Output dir label: {output_dir}")
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
                "grid_search_results": grid_search_results_path,
                "grid_search_rows": grid_search_rows,
                "grid_search_diagnostics": diagnostics_path,
                "grid_search_diagnostic_rows": diagnostics_rows,
                "status": status,
                "elapsed_minutes": f"{elapsed / 60:.2f}",
                "command": " ".join(cmd),
            })

        print("\n" + "=" * 100)
        print(f"Worker combo done: {workload_name} × {exp_name}")
        print(f"Status: {status}")
        print(f"Elapsed: {format_minutes(elapsed)}")
        print(f"Manifest saved to: {manifest_path}")
        print("=" * 100)

    return final_return_code


def write_worker_bat_files(args, workload_configs: list[str], experiment_configs: list[str]) -> list[tuple[str, str, Path]]:
    batch_dir = REPO_ROOT / ".flexdc_worker_bats"
    batch_dir.mkdir(exist_ok=True)

    script_path = Path(__file__).resolve()
    bat_files = []
    common_args = get_common_launch_args(args)

    for experiment_config in experiment_configs:
        exp_name = safe_name(experiment_config)

        for workload_config in workload_configs:
            workload_name = safe_name(workload_config)
            bat_path = batch_dir / f"run_worker_{args.policy_name}_{workload_name}_{exp_name}.bat"

            worker_cmd = [
                sys.executable,
                "-u",
                str(script_path),
                "--worker-experiment",
                experiment_config,
                "--worker-workload",
                workload_config,
            ] + common_args

            worker_cmd_text = subprocess.list2cmdline(worker_cmd)

            bat_text = f"""@echo off
title FlexDC {args.policy_name} {workload_name} {exp_name}
cd /d "{REPO_ROOT}"
set PYTHONUNBUFFERED=1
set PYTHONPATH={SRC_DIR};%PYTHONPATH%
echo Starting FlexDC worker: {args.policy_name} {workload_name} {exp_name}
echo Repo root: {REPO_ROOT}
echo Command:
echo {worker_cmd_text}
echo.
{worker_cmd_text}
echo.
echo Worker finished: {args.policy_name} {workload_name} {exp_name}
echo Press any key to close this window.
pause > nul
"""

            bat_path.write_text(bat_text)
            bat_files.append((workload_name, exp_name, bat_path))

    return bat_files


def launch_workers(args) -> None:
    workload_configs = parse_csv_list(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_configs = parse_csv_list(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(args, workload_configs, experiment_configs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_manifest_path = REPO_ROOT / f"dataset_parallel_combo_launch_manifest_{args.policy_name}_{timestamp}.csv"
    total_workers = len(workload_configs) * len(experiment_configs)

    print(f"Launching {total_workers} visible Command Prompt worker window(s)...")
    print("Each worker runs exactly one workload config and one experiment/server-count config.")
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
    print("- It launches one worker for each workload/server combination.")
    print("- With the default range-finding subset, this launches 4 workers: 4 workloads × 1 server config.")
    print()

    if os.name != "nt":
        print("This visible-window launcher is written for Windows.")
        print("On macOS/Linux, run workers manually with --worker-experiment and --worker-workload.")
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
                "batch_file",
                "status",
            ],
        )
        writer.writeheader()

        for launch_index, (workload_name, exp_name, bat_path) in enumerate(bat_files, start=1):
            subprocess.Popen(
                [
                    "cmd.exe",
                    "/c",
                    "start",
                    f"FlexDC {args.policy_name} {workload_name} {exp_name}",
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
                "batch_file": str(bat_path),
                "status": "launched",
            })
            f.flush()

            print(f"Launched {launch_index}/{len(bat_files)}: {args.policy_name} {workload_name} {exp_name}")
            print(f"  Batch file: {bat_path}")

    print()
    print("All workers launched.")
    print("Check the Windows taskbar for Command Prompt windows.")
    print("Each window should eventually print: Status: completed")
    print(f"Launch manifest saved to: {launch_manifest_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.worker_experiment or args.worker_workload:
        raise SystemExit(run_one_combo_worker(args))

    launch_workers(args)


if __name__ == "__main__":
    main()
