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

DEFAULT_WORKLOAD_CONFIGS = [
    "../../configs/workload/W8.ini",
    "../../configs/workload/W9.ini",
    "../../configs/workload/W10.ini",
    "../../configs/workload/W11.ini",
    "../../configs/workload/W12.ini",
    "../../configs/workload/W13.ini",
]

DEFAULT_EXPERIMENT_CONFIGS = [
    "../../configs/experiment/generated_server_counts/exp_servers_64.ini",
    "../../configs/experiment/generated_server_counts/exp_servers_100.ini",
    "../../configs/experiment/generated_server_counts/exp_servers_500.ini",
    "../../configs/experiment/generated_server_counts/exp_servers_1000.ini",
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
    optional_args = []

    passthrough_pairs = [
        ("--node-count-control", args.node_count_control),
        ("--pbar-range", args.pbar_range),
        ("--r-range", args.r_range),
        ("--pbar-values", args.pbar_values),
        ("--r-values", args.r_values),
        ("--utilization-values", args.utilization_values),
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

    if args.pbar_range is not None and args.pbar_values is not None:
        raise ValueError("Use either --pbar-range or --pbar-values, not both.")

    if args.r_range is not None and args.r_values is not None:
        raise ValueError("Use either --r-range or --r-values, not both.")

    if args.weight_vectors is not None and args.weight_file is not None:
        raise ValueError("Use either --weight-vectors or --weight-file, not both.")


def validate_grid_search_output(output_dir_label: str) -> tuple[str, int]:
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

    csv_path = matches[-1]

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        required_columns = [
            "Iteration",
            "Weight_Sample_ID",
            "Policy_Name",
            "Node_Count_Control",
            "Pbar_ratio",
            "R_ratio",
            "P_actual_watts",
            "R_actual_watts",
            "server_count",
            "utilization",
            "workload_mix_size",
            "workload_mix",
            "Raw_Total_Cost",
            "Power_Cost",
            "Raw_Tracking_Cost",
            "Raw_QoS_Cost",
        ]

        missing_columns = [col for col in required_columns if col not in headers]
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

        row_count = sum(1 for _ in reader)

    if row_count <= 0:
        raise ValueError(f"{csv_path} has no data rows.")

    return str(csv_path), row_count


def build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = str(SRC_DIR)

    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch or run parallel FlexDC dataset workers.")

    parser.add_argument(
        "--worker-experiment",
        type=str,
        default=None,
        help="Internal worker mode: run all workloads for one experiment config."
    )

    parser.add_argument("--gradient-config", type=str, default=DEFAULT_GRADIENT_CONFIG,
                        help=f"Gradient config path. Default: {DEFAULT_GRADIENT_CONFIG}")

    parser.add_argument("--cluster-config", type=str, default=DEFAULT_CLUSTER_CONFIG,
                        help=f"Cluster config path. Default: {DEFAULT_CLUSTER_CONFIG}")

    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY_NAME,
                        choices=SUPPORTED_RUNTIME_POLICIES,
                        help=f"Runtime policy. Default: {DEFAULT_POLICY_NAME}")

    parser.add_argument("--workload-configs", type=str, default=None,
                        help="Comma-separated workload config paths. Default: W8-W13.")

    parser.add_argument("--experiment-configs", type=str, default=None,
                        help="Comma-separated experiment config paths. Default: 64, 100, 500, 1000 server configs.")

    parser.add_argument("--output-prefix", type=str, default="dataset",
                        help="Prefix for output folder labels. Default: dataset")

    parser.add_argument("--node-count-control", type=str, default=None,
                        help="Optional override: true or false.")

    parser.add_argument("--pbar-range", type=str, default=None,
                        help="Optional compact Pbar range: 'min,max,num_points'.")

    parser.add_argument("--r-range", type=str, default=None,
                        help="Optional compact R range: 'min,max,num_points'.")

    parser.add_argument("--pbar-values", type=str, default=None,
                        help="Optional exact comma-separated Pbar values.")

    parser.add_argument("--r-values", type=str, default=None,
                        help="Optional exact comma-separated R values.")

    parser.add_argument("--utilization-values", type=str, default=None,
                        help="Optional comma-separated utilization values.")

    parser.add_argument("--weight-vectors", type=str, default=None,
                        help="Optional inline custom workload weight vectors.")

    parser.add_argument("--weight-file", type=str, default=None,
                        help="Optional CSV or JSON weight file.")

    parser.add_argument("--power-cap-percentage", type=float, default=None,
                        help="PriorityCap alpha value in [0,1]. Only used for priority-cap.")

    parser.add_argument("--probabilities", type=str, default=None,
                        help="FlexResource probabilities, one value per job type.")

    return parser


def get_common_launch_args(args) -> list[str]:
    """
    Args that should be forwarded from launcher mode into each worker process.
    """
    common_args = [
        "--gradient-config", args.gradient_config,
        "--cluster-config", args.cluster_config,
        "--policy-name", args.policy_name,
        "--output-prefix", args.output_prefix,
    ]

    if args.workload_configs is not None:
        common_args.extend(["--workload-configs", args.workload_configs])

    # Do NOT forward --experiment-configs into each worker.
    # Each worker receives exactly one --worker-experiment instead.

    common_args.extend(build_optional_generator_args(args))

    return common_args


def run_worker(args) -> int:
    """
    Run all workload configs sequentially for one experiment/server-count config.
    Each visible worker terminal calls this with --worker-experiment.
    """
    workload_configs = parse_csv_list(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_config = args.worker_experiment
    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(args, workload_configs, [experiment_config])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = safe_name(experiment_config)
    manifest_path = REPO_ROOT / f"dataset_worker_manifest_{args.policy_name}_{exp_name}_{timestamp}.csv"

    runs = []
    for workload_config in workload_configs:
        workload_name = safe_name(workload_config)
        output_dir = f"{args.output_prefix}_{args.policy_name}_{workload_name}_{exp_name}"

        cmd = [
            sys.executable,
            "-u",
            DATA_GENERATOR,
            "--gradient-config", args.gradient_config,
            "--experiment-config", experiment_config,
            "--cluster-config", args.cluster_config,
            "--policy-name", args.policy_name,
            "--job-config", workload_config,
            "--output-dir", output_dir,
        ] + optional_generator_args

        runs.append({
            "experiment_config": experiment_config,
            "workload_config": workload_config,
            "output_dir": output_dir,
            "command": cmd,
        })

    print("=" * 100)
    print(f"Worker for experiment config: {experiment_config}")
    print(f"Policy: {args.policy_name}")
    print(f"Workloads in this worker: {len(runs)}")
    print(f"Workload configs: {workload_configs}")
    print(f"Optional generator args: {optional_generator_args if optional_generator_args else '[generator defaults]'}")
    print(f"Working directory: {PEACSIM_DIR}")
    print(f"PYTHONPATH includes: {SRC_DIR}")
    print(f"Manifest: {manifest_path}")
    print("=" * 100)

    env = build_env()
    overall_start = time.time()
    completed_runs = 0
    final_return_code = 0

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "total_runs",
                "policy_name",
                "experiment_config",
                "workload_config",
                "output_dir",
                "grid_search_results",
                "grid_search_rows",
                "status",
                "elapsed_minutes",
                "command",
            ],
        )
        writer.writeheader()
        f.flush()

        for idx, run in enumerate(runs, start=1):
            print("\n" + "=" * 100)
            print(f"[{exp_name}] Starting workload run {idx}/{len(runs)}")
            print(f"Experiment config: {run['experiment_config']}")
            print(f"Workload config:   {run['workload_config']}")
            print(f"Output dir label:  {run['output_dir']}")
            print("=" * 100)

            status = "running"
            grid_search_results_path = ""
            grid_search_rows = ""
            run_start = time.time()

            try:
                subprocess.run(
                    run["command"],
                    cwd=PEACSIM_DIR,
                    check=True,
                    env=env,
                )

                grid_search_results_path, grid_search_rows = validate_grid_search_output(
                    run["output_dir"]
                )

                status = "completed"
                completed_runs += 1

                print(f"Validated output CSV: {grid_search_results_path}")
                print(f"Validated output rows: {grid_search_rows}")

            except subprocess.CalledProcessError as e:
                status = f"failed_returncode_{e.returncode}"
                final_return_code = e.returncode
                print(f"\n[{exp_name}] Run failed with return code {e.returncode}.")
                print("Stopping this worker so you can debug before continuing.")

            except KeyboardInterrupt:
                status = "interrupted_by_user"
                final_return_code = 130
                print(f"\n[{exp_name}] Interrupted by user.")
                print("Stopping after writing this run's status to the manifest.")

            except Exception as e:
                status = f"failed_validation_or_runtime_error_{type(e).__name__}"
                final_return_code = 1
                print(f"\n[{exp_name}] Run failed during validation or runtime checks.")
                print(f"Error type: {type(e).__name__}")
                print(f"Error: {e}")
                print("Stopping this worker so you can debug before continuing.")

            finally:
                elapsed = time.time() - run_start
                total_elapsed = time.time() - overall_start

                writer.writerow({
                    "run_index": idx,
                    "total_runs": len(runs),
                    "policy_name": args.policy_name,
                    "experiment_config": run["experiment_config"],
                    "workload_config": run["workload_config"],
                    "output_dir": run["output_dir"],
                    "grid_search_results": grid_search_results_path,
                    "grid_search_rows": grid_search_rows,
                    "status": status,
                    "elapsed_minutes": f"{elapsed / 60:.2f}",
                    "command": " ".join(run["command"]),
                })
                f.flush()

                print(f"\n[{exp_name}] Workload run {idx}/{len(runs)} status: {status}")
                print(f"[{exp_name}] Workload elapsed time: {format_minutes(elapsed)}")

                if completed_runs > 0:
                    avg_per_completed = total_elapsed / completed_runs
                    remaining_runs = len(runs) - idx
                    eta_seconds = avg_per_completed * remaining_runs
                    print(f"[{exp_name}] Average completed workload time: {format_minutes(avg_per_completed)}")
                    print(f"[{exp_name}] Estimated remaining time for this worker: {format_minutes(eta_seconds)}")

            if status != "completed":
                break

    print("\n" + "=" * 100)
    print(f"Worker done for: {experiment_config}")
    print(f"Completed workload runs: {completed_runs}/{len(runs)}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 100)

    return final_return_code


def write_worker_bat_files(args, experiment_configs: list[str]) -> list[tuple[str, Path]]:
    batch_dir = REPO_ROOT / ".flexdc_worker_bats"
    batch_dir.mkdir(exist_ok=True)

    script_path = Path(__file__).resolve()
    bat_files = []
    common_args = get_common_launch_args(args)

    for experiment_config in experiment_configs:
        exp_name = safe_name(experiment_config)
        bat_path = batch_dir / f"run_worker_{args.policy_name}_{exp_name}.bat"

        worker_cmd = [
            sys.executable,
            "-u",
            str(script_path),
            "--worker-experiment",
            experiment_config,
        ] + common_args

        worker_cmd_text = subprocess.list2cmdline(worker_cmd)

        bat_text = f"""@echo off
title FlexDC {args.policy_name} {exp_name}
cd /d "{REPO_ROOT}"
set PYTHONUNBUFFERED=1
set PYTHONPATH={SRC_DIR};%PYTHONPATH%
echo Starting FlexDC worker: {args.policy_name} {exp_name}
echo Repo root: {REPO_ROOT}
echo Command:
echo {worker_cmd_text}
echo.
{worker_cmd_text}
echo.
echo Worker finished: {args.policy_name} {exp_name}
echo Press any key to close this window.
pause > nul
"""

        bat_path.write_text(bat_text)
        bat_files.append((exp_name, bat_path))

    return bat_files


def launch_workers(args) -> None:
    workload_configs = parse_csv_list(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_configs = parse_csv_list(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(args, workload_configs, experiment_configs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_manifest_path = REPO_ROOT / f"dataset_parallel_launch_manifest_{args.policy_name}_{timestamp}.csv"

    print(f"Launching {len(experiment_configs)} visible Command Prompt worker windows...")
    print("Each worker will run all workload configs sequentially for one experiment/server-count config.")
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
    print("- This does not change simulator inputs or labels.")
    print("- It only runs independent experiment-config batches at the same time.")
    print("- Each worker writes separate timestamped output folders and its own worker manifest.")
    print("- Each worker validates grid_search_results.csv after each workload run.")
    print()

    if os.name != "nt":
        print("This visible-window launcher is written for Windows.")
        print("On macOS/Linux, run workers manually with --worker-experiment.")
        return

    bat_files = write_worker_bat_files(args, experiment_configs)

    with launch_manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "launch_index",
                "policy_name",
                "experiment_config",
                "batch_file",
                "status",
            ],
        )
        writer.writeheader()

        for launch_index, (exp_name, bat_path) in enumerate(bat_files, start=1):
            subprocess.Popen(
                ["cmd.exe", "/c", "start", f"FlexDC {args.policy_name} {exp_name}", "cmd.exe", "/k", str(bat_path)],
                cwd=REPO_ROOT,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

            experiment_config = experiment_configs[launch_index - 1]

            writer.writerow({
                "launch_index": launch_index,
                "policy_name": args.policy_name,
                "experiment_config": experiment_config,
                "batch_file": str(bat_path),
                "status": "launched",
            })
            f.flush()

            print(f"Launched visible cmd window for {args.policy_name} {exp_name}")
            print(f"  Batch file: {bat_path}")

    print()
    print("All workers launched.")
    print("If you do not see windows, check the Windows taskbar for Command Prompt icons.")
    print("The batch files are in .flexdc_worker_bats, so you can double-click them manually too.")
    print(f"Launch manifest saved to: {launch_manifest_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.worker_experiment:
        raise SystemExit(run_worker(args))

    launch_workers(args)


if __name__ == "__main__":
    main()