import argparse
import subprocess
import sys
from pathlib import Path
import datetime
import csv
import time
import os


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

# Default CPU/AQA-style workload family.
DEFAULT_WORKLOAD_CONFIGS = [
    "../../configs/workload/W8.ini",
    "../../configs/workload/W9.ini",
    "../../configs/workload/W10.ini",
    "../../configs/workload/W11.ini",
    "../../configs/workload/W12.ini",
    "../../configs/workload/W13.ini",
]

# Default CONDOR-style server-count coverage.
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

def parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    """
    This takes a thing like this and properly parses it:
    "a.ini,b.ini,c.ini" -> ["a.ini", "b.ini", "c.ini"]

    If value is omitted, return the provided default list.
    """
    if value is None or str(value).strip() == "":
        return list(default)

    items = [item.strip() for item in str(value).split(",") if item.strip()]

    if not items:
        raise ValueError(f"Expected at least one item, got: {value}")

    return items


def validate_existing_file(path_str: str, base_dir: Path, label: str) -> None:
    """
    Validate paths before launching long runs.
    """
    path = Path(path_str)

    if not path.is_absolute():
        path = base_dir / path

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path_str} -> resolved as {path}")


def build_optional_generator_args(args) -> list[str]:
    """
    Pass optional modular arguments through to am_data_extraction_wizard.py.
    If no optional value is chosen, its None, then the generator uses its own defaults.
    """
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
    """
    Validate that a completed generator run actually produced usable data.
    This checks the grid_search_results.csv file where it then checks required columns,
    and verifies that at least one data row exists.
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
            raise ValueError(
                f"{csv_path} is missing required columns: {missing_columns}"
            )

        row_count = sum(1 for _ in reader)

    if row_count <= 0:
        raise ValueError(f"{csv_path} has no data rows.")

    return str(csv_path), row_count

def main():
    parser = argparse.ArgumentParser(description="Run FlexDC dataset generation sequentially.")

    parser.add_argument("--gradient-config", type=str, default=DEFAULT_GRADIENT_CONFIG,
                        help=f"Gradient config path. Default: {DEFAULT_GRADIENT_CONFIG}")

    parser.add_argument("--cluster-config", type=str, default=DEFAULT_CLUSTER_CONFIG,
                        help=f"Cluster config path. Default: {DEFAULT_CLUSTER_CONFIG}")

    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY_NAME,
                        choices=SUPPORTED_RUNTIME_POLICIES,
                        help=f"Runtime policy. Default: {DEFAULT_POLICY_NAME}")

    parser.add_argument("--workload-configs", type=str, default=None,
                        help="Comma-separated workload config paths. "
                             "Default: W8-W13 CPU/AQA-style workload family.")

    parser.add_argument("--experiment-configs", type=str, default=None,
                        help="Comma-separated experiment config paths. "
                             "Default: exp_servers_64, 100, 500, 1000.")

    parser.add_argument("--output-prefix", type=str, default="dataset",
                        help="Prefix for output folder labels. Default: dataset")

    # The below are optional arguments!!!
    # If omitted, am_data_extraction_wizard.py uses its own defaults.
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

    args = parser.parse_args()

    workload_configs = parse_csv_list(args.workload_configs, DEFAULT_WORKLOAD_CONFIGS)
    experiment_configs = parse_csv_list(args.experiment_configs, DEFAULT_EXPERIMENT_CONFIGS)
    optional_generator_args = build_optional_generator_args(args)

    preflight_validate(args, workload_configs, experiment_configs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = REPO_ROOT / f"dataset_run_manifest_{timestamp}.csv"

    runs = []

    for experiment_config in experiment_configs:
        for workload_config in workload_configs:
            exp_name = safe_name(experiment_config)
            workload_name = safe_name(workload_config)

            output_dir = f"{args.output_prefix}_{args.policy_name}_{workload_name}_{exp_name}"

            cmd = [
                sys.executable,
                "-u",  # unbuffered output so progress prints immediately
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

    print(f"Planned runs: {len(runs)}")
    print(f"Policy: {args.policy_name}")
    print(f"Workload configs: {workload_configs}")
    print(f"Experiment configs: {experiment_configs}")
    print(f"Optional generator args: {optional_generator_args if optional_generator_args else '[generator defaults]'}")
    print(f"Working directory: {PEACSIM_DIR}")
    print(f"PYTHONPATH includes: {SRC_DIR}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Note:")
    print("- This driver tracks progress at the workload/server-count run level.")
    print("- am_data_extraction_wizard.py tracks progress inside each run with Iteration x/y.")
    print("- Each completed simulator iteration is appended to that run's grid_search_results.csv.")

    overall_start = time.time()
    completed_runs = 0

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
            print(f"Starting run {idx}/{len(runs)}")
            print(f"Experiment config: {run['experiment_config']}")
            print(f"Workload config:   {run['workload_config']}")
            print(f"Output dir label:  {run['output_dir']}")
            print("=" * 100)

            status = "running"
            grid_search_results_path = ""
            grid_search_rows = ""
            run_start = time.time()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            
            # We run from src/peacsim so FlexDC's ../../data ISO path still works.
            # But from src/peacsim, Python cannot see the top-level peacsim package
            # unless repo/src is on PYTHONPATH.
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + existing_pythonpath
            else:
                env["PYTHONPATH"] = str(SRC_DIR)

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
                print(f"\nRun failed with return code {e.returncode}.")
                print("Stopping so you can debug before continuing.")

            except KeyboardInterrupt:
                status = "interrupted_by_user"
                print("\nInterrupted by user.")
                print("Stopping after writing this run's status to the manifest.")

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

                print(f"\nRun {idx}/{len(runs)} status: {status}")
                print(f"Run elapsed time: {format_minutes(elapsed)}")

                if completed_runs > 0:
                    avg_per_completed = total_elapsed / completed_runs
                    remaining_runs = len(runs) - idx
                    eta_seconds = avg_per_completed * remaining_runs
                    print(f"Average completed-run time: {format_minutes(avg_per_completed)}")
                    print(f"Estimated remaining time: {format_minutes(eta_seconds)}")

            if status != "completed":
                break

    print("\nDone.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
