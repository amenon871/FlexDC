import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# DEFAULT TARGET RUN
# ============================================================

DEFAULT_OUTPUT_PREFIX = "rangefind64_p020_084_rtargeted"
DEFAULT_POLICY_NAME = "AQA"
DEFAULT_EXPERIMENT_NAME = "exp_servers_64"

EXPECTED_WORKLOAD_NAMES = [
    "W9",
    "W10",
    "W_train_2app",
    "W_inference_2app",
]

DEFAULT_OPTIMIZATION_DIR = Path("src") / "peacsim" / "output" / "optimization"
DEFAULT_COMBINED_DIR = Path("combined_outputs_new_2")


# ============================================================
# HELPERS
# ============================================================

def get_weight_columns(columns):
    weight_cols = []

    for col in columns:
        match = re.fullmatch(r"Weight_(\d+)", str(col))
        if match:
            weight_cols.append((int(match.group(1)), col))

    return [col for _, col in sorted(weight_cols)]


def combine_weight_columns(row, weight_cols):
    weights = []

    for col in weight_cols:
        value = row.get(col)

        if pd.isna(value):
            continue

        weights.append(float(value))

    return json.dumps(weights)


def workload_name_from_output_dir(output_dir_name, output_prefix, policy_name, experiment_name):
    """
    Expected output dir format from the runner/wizard:

    {output_prefix}_{policy}_{workload_name}_{experiment_name}_{timestamp}

    Example:
    rangefind64_p020_084_r001_020_AQA_W9_exp_servers_64_20260609010101
    """
    start = f"{output_prefix}_{policy_name}_"
    end_marker = f"_{experiment_name}_"

    if not output_dir_name.startswith(start):
        return None

    if end_marker not in output_dir_name:
        return None

    middle = output_dir_name[len(start):]
    workload_name = middle.split(end_marker)[0]

    return workload_name


def discover_run_dirs(optimization_dir, output_prefix, policy_name, experiment_name, expected_workloads):
    if not optimization_dir.exists():
        raise FileNotFoundError(f"Optimization dir does not exist: {optimization_dir}")

    pattern = f"{output_prefix}_{policy_name}_*_{experiment_name}_*"
    candidates = sorted([p for p in optimization_dir.glob(pattern) if p.is_dir()])

    matched_by_workload = {workload: [] for workload in expected_workloads}

    for output_dir in candidates:
        workload_name = workload_name_from_output_dir(
            output_dir.name,
            output_prefix=output_prefix,
            policy_name=policy_name,
            experiment_name=experiment_name,
        )

        if workload_name in matched_by_workload:
            raw_csv = output_dir / "grid_search_results.csv"
            diagnostics_csv = output_dir / "grid_search_diagnostics.csv"

            if raw_csv.exists():
                matched_by_workload[workload_name].append({
                    "workload_name": workload_name,
                    "output_dir": output_dir,
                    "raw_csv": raw_csv,
                    "diagnostics_csv": diagnostics_csv if diagnostics_csv.exists() else None,
                    "mtime": raw_csv.stat().st_mtime,
                })

    missing = [workload for workload, matches in matched_by_workload.items() if not matches]
    if missing:
        raise FileNotFoundError(
            "Missing expected workload outputs:\n"
            + "\n".join(f"  - {workload}" for workload in missing)
            + f"\n\nSearched under: {optimization_dir}\nPattern: {pattern}"
        )

    selected = []

    for workload, matches in matched_by_workload.items():
        matches = sorted(matches, key=lambda item: item["mtime"], reverse=True)

        if len(matches) > 1:
            print()
            print(f"WARNING: Found multiple matching outputs for workload {workload}.")
            print("Selecting the most recent one:")
            print(f"  SELECTED: {matches[0]['output_dir']}")
            print("  Other matches:")
            for item in matches[1:]:
                print(f"    {item['output_dir']}")

        selected.append(matches[0])

    selected = sorted(selected, key=lambda item: item["workload_name"])

    return selected


def add_source_columns(df, item):
    df = df.copy()
    df.insert(0, "Source_Output_Dir", item["output_dir"].name)
    df.insert(1, "Source_Workload_Name", item["workload_name"])
    return df


def combine_raw_csvs(selected_items):
    frames = []

    for item in selected_items:
        df = pd.read_csv(item["raw_csv"])
        df = add_source_columns(df, item)

        weight_cols = get_weight_columns(df.columns)

        if weight_cols:
            df["Weights"] = df.apply(lambda row: combine_weight_columns(row, weight_cols), axis=1)
            df = df.drop(columns=weight_cols)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    return combined


def combine_diagnostics_csvs(selected_items):
    frames = []

    for item in selected_items:
        if item["diagnostics_csv"] is None:
            continue

        df = pd.read_csv(item["diagnostics_csv"])
        df = add_source_columns(df, item)
        frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True, sort=False)


def validate_combined_raw(df, expected_workloads):
    required_columns = [
        "Source_Output_Dir",
        "Source_Workload_Name",
        "Iteration",
        "Weight_Sample_ID",
        "Policy_Name",
        "Node_Count_Control",
        "Pbar_kw_per_server",
        "R_kw_per_server",
        "P_actual_watts",
        "R_actual_watts",
        "Pbar_ratio",
        "R_ratio",
        "server_count",
        "utilization",
        "workload_mix_size",
        "workload_mix",
        "Weights",
        "Simulator_RSR_Total_Cost",
        "Simulator_Power_Cost",
        "Simulator_Tracking_Cost",
        "QoS_Delay_Probability_Sum",
        "QoS_Violation_Ratio",
        "QoS_Delay_Probabilities",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Combined raw CSV is missing required columns: {missing}")

    actual_workloads = sorted(df["Source_Workload_Name"].dropna().unique().tolist())
    expected_sorted = sorted(expected_workloads)

    if actual_workloads != expected_sorted:
        raise ValueError(
            f"Unexpected workloads.\n"
            f"Expected: {expected_sorted}\n"
            f"Actual:   {actual_workloads}"
        )

    if df.empty:
        raise ValueError("Combined raw dataframe has zero rows.")

    if "Weight_0" in df.columns:
        raise ValueError("Weight columns were not removed correctly.")

    if df["Weights"].isna().any():
        raise ValueError("Some rows have missing Weights arrays.")


def write_manifest(selected_items, output_path):
    rows = []

    for item in selected_items:
        rows.append({
            "workload_name": item["workload_name"],
            "output_dir": str(item["output_dir"]),
            "raw_csv": str(item["raw_csv"]),
            "diagnostics_csv": str(item["diagnostics_csv"]) if item["diagnostics_csv"] else "",
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combine specific rangefind64 FlexDC worker CSVs into one dataset."
    )

    parser.add_argument("--optimization-dir", type=str, default=str(DEFAULT_OPTIMIZATION_DIR),
                        help=f"Path to peacsim output/optimization dir. Default: {DEFAULT_OPTIMIZATION_DIR}")

    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX,
                        help=f"Output prefix to target. Default: {DEFAULT_OUTPUT_PREFIX}")

    parser.add_argument("--policy-name", type=str, default=DEFAULT_POLICY_NAME,
                        help=f"Policy name to target. Default: {DEFAULT_POLICY_NAME}")

    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help=f"Experiment/server config stem to target. Default: {DEFAULT_EXPERIMENT_NAME}")

    parser.add_argument("--combined-dir", type=str, default=str(DEFAULT_COMBINED_DIR),
                        help=f"Directory for combined outputs. Default: {DEFAULT_COMBINED_DIR}")

    args = parser.parse_args()

    optimization_dir = Path(args.optimization_dir)
    combined_dir = Path(args.combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    selected_items = discover_run_dirs(
        optimization_dir=optimization_dir,
        output_prefix=args.output_prefix,
        policy_name=args.policy_name,
        experiment_name=args.experiment_name,
        expected_workloads=EXPECTED_WORKLOAD_NAMES,
    )

    print()
    print("Selected CSVs for this combine:")
    for item in selected_items:
        print(f"  {item['workload_name']}: {item['raw_csv']}")

    combined_raw = combine_raw_csvs(selected_items)
    validate_combined_raw(combined_raw, EXPECTED_WORKLOAD_NAMES)

    combined_diagnostics = combine_diagnostics_csvs(selected_items)

    raw_output_path = combined_dir / f"{args.output_prefix}_{args.policy_name}_{args.experiment_name}_combined_raw.csv"
    diagnostics_output_path = combined_dir / f"{args.output_prefix}_{args.policy_name}_{args.experiment_name}_combined_diagnostics.csv"
    manifest_output_path = combined_dir / f"{args.output_prefix}_{args.policy_name}_{args.experiment_name}_combine_manifest.csv"

    combined_raw.to_csv(raw_output_path, index=False)

    if combined_diagnostics is not None:
        combined_diagnostics.to_csv(diagnostics_output_path, index=False)

    write_manifest(selected_items, manifest_output_path)

    print()
    print("Done.")
    print(f"Combined raw CSV:         {raw_output_path}")
    print(f"Rows:                     {len(combined_raw)}")
    print(f"Columns:                  {len(combined_raw.columns)}")

    if combined_diagnostics is not None:
        print(f"Combined diagnostics CSV: {diagnostics_output_path}")
        print(f"Diagnostics rows:         {len(combined_diagnostics)}")
        print(f"Diagnostics columns:      {len(combined_diagnostics.columns)}")

    print(f"Combine manifest:         {manifest_output_path}")
    print()
    print("Workload row counts:")
    print(combined_raw["Source_Workload_Name"].value_counts().sort_index().to_string())
    print()
    print("Server counts:")
    print(sorted(combined_raw["server_count"].dropna().unique().tolist()))
    print()
    print("Pbar kW/server values:")
    print(sorted(combined_raw["Pbar_kw_per_server"].dropna().unique().tolist()))
    print()
    print("R kW/server values:")
    print(sorted(combined_raw["R_kw_per_server"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()