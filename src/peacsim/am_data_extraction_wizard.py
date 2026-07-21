import argparse
import numpy as np
import pandas as pd
import json
import csv
from peacsim.simulator import Simulator
from peacsim.run_simulator import init_job_table, init_node_table
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.flex_resource_policy import FlexResourcePolicy
from peacsim.proportional_cap_policy import ProportionalCapPolicy
from peacsim.priority_cap_policy import PriorityCapPolicy
from peacsim.parsing import gradient_descent_reader as gradient_parser
import datetime
import os
import itertools
import random
from pathlib import Path
from calculate_qos_cost import calculate_qos_cost


#Here are my default sweep settings I added
DEFAULT_UTILIZATION_VALUES = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]

# Physical RSR bid sweep values. These are kW/server, not FlexDC normalized ratios.
# Format: min, max, num_points
DEFAULT_PBAR_KW_PER_SERVER_RANGE = (0.2, 0.6, 10)
DEFAULT_R_KW_PER_SERVER_RANGE = (0.0, 0.12, 10)

# Workload-specific targeted P/R sweep defaults.
# In auto mode, Pbar is derived from workload Pmin/Pmax and each Pbar gets its own R linspace.
DEFAULT_AUTO_WORKLOAD_PR_SWEEP = True
DEFAULT_PBAR_LOWER_FACTOR = 0.9
DEFAULT_PBAR_UPPER_FACTOR = 1.0
DEFAULT_PR_UPPER_FACTOR = 1.2 # This is what we called alpha
DEFAULT_PBAR_POINTS = 10
DEFAULT_R_POINTS_PER_PBAR = 10
DEFAULT_R_LOWER_KW_PER_SERVER = 0.01
PR_SWEEP_EPS = 1e-9

# Paper tracking constraint gamma for Ctrack residual columns.
TRACKING_ERROR_CONSTRAINT = 0.3

# Constants used by simulator.py::final_output() for the simulator monetary tracking term.
# Keep these explicit so Mtrack_Cost lineage is audit-friendly.
MTRACK_HOUR_SECONDS = 3600
MTRACK_PRICE_COEFFICIENT_PIE = 0.1 / (MTRACK_HOUR_SECONDS * 1000)

DEFAULT_NUM_WEIGHT_SAMPLES = 100
DEFAULT_WEIGHT_SEED = 42
DEFAULT_WEIGHT_TEMPERATURE = 2.0
DEFAULT_INCLUDE_EQUAL_WEIGHTS = True
DEFAULT_WEIGHT_MIN_FRACTION_OF_EQUAL = 0.1
DEFAULT_WEIGHT_MAX_MULTIPLE_OF_EQUAL = 4.0

#############################
# Runtime policy defaults
#############################
SUPPORTED_RUNTIME_POLICIES = [
    "AQA",
    "flex-resource",
    "priority-cap",
    "proportional-cap",
]

# To be consistent with the paper I used the following defaults:
# AQA and FlexResource use node-count control.
# PriorityCap and ProportionalCap rely only on power capping.
DEFAULT_NODE_COUNT_CONTROL_BY_POLICY = {
    "AQA": True,
    "flex-resource": True,
    "priority-cap": False,
    "proportional-cap": False,
}

# PriorityCap uses alpha, the max allowed cap percentage of each job type's power range.
DEFAULT_POWER_CAP_PERCENTAGE = 1.0

# FlexResource requires one probability per job type.
DEFAULT_FLEX_RESOURCE_PROBABILITY = 0.8



def init_output(output_dir, experiment_config, policy_config):
    with open(output_dir + '/power_trace.csv', 'w') as f:
        f.write(
            'time,QoSemergency,numNeededForTracking,waitingSum,target,realSum,trackingError,QoSJobSum,standbyPower')
        for i in range(experiment_config.server_count):
            f.write(',client%d' % (i + 1))
        f.write(',estimateSum')
        for i in range(experiment_config.server_count):
            f.write(',state%d' % (i + 1))
        f.write('\n')
    with open(output_dir + '/PRtable.csv', 'w') as f:
        f.write('phase,P,R\n')
        f.write('%d,%d,%d\n' % (0, policy_config.P, policy_config.R))
    with open(os.path.join(output_dir, 'job_table.csv'), 'w') as f:
        f.write('job_id,job_type_id,arrival_time,start_time,end_time,estimate_finished,update_time,realtime_qos,'
                'min_execution_time,qos_constraint,job_size\n')


def softplus(x):
    return np.log1p(np.exp(x))

def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def parse_float_list(value, argument_name):
    if value is None or str(value).strip() == "":
        return None

    try:
        parsed = [float(item.strip()) for item in str(value).split(",") if item.strip() != ""]
    except ValueError as exc:
        raise ValueError(f"Could not parse {argument_name} as comma-separated floats: {value}") from exc

    if not parsed:
        raise ValueError(f"{argument_name} was provided but no numeric values were found.")

    return parsed


def parse_bool_or_none(value, argument_name):
    """
    This parses optional true/false command-line values.
    """
    if value is None or str(value).strip() == "":
        return None

    normalized = str(value).strip().lower()

    if normalized in ["true", "t", "1", "yes", "y"]:
        return True

    if normalized in ["false", "f", "0", "no", "n"]:
        return False

    raise ValueError(f"{argument_name} must be true or false. Got: {value}")


def parse_probability_values(value, job_type_count):
    """
    Parse FlexResource probability values.
    If the user doesn give any custom probabilities, default to the same probability for every job type.
    """
    if value is None or str(value).strip() == "":
        probabilities = [DEFAULT_FLEX_RESOURCE_PROBABILITY] * job_type_count
    else:
        probabilities = parse_float_list(value, "--probabilities")

    if len(probabilities) != job_type_count:
        raise ValueError(
            f"--probabilities must contain exactly {job_type_count} values for this workload. "
            f"Got {len(probabilities)} values: {probabilities}"
        )

    for p in probabilities:
        if p < 0 or p > 1:
            raise ValueError(f"FlexResource probabilities must be in [0, 1]. Got: {probabilities}")

    return probabilities


def parse_range_spec(value, default_range, argument_name):
    """
    Ex. "0.7,1.8,10" -> np.linspace(0.7, 1.8, 10)
    """
    if value is None or str(value).strip() == "":
        min_value, max_value, num_points = default_range
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip() != ""]

        if len(parts) != 3:
            raise ValueError(
                f"{argument_name} must have format 'min,max,num_points'. "
                f"Example: '0.7,1.8,10'. Got: {value}"
            )

        try:
            min_value = float(parts[0])
            max_value = float(parts[1])
            num_points = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Could not parse {argument_name}. Expected 'float,float,int'. Got: {value}"
            ) from exc

    if num_points <= 0:
        raise ValueError(f"{argument_name} num_points must be positive. Got: {num_points}")

    if max_value < min_value:
        raise ValueError(
            f"{argument_name} max must be greater than or equal to min. "
            f"Got min={min_value}, max={max_value}"
        )

    return np.linspace(min_value, max_value, num_points).tolist()


def parse_weight_vectors_from_string(value):
    """
    this parses custom workload weight vectors from a command-line string.
    Format:
        "0.25,0.25,0.25,0.25;0.4,0.3,0.2,0.1"
    Each semicolon separates one full weight vector.
    """
    if value is None or str(value).strip() == "":
        return None

    vectors = []
    for vector_text in str(value).split(";"):
        vector_text = vector_text.strip()
        if not vector_text:
            continue

        vectors.append(parse_float_list(vector_text, "--weight-vectors"))

    if not vectors:
        raise ValueError("--weight-vectors was provided but no valid vectors were found.")

    return vectors


def load_weight_vectors_from_file(path):
    """
    This function loads custom workload weight vectors from a CSV or JSON file.
    For CSV:
        0.25,0.25,0.25,0.25
        0.4,0.3,0.2,0.1
    For JSON:
        [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]]
        {"weights": [[0.25, 0.25, 0.25, 0.25]]}
    """
    if path is None:
        return None

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file does not exist: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "weights" not in data:
                raise ValueError("JSON weight file must either be a list of vectors or contain a 'weights' key.")
            data = data["weights"]

        return [[float(x) for x in row] for row in data]

    vectors = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            try:
                vector = [float(x) for x in row if str(x).strip() != ""]
            except ValueError:
                # Allows a simple header row to be skipped.
                continue

            if vector:
                vectors.append(vector)

    if not vectors:
        raise ValueError(f"No valid weight vectors found in file: {path}")

    return vectors



PLAN_PROVENANCE_COLUMNS = [
    'Plan_Row_ID',
    'Base_Plan_Row_ID',
    'Design_Version',
    'Context_ID',
    'PR_ID',
    'PR_Stratum',
    'PR_Stratum_Detail',
    'Weight_ID',
    'Weight_Stratum',
    'Weight_Family',
    'Weight_Detail',
    'Dirichlet_Alpha',
    'Anchor_Type',
    'Data_Split',
    'R_Over_P',
    'Configured_R_Over_P_Max',
    'Configured_Weight_Lower',
    'Configured_Weight_Upper',
    'Simulation_Seed',
    'Is_Replicate',
    'Replicate_Of_Plan_Row_ID',
    'Plan_Source_File',
]


def parse_json_or_python_list(value, argument_name):
    """Parse a JSON/Python-style list used by plan CSV cells."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{argument_name} must contain a list. Got: {value}")
    return [float(x) for x in parsed]


def normalize_plan_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes', 'y', 't'}


def optional_plan_float(value):
    """Return None for blank/NaN plan cells, otherwise a finite float."""
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none'}:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        return None
    return parsed


def same_config_reference(plan_value, cli_value):
    """Compare path references by resolved path when possible, otherwise by filename stem."""
    if plan_value is None or str(plan_value).strip() == '':
        return True
    plan_path = Path(str(plan_value))
    cli_path = Path(str(cli_value))
    try:
        if plan_path.resolve() == cli_path.resolve():
            return True
    except Exception:
        pass
    return plan_path.stem == cli_path.stem


def plan_provenance_values(plan_row_metadata):
    row = plan_row_metadata or {}
    return [
        row.get('plan_row_id', ''),
        row.get('base_plan_row_id', ''),
        row.get('design_version', ''),
        row.get('context_id', ''),
        row.get('pr_id', ''),
        row.get('pr_stratum', ''),
        row.get('pr_stratum_detail', ''),
        row.get('weight_id', ''),
        row.get('weight_stratum', ''),
        row.get('weight_family', ''),
        row.get('weight_detail', ''),
        row.get('dirichlet_alpha', ''),
        row.get('anchor_type', ''),
        row.get('data_split', ''),
        row.get('R_over_P', row.get('r_over_p', '')),
        row.get('configured_r_over_p_max', ''),
        row.get('configured_weight_lower', ''),
        row.get('configured_weight_upper', ''),
        row.get('simulation_seed', ''),
        row.get('is_replicate', ''),
        row.get('replicate_of_plan_row_id', ''),
        row.get('plan_source_file', ''),
    ]


def load_sweep_plan_rows(path, args, job_config, experiment_config):
    """Load, validate, filter, and deterministically chunk an exact sweep plan.

    Plan mode executes one exact (utilization, Pbar, R, weight, seed) row at a time.
    It does not form a Cartesian product.
    """
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Sweep plan file does not exist: {plan_path}")

    if plan_path.suffix.lower() == '.json':
        data = json.loads(plan_path.read_text())
        if isinstance(data, dict):
            data = data.get('rows', data.get('plan', data))
        plan_df = pd.DataFrame(data)
    else:
        plan_df = pd.read_csv(plan_path)

    required = [
        'plan_row_id', 'utilization', 'Pbar_kw_per_server',
        'R_kw_per_server', 'weights', 'simulation_seed'
    ]
    missing = [c for c in required if c not in plan_df.columns]
    if missing:
        raise ValueError(f"Sweep plan is missing required columns: {missing}")
    if plan_df['plan_row_id'].astype(str).duplicated().any():
        dupes = plan_df.loc[plan_df['plan_row_id'].astype(str).duplicated(), 'plan_row_id'].head(10).tolist()
        raise ValueError(f"Sweep plan contains duplicate plan_row_id values, examples: {dupes}")

    # A plan chunk passed by the runner should contain one workload/config context.
    if 'workload_config' in plan_df.columns:
        bad = ~plan_df['workload_config'].map(lambda x: same_config_reference(x, args.job_config))
        if bad.any():
            examples = plan_df.loc[bad, 'workload_config'].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"Sweep plan rows do not match --job-config {args.job_config}. Examples: {examples}"
            )
    if 'experiment_config' in plan_df.columns:
        bad = ~plan_df['experiment_config'].map(lambda x: same_config_reference(x, args.experiment_config))
        if bad.any():
            examples = plan_df.loc[bad, 'experiment_config'].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"Sweep plan rows do not match --experiment-config {args.experiment_config}. Examples: {examples}"
            )
    if 'server_count' in plan_df.columns:
        bad = plan_df['server_count'].astype(int) != int(experiment_config.server_count)
        if bad.any():
            raise ValueError(
                f"Sweep plan server_count does not match experiment config server_count={experiment_config.server_count}."
            )
    if 'job_type_count' in plan_df.columns:
        bad = plan_df['job_type_count'].astype(int) != int(job_config.job_type_count)
        if bad.any():
            raise ValueError(
                f"Sweep plan job_type_count does not match workload J={job_config.job_type_count}."
            )

    bounds = calculate_workload_pr_bounds(
        job_config=job_config,
        pbar_lower_factor=args.pbar_lower_factor,
        pbar_upper_factor=args.pbar_upper_factor,
        pr_upper_factor=args.pr_upper_factor,
    )
    legacy_weight_bounds = calculate_weight_bounds(
        job_config.job_type_count,
        experiment_config.server_count,
    )

    validated = []
    for row_index, row in plan_df.reset_index(drop=True).iterrows():
        pbar = float(row['Pbar_kw_per_server'])
        reserve = float(row['R_kw_per_server'])
        utilization = float(row['utilization'])
        seed = int(row['simulation_seed'])
        weights = parse_json_or_python_list(row['weights'], f"plan row {row_index} weights")
        validate_weight_vectors(
            [weights],
            job_type_count=job_config.job_type_count,
            server_count=experiment_config.server_count,
            enforce_weight_bounds=True,
        )

        plan_r_over_p_max = optional_plan_float(row.get('configured_r_over_p_max'))
        cli_r_over_p_max = optional_plan_float(args.r_over_p_max)
        if (
            plan_r_over_p_max is not None
            and cli_r_over_p_max is not None
            and not np.isclose(plan_r_over_p_max, cli_r_over_p_max, atol=1e-9)
        ):
            raise ValueError(
                f"Plan row {row['plan_row_id']} configured_r_over_p_max="
                f"{plan_r_over_p_max} does not match CLI --r-over-p-max="
                f"{cli_r_over_p_max}."
            )
        effective_r_over_p_max = (
            cli_r_over_p_max if cli_r_over_p_max is not None else plan_r_over_p_max
        )

        plan_weight_lower = optional_plan_float(row.get('configured_weight_lower'))
        plan_weight_upper = optional_plan_float(row.get('configured_weight_upper'))
        cli_weight_lower = optional_plan_float(args.configured_weight_lower)
        cli_weight_upper = optional_plan_float(args.configured_weight_upper)
        if (
            plan_weight_lower is not None
            and cli_weight_lower is not None
            and not np.isclose(plan_weight_lower, cli_weight_lower, atol=1e-9)
        ):
            raise ValueError(
                f"Plan row {row['plan_row_id']} configured_weight_lower="
                f"{plan_weight_lower} does not match CLI --configured-weight-lower="
                f"{cli_weight_lower}."
            )
        if (
            plan_weight_upper is not None
            and cli_weight_upper is not None
            and not np.isclose(plan_weight_upper, cli_weight_upper, atol=1e-9)
        ):
            raise ValueError(
                f"Plan row {row['plan_row_id']} configured_weight_upper="
                f"{plan_weight_upper} does not match CLI --configured-weight-upper="
                f"{cli_weight_upper}."
            )
        configured_weight_lower = (
            cli_weight_lower if cli_weight_lower is not None else plan_weight_lower
        )
        configured_weight_upper = (
            cli_weight_upper if cli_weight_upper is not None else plan_weight_upper
        )
        effective_weight_lower = max(
            legacy_weight_bounds['Weight_Final_Lower_Bound'],
            configured_weight_lower
            if configured_weight_lower is not None
            else legacy_weight_bounds['Weight_Final_Lower_Bound'],
        )
        effective_weight_upper = min(
            legacy_weight_bounds['Weight_Final_Upper_Bound'],
            configured_weight_upper
            if configured_weight_upper is not None
            else legacy_weight_bounds['Weight_Final_Upper_Bound'],
        )
        if effective_weight_upper < effective_weight_lower - 1e-9:
            raise ValueError(
                f"Plan row {row['plan_row_id']} has infeasible effective weight bounds "
                f"[{effective_weight_lower}, {effective_weight_upper}]."
            )
        if min(weights) < effective_weight_lower - 1e-6:
            raise ValueError(
                f"Plan row {row['plan_row_id']} has weight below effective lower bound: "
                f"min={min(weights)}, lower={effective_weight_lower}."
            )
        if max(weights) > effective_weight_upper + 1e-6:
            raise ValueError(
                f"Plan row {row['plan_row_id']} has weight above effective upper bound: "
                f"max={max(weights)}, upper={effective_weight_upper}."
            )

        if pbar < bounds['Pbar_lower_bound_kw_per_server'] - PR_SWEEP_EPS:
            raise ValueError(f"Plan row {row['plan_row_id']} has Pbar below workload bound: {pbar}")
        if pbar > bounds['Pbar_upper_bound_kw_per_server'] + PR_SWEEP_EPS:
            raise ValueError(f"Plan row {row['plan_row_id']} has Pbar above workload bound: {pbar}")
        if reserve < args.r_lower_kw_per_server - PR_SWEEP_EPS:
            raise ValueError(f"Plan row {row['plan_row_id']} has R below lower bound: {reserve}")
        if reserve > pbar + PR_SWEEP_EPS:
            raise ValueError(f"Plan row {row['plan_row_id']} has R>Pbar: Pbar={pbar}, R={reserve}")
        if (
            effective_r_over_p_max is not None
            and reserve > effective_r_over_p_max * pbar + PR_SWEEP_EPS
        ):
            raise ValueError(
                f"Plan row {row['plan_row_id']} violates R/P cap: "
                f"R={reserve}, Pbar={pbar}, R/P={reserve / pbar}, "
                f"limit={effective_r_over_p_max}."
            )
        if pbar + reserve > bounds['PR_upper_bound_kw_per_server'] + PR_SWEEP_EPS:
            raise ValueError(
                f"Plan row {row['plan_row_id']} violates Pbar+R bound: "
                f"{pbar + reserve}>{bounds['PR_upper_bound_kw_per_server']}"
            )
        if utilization <= 0 or utilization > 1:
            raise ValueError(f"Plan row {row['plan_row_id']} has invalid utilization={utilization}")
        if seed < 0:
            raise ValueError(f"Plan row {row['plan_row_id']} has negative simulation_seed={seed}")

        metadata = row.to_dict()
        metadata['weights'] = weights
        metadata['Pbar_kw_per_server'] = pbar
        metadata['R_kw_per_server'] = reserve
        metadata['utilization'] = utilization
        metadata['simulation_seed'] = seed
        metadata['is_replicate'] = normalize_plan_bool(metadata.get('is_replicate', False))
        metadata['R_over_P'] = float(reserve / pbar)
        metadata['configured_r_over_p_max'] = (
            '' if effective_r_over_p_max is None else float(effective_r_over_p_max)
        )
        metadata['configured_weight_lower'] = (
            '' if configured_weight_lower is None else float(configured_weight_lower)
        )
        metadata['configured_weight_upper'] = (
            '' if configured_weight_upper is None else float(configured_weight_upper)
        )
        metadata['effective_weight_lower'] = float(effective_weight_lower)
        metadata['effective_weight_upper'] = float(effective_weight_upper)
        metadata['plan_source_file'] = str(plan_path)
        validated.append(metadata)

    if args.plan_num_chunks <= 0:
        raise ValueError(f"--plan-num-chunks must be positive, got {args.plan_num_chunks}")
    if args.plan_chunk_index < 0 or args.plan_chunk_index >= args.plan_num_chunks:
        raise ValueError(
            f"--plan-chunk-index must be in [0,{args.plan_num_chunks - 1}], got {args.plan_chunk_index}"
        )

    chunks = np.array_split(np.arange(len(validated)), args.plan_num_chunks)
    selected_indices = chunks[args.plan_chunk_index].tolist()
    selected = [validated[int(i)] for i in selected_indices]
    if not selected:
        raise ValueError(
            f"Plan chunk {args.plan_chunk_index + 1}/{args.plan_num_chunks} is empty. "
            f"Plan rows={len(validated)}"
        )

    metadata = {
        'PR_Sweep_Mode': 'exact_sweep_plan',
        **bounds,
        'R_lower_bound_kw_per_server': float(args.r_lower_kw_per_server),
        'Pbar_points': '',
        'R_points_per_Pbar': '',
        'PR_Total_Pairs_Before_Chunk': int(len(validated)),
        'PR_Chunk_Index': int(args.plan_chunk_index),
        'PR_Num_Chunks': int(args.plan_num_chunks),
        'PR_Chunk_Start_Index': int(selected_indices[0]),
        'PR_Chunk_End_Index_Exclusive': int(selected_indices[-1] + 1),
        'PR_Pairs_In_Chunk': int(len(selected)),
        'Plan_Source_File': str(plan_path),
    }
    return selected, metadata


def calculate_weight_bounds(job_type_count, server_count):
    """
    Calculate finalized workload-weight bounds from the current experiment server count
    and current workload job-type count.
    """
    job_type_count = int(job_type_count)
    server_count = int(server_count)

    if job_type_count <= 0:
        raise ValueError(f"job_type_count must be positive. Got: {job_type_count}")

    if server_count <= 0:
        raise ValueError(f"server_count must be positive. Got: {server_count}")

    if server_count < job_type_count:
        raise ValueError(
            f"Weight feasibility is impossible because server_count={server_count} "
            f"is smaller than job_type_count={job_type_count}."
        )

    equal_weight = 1.0 / job_type_count

    relative_lower = DEFAULT_WEIGHT_MIN_FRACTION_OF_EQUAL * equal_weight
    relative_upper = DEFAULT_WEIGHT_MAX_MULTIPLE_OF_EQUAL * equal_weight

    # New server-feasibility lower bound: S * w_i >= 1.
    server_lower = 1.0 / server_count
    final_lower = max(relative_lower, server_lower)

    # New upper bound implied by the chosen lower bound.
    # If J - 1 job types each get final_lower, the remaining job type can get at most this.
    upper_from_lower = 1.0 - (job_type_count - 1) * final_lower
    final_upper = min(relative_upper, upper_from_lower)

    if final_upper < final_lower:
        raise ValueError(
            f"Weight bounds are infeasible for job_type_count={job_type_count}, "
            f"server_count={server_count}: final_lower={final_lower}, final_upper={final_upper}"
        )

    return {
        "Weight_Equal_Value": float(equal_weight),
        "Weight_Relative_Lower_Bound": float(relative_lower),
        "Weight_Relative_Upper_Bound": float(relative_upper),
        "Weight_Server_Lower_Bound": float(server_lower),
        "Weight_Upper_From_Lower_Bound": float(upper_from_lower),
        "Weight_Final_Lower_Bound": float(final_lower),
        "Weight_Final_Upper_Bound": float(final_upper),
    }


def weight_vector_passes_constraints(weights, weight_bounds, tolerance=1e-6):
    min_weight = float(np.min(weights))
    max_weight = float(np.max(weights))

    if min_weight < weight_bounds["Weight_Final_Lower_Bound"] - tolerance:
        return False

    if max_weight > weight_bounds["Weight_Final_Upper_Bound"] + tolerance:
        return False

    return True


def validate_weight_vectors(weight_combinations, job_type_count, server_count=None, enforce_weight_bounds=False,
                            tolerance=1e-6):
    """
    Validate that every workload-weight vector has the correct length, sums to 1,
    and optionally satisfies the finalized weight bounds.
    """
    validated = []
    weight_bounds = None

    if enforce_weight_bounds:
        if server_count is None:
            raise ValueError("server_count is required when enforce_weight_bounds=True")
        weight_bounds = calculate_weight_bounds(job_type_count, server_count)

    for idx, weights in enumerate(weight_combinations):
        weights = [float(w) for w in weights]

        if len(weights) != job_type_count:
            raise ValueError(
                f"Weight vector {idx} has length {len(weights)}, but this workload has "
                f"{job_type_count} job types."
            )

        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > tolerance:
            raise ValueError(
                f"Weight vector {idx} sums to {weight_sum}, but it must sum to 1. "
                f"Vector: {weights}"
            )

        if any(w < 0 for w in weights):
            raise ValueError(f"Weight vector {idx} has a negative value. Vector: {weights}")

        if enforce_weight_bounds and not weight_vector_passes_constraints(weights, weight_bounds, tolerance=tolerance):
            raise ValueError(
                f"Weight vector {idx} violates finalized weight bounds. "
                f"min={min(weights)}, max={max(weights)}, "
                f"lower={weight_bounds['Weight_Final_Lower_Bound']}, "
                f"upper={weight_bounds['Weight_Final_Upper_Bound']}, "
                f"Vector: {weights}"
            )

        validated.append(weights)

    return validated

def to_float(value, argument_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{argument_name} must be a float. Got: {value}") from exc


def to_positive_int(value, argument_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{argument_name} must be an integer. Got: {value}") from exc

    if parsed <= 0:
        raise ValueError(f"{argument_name} must be positive. Got: {parsed}")

    return parsed


def to_nonnegative_int(value, argument_name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{argument_name} must be an integer. Got: {value}") from exc

    if parsed < 0:
        raise ValueError(f"{argument_name} must be non-negative. Got: {parsed}")

    return parsed


def round_or_blank(value, digits=6):
    if value is None or value == "":
        return ""
    return round(float(value), digits)


def calculate_workload_pr_bounds(job_config, pbar_lower_factor, pbar_upper_factor, pr_upper_factor):
    """
    Calculate workload-specific physical kW/server P/R bounds from the loaded workload config.

    Pmin and Pmax come from the current workload's per-job-type power fields.
    This means each workload config automatically gets its own Pbar range and R envelope.
    """
    min_job_powers = [float(x) for x in job_config.all_min_job_power.values()]
    max_job_powers = [float(x) for x in job_config.all_max_job_power.values()]

    if not min_job_powers:
        raise ValueError("job_config.all_min_job_power is empty; cannot derive Pmin.")

    if not max_job_powers:
        raise ValueError("job_config.all_max_job_power is empty; cannot derive Pmax.")

    pmin_kw_per_server = min(min_job_powers) / 1000.0
    pmax_kw_per_server = max(max_job_powers) / 1000.0

    pbar_lower_bound = pbar_lower_factor * pmin_kw_per_server
    pbar_upper_bound = pbar_upper_factor * pmax_kw_per_server
    pr_upper_bound = pr_upper_factor * pmax_kw_per_server

    if pbar_lower_bound <= 0:
        raise ValueError(f"Invalid Pbar lower bound: {pbar_lower_bound}")

    if pbar_upper_bound < pbar_lower_bound:
        raise ValueError(
            f"Invalid Pbar range: lower={pbar_lower_bound}, upper={pbar_upper_bound}"
        )

    if pr_upper_bound <= 0:
        raise ValueError(f"Invalid Pbar+R upper bound: {pr_upper_bound}")

    return {
        "Pmin_kw_per_server": float(pmin_kw_per_server),
        "Pmax_kw_per_server": float(pmax_kw_per_server),
        "Pbar_lower_bound_kw_per_server": float(pbar_lower_bound),
        "Pbar_upper_bound_kw_per_server": float(pbar_upper_bound),
        "PR_upper_bound_kw_per_server": float(pr_upper_bound),
        "Pbar_lower_factor": float(pbar_lower_factor),
        "Pbar_upper_factor": float(pbar_upper_factor),
        "PR_upper_factor": float(pr_upper_factor),
    }


def build_auto_workload_pr_pairs(job_config, pbar_points, r_points_per_pbar, r_lower_kw_per_server,
                                 pbar_lower_factor, pbar_upper_factor, pr_upper_factor):
    """
    Build targeted physical P/R pairs using the same logic as the spreadsheet helper script:

        Pbar values = linspace(0.9 * Pmin, 1.1 * Pmax, pbar_points)
        R max for each Pbar = min(Pbar, 1.3 * Pmax - Pbar)
        R values for each Pbar = linspace(R_lower, R max for this Pbar, r_points_per_pbar)

    The output is a flat list of valid (Pbar_kw_per_server, R_kw_per_server) pairs.
    """
    bounds = calculate_workload_pr_bounds(
        job_config=job_config,
        pbar_lower_factor=pbar_lower_factor,
        pbar_upper_factor=pbar_upper_factor,
        pr_upper_factor=pr_upper_factor,
    )

    pbar_values = np.linspace(
        bounds["Pbar_lower_bound_kw_per_server"],
        bounds["Pbar_upper_bound_kw_per_server"],
        pbar_points,
    )

    pr_pairs = []
    allowed_values = {}
    r_max_by_pbar = {}

    for pbar in pbar_values:
        pbar = float(pbar)
        max_allowed_r = min(
            pbar,
            bounds["PR_upper_bound_kw_per_server"] - pbar,
        )

        pbar_key = round(float(pbar), 4)
        r_max_by_pbar[pbar_key] = round(float(max_allowed_r), 6)

        if max_allowed_r < r_lower_kw_per_server - PR_SWEEP_EPS:
            allowed_values[pbar_key] = []
            continue

        r_values_for_pbar = np.linspace(
            r_lower_kw_per_server,
            max_allowed_r,
            r_points_per_pbar,
        )

        allowed_values[pbar_key] = [round(float(r), 4) for r in r_values_for_pbar]

        for r in r_values_for_pbar:
            r = float(r)

            # Defensive guard against numerical drift.
            if r > pbar + PR_SWEEP_EPS:
                raise ValueError(f"Generated invalid R>Pbar pair: Pbar={pbar}, R={r}")

            if pbar + r > bounds["PR_upper_bound_kw_per_server"] + PR_SWEEP_EPS:
                raise ValueError(
                    f"Generated invalid Pbar+R pair: Pbar={pbar}, R={r}, "
                    f"PR upper={bounds['PR_upper_bound_kw_per_server']}"
                )

            pr_pairs.append((pbar, r))

    metadata = {
        "PR_Sweep_Mode": "auto_workload_pr",
        **bounds,
        "R_lower_bound_kw_per_server": float(r_lower_kw_per_server),
        "Pbar_points": int(pbar_points),
        "R_points_per_Pbar": int(r_points_per_pbar),
        "Pbar_values": [float(x) for x in pbar_values],
        "Allowed_R_values_by_Pbar": allowed_values,
        "R_max_by_Pbar": r_max_by_pbar,
    }

    return pr_pairs, metadata


def build_manual_pr_pairs(pbar_kw_per_server_values, r_kw_per_server_values, job_config,
                          pbar_lower_factor, pbar_upper_factor, pr_upper_factor):
    """
    Preserve the old manual/global P x R behavior when auto-workload sweep is disabled.
    Manual mode still records workload-specific bounds in diagnostics for traceability.
    """
    bounds = calculate_workload_pr_bounds(
        job_config=job_config,
        pbar_lower_factor=pbar_lower_factor,
        pbar_upper_factor=pbar_upper_factor,
        pr_upper_factor=pr_upper_factor,
    )

    pr_pairs = []
    for pbar_kw_per_server in pbar_kw_per_server_values:
        for r_kw_per_server in r_kw_per_server_values:
            if pbar_kw_per_server <= r_kw_per_server:
                continue
            pr_pairs.append((float(pbar_kw_per_server), float(r_kw_per_server)))

    metadata = {
        "PR_Sweep_Mode": "manual_global_pr",
        **bounds,
        "R_lower_bound_kw_per_server": min(r_kw_per_server_values) if r_kw_per_server_values else "",
        "Pbar_points": len(pbar_kw_per_server_values),
        "R_points_per_Pbar": "manual_global_r_values",
        "Pbar_values": [float(x) for x in pbar_kw_per_server_values],
        "Allowed_R_values_by_Pbar": {
            round(float(p), 4): [round(float(r), 4) for r in r_kw_per_server_values if float(p) > float(r)]
            for p in pbar_kw_per_server_values
        },
        "R_max_by_Pbar": {},
    }

    return pr_pairs, metadata


def apply_pr_chunk(pr_pairs, pr_sweep_metadata, pr_chunk_index, pr_num_chunks):
    """
    Select a deterministic contiguous chunk of the already-built P/R pair list.
    This changes only which P/R pairs this worker runs; it does not change how P/R
    pairs are generated or how the simulator evaluates each selected pair.
    """
    pr_chunk_index = int(pr_chunk_index)
    pr_num_chunks = int(pr_num_chunks)

    if pr_num_chunks <= 0:
        raise ValueError(f"--pr-num-chunks must be positive. Got: {pr_num_chunks}")

    if pr_chunk_index < 0 or pr_chunk_index >= pr_num_chunks:
        raise ValueError(
            f"--pr-chunk-index must be in [0, {pr_num_chunks - 1}]. Got: {pr_chunk_index}"
        )

    total_pairs = len(pr_pairs)
    if total_pairs <= 0:
        raise ValueError("Cannot chunk an empty P/R pair list.")

    base_chunk_size = total_pairs // pr_num_chunks
    remainder = total_pairs % pr_num_chunks

    start_idx = pr_chunk_index * base_chunk_size + min(pr_chunk_index, remainder)
    chunk_size = base_chunk_size + (1 if pr_chunk_index < remainder else 0)
    end_idx = start_idx + chunk_size

    if chunk_size <= 0:
        raise ValueError(
            f"P/R chunk {pr_chunk_index} of {pr_num_chunks} is empty because there are only "
            f"{total_pairs} P/R pairs. Reduce --pr-num-chunks."
        )

    chunked_pairs = pr_pairs[start_idx:end_idx]
    metadata = dict(pr_sweep_metadata)
    metadata.update({
        "PR_Total_Pairs_Before_Chunk": int(total_pairs),
        "PR_Chunk_Index": int(pr_chunk_index),
        "PR_Num_Chunks": int(pr_num_chunks),
        "PR_Chunk_Start_Index": int(start_idx),
        "PR_Chunk_End_Index_Exclusive": int(end_idx),
        "PR_Pairs_In_Chunk": int(len(chunked_pairs)),
    })

    return chunked_pairs, metadata


def print_pr_sweep_summary(pr_sweep_metadata, utilization_values, weight_combinations):
    """
    Print a human-checkable P/R table before simulator runs start.
    This mirrors the standalone helper script used to fill the spreadsheet.
    """
    print("\nWorkload-specific P/R sweep summary:")
    print(f"  mode = {pr_sweep_metadata.get('PR_Sweep_Mode')}")
    print(f"  Pmin_kw_per_server = {pr_sweep_metadata.get('Pmin_kw_per_server'):.6f}")
    print(f"  Pmax_kw_per_server = {pr_sweep_metadata.get('Pmax_kw_per_server'):.6f}")
    print(
        f"  Pbar range = "
        f"[{pr_sweep_metadata.get('Pbar_lower_bound_kw_per_server'):.6f}, "
        f"{pr_sweep_metadata.get('Pbar_upper_bound_kw_per_server'):.6f}] kW/server"
    )
    print(f"  Pbar + R upper bound = {pr_sweep_metadata.get('PR_upper_bound_kw_per_server'):.6f} kW/server")
    print(f"  R lower bound = {pr_sweep_metadata.get('R_lower_bound_kw_per_server')}")
    print(f"  Pbar points = {pr_sweep_metadata.get('Pbar_points')}")
    print(f"  R points per Pbar = {pr_sweep_metadata.get('R_points_per_Pbar')}")
    print(
        f"  P/R chunk = {pr_sweep_metadata.get('PR_Chunk_Index', 0)} "
        f"of {pr_sweep_metadata.get('PR_Num_Chunks', 1)} "
        f"({pr_sweep_metadata.get('PR_Pairs_In_Chunk', 'all')} selected from "
        f"{pr_sweep_metadata.get('PR_Total_Pairs_Before_Chunk', pr_sweep_metadata.get('PR_Pairs_In_Chunk', 'all'))})"
    )

    pbar_values = pr_sweep_metadata.get("Pbar_values", [])
    print(f"\nPbar values generated from workload bounds:\n{np.array(pbar_values)}\n")

    print("Allowed values of Pbar and R:")
    allowed_values = pr_sweep_metadata.get("Allowed_R_values_by_Pbar", {})
    for pbar, r_list in allowed_values.items():
        print(f"Pbar = {pbar}: R = {r_list}")

    print("\nAllowed values with Pbar+R and Pbar-R:")
    for pbar, r_list in allowed_values.items():
        p_plus_r_list = [round(float(pbar + r), 4) for r in r_list]
        p_minus_r_list = [round(float(pbar - r), 4) for r in r_list]
        print(
            f"Pbar = {pbar}: "
            f"R = {r_list}, "
            f"Pbar+R = {p_plus_r_list}, "
            f"Pbar-R = {p_minus_r_list}"
        )

    valid_pair_count = pr_sweep_metadata.get("PR_Pairs_In_Chunk", sum(len(r_list) for r_list in allowed_values.values()))
    data_rows_per_workload = int(valid_pair_count) * len(utilization_values) * len(weight_combinations)
    print(
        f"\nData rows for this workload = {valid_pair_count} valid P/R pairs "
        f"* {len(utilization_values)} utilization values "
        f"* {len(weight_combinations)} weight combinations = {data_rows_per_workload} rows"
    )

def build_sweep_values(args, job_config, experiment_config):
    utilization_values = parse_float_list(args.utilization_values, "--utilization-values")
    if utilization_values is None:
        utilization_values = DEFAULT_UTILIZATION_VALUES

    auto_workload_pr_sweep = parse_bool_or_none(
        args.auto_workload_pr_sweep,
        "--auto-workload-pr-sweep"
    )
    if auto_workload_pr_sweep is None:
        auto_workload_pr_sweep = DEFAULT_AUTO_WORKLOAD_PR_SWEEP

    pbar_lower_factor = to_float(args.pbar_lower_factor, "--pbar-lower-factor")
    pbar_upper_factor = to_float(args.pbar_upper_factor, "--pbar-upper-factor")
    pr_upper_factor = to_float(args.pr_upper_factor, "--pr-upper-factor")
    pbar_points = to_positive_int(args.pbar_points, "--pbar-points")
    r_points_per_pbar = to_positive_int(args.r_points_per_pbar, "--r-points-per-pbar")
    r_lower_kw_per_server = to_float(args.r_lower_kw_per_server, "--r-lower-kw-per-server")
    pr_chunk_index = to_nonnegative_int(args.pr_chunk_index, "--pr-chunk-index")
    pr_num_chunks = to_positive_int(args.pr_num_chunks, "--pr-num-chunks")

    if r_lower_kw_per_server < 0:
        raise ValueError(f"--r-lower-kw-per-server must be non-negative. Got: {r_lower_kw_per_server}")

    if args.pbar_kw_per_server_values is not None and args.pbar_kw_per_server_range is not None:
        raise ValueError("Use either --pbar-kw-per-server-values or --pbar-kw-per-server-range, not both.")

    if args.r_kw_per_server_values is not None and args.r_kw_per_server_range is not None:
        raise ValueError("Use either --r-kw-per-server-values or --r-kw-per-server-range, not both.")

    inline_weight_vectors = parse_weight_vectors_from_string(args.weight_vectors)
    file_weight_vectors = load_weight_vectors_from_file(args.weight_file)

    if inline_weight_vectors is not None and file_weight_vectors is not None:
        raise ValueError("Use either --weight-vectors or --weight-file, not both.")

    if inline_weight_vectors is not None:
        weight_combinations = inline_weight_vectors

    elif file_weight_vectors is not None:
        weight_combinations = file_weight_vectors

    else:
        weight_combinations = []

        if DEFAULT_INCLUDE_EQUAL_WEIGHTS:
            equal_weights = [1 / job_config.job_type_count] * job_config.job_type_count
            weight_combinations.append(equal_weights)

        weight_combinations.extend(
            generate_random_softmax_weights(
                job_type_count=job_config.job_type_count,
                server_count=experiment_config.server_count,
                num_weight_samples=DEFAULT_NUM_WEIGHT_SAMPLES,
                seed=DEFAULT_WEIGHT_SEED,
                temperature=DEFAULT_WEIGHT_TEMPERATURE,
            )
        )

    weight_combinations = validate_weight_vectors(
        weight_combinations,
        job_config.job_type_count,
        server_count=experiment_config.server_count,
        enforce_weight_bounds=True,
    )

    if auto_workload_pr_sweep:
        manual_pr_args = [
            args.pbar_kw_per_server_values,
            args.pbar_kw_per_server_range,
            args.r_kw_per_server_values,
            args.r_kw_per_server_range,
        ]
        if any(value is not None for value in manual_pr_args):
            raise ValueError(
                "Auto workload P/R sweep is enabled, so do not pass manual --pbar-* or --r-* sweep flags. "
                "Use --auto-workload-pr-sweep false if you intentionally want the old manual/global P x R behavior."
            )

        pr_pairs, pr_sweep_metadata = build_auto_workload_pr_pairs(
            job_config=job_config,
            pbar_points=pbar_points,
            r_points_per_pbar=r_points_per_pbar,
            r_lower_kw_per_server=r_lower_kw_per_server,
            pbar_lower_factor=pbar_lower_factor,
            pbar_upper_factor=pbar_upper_factor,
            pr_upper_factor=pr_upper_factor,
        )

    else:
        pbar_kw_per_server_values = parse_float_list(
            args.pbar_kw_per_server_values,
            "--pbar-kw-per-server-values"
        )
        if pbar_kw_per_server_values is None:
            pbar_kw_per_server_values = parse_range_spec(
                args.pbar_kw_per_server_range,
                DEFAULT_PBAR_KW_PER_SERVER_RANGE,
                "--pbar-kw-per-server-range"
            )

        r_kw_per_server_values = parse_float_list(
            args.r_kw_per_server_values,
            "--r-kw-per-server-values"
        )
        if r_kw_per_server_values is None:
            r_kw_per_server_values = parse_range_spec(
                args.r_kw_per_server_range,
                DEFAULT_R_KW_PER_SERVER_RANGE,
                "--r-kw-per-server-range"
            )

        pbar_kw_per_server_values = [float(x) for x in pbar_kw_per_server_values]
        r_kw_per_server_values = [float(x) for x in r_kw_per_server_values]

        pr_pairs, pr_sweep_metadata = build_manual_pr_pairs(
            pbar_kw_per_server_values=pbar_kw_per_server_values,
            r_kw_per_server_values=r_kw_per_server_values,
            job_config=job_config,
            pbar_lower_factor=pbar_lower_factor,
            pbar_upper_factor=pbar_upper_factor,
            pr_upper_factor=pr_upper_factor,
        )

    utilization_values = [float(x) for x in utilization_values]

    if not pr_pairs:
        raise ValueError("No valid P/R pairs were generated. Check workload Pmin/Pmax and R lower bound.")

    pr_pairs, pr_sweep_metadata = apply_pr_chunk(
        pr_pairs=pr_pairs,
        pr_sweep_metadata=pr_sweep_metadata,
        pr_chunk_index=pr_chunk_index,
        pr_num_chunks=pr_num_chunks,
    )

    return utilization_values, pr_pairs, weight_combinations, pr_sweep_metadata


def build_policy_parameters(args, job_type_count):
    """
    From the FlexDC paper here are the defaults:
        AQA: node_count_control=True
        flex-resource: node_count_control=True, probabilities required/defaulted
        priority-cap: node_count_control=False, power_cap_percentage required/defaulted
        proportional-cap: node_count_control=False

    I have also allowed the user to override the functionality as seen below:
        --node-count-control true
        --node-count-control false
    """
    if args.policy_name not in DEFAULT_NODE_COUNT_CONTROL_BY_POLICY:
        raise ValueError(f"Unsupported policy: {args.policy_name}")

    node_count_control_override = parse_bool_or_none(
        args.node_count_control,
        "--node-count-control"
    )

    if node_count_control_override is None:
        node_count_control = DEFAULT_NODE_COUNT_CONTROL_BY_POLICY[args.policy_name]
    else:
        node_count_control = node_count_control_override

    policy_parameters = {
        "node_count_control": node_count_control
    }

    if args.policy_name == "priority-cap":
        if args.power_cap_percentage is None:
            power_cap_percentage = DEFAULT_POWER_CAP_PERCENTAGE
        else:
            power_cap_percentage = float(args.power_cap_percentage)

        if power_cap_percentage < 0 or power_cap_percentage > 1:
            raise ValueError(
                f"--power-cap-percentage should be in [0, 1]. Got: {power_cap_percentage}"
            )

        policy_parameters["power_cap_percentage"] = power_cap_percentage

    elif args.power_cap_percentage is not None:
        print(
            f"Warning: --power-cap-percentage was provided but policy {args.policy_name} "
            f"does not use it. It will be ignored."
        )

    if args.policy_name == "flex-resource":
        policy_parameters["probabilities"] = parse_probability_values(
            args.probabilities,
            job_type_count
        )

    elif args.probabilities is not None:
        print(
            f"Warning: --probabilities was provided but policy {args.policy_name} "
            f"does not use FlexResource probabilities. It will be ignored."
        )

    return policy_parameters


def count_valid_parameter_combinations(utilization_values, pr_pairs, weight_combinations):
    return len(utilization_values) * len(pr_pairs) * len(weight_combinations)


def set_experiment_utilization(experiment_config, utilization):
    if hasattr(experiment_config, "_utilization"):
        experiment_config._utilization = utilization
    else:
        experiment_config.utilization = utilization

def calculate_flexdc_pr_denominators(experiment_config, job_config):
    """
    FlexDC's internal P_ratio and R_ratio denominators.
    These are kept only so we can log the equivalent internal ratios.
    The sweep itself should use Pbar_kw_per_server and R_kw_per_server.
    """
    avg_max_job_power = sum(job_config.all_max_job_power.values()) / len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers

    pbar_denominator_watts = (
        avg_max_job_power * avg_num_servers
        + experiment_config.idle_power * avg_idle_servers
    )

    r_denominator_watts = (
        avg_max_job_power * experiment_config.server_count
        - experiment_config.idle_power * experiment_config.server_count
    ) / 2

    if pbar_denominator_watts <= 0:
        raise ValueError(f"Invalid Pbar denominator: {pbar_denominator_watts}")

    if r_denominator_watts <= 0:
        raise ValueError(f"Invalid R denominator: {r_denominator_watts}")

    return pbar_denominator_watts, r_denominator_watts

def convert_kw_per_server_to_flexdc_pr(pbar_kw_per_server, r_kw_per_server, experiment_config, job_config):
    """
    Convert physical per-server bids into the actual watt bids used by the simulator,
    plus the equivalent FlexDC internal ratios for logging/debugging.
    """
    p_actual_watts = int(round(float(pbar_kw_per_server) * 1000 * experiment_config.server_count))
    r_actual_watts = int(round(float(r_kw_per_server) * 1000 * experiment_config.server_count))

    pbar_denominator_watts, r_denominator_watts = calculate_flexdc_pr_denominators(
        experiment_config,
        job_config
    )

    pbar_ratio = p_actual_watts / pbar_denominator_watts
    r_ratio = r_actual_watts / r_denominator_watts

    return {
        "Pbar_kw_per_server": float(pbar_kw_per_server),
        "R_kw_per_server": float(r_kw_per_server),
        "P_actual_watts": p_actual_watts,
        "R_actual_watts": r_actual_watts,
        "Pbar_ratio": round(float(pbar_ratio), 6),
        "R_ratio": round(float(r_ratio), 6),
        "Pbar_denominator_watts": float(pbar_denominator_watts),
        "R_denominator_watts": float(r_denominator_watts),
    }

def create_simulator_object(pbar_kw_per_server, r_kw_per_server, experiment_config, job_config, cluster_config, policy_config, output_dir):
    # Re-seed BEFORE job-table construction. Job arrivals/prefill are generated in
    # init_job_table() using Python's random module. Simulator.run() also seeds,
    # but that happens after the job table already exists. Without this reset, a
    # row's workload trace depends on process startup and on every row simulated
    # before it, so the same P/R/weights/config cannot be reproduced standalone.
    effective_seed = int(experiment_config.random_seed)
    np.random.seed(effective_seed)
    random.seed(effective_seed)

    job_table = init_job_table(experiment_config, job_config)
    node_table = init_node_table(experiment_config)

    pr_values = convert_kw_per_server_to_flexdc_pr(
        pbar_kw_per_server=pbar_kw_per_server,
        r_kw_per_server=r_kw_per_server,
        experiment_config=experiment_config,
        job_config=job_config
    )

    # Set the simulator's actual bids in watts. Ratios are only logged for traceability.
    policy_config._P = pr_values["P_actual_watts"]
    policy_config._R = pr_values["R_actual_watts"]
    policy_config._P_ratio = pr_values["Pbar_ratio"]
    policy_config._R_ratio = pr_values["R_ratio"]

    policy_mapping = {
        "AQA": (
            AQARuntimePolicy,
            (policy_config, experiment_config, job_config, node_table, job_table)
        ),
        "flex-resource": (
            FlexResourcePolicy,
            (policy_config, experiment_config, job_config, node_table, job_table)
        ),
        "priority-cap": (
            PriorityCapPolicy,
            (policy_config, experiment_config, job_config, node_table, job_table)
        ),
        "proportional-cap": (
            ProportionalCapPolicy,
            (policy_config, experiment_config, job_config, node_table, job_table)
        ),
    }

    policy_class, constructor_args = policy_mapping[policy_config.runtime_policy_name]
    runtime_policy = policy_class(*constructor_args)

    simulator = Simulator(job_table, node_table, runtime_policy, experiment_config,
                          job_config, cluster_config, output_dir=output_dir)
    return simulator, pr_values

def build_workload_mix_json(job_config, weights=None):
    """
    Important:
    job_config.all_jobs maps numeric job_type_id -> job name.
    The feature dictionaries are keyed by numeric job_type_id.
    This means that we should iterate over the numeric keys, not all_jobs.values().

    The workload_mix column intentionally stores only workload/job features.
    Weights are already saved in base_weights.csv and the individual Weight_i columns.
    """
    workload_mix = []

    job_type_ids = sorted(job_config.all_min_job_power.keys())

    for job_type_id in job_type_ids:
        row = [
            float(job_config.all_min_job_power[job_type_id]),
            float(job_config.all_max_job_power[job_type_id]),
            float(job_config.all_min_execution_time[job_type_id]),
            float(job_config.all_max_execution_time[job_type_id]),
            float(job_config.all_job_qos_constraints[job_type_id]),
            float(job_config.all_job_size[job_type_id]),
        ]
        workload_mix.append(row)

    return json.dumps(workload_mix)

def calculate_qos_raw_metrics(sim_hour, J, job_table, qos_constraints, min_execution_times, cost_function_config):
    from calculate_qos_cost import calculate_delay_prob

    delay_prob = calculate_delay_prob(
        sim_hour,
        J,
        job_table,
        qos_constraints,
        min_execution_times
    )

    delta = cost_function_config.calculate_gradient_variables['qos_threshold']

    qos_delay_probabilities = [float(delay_prob[itype]) for itype in range(J)]
    qos_delay_probability_sum = sum(qos_delay_probabilities)

    # Diagnostic only: this subtracts the QoS threshold, so it is a residual, not a raw cost.
    qos_delay_probability_residuals = [delay_prob[itype] - delta for itype in range(J)]
    qos_delay_probability_residual_sum = sum(qos_delay_probability_residuals)

    return (
        qos_delay_probability_sum,
        qos_delay_probabilities,
        qos_delay_probability_residual_sum,
        qos_delay_probability_residuals,
    )



def calculate_mtrack_raw_metrics(output_dir, r_actual_watts, simulation_duration_seconds):
    """
    Extract raw simulator monetary tracking-cost lineage from power_trace.csv.

    This deliberately mirrors simulator.py::final_output() for the one-hour experiments
    used in the dataset workflow:

        tracking_cost += piE * R * mean(abs(trackingError_phase)) * hour

    The simulator itself still returns Mtrack_Cost through simulator.run(); this helper
    only logs the underlying components with clear names.
    """
    if int(simulation_duration_seconds) != MTRACK_HOUR_SECONDS:
        raise ValueError(
            "Mtrack component labels currently assume a one-hour simulation_duration=3600. "
            f"Got simulation_duration={simulation_duration_seconds}. For multi-hour runs, add explicit "
            "per-phase or phase-sum component columns before collecting data."
        )

    power_trace_path = os.path.join(output_dir, 'power_trace.csv')
    if not os.path.exists(power_trace_path):
        raise FileNotFoundError(f"power_trace.csv not found for Mtrack extraction: {power_trace_path}")

    trace_df = pd.read_csv(power_trace_path, usecols=['time', 'trackingError'])
    if trace_df.empty:
        raise ValueError(f"power_trace.csv has no rows: {power_trace_path}")

    trace_df['time'] = pd.to_numeric(trace_df['time'], errors='coerce')
    trace_df['trackingError'] = pd.to_numeric(trace_df['trackingError'], errors='coerce')
    phase_error = trace_df[
        (trace_df['time'] >= 0) &
        (trace_df['time'] <= MTRACK_HOUR_SECONDS)
    ]['trackingError'].dropna()

    # Match simulator.py::final_output(): ['trackingError'][1:] drops the first sample.
    phase_error = phase_error.iloc[1:]
    if phase_error.empty:
        raise ValueError(f"No usable trackingError samples for Mtrack extraction: {power_trace_path}")

    mean_abs_normalized = float(np.mean(np.abs(phase_error.to_numpy(dtype=float))))
    mean_abs_watts = float(r_actual_watts) * mean_abs_normalized

    return {
        'Mtrack_Error_MeanAbs_Normalized': mean_abs_normalized,
        'Mtrack_Error_MeanAbs_Watts': mean_abs_watts,
        'Mtrack_Price_Coefficient_piE': float(MTRACK_PRICE_COEFFICIENT_PIE),
        'Mtrack_Hour_Seconds': float(MTRACK_HOUR_SECONDS),
    }


def objective_function(params, cost_function_config, policy_config, experiment_config, cluster_config, job_config,
                       output_dir, iteration_idx=None, total_iterations=None, weight_sample_id=None,
                       pr_sweep_metadata=None, plan_row_metadata=None):
    pbar_kw_per_server, r_kw_per_server, *weights = params
    if iteration_idx is not None and total_iterations is not None:
        print(f"\n========== Iteration {iteration_idx}/{total_iterations} ==========")
        print(
            f"utilization={experiment_config.utilization}, "
            f"Pbar_kw_per_server={pbar_kw_per_server:.6f}, "
            f"R_kw_per_server={r_kw_per_server:.6f}, weights={weights}"
        )

    base_weights_path = os.path.join(output_dir, 'base_weights.csv')
    pd.DataFrame({
        'job_type_id': np.array(list(job_config.all_jobs.values())),
        'weights': weights
    }).to_csv(base_weights_path, index=False)

    policy_config.policy_parameters['weights_path'] = base_weights_path
    simulator, pr_values = create_simulator_object(
        pbar_kw_per_server,
        r_kw_per_server,
        experiment_config,
        job_config,
        cluster_config,
        policy_config,
        output_dir
    )

    init_output(output_dir, experiment_config, policy_config)

    # simulator.run() returns the simulator's RSR monetary components.
    # In simulator.final_output():
    #   power cost = (piP * P - piR * R) * hour
    #   Mtrack    = piE * R * mean(abs(normalized_tracking_error)) * hour
    # We no longer write a column named Simulator_Tracking_Cost because that name is ambiguous.
    simulator_power_cost, simulator_mtrack_cost = simulator.run()

    mtrack_metrics = calculate_mtrack_raw_metrics(
        output_dir=output_dir,
        r_actual_watts=pr_values["R_actual_watts"],
        simulation_duration_seconds=experiment_config.simulation_duration,
    )
    mtrack_cost = float(simulator_mtrack_cost)
    reconstructed_mtrack_cost = (
        mtrack_metrics['Mtrack_Price_Coefficient_piE']
        * mtrack_metrics['Mtrack_Error_MeanAbs_Watts']
        * mtrack_metrics['Mtrack_Hour_Seconds']
    )
    if not np.isclose(reconstructed_mtrack_cost, mtrack_cost, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "Mtrack component reconstruction mismatch: "
            f"simulator_mtrack_cost={mtrack_cost}, "
            f"reconstructed_mtrack_cost={reconstructed_mtrack_cost}"
        )

    simulator_rsr_total_cost = simulator_power_cost + mtrack_cost

    ctrack_epsilon_90th = float(simulator.tracking_error_at_90)
    ctrack_gamma = float(TRACKING_ERROR_CONSTRAINT)
    ctrack_residual = ctrack_epsilon_90th - ctrack_gamma

    qos_constraints = list(simulator._job_config.all_job_qos_constraints.values())
    min_execution_times = list(simulator._job_config.all_min_execution_time.values())

    (
        qos_delay_probability_sum,
        qos_delay_probabilities,
        qos_delay_probability_residual_sum,
        qos_delay_probability_residuals,
    ) = calculate_qos_raw_metrics(
        sim_hour=1,
        J=simulator._job_type_count,
        job_table=simulator._job_table,
        qos_constraints=qos_constraints,
        min_execution_times=min_execution_times,
        cost_function_config=cost_function_config
    )

    # Ctrack diagnostics only. These are intentionally not written as raw labels.
    ctrack_psi = float(cost_function_config.calculate_gradient_variables['psi1'])
    ctrack_mu = float(cost_function_config.calculate_gradient_variables['psi2'])
    ctrack_mu_scaled_residual = ctrack_mu * ctrack_residual
    ctrack_softplus_value = float(np.logaddexp(0, ctrack_mu_scaled_residual))
    ctrack_weighted_cost = ctrack_psi * ctrack_softplus_value

    diagnostic_flexdc_softplus_qos_cost = calculate_qos_cost(
        experiment_config=experiment_config,
        job_config=job_config,
        policy_config=policy_config,
        simulator=simulator,
        fit=True,
        fitdown=0.85,
        fitup=0.95,
        sim_hour=1,
        J=simulator._job_type_count,
        job_table=simulator._job_table,
        QoS_constraint=qos_constraints,
        min_execution_time=min_execution_times,
        cost_function_config=cost_function_config
    )

    diagnostic_full_paper_objective_cost = (
        simulator_rsr_total_cost
        + ctrack_weighted_cost
        + diagnostic_flexdc_softplus_qos_cost
    )

    print(
        f"Iteration Variables: Pbar_kw_per_server={pbar_kw_per_server}, "
        f"R_kw_per_server={r_kw_per_server}, "
        f"Pbar_ratio={pr_values['Pbar_ratio']}, R_ratio={pr_values['R_ratio']}, "
        f"Weights={weights}, "
        f"Simulator RSR Total Cost={simulator_rsr_total_cost}, "
        f"Simulator Power Cost={simulator_power_cost}, "
        f"Mtrack Cost={mtrack_cost}, "
        f"Mtrack MeanAbs Normalized={mtrack_metrics['Mtrack_Error_MeanAbs_Normalized']}, "
        f"Ctrack Epsilon 90th={ctrack_epsilon_90th}, "
        f"Ctrack Weighted Cost={ctrack_weighted_cost}, "
        f"QoS Delay Probability Sum={qos_delay_probability_sum}, "
        f"Diagnostic Full Paper Objective Cost={diagnostic_full_paper_objective_cost}"
    )

    workload_mix_json = build_workload_mix_json(job_config, weights)

    pr_sweep_metadata = pr_sweep_metadata or {}
    workload_name = pr_sweep_metadata.get("Workload_Name", "")
    workload_config = pr_sweep_metadata.get("Workload_Config", "")

    raw_results_file = os.path.join(output_dir, 'grid_search_results.csv')
    raw_results = [
        iteration_idx,
        weight_sample_id,
        policy_config.runtime_policy_name,
        workload_name,
        workload_config,
    ] + plan_provenance_values(plan_row_metadata) + [
        policy_config.policy_parameters.get("node_count_control"),
        policy_config.policy_parameters.get("power_cap_percentage", ""),
        json.dumps(policy_config.policy_parameters.get("probabilities", "")),
        pr_values["Pbar_kw_per_server"],
        pr_values["R_kw_per_server"],
        pr_values["P_actual_watts"],
        pr_values["R_actual_watts"],
        pr_values["Pbar_ratio"],
        pr_values["R_ratio"],
        pr_values["Pbar_denominator_watts"],
        pr_values["R_denominator_watts"],
        experiment_config.server_count,
        experiment_config.utilization,
        job_config.job_type_count,
        workload_mix_json,
    ] + list(weights) + [
        simulator_rsr_total_cost,
        simulator_power_cost,
        mtrack_metrics['Mtrack_Error_MeanAbs_Normalized'],
        mtrack_metrics['Mtrack_Error_MeanAbs_Watts'],
        mtrack_metrics['Mtrack_Price_Coefficient_piE'],
        mtrack_metrics['Mtrack_Hour_Seconds'],
        mtrack_cost,
        qos_delay_probability_sum,
        simulator.ratio_of_job_type_qos_violation,
        json.dumps(qos_delay_probabilities),
    ]

    with open(raw_results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(raw_results)

    diagnostics_file = os.path.join(output_dir, 'grid_search_diagnostics.csv')

    pr_upper_bound_kw = pr_sweep_metadata.get("PR_upper_bound_kw_per_server")
    if pr_upper_bound_kw in (None, ""):
        r_max_for_this_pbar_kw = ""
    else:
        r_max_candidates = [
            float(pbar_kw_per_server),
            float(pr_upper_bound_kw) - float(pbar_kw_per_server),
        ]
        plan_r_over_p_max = optional_plan_float(
            (plan_row_metadata or {}).get('configured_r_over_p_max')
        )
        if plan_r_over_p_max is not None:
            r_max_candidates.append(plan_r_over_p_max * float(pbar_kw_per_server))
        r_max_for_this_pbar_kw = min(r_max_candidates)

    p_plus_r_kw = float(pbar_kw_per_server) + float(r_kw_per_server)
    p_minus_r_kw = float(pbar_kw_per_server) - float(r_kw_per_server)

    weight_bounds = calculate_weight_bounds(job_config.job_type_count, experiment_config.server_count)
    configured_weight_lower = optional_plan_float(
        (plan_row_metadata or {}).get('configured_weight_lower')
    )
    configured_weight_upper = optional_plan_float(
        (plan_row_metadata or {}).get('configured_weight_upper')
    )
    if configured_weight_lower is not None:
        weight_bounds['Weight_Final_Lower_Bound'] = max(
            weight_bounds['Weight_Final_Lower_Bound'],
            configured_weight_lower,
        )
    if configured_weight_upper is not None:
        weight_bounds['Weight_Final_Upper_Bound'] = min(
            weight_bounds['Weight_Final_Upper_Bound'],
            configured_weight_upper,
        )
    weight_min = float(np.min(weights))
    weight_max = float(np.max(weights))

    diagnostic_results = [
        iteration_idx,
        weight_sample_id,
        workload_name,
        workload_config,
    ] + plan_provenance_values(plan_row_metadata) + [
        pr_values["Pbar_kw_per_server"],
        pr_values["R_kw_per_server"],
        pr_values["P_actual_watts"],
        pr_values["R_actual_watts"],
        pr_values["Pbar_ratio"],
        pr_values["R_ratio"],
        experiment_config.server_count,
        experiment_config.utilization,
        pr_sweep_metadata.get("PR_Sweep_Mode", ""),
        round_or_blank(pr_sweep_metadata.get("Pmin_kw_per_server")),
        round_or_blank(pr_sweep_metadata.get("Pmax_kw_per_server")),
        round_or_blank(pr_sweep_metadata.get("Pbar_lower_bound_kw_per_server")),
        round_or_blank(pr_sweep_metadata.get("Pbar_upper_bound_kw_per_server")),
        round_or_blank(pr_sweep_metadata.get("PR_upper_bound_kw_per_server")),
        round_or_blank(pr_sweep_metadata.get("R_lower_bound_kw_per_server")),
        round_or_blank(r_max_for_this_pbar_kw),
        round_or_blank(p_plus_r_kw),
        round_or_blank(p_minus_r_kw),
        pr_sweep_metadata.get("Pbar_points", ""),
        pr_sweep_metadata.get("R_points_per_Pbar", ""),
        pr_sweep_metadata.get("PR_Total_Pairs_Before_Chunk", ""),
        pr_sweep_metadata.get("PR_Chunk_Index", ""),
        pr_sweep_metadata.get("PR_Num_Chunks", ""),
        pr_sweep_metadata.get("PR_Chunk_Start_Index", ""),
        pr_sweep_metadata.get("PR_Chunk_End_Index_Exclusive", ""),
        pr_sweep_metadata.get("PR_Pairs_In_Chunk", ""),
        weight_bounds["Weight_Equal_Value"],
        weight_bounds["Weight_Relative_Lower_Bound"],
        weight_bounds["Weight_Relative_Upper_Bound"],
        weight_bounds["Weight_Server_Lower_Bound"],
        weight_bounds["Weight_Upper_From_Lower_Bound"],
        weight_bounds["Weight_Final_Lower_Bound"],
        weight_bounds["Weight_Final_Upper_Bound"],
        weight_min,
        weight_max,
        ctrack_epsilon_90th,
        ctrack_gamma,
        ctrack_residual,
        ctrack_mu,
        ctrack_mu_scaled_residual,
        ctrack_softplus_value,
        ctrack_psi,
        ctrack_weighted_cost,
        qos_delay_probability_residual_sum,
        json.dumps(qos_delay_probability_residuals),
        diagnostic_flexdc_softplus_qos_cost,
        diagnostic_full_paper_objective_cost,
    ]

    with open(diagnostics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(diagnostic_results)

    return simulator_rsr_total_cost


def generate_weight_grid(lower, upper, step, count):
    values = np.arange(lower, upper + step / 2, step)
    weight_combinations = []
    for combo in itertools.product(values, repeat=count):
        if abs(sum(combo) - 1.0) < 1e-6:
            weight_combinations.append(combo)
    return weight_combinations


def generate_random_softmax_weights(job_type_count, server_count, num_weight_samples, seed=42, temperature=2.0):
    #Temp is how spread out they are btw
    rng = np.random.default_rng(seed)
    weight_combinations = []
    weight_bounds = calculate_weight_bounds(job_type_count, server_count)

    attempts = 0
    max_attempts = num_weight_samples * 1000

    while len(weight_combinations) < num_weight_samples and attempts < max_attempts:
        attempts += 1

        raw_scores = rng.normal(loc=0.0, scale=1.0, size=job_type_count)
        weights = softmax(raw_scores / temperature)

        if not weight_vector_passes_constraints(weights, weight_bounds):
            continue

        weight_combinations.append(weights.tolist())

    if len(weight_combinations) < num_weight_samples:
        raise ValueError(
            f"Only generated {len(weight_combinations)}/{num_weight_samples} valid weight vectors "
            f"for job_type_count={job_type_count}, server_count={server_count}. "
            f"Check the finalized weight bounds: {weight_bounds}"
        )

    return weight_combinations


def grid_search_plan_rows(cost_function_config, policy_config, experiment_config, cluster_config, job_config,
                          output_dir, plan_rows, pr_sweep_metadata):
    """Execute exact plan rows without a P/R x weight Cartesian product."""
    best_cost = float('inf')
    best_params = None
    initialize_grid_search_output_files(output_dir, job_config.job_type_count)
    total_iterations = len(plan_rows)
    print("\nExact sweep-plan configuration:")
    print(f"  plan rows in this worker = {total_iterations}")
    print(f"  job_type_count = {job_config.job_type_count}")
    print(f"  policy_name = {policy_config.runtime_policy_name}")
    print(f"  plan source = {pr_sweep_metadata.get('Plan_Source_File', '')}")

    for iteration_idx, plan_row in enumerate(plan_rows, start=1):
        utilization = float(plan_row['utilization'])
        pbar = float(plan_row['Pbar_kw_per_server'])
        reserve = float(plan_row['R_kw_per_server'])
        weights = [float(x) for x in plan_row['weights']]
        simulation_seed = int(plan_row['simulation_seed'])

        set_experiment_utilization(experiment_config, utilization)
        # ExperimentConfigReader exposes random_seed read-only, but Simulator reads the
        # internal value when it is constructed. This exact per-row override is recorded
        # in both CSVs through plan provenance.
        experiment_config._random_seed = simulation_seed

        row_metadata = dict(plan_row)
        row_pr_metadata = dict(pr_sweep_metadata)
        row_pr_metadata.update({
            'Workload_Name': os.path.splitext(os.path.basename(row_metadata.get('workload_config', '')))[0]
                or pr_sweep_metadata.get('Workload_Name', ''),
            'Workload_Config': row_metadata.get('workload_config', pr_sweep_metadata.get('Workload_Config', '')),
        })

        params = [pbar, reserve] + weights
        cost = objective_function(
            params,
            cost_function_config,
            policy_config,
            experiment_config,
            cluster_config,
            job_config,
            output_dir,
            iteration_idx=iteration_idx,
            total_iterations=total_iterations,
            weight_sample_id=int(plan_row.get('weight_index_within_pr', iteration_idx - 1)),
            pr_sweep_metadata=row_pr_metadata,
            plan_row_metadata=row_metadata,
        )
        if cost < best_cost:
            best_cost = cost
            best_params = [utilization] + params
    return best_params


def initialize_grid_search_output_files(output_dir, job_type_count):
        # CSV file for raw simulator outputs only.
        results_file = os.path.join(output_dir, 'grid_search_results.csv')
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                'Iteration',
                'Weight_Sample_ID',
                'Policy_Name',
                'Workload_Name',
                'Workload_Config',
                ] + PLAN_PROVENANCE_COLUMNS + [
                'Node_Count_Control',
                'Power_Cap_Percentage',
                'Flex_Resource_Probabilities',
                'Pbar_kw_per_server',
                'R_kw_per_server',
                'P_actual_watts',
                'R_actual_watts',
                'Pbar_ratio',
                'R_ratio',
                'Pbar_denominator_watts',
                'R_denominator_watts',
                'server_count',
                'utilization',
                'workload_mix_size',
                'workload_mix',
                ] + [f'Weight_{i}' for i in range(job_type_count)] + [
                'Simulator_RSR_Total_Cost',
                'Simulator_Power_Cost',
                'Mtrack_Error_MeanAbs_Normalized',
                'Mtrack_Error_MeanAbs_Watts',
                'Mtrack_Price_Coefficient_piE',
                'Mtrack_Hour_Seconds',
                'Mtrack_Cost',
                'QoS_Delay_Probability_Sum',
                'QoS_Violation_Ratio',
                'QoS_Delay_Probabilities',
                ]
            writer.writerow(headers)

        # Separate diagnostics file for residuals, percentiles, activation-function terms,
        # and P/R sweep audit metadata. These extra P/R metadata columns are diagnostic only.
        diagnostics_file = os.path.join(output_dir, 'grid_search_diagnostics.csv')
        with open(diagnostics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            diagnostic_headers = [
                'Iteration',
                'Weight_Sample_ID',
                'Workload_Name',
                'Workload_Config',
                ] + PLAN_PROVENANCE_COLUMNS + [
                'Pbar_kw_per_server',
                'R_kw_per_server',
                'P_actual_watts',
                'R_actual_watts',
                'Pbar_ratio',
                'R_ratio',
                'server_count',
                'utilization',
                'PR_Sweep_Mode',
                'Pmin_kw_per_server',
                'Pmax_kw_per_server',
                'Pbar_lower_bound_kw_per_server',
                'Pbar_upper_bound_kw_per_server',
                'PR_upper_bound_kw_per_server',
                'R_lower_bound_kw_per_server',
                'R_max_for_this_Pbar_kw_per_server',
                'P_plus_R_kw_per_server',
                'P_minus_R_kw_per_server',
                'Pbar_points',
                'R_points_per_Pbar',
                'PR_Total_Pairs_Before_Chunk',
                'PR_Chunk_Index',
                'PR_Num_Chunks',
                'PR_Chunk_Start_Index',
                'PR_Chunk_End_Index_Exclusive',
                'PR_Pairs_In_Chunk',
                'Weight_Equal_Value',
                'Weight_Relative_Lower_Bound',
                'Weight_Relative_Upper_Bound',
                'Weight_Server_Lower_Bound',
                'Weight_Upper_From_Lower_Bound',
                'Weight_Final_Lower_Bound',
                'Weight_Final_Upper_Bound',
                'Weight_Min',
                'Weight_Max',
                'Ctrack_Epsilon_90th',
                'Ctrack_Gamma',
                'Ctrack_Residual',
                'Ctrack_Mu',
                'Ctrack_MuScaled_Residual',
                'Ctrack_SoftPlus_Value',
                'Ctrack_Psi',
                'Ctrack_Weighted_Cost',
                'QoS_Delay_Probability_Residual_Sum',
                'QoS_Delay_Probability_Residuals',
                'Diagnostic_FlexDC_SoftPlus_QoS_Cost',
                'Diagnostic_FullPaperObjective_Cost',
                ]
            writer.writerow(diagnostic_headers)

def grid_search_parameters(cost_function_config, policy_config, experiment_config, cluster_config, job_config, output_dir,
                           utilization_values, pr_pairs, weight_combinations, pr_sweep_metadata):
    """
    Performs a sweep over:
    - utilization values
    - targeted physical (Pbar_kw_per_server, R_kw_per_server) pairs
    - workload weight vectors
    """
    best_cost = float('inf')
    best_params = None

    print("\nSweep configuration:")
    print(f"  utilization_values = {utilization_values}")
    print(f"  number of P/R pairs = {len(pr_pairs)}")
    print(f"  number of weight vectors = {len(weight_combinations)}")
    print(f"  job_type_count = {job_config.job_type_count}")
    print(f"  policy_name = {policy_config.runtime_policy_name}")
    print(f"  policy_parameters = {policy_config.policy_parameters}")

    print_pr_sweep_summary(
        pr_sweep_metadata=pr_sweep_metadata,
        utilization_values=utilization_values,
        weight_combinations=weight_combinations,
    )

    initialize_grid_search_output_files(output_dir, job_config.job_type_count)


    total_iterations = count_valid_parameter_combinations(
        utilization_values,
        pr_pairs,
        weight_combinations
    )

    print(f"Total simulator runs to execute: {total_iterations}")

    iteration_idx = 0

    for utilization in utilization_values:
        set_experiment_utilization(experiment_config, utilization)
        print(f"\n===== Starting utilization = {utilization} =====")

        for pbar_kw_per_server, r_kw_per_server in pr_pairs:
            # Defensive constraints. Auto mode should already generate only valid pairs.
            if pbar_kw_per_server + PR_SWEEP_EPS < r_kw_per_server:
                raise ValueError(f"Invalid P/R pair: Pbar={pbar_kw_per_server}, R={r_kw_per_server}")

            pr_upper_bound = pr_sweep_metadata.get("PR_upper_bound_kw_per_server")
            if pr_sweep_metadata.get("PR_Sweep_Mode") == "auto_workload_pr":
                if pbar_kw_per_server + r_kw_per_server > float(pr_upper_bound) + PR_SWEEP_EPS:
                    raise ValueError(
                        f"Invalid Pbar+R pair: Pbar={pbar_kw_per_server}, R={r_kw_per_server}, "
                        f"PR upper={pr_upper_bound}"
                    )

            # Loop over all valid weight combinations.
            for weight_sample_id, weights in enumerate(weight_combinations):
                iteration_idx += 1
                params = [pbar_kw_per_server, r_kw_per_server] + list(weights)

                cost = objective_function(
                    params,
                    cost_function_config,
                    policy_config,
                    experiment_config,
                    cluster_config,
                    job_config,
                    output_dir,
                    iteration_idx=iteration_idx,
                    total_iterations=total_iterations,
                    weight_sample_id=weight_sample_id,
                    pr_sweep_metadata=pr_sweep_metadata,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_params = [utilization] + params

    return best_params


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Parse configuration file paths.")

    # Configuration files under configs folder
    parser.add_argument('--gradient-config', type=str, required=True,
                        help="Path to the gradient descent configuration file (e.g., gradient_descent.ini)")
    parser.add_argument('--experiment-config', type=str, required=True,
                        help="Path to the experiment configuration file (e.g., exp.ini)")
    parser.add_argument('--cluster-config', type=str, required=True,
                        help="Path to the cluster configuration file (e.g., cluster.ini)")
    parser.add_argument('--policy-name', type=str, required=True,
                        choices=SUPPORTED_RUNTIME_POLICIES,
                        help="Name of the runtime policy. Options: AQA, flex-resource, priority-cap, proportional-cap")
    parser.add_argument('--job-config', type=str, required=True,
                        help="Path to the workload mix having job power performance configuration file (e.g., workload/W10.ini)")

    # Additional parameters
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Custom output directory name for simulation results under src.peacsim.output.")
    parser.add_argument('--logger', type=str, required=False, default='', help="Available loggers: ['wandb']")

    # Modular sweep parameters.
    parser.add_argument('--utilization-values', type=str, default=None,
                        help="Optional comma-separated utilization values, e.g. '0.5,0.75,0.9'. "
                             "Default: 0.5,0.75,0.8,0.85,0.9,0.95")

    parser.add_argument('--auto-workload-pr-sweep', type=str, default='true',
                        help="If true, derive Pbar/R sweep from the current workload config's Pmin/Pmax. Default: true.")

    parser.add_argument('--pbar-points', type=int, default=DEFAULT_PBAR_POINTS,
                        help="Number of Pbar points for auto workload P/R sweep. Default: 10.")

    parser.add_argument('--r-points-per-pbar', type=int, default=DEFAULT_R_POINTS_PER_PBAR,
                        help="Number of R points generated inside each Pbar's valid interval. Default: 10.")

    parser.add_argument('--pr-chunk-index', type=int, default=0,
                        help="Zero-based P/R chunk index for this worker. Default: 0.")

    parser.add_argument('--pr-num-chunks', type=int, default=1,
                        help="Total number of deterministic P/R chunks. Default: 1, meaning no P/R split.")

    parser.add_argument('--r-lower-kw-per-server', type=float, default=DEFAULT_R_LOWER_KW_PER_SERVER,
                        help="Lower R value for auto workload P/R sweep in kW/server. Default: 0.01.")

    parser.add_argument('--pbar-lower-factor', type=float, default=DEFAULT_PBAR_LOWER_FACTOR,
                        help="Lower Pbar factor applied to workload Pmin. Default: 0.9.")

    parser.add_argument('--pbar-upper-factor', type=float, default=DEFAULT_PBAR_UPPER_FACTOR,
                        help="Upper Pbar factor applied to workload Pmax. Default: 1.1.")

    parser.add_argument('--pr-upper-factor', type=float, default=DEFAULT_PR_UPPER_FACTOR,
                        help="Upper Pbar+R factor applied to workload Pmax. Default: 1.3.")

    parser.add_argument('--r-over-p-max', type=float, default=None,
                        help="Optional additional reserve cap R <= value * Pbar. "
                             "Exact-plan rows may also carry configured_r_over_p_max.")

    parser.add_argument('--configured-weight-lower', type=float, default=None,
                        help="Optional additional lower bound for every scheduling weight. "
                             "Intersected with existing server/job-count feasibility bounds.")

    parser.add_argument('--configured-weight-upper', type=float, default=None,
                        help="Optional additional upper bound for every scheduling weight. "
                             "Intersected with existing server/job-count feasibility bounds.")

    parser.add_argument('--pbar-kw-per-server-range', type=str, default=None,
                        help="Optional compact physical Pbar linspace range in kW/server as 'min,max,num_points', "
                             "e.g. '0.2,0.6,10'. Default: '0.2,0.6,10'")

    parser.add_argument('--r-kw-per-server-range', type=str, default=None,
                        help="Optional compact physical R linspace range in kW/server as 'min,max,num_points', "
                             "e.g. '0.0,0.12,10'. Default: '0.0,0.12,10'")

    parser.add_argument('--pbar-kw-per-server-values', type=str, default=None,
                        help="Optional exact comma-separated physical Pbar values in kW/server, e.g. '0.2,0.4,0.6'. "
                             "Overrides --pbar-kw-per-server-range if used alone.")

    parser.add_argument('--r-kw-per-server-values', type=str, default=None,
                        help="Optional exact comma-separated physical R values in kW/server, e.g. '0.0,0.06,0.12'. "
                             "Overrides --r-kw-per-server-range if used alone.")

    parser.add_argument('--weight-vectors', type=str, default=None,
                        help="Optional custom weight vectors. Format: "
                             "'0.25,0.25,0.25,0.25;0.4,0.3,0.2,0.1'. "
                             "Each vector must match job_type_count and sum to 1.")

    parser.add_argument('--weight-file', type=str, default=None,
                        help="Optional CSV or JSON file containing custom weight vectors. "
                             "CSV rows should be one vector per row. "
                             "JSON can be a list of vectors or {'weights': [...] }.")

    parser.add_argument('--sweep-plan-file', type=str, default=None,
                        help="Optional exact CSV/JSON plan. Each row specifies utilization, Pbar, R, weights, "
                             "and simulation_seed. Plan mode bypasses the legacy Cartesian sweep.")
    parser.add_argument('--plan-chunk-index', type=int, default=0,
                        help="Zero-based exact-plan row chunk index. Default: 0.")
    parser.add_argument('--plan-num-chunks', type=int, default=1,
                        help="Number of deterministic exact-plan row chunks. Default: 1.")
    parser.add_argument('--validate-sweep-plan-only', action='store_true',
                        help="Validate and summarize the selected plan chunk, then exit without simulation.")

    # Below is for runtime policies
    # I added node count control per Kerim's request
    # I hope this works!
    parser.add_argument('--node-count-control', type=str, default=None,
                        help="Optional override for node_count_control: true or false. "
                             "If omitted, uses the paper-consistent policy default: "
                             "AQA=true, flex-resource=true, priority-cap=false, proportional-cap=false.")

    parser.add_argument('--power-cap-percentage', type=float, default=None,
                        help="PriorityCap alpha value in [0, 1]. "
                             "Only used when --policy-name priority-cap. Default: 1.0")

    parser.add_argument('--probabilities', type=str, default=None,
                        help="FlexResource probabilities, comma-separated with one value per job type, "
                             "e.g. '0.8,0.8,0.8,0.8'. "
                             "Only used when --policy-name flex-resource. "
                             "Default: 0.8 for every job type.")

    args = parser.parse_args()

    # The gradient config is no longer used in grid search but is kept for compatibility.
    cost_function_config = gradient_parser.GradientDescentReader(args.gradient_config)

    experiment_config = experiment_parser.ExperimentConfigReader(args.experiment_config)
    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)
    job_config = job_parser.JobProfileReader(args.job_config)

    policy_parameters = build_policy_parameters(
        args=args,
        job_type_count=job_config.job_type_count
    )

    policy_config = policy_parser.PolicyConfigReader.from_parameters(
        policy_name=args.policy_name,
        P=-1,
        R=-1,
        P_ratio=-1,
        R_ratio=-1,
        policy_parameters=policy_parameters
    )

    if args.sweep_plan_file:
        plan_rows, pr_sweep_metadata = load_sweep_plan_rows(
            path=args.sweep_plan_file,
            args=args,
            job_config=job_config,
            experiment_config=experiment_config,
        )
        pr_sweep_metadata["Workload_Name"] = os.path.splitext(os.path.basename(args.job_config))[0]
        pr_sweep_metadata["Workload_Config"] = args.job_config
        print(f"Validated exact sweep plan rows in selected chunk: {len(plan_rows)}")
        if args.validate_sweep_plan_only:
            print("Plan validation completed successfully; simulation was not run.")
            raise SystemExit(0)
    else:
        utilization_values, pr_pairs, weight_combinations, pr_sweep_metadata = build_sweep_values(
            args=args,
            job_config=job_config,
            experiment_config=experiment_config
        )
        pr_sweep_metadata["Workload_Name"] = os.path.splitext(os.path.basename(args.job_config))[0]
        pr_sweep_metadata["Workload_Config"] = args.job_config

    output_dir = 'output/optimization/' + args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'
    os.makedirs(output_dir, exist_ok=True)
    print("saving results in", output_dir)

    if args.sweep_plan_file:
        optimal_params = grid_search_plan_rows(
            cost_function_config=cost_function_config,
            policy_config=policy_config,
            experiment_config=experiment_config,
            cluster_config=cluster_config,
            job_config=job_config,
            output_dir=output_dir,
            plan_rows=plan_rows,
            pr_sweep_metadata=pr_sweep_metadata,
        )
    else:
        optimal_params = grid_search_parameters(
            cost_function_config=cost_function_config,
            policy_config=policy_config,
            experiment_config=experiment_config,
            cluster_config=cluster_config,
            job_config=job_config,
            output_dir=output_dir,
            utilization_values=utilization_values,
            pr_pairs=pr_pairs,
            weight_combinations=weight_combinations,
            pr_sweep_metadata=pr_sweep_metadata
        )
    print("Best Parameters Under Simulator RSR Total Cost Only:", optimal_params)
