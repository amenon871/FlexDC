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
from denormalize_PR import denormalize_PR
import os
import itertools
from calculate_qos_cost import calculate_qos_cost


#Here are my default sweep settings I added
DEFAULT_UTILIZATION_VALUES = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]

# Format: min, max, num_points
DEFAULT_PBAR_RANGE = (0.7, 1.8, 10)
DEFAULT_R_RANGE = (0.05, 1.0, 10)

DEFAULT_NUM_WEIGHT_SAMPLES = 5
DEFAULT_WEIGHT_SEED = 42
DEFAULT_WEIGHT_TEMPERATURE = 1.0
DEFAULT_INCLUDE_EQUAL_WEIGHTS = True

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


def validate_weight_vectors(weight_combinations, job_type_count, tolerance=1e-6):
    """
    Validate that every workload-weight vector has the correct length and sums to 1.
    """
    validated = []

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

        validated.append(weights)

    return validated


def build_sweep_values(args, job_type_count):
    utilization_values = parse_float_list(args.utilization_values, "--utilization-values")
    if utilization_values is None:
        utilization_values = DEFAULT_UTILIZATION_VALUES

    if args.pbar_values is not None and args.pbar_range is not None:
        raise ValueError("Use either --pbar-values or --pbar-range, not both.")

    if args.r_values is not None and args.r_range is not None:
        raise ValueError("Use either --r-values or --r-range, not both.")

    pbar_values = parse_float_list(args.pbar_values, "--pbar-values")
    if pbar_values is None:
        pbar_values = parse_range_spec(args.pbar_range, DEFAULT_PBAR_RANGE, "--pbar-range")

    r_values = parse_float_list(args.r_values, "--r-values")
    if r_values is None:
        r_values = parse_range_spec(args.r_range, DEFAULT_R_RANGE, "--r-range")

    inline_weight_vectors = parse_weight_vectors_from_string(args.weight_vectors)
    file_weight_vectors = load_weight_vectors_from_file(args.weight_file)

    if inline_weight_vectors is not None and file_weight_vectors is not None:
        raise ValueError("Use either --weight-vectors or --weight-file, not both.")

    if inline_weight_vectors is not None:
        weight_combinations = inline_weight_vectors

    elif file_weight_vectors is not None:
        weight_combinations = file_weight_vectors

    else:
        weight_combinations = generate_random_softmax_weights(
            job_type_count=job_type_count,
            num_weight_samples=DEFAULT_NUM_WEIGHT_SAMPLES,
            seed=DEFAULT_WEIGHT_SEED,
            temperature=DEFAULT_WEIGHT_TEMPERATURE
        )

        if DEFAULT_INCLUDE_EQUAL_WEIGHTS:
            equal_weights = [1 / job_type_count] * job_type_count
            weight_combinations.append(equal_weights)

    weight_combinations = validate_weight_vectors(weight_combinations, job_type_count)

    pbar_values = [float(x) for x in pbar_values]
    r_values = [float(x) for x in r_values]
    utilization_values = [float(x) for x in utilization_values]

    return utilization_values, pbar_values, r_values, weight_combinations


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


def count_valid_parameter_combinations(utilization_values, pbar_values, r_values, weight_combinations):
    total = 0
    for _ in utilization_values:
        for Pbar in pbar_values:
            for R in r_values:
                if Pbar <= R:
                    continue
                total += len(weight_combinations)
    return total

def set_experiment_utilization(experiment_config, utilization):
    if hasattr(experiment_config, "_utilization"):
        experiment_config._utilization = utilization
    else:
        experiment_config.utilization = utilization

def create_simulator_object(P_ratio, R_ratio, experiment_config, job_config, cluster_config, policy_config, output_dir):
    job_table = init_job_table(experiment_config, job_config)
    node_table = init_node_table(experiment_config)

    avg_max_job_power = sum(job_config.all_max_job_power.values()) / len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers
    P_ratio = round(P_ratio, 6)
    R_ratio = round(R_ratio, 6)
    policy_config._P_ratio = P_ratio
    policy_config._R_ratio = R_ratio
    policy_config._P = int(P_ratio * (avg_max_job_power * avg_num_servers +
                                      experiment_config.idle_power * avg_idle_servers))
    policy_config._R = int(R_ratio * (avg_max_job_power * experiment_config.server_count -
                                      (experiment_config.idle_power * experiment_config.server_count)) / 2)

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
    return simulator

def build_workload_mix_json(job_config, weights):
    """
    Important:
    job_config.all_jobs maps numeric job_type_id -> job name.
    The feature dictionaries are keyed by numeric job_type_id.
    This means that we should iterate over the numeric keys, not all_jobs.values().
    """
    workload_mix = []

    job_type_ids = sorted(job_config.all_min_job_power.keys())

    for idx, job_type_id in enumerate(job_type_ids):
        row = [
            float(job_config.all_min_job_power[job_type_id]),
            float(job_config.all_max_job_power[job_type_id]),
            float(job_config.all_min_execution_time[job_type_id]),
            float(job_config.all_max_execution_time[job_type_id]),
            float(job_config.all_job_qos_constraints[job_type_id]),
            float(job_config.all_job_size[job_type_id]),
            float(weights[idx]),
        ]
        workload_mix.append(row)

    return json.dumps(workload_mix)


def calculate_raw_qos_cost(sim_hour, J, job_table, qos_constraints, min_execution_times, cost_function_config):
    from calculate_qos_cost import calculate_delay_prob

    delay_prob = calculate_delay_prob(
        sim_hour,
        J,
        job_table,
        qos_constraints,
        min_execution_times
    )

    delta = cost_function_config.calculate_gradient_variables['qos_threshold']

    raw_qos_terms = [delay_prob[itype] - delta for itype in range(J)]
    raw_qos_cost = sum(raw_qos_terms)

    return raw_qos_cost, raw_qos_terms, delay_prob



def objective_function(params, cost_function_config, policy_config, experiment_config, cluster_config, job_config,
                       output_dir, iteration_idx=None, total_iterations=None, weight_sample_id=None):
    Pbar, R, *weights = params
    if iteration_idx is not None and total_iterations is not None:
        print(f"\n========== Iteration {iteration_idx}/{total_iterations} ==========")
        print(
            f"utilization={experiment_config.utilization}, "
            f"Pbar={Pbar:.6f}, R={R:.6f}, weights={weights}"
        )

    base_weights_path = os.path.join(output_dir, 'base_weights.csv')
    pd.DataFrame({
        'job_type_id': np.array(list(job_config.all_jobs.values())),
        'weights': weights
    }).to_csv(base_weights_path, index=False)

    policy_config.policy_parameters['weights_path'] = base_weights_path
    simulator = create_simulator_object(Pbar, R, experiment_config, job_config, cluster_config,
                                        policy_config, output_dir)

    # Actual watt values after converting normalized Pbar/R ratios
    p_actual = getattr(policy_config, "P", getattr(policy_config, "_P", None))
    r_actual = getattr(policy_config, "R", getattr(policy_config, "_R", None))

    init_output(output_dir, experiment_config, policy_config)
    cost_power, simulator_tracking_error = simulator.run()

    # Raw tracking term for data generation.
    raw_tracking_cost = simulator_tracking_error - 0.3

    # I kept the original old tracking penalty too for debugging/comparison.
    psi1, psi2 = cost_function_config.calculate_gradient_variables['psi1'], \
        cost_function_config.calculate_gradient_variables['psi2']

    #softplus_tracking_cost = psi1 * simulator_tracking_error * (
    #    1 + np.log(1 + np.exp(psi2 * (simulator_tracking_error - 0.3)))
    #)
    
    #Same math but this version prevents overflow
    softplus_tracking_cost = psi1 * simulator_tracking_error * (
    1 + np.logaddexp(0, psi2 * (simulator_tracking_error - 0.3))
    )

    qos_constraints = list(simulator._job_config.all_job_qos_constraints.values())
    min_execution_times = list(simulator._job_config.all_min_execution_time.values())

    # Existing FlexDC/old objective QoS cost, kept for comparison.
    softplus_qos_cost = calculate_qos_cost(
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

    # Raw QoS term for data generation.
    raw_qos_cost, raw_qos_terms, delay_prob = calculate_raw_qos_cost(
        sim_hour=1,
        J=simulator._job_type_count,
        job_table=simulator._job_table,
        qos_constraints=qos_constraints,
        min_execution_times=min_execution_times,
        cost_function_config=cost_function_config
    )

    # Raw objective for data generation.
    raw_total_cost = cost_power + raw_tracking_cost + raw_qos_cost

    # Existing old objective, useful for comparison/debugging.
    softplus_total_cost = cost_power + softplus_tracking_cost + softplus_qos_cost

    
    print(
        f"Iteration Variables: Pbar={Pbar}, R={R}, Weights={weights}, "
        f"Raw Total Cost={raw_total_cost}, "
        f"Power Cost={cost_power}, "
        f"Raw Tracking Cost={raw_tracking_cost}, "
        f"Raw QoS Cost={raw_qos_cost}, "
        f"SoftPlus Total Cost={softplus_total_cost}"
    )

    results_file = os.path.join(output_dir, 'grid_search_results.csv')

    workload_mix_json = build_workload_mix_json(job_config, weights)

    results = [
        iteration_idx,
        weight_sample_id,
        policy_config.runtime_policy_name,
        policy_config.policy_parameters.get("node_count_control"),
        policy_config.policy_parameters.get("power_cap_percentage", ""),
        json.dumps(policy_config.policy_parameters.get("probabilities", "")),
        Pbar,                  # normalized Pbar ratio
        R,                     # normalized R ratio
        p_actual,                     # actual P in watts
        r_actual,                     # actual R in watts
        experiment_config.server_count,
        experiment_config.utilization,
        job_config.job_type_count,
        workload_mix_json,
    ] + list(weights) + [
        raw_total_cost,
        cost_power,
        raw_tracking_cost,
        raw_qos_cost,
        simulator_tracking_error,
        softplus_total_cost,
        softplus_tracking_cost,
        softplus_qos_cost,
        simulator.ratio_of_job_type_qos_violation,
        simulator.tracking_error_at_90,
        json.dumps(delay_prob),
        json.dumps(raw_qos_terms),
    ]

    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)

    return raw_total_cost


def generate_weight_grid(lower, upper, step, count):
    values = np.arange(lower, upper + step / 2, step)
    weight_combinations = []
    for combo in itertools.product(values, repeat=count):
        if abs(sum(combo) - 1.0) < 1e-6:
            weight_combinations.append(combo)
    return weight_combinations


def generate_random_softmax_weights(job_type_count, num_weight_samples, seed=42, temperature=1.0):
    #Temp is how spread out they are btw
    rng = np.random.default_rng(seed)
    weight_combinations = []

    for _ in range(num_weight_samples):
        raw_scores = rng.normal(loc=0.0, scale=1.0, size=job_type_count)
        weights = softmax(raw_scores / temperature)
        weight_combinations.append(weights.tolist())

    return weight_combinations

def grid_search_parameters(cost_function_config, policy_config, experiment_config, cluster_config, job_config, output_dir,
                           utilization_values, pbar_values, r_values, weight_combinations):
    """
    Performs a sweep over:
    - utilization values
    - Pbar/R bid ratios
    - randomly sampled workload weight vectors
    """
    best_cost = float('inf')
    best_params = None

    print("\nSweep configuration:")
    print(f"  utilization_values = {utilization_values}")
    print(f"  pbar_values = {pbar_values}")
    print(f"  r_values = {r_values}")
    print(f"  number of weight vectors = {len(weight_combinations)}")
    print(f"  job_type_count = {job_config.job_type_count}")
    print(f"  policy_name = {policy_config.runtime_policy_name}")
    print(f"  policy_parameters = {policy_config.policy_parameters}")

    # CSV file for logging iterations
    results_file = os.path.join(output_dir, 'grid_search_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = [
            'Iteration',
            'Weight_Sample_ID',
            'Policy_Name',
            'Node_Count_Control',
            'Power_Cap_Percentage',
            'Flex_Resource_Probabilities',
            'Pbar_ratio',
            'R_ratio',
            'P_actual_watts',
            'R_actual_watts',
            'server_count',
            'utilization',
            'workload_mix_size',
            'workload_mix',
            ] + [f'Weight_{i}' for i in range(job_config.job_type_count)] + [
                'Raw_Total_Cost',
                'Power_Cost',
                'Raw_Tracking_Cost',
                'Raw_QoS_Cost',
                'Simulator_Tracking_Error',
                'SoftPlus_Total_Cost',
                'SoftPlus_Tracking_Cost',
                'SoftPlus_QoS_Cost',
                'QoS_Violation_Ratio',
                'Tracking_Error_90th',
                'Delay_Probabilities',
                'Raw_QoS_Terms',
            ]
        writer.writerow(headers)



    total_iterations = count_valid_parameter_combinations(
        utilization_values,
        pbar_values,
        r_values,
        weight_combinations
    )

    print(f"Total simulator runs to execute: {total_iterations}")

    iteration_idx = 0

    for utilization in utilization_values:
        set_experiment_utilization(experiment_config, utilization)
        print(f"\n===== Starting utilization = {utilization} =====")

        for Pbar in pbar_values:
            for R in r_values:
                # Enforce constraint: Pbar must be greater than R.
                if Pbar <= R:
                    continue

                # Loop over all valid weight combinations
                for weight_sample_id, weights in enumerate(weight_combinations):
                    iteration_idx += 1
                    params = [Pbar, R] + list(weights)

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
                        weight_sample_id=weight_sample_id
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

    parser.add_argument('--pbar-range', type=str, default=None,
                        help="Optional compact Pbar linspace range as 'min,max,num_points', e.g. '0.7,1.8,10'. "
                             "Default: '0.7,1.8,10'")

    parser.add_argument('--r-range', type=str, default=None,
                        help="Optional compact R linspace range as 'min,max,num_points', e.g. '0.05,1.0,10'. "
                             "Default: '0.05,1.0,10'")

    parser.add_argument('--pbar-values', type=str, default=None,
                        help="Optional exact comma-separated Pbar ratio values, e.g. '0.7,1.0,1.3'. "
                             "Overrides --pbar-range if used alone.")

    parser.add_argument('--r-values', type=str, default=None,
                        help="Optional exact comma-separated R ratio values, e.g. '0.05,0.2,0.5'. "
                             "Overrides --r-range if used alone.")

    parser.add_argument('--weight-vectors', type=str, default=None,
                        help="Optional custom weight vectors. Format: "
                             "'0.25,0.25,0.25,0.25;0.4,0.3,0.2,0.1'. "
                             "Each vector must match job_type_count and sum to 1.")

    parser.add_argument('--weight-file', type=str, default=None,
                        help="Optional CSV or JSON file containing custom weight vectors. "
                             "CSV rows should be one vector per row. "
                             "JSON can be a list of vectors or {'weights': [...]}.")

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

    utilization_values, pbar_values, r_values, weight_combinations = build_sweep_values(
        args=args,
        job_type_count=job_config.job_type_count
    )

    output_dir = 'output/optimization/' + args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'
    os.makedirs(output_dir, exist_ok=True)
    print("saving results in", output_dir)

    optimal_params = grid_search_parameters(
        cost_function_config=cost_function_config,
        policy_config=policy_config,
        experiment_config=experiment_config,
        cluster_config=cluster_config,
        job_config=job_config,
        output_dir=output_dir,
        utilization_values=utilization_values,
        pbar_values=pbar_values,
        r_values=r_values,
        weight_combinations=weight_combinations
    )
    print("Best Parameters Under Raw Total Cost:", optimal_params)
