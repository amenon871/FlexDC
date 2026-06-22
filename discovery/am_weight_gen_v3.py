import numpy as np
import pandas as pd

JOB_TYPE_COUNTS = [2, 4, 8, 16]
NUM_SAMPLES_PER_SIZE = 1000
TEMPERATURE = 2.0
SEED = 42

SERVER_COUNT = 64

print(f"Temperature Value {TEMPERATURE}")

#Our arbitrary thresholds
MIN_FRACTION_OF_EQUAL = 0.1   # Example: 0.05 means min weight >= 5% of equal weight
MAX_MULTIPLE_OF_EQUAL = 4.0   # Example: 3.0 means max weight <= 3x equal weight

OUTPUT_CSV = "generated_softmax_weights2.csv"


def softmax_temperature(z, T):
    # This is exactly exp(z_i / T) / sum_j exp(z_j / T)
    exp_values = np.exp(z / T)
    return exp_values / np.sum(exp_values)


def get_bounds(J):
    equal_weight = 1.0 / J

    # This is the relative lower and upper bounds (the arbitrary ones)
    if MIN_FRACTION_OF_EQUAL is None:
        relative_lower = 0.0
    else:
        relative_lower = MIN_FRACTION_OF_EQUAL * equal_weight

    if MAX_MULTIPLE_OF_EQUAL is None:
        relative_upper = 1.0
    else:
        relative_upper = MAX_MULTIPLE_OF_EQUAL * equal_weight

    # This is the new server-based lower bound we discussed from S * w_min >= 1
    server_lower = 1.0 / SERVER_COUNT

    # We keep only the stricter lower bound
    if relative_lower >= server_lower:
        final_lower = relative_lower
        lower_bound_source = "relative"
    else:
        final_lower = server_lower
        lower_bound_source = "server"

    # This is the server-based upper bound that is based on the chosen lower bound
    # If J - 1 jobs each get final_lower, then the remaining job can get at most this
    upper_from_lower = 1.0 - (J - 1) * final_lower

    # Again we keep only stricter
    if relative_upper <= upper_from_lower:
        final_upper = relative_upper
        upper_bound_source = "relative"
    else:
        final_upper = upper_from_lower
        upper_bound_source = "server"

    return {
        "relative_lower": relative_lower,
        "server_lower": server_lower,
        "lower_bound": final_lower,
        "lower_bound_source": lower_bound_source,
        "relative_upper": relative_upper,
        "upper_from_lower": upper_from_lower,
        "upper_bound": final_upper,
        "upper_bound_source": upper_bound_source,
    }


def passes_threshold(weights, J):
    bounds = get_bounds(J)

    if np.min(weights) < bounds["lower_bound"]:
        return False

    if np.max(weights) > bounds["upper_bound"]:
        return False

    return True


rng = np.random.default_rng(SEED)
rows = []

for J in JOB_TYPE_COUNTS:
    generated = 0
    attempts = 0
    max_attempts = NUM_SAMPLES_PER_SIZE * 1000

    while generated < NUM_SAMPLES_PER_SIZE and attempts < max_attempts:
        attempts += 1

        z = rng.normal(loc=0.0, scale=1.0, size=J)
        weights = softmax_temperature(z, TEMPERATURE)

        if not passes_threshold(weights, J):
            continue

        bounds = get_bounds(J)

        row = {
            "workload_mix_size": J,
            "sample_id": generated,
            "temperature": TEMPERATURE,
            "server_count": SERVER_COUNT,
            "equal_weight": 1.0 / J,

            "relative_lower": bounds["relative_lower"],
            "server_lower": bounds["server_lower"],
            "lower_bound": bounds["lower_bound"],
            "lower_bound_source": bounds["lower_bound_source"],

            "relative_upper": bounds["relative_upper"],
            "upper_from_lower": bounds["upper_from_lower"],
            "upper_bound": bounds["upper_bound"],
            "upper_bound_source": bounds["upper_bound_source"],

            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "std_weight": float(np.std(weights)),
        }

        for i, w in enumerate(weights):
            row[f"Weight_{i}"] = float(w)

        rows.append(row)
        generated += 1

    if generated < NUM_SAMPLES_PER_SIZE:
        print(f"Warning: only generated {generated}/{NUM_SAMPLES_PER_SIZE} samples for J={J}")


df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved weights to {OUTPUT_CSV}")

summary = (
    df.groupby("workload_mix_size")
    .agg(
        samples=("sample_id", "count"),
        equal_weight=("equal_weight", "first"),

        relative_lower=("relative_lower", "first"),
        server_lower=("server_lower", "first"),
        lower_bound=("lower_bound", "first"),
        lower_bound_source=("lower_bound_source", "first"),

        relative_upper=("relative_upper", "first"),
        upper_from_lower=("upper_from_lower", "first"),
        upper_bound=("upper_bound", "first"),
        upper_bound_source=("upper_bound_source", "first"),

        min_weight_seen=("min_weight", "min"),
        median_min_weight=("min_weight", "median"),
        median_max_weight=("max_weight", "median"),
        max_weight_seen=("max_weight", "max"),
        median_std_weight=("std_weight", "median"),
    )
    .reset_index()
)

print("\nSummary:")
print(summary)