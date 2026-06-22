import numpy as np
import pandas as pd

# Settings
JOB_TYPE_COUNTS = [2, 4, 8, 16]
NUM_SAMPLES_PER_SIZE = 1000
TEMPERATURE = 1.0
SEED = 42

# Optional threshold settings.
# Example: 0.05 means min weight must be at least 5% of equal weight.
# For J=10, equal weight = 0.1, so lower bound = 0.005.
MIN_FRACTION_OF_EQUAL = None  # try 0.05 later if needed
MAX_MULTIPLE_OF_EQUAL = None  # try 3.0 later if needed

OUTPUT_CSV = "generated_softmax_weights.csv"


def softmax_temperature(z, T):
    z = np.asarray(z, dtype=float) / T
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def passes_threshold(weights, J):
    equal_weight = 1.0 / J

    if MIN_FRACTION_OF_EQUAL is not None:
        lower_bound = MIN_FRACTION_OF_EQUAL * equal_weight
        if np.min(weights) < lower_bound:
            return False

    if MAX_MULTIPLE_OF_EQUAL is not None:
        upper_bound = MAX_MULTIPLE_OF_EQUAL * equal_weight
        if np.max(weights) > upper_bound:
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

        row = {
            "workload_mix_size": J,
            "sample_id": generated,
            "temperature": TEMPERATURE,
            "equal_weight": 1.0 / J,
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