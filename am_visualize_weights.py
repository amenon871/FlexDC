import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "generated_softmax_weights.csv"
OUTPUT_DIR = "weight_distribution_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

weight_cols = [c for c in df.columns if c.startswith("Weight_")]

# Convert wide Weight_0, Weight_1, ... columns into one long table.
weights_long = df.melt(
    id_vars=["workload_mix_size", "temperature", "sample_id", "equal_weight"],
    value_vars=weight_cols,
    var_name="weight_index",
    value_name="weight",
).dropna(subset=["weight"])

print("Loaded:", INPUT_CSV)
print("Rows:", len(df))
print("Individual weights:", len(weights_long))
print("Output folder:", OUTPUT_DIR)

for J in sorted(weights_long["workload_mix_size"].unique()):
    sub = weights_long[weights_long["workload_mix_size"] == J].copy()
    weights = sub["weight"].to_numpy()
    equal_weight = 1.0 / J
    temp = sub["temperature"].iloc[0]

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(weights, bins=40)
    plt.axvline(equal_weight, linestyle="--", label=f"Equal weight = {equal_weight:.4f}")
    plt.title(f"Weight Histogram, J={J}, T={temp}")
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"histogram_J{J}.png"), dpi=200)
    plt.close()

    # CDF
    sorted_weights = np.sort(weights)
    cdf = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_weights, cdf)
    plt.axvline(equal_weight, linestyle="--", label=f"Equal weight = {equal_weight:.4f}")
    plt.title(f"Weight CDF, J={J}, T={temp}")
    plt.xlabel("Weight value")
    plt.ylabel("Fraction of weights <= value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cdf_J{J}.png"), dpi=200)
    plt.close()

print("\nCreated plots:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(os.path.join(OUTPUT_DIR, filename))