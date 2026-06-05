import pandas as pd
from pathlib import Path
import re

ROOT = Path("src/peacsim/output/optimization")

all_frames = []

for csv_file in ROOT.rglob("grid_search_results.csv"):

    try:
        df = pd.read_csv(csv_file)

        folder = csv_file.parent.name

        df["source_folder"] = folder

        workload_match = re.search(r"_W(\d+)_", folder)
        if workload_match:
            df["workload_name"] = f"W{workload_match.group(1)}"
        else:
            df["workload_name"] = "unknown"

        server_match = re.search(r"exp_servers_(\d+)", folder)
        if server_match:
            df["experiment_name"] = f"exp_servers_{server_match.group(1)}"
        else:
            df["experiment_name"] = "unknown"

        all_frames.append(df)

        print(
            f"Loaded {csv_file} "
            f"({len(df)} rows)"
        )

    except Exception as e:
        print(f"FAILED: {csv_file}")
        print(e)

combined = pd.concat(
    all_frames,
    ignore_index=True
)

combined.to_csv(
    "flexdc_all_data.csv",
    index=False
)

print()
print("====================================")
print("Rows:", len(combined))
print("Columns:", len(combined.columns))
print("Saved: flexdc_all_data.csv")
print("====================================")

df = pd.read_csv("flexdc_all_data.csv")
print(df.head())
print(df.info())

print(df[df["Weight_8"].notna()][
    ["workload_name","experiment_name"]
].drop_duplicates())