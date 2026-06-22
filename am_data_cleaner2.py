import pandas as pd

INPUT_RESULTS_CSV = "grid_search_results.csv"
INPUT_DIAGNOSTICS_CSV = "grid_search_diagnostics.csv"
OUTPUT_CSV = "training_aqa.csv"

# Columns we want in the final training CSV.
# Fill this with the actual training inputs + labels.
FINAL_COLUMNS = [
    "server_count",
    "utilization",
    "P_actual_watts",
    "R_actual_watts",
]

df_results = pd.read_csv(INPUT_RESULTS_CSV)
df_diag = pd.read_csv(INPUT_DIAGNOSTICS_CSV)

# Optional: AQA-only training file.
if "Policy_Name" in df_results.columns:
    df_results = df_results[df_results["Policy_Name"] == "AQA"].copy()

# Use only merge keys that exist in both files.
candidate_merge_keys = [
    "Source_Output_Dir",
    "Iteration",
    "Weight_Sample_ID",
    "Workload_Name",
    "Workload_Config",
    "Pbar_kw_per_server",
    "R_kw_per_server",
    "P_actual_watts",
    "R_actual_watts",
    "server_count",
    "utilization",
]

merge_keys = [c for c in candidate_merge_keys if c in df_results.columns and c in df_diag.columns]

if not merge_keys:
    raise ValueError("No shared merge keys found between results and diagnostics CSVs.")

merged = df_results.merge(
    df_diag,
    on=merge_keys,
    how="left",
    validate="one_to_one",
    suffixes=("", "_diagnostics")
)

# Keep only columns that actually exist.
existing_final_columns = [c for c in FINAL_COLUMNS if c in merged.columns]
missing_final_columns = [c for c in FINAL_COLUMNS if c not in merged.columns]

if missing_final_columns:
    print("Warning: these requested columns were not found and will be skipped:")
    print(missing_final_columns)

final_df = merged[existing_final_columns].copy()

final_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved final training CSV to {OUTPUT_CSV}")
print(f"Rows: {len(final_df)}")
print(f"Columns: {len(final_df.columns)}")
print(final_df.columns.tolist())