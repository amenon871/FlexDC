import numpy as np
import pandas as pd
INPUT_CSV1 = "grid_search_results.csv"
INPUT_CSV2 = "diagnostics.csv"

FINAL_COLUMNS = [] #What columns we wish to keep

df1 = pd.read_csv(INPUT_CSV1)
df2 = pd.read_csv(INPUT_CSV2)

df1_cols = list(df1.columns)
df2_cols = list(df2.columns)

final_df = pd.DataFrame()

for col in df1_cols:
    if col in FINAL_COLUMNS:
        final_df[col] = df1[col]
for col in df2_cols:
    if col in FINAL_COLUMNS:
        final_df[col] = df2[col]

final_df.to_csv("merged_results.csv", index=False)