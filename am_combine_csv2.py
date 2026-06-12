from pathlib import Path
import argparse
import json
import re

import pandas as pd


RAW_NAME = "grid_search_results.csv"
DIAG_NAME = "grid_search_diagnostics.csv"


def collapse_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [
        col for col in df.columns
        if re.fullmatch(r"Weight_\d+", str(col))
    ]

    if not weight_cols:
        return df

    weight_cols = sorted(weight_cols, key=lambda c: int(c.split("_")[1]))
    first_weight_col_index = df.columns.get_loc(weight_cols[0])

    def row_to_weights(row):
        weights = []
        for col in weight_cols:
            value = row[col]
            if pd.isna(value):
                continue
            weights.append(float(value))
        return json.dumps(weights)

    weights_series = df.apply(row_to_weights, axis=1)

    df = df.drop(columns=weight_cols)
    df.insert(first_weight_col_index, "weights", weights_series)

    return df


def combine_csvs(opt_dir: Path, run_prefix: str, filename: str, output_path: Path) -> None:
    csv_paths = sorted(opt_dir.glob(f"{run_prefix}*/{filename}"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No {filename} files found under {opt_dir} with prefix '{run_prefix}'"
        )

    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)

        df.insert(0, "Source_Output_Dir", path.parent.name)
        df.insert(1, "Source_CSV_Path", str(path))

        df = collapse_weight_columns(df)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined.to_csv(output_path, index=False)

    print(f"Combined {len(csv_paths)} files matching prefix '{run_prefix}'")
    print(f"Output: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="targeted_pr_test",
        help="Only combine optimization folders starting with this prefix."
    )
    args = parser.parse_args()

    root = Path.cwd()
    opt_dir = root / "src" / "peacsim" / "output" / "optimization"

    out_dir = root / "06112026combined_output_targetedpr"
    out_dir.mkdir(exist_ok=True)

    raw_out = out_dir / f"{args.prefix}_combined_grid_search_results.csv"
    diag_out = out_dir / f"{args.prefix}_combined_grid_search_diagnostics.csv"

    print(f"Searching under: {opt_dir}")
    print(f"Using prefix: {args.prefix}")
    print(f"Saving combined outputs to: {out_dir}")
    print()

    combine_csvs(opt_dir, args.prefix, RAW_NAME, raw_out)
    combine_csvs(opt_dir, args.prefix, DIAG_NAME, diag_out)

    print("Done.")


if __name__ == "__main__":
    main()