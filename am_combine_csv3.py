from pathlib import Path
import argparse
import json
import re
import sys

import pandas as pd


RAW_NAME = "grid_search_results.csv"
DIAG_NAME = "grid_search_diagnostics.csv"

DEFAULT_PREFIX = "final_aqa_s250_1000_u060_080_prchunk10"

# Current expected run:
# 2 workloads × 2 server configs × 10 P/R chunks = 40 folders
# each folder: 2 utilizations × 10 P/R pairs × 101 weights = 2020 rows
EXPECTED_FILES = 40
EXPECTED_ROWS_PER_FILE = 2020
EXPECTED_TOTAL_ROWS = EXPECTED_FILES * EXPECTED_ROWS_PER_FILE


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


def find_csv_paths(opt_dir: Path, run_prefix: str, filename: str):
    csv_paths = sorted(opt_dir.glob(f"{run_prefix}*/{filename}"))

    # Safety: only combine chunked folders for this run.
    csv_paths = [p for p in csv_paths if "_prchunk" in p.parent.name]

    return csv_paths


def audit_chunk_coverage(csv_paths, filename: str):
    folders = [p.parent.name for p in csv_paths]

    chunk_ids = []
    for folder in folders:
        match = re.search(r"_prchunk(\d+)of(\d+)", folder)
        if match:
            chunk_ids.append((int(match.group(1)), int(match.group(2)), folder))

    if not chunk_ids:
        print(f"WARNING: could not parse P/R chunk IDs from {filename} folders.")
        return

    chunk_counts = sorted(set(total for _, total, _ in chunk_ids))
    if len(chunk_counts) != 1:
        print(f"WARNING: mixed prchunk totals detected for {filename}: {chunk_counts}")
        return

    total_chunks = chunk_counts[0]

    # This checks coverage per workload/server combination.
    # It groups folders by everything before _prchunkXXofYY.
    groups = {}
    for chunk_id, total, folder in chunk_ids:
        group_name = re.sub(r"_prchunk\d+of\d+.*$", "", folder)
        groups.setdefault(group_name, []).append(chunk_id)

    for group_name, chunks in sorted(groups.items()):
        expected = set(range(1, total_chunks + 1))
        actual = set(chunks)

        missing = sorted(expected - actual)
        duplicate_count = len(chunks) - len(actual)

        if missing:
            print(f"WARNING: {group_name} is missing chunks: {missing}")

        if duplicate_count:
            print(f"WARNING: {group_name} has {duplicate_count} duplicate chunk folders")

    print(f"Chunk audit for {filename}: {len(groups)} workload/server groups, {total_chunks} chunks each")


def combine_csvs(
    opt_dir: Path,
    run_prefix: str,
    filename: str,
    output_path: Path,
    collapse_weights: bool,
    expected_files: int,
    expected_rows_per_file: int,
    expected_total_rows: int,
) -> pd.DataFrame:
    csv_paths = find_csv_paths(opt_dir, run_prefix, filename)

    if not csv_paths:
        raise FileNotFoundError(
            f"No {filename} files found under {opt_dir} with prefix '{run_prefix}'"
        )

    print(f"Found {len(csv_paths)} {filename} files matching prefix '{run_prefix}'")

    if len(csv_paths) != expected_files:
        print(f"WARNING: expected {expected_files} files, but found {len(csv_paths)}")

    audit_chunk_coverage(csv_paths, filename)

    dfs = []
    bad_row_files = []

    for path in csv_paths:
        df = pd.read_csv(path)

        if len(df) != expected_rows_per_file:
            bad_row_files.append((path.parent.name, len(df)))

        df.insert(0, "Source_Output_Dir", path.parent.name)
        df.insert(1, "Source_CSV_Path", str(path))

        if collapse_weights:
            df = collapse_weight_columns(df)

        dfs.append(df)

    if bad_row_files:
        print()
        print(f"WARNING: these {filename} files do not have {expected_rows_per_file} rows:")
        for folder, rows in bad_row_files:
            print(f"  {folder}: {rows} rows")

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    if len(combined) != expected_total_rows:
        print(f"WARNING: expected {expected_total_rows} total rows, but found {len(combined)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print()
    print(f"Combined {len(csv_paths)} files")
    print(f"Output: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")
    print()

    return combined


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Only combine optimization folders starting with this prefix."
    )

    parser.add_argument(
        "--opt-dir",
        type=str,
        default=None,
        help="Optimization output directory. Default: src/peacsim/output/optimization"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="combined_output_final_aqa_s250_1000_u060_080_prchunk10",
        help="Directory where combined CSVs will be saved."
    )

    parser.add_argument(
        "--collapse-weights",
        action="store_true",
        help="Collapse Weight_0...Weight_k columns into one JSON weights column."
    )

    parser.add_argument(
        "--expected-files",
        type=int,
        default=EXPECTED_FILES,
    )

    parser.add_argument(
        "--expected-rows-per-file",
        type=int,
        default=EXPECTED_ROWS_PER_FILE,
    )

    args = parser.parse_args()

    root = Path.cwd()

    if args.opt_dir is None:
        opt_dir = root / "src" / "peacsim" / "output" / "optimization"
    else:
        opt_dir = Path(args.opt_dir)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_total_rows = args.expected_files * args.expected_rows_per_file

    raw_out = out_dir / f"{args.prefix}_combined_grid_search_results.csv"
    diag_out = out_dir / f"{args.prefix}_combined_grid_search_diagnostics.csv"

    print("Combine configuration")
    print("---------------------")
    print(f"FlexDC root: {root}")
    print(f"Searching under: {opt_dir}")
    print(f"Using prefix: {args.prefix}")
    print(f"Saving combined outputs to: {out_dir}")
    print(f"Collapse weights: {args.collapse_weights}")
    print(f"Expected files: {args.expected_files}")
    print(f"Expected rows per file: {args.expected_rows_per_file}")
    print(f"Expected total rows: {expected_total_rows}")
    print()

    raw_df = combine_csvs(
        opt_dir=opt_dir,
        run_prefix=args.prefix,
        filename=RAW_NAME,
        output_path=raw_out,
        collapse_weights=args.collapse_weights,
        expected_files=args.expected_files,
        expected_rows_per_file=args.expected_rows_per_file,
        expected_total_rows=expected_total_rows,
    )

    diag_df = combine_csvs(
        opt_dir=opt_dir,
        run_prefix=args.prefix,
        filename=DIAG_NAME,
        output_path=diag_out,
        collapse_weights=args.collapse_weights,
        expected_files=args.expected_files,
        expected_rows_per_file=args.expected_rows_per_file,
        expected_total_rows=expected_total_rows,
    )

    if len(raw_df) != len(diag_df):
        print(f"WARNING: raw rows ({len(raw_df)}) != diagnostics rows ({len(diag_df)})")
    else:
        print(f"Raw and diagnostics row counts match: {len(raw_df)}")

    print()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("FAILED:")
        print(exc)
        sys.exit(1)