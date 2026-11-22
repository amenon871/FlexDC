"""Bounded one-hour AQA parity checks against the pre-speedup checkout."""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .collection import plan_path, worker_env
from .design import CLUSTER
from .io import ROOT, csv_rows, csv_write, json_write, load, sha
from .sampling import FIELDS

REFERENCE = "c50aea78071a903b20a581891d4106a9da3d2d57"
SIMULATOR_OUTPUTS = (
    "power_trace.csv",
    "job_table.csv",
    "PRtable.csv",
    "base_weights.csv",
    "qos_summary.csv",
    "QoSDegAvg.csv",
    "cost_power_rsr.csv",
)
COMPACT = ("grid_search_results.csv", "grid_search_diagnostics.csv")


def select_cases(manifest):
    choices = [
        (2, 0.6, "training", "measured"),
        (2, 0.8, "inference", "runtime"),
        (4, 0.8, "mixed", "power"),
        (6, 0.6, "mixed", "runtime"),
        (7, 0.8, "mixed", "power"),
        (8, 0.6, "mixed", "measured"),
    ]
    cases = []
    for j, u, mix, kind in choices:
        for cid, c in manifest["contexts"].items():
            w = manifest["workloads"][c["workload"]]
            roles = {manifest["profiles"][name]["role"] for name in w["jobs"]}
            kinds = {manifest["profiles"][name]["kind"] for name in w["jobs"]}
            if (c["j"], c["u"]) == (j, u) and roles == {
                "training": {"T"},
                "inference": {"I"},
                "mixed": {"T", "I"},
            }[mix]:
                if kinds == {"measured"} if kind == "measured" else kind in kinds:
                    cases.append((cid, j, u, mix, kind))
                    break
        else:
            raise ValueError(f"Missing parity case {j}/{u}/{mix}/{kind}")
    return cases


def run_one(checkout, config_root, manifest, row, output, label, fast):
    data = dict(row)
    for key in ("workload_config", "experiment_config"):
        data[key] = (ROOT / data[key]).as_posix()
    plan = output / (label + ".csv")
    csv_write(plan, [data], FIELDS)
    wizard = checkout / "src/peacsim/am_data_extraction_wizard.py"
    env = worker_env()
    import os

    env["PYTHONPATH"] = os.pathsep.join([str(checkout / "src"), str(checkout)])
    if not fast:
        env.pop("FLEXDC_P100_DOE_EVIDENCE", None)
    command = [
        sys.executable,
        "-B",
        str(wizard),
        "--sweep-plan-file",
        str(plan),
        "--job-config",
        data["workload_config"],
        "--experiment-config",
        data["experiment_config"],
        "--gradient-config",
        str(config_root / manifest["gradient"]),
        "--cluster-config",
        str(CLUSTER),
        "--output-dir",
        label,
        "--policy-name",
        "AQA",
        "--node-count-control",
        "true",
        "--pbar-lower-factor",
        "0.9",
        "--pbar-upper-factor",
        "1.0",
        "--pr-upper-factor",
        "1.2",
        "--r-lower-kw-per-server",
        "0.01",
        "--r-over-p-max",
        "0.6",
    ]
    with (output / (label + ".log")).open("wb") as stream:
        start = time.perf_counter()
        subprocess.run(
            command,
            cwd=wizard.parent,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        elapsed = time.perf_counter() - start
    folders = list((wizard.parent / "output/optimization").glob(label + "_*"))
    if len(folders) != 1:
        raise ValueError(f"Expected one output for {label}")
    folder = folders[0]
    signatures = {
        name: sha(folder / name)
        for name in SIMULATOR_OUTPUTS
        if (folder / name).exists()
    }
    compact = {}
    for name in COMPACT:
        (compact[name],) = list(csv_rows(folder / name))
        compact[name].pop("Plan_Source_File", None)
    size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    return dict(
        seconds=elapsed, signatures=signatures, compact=compact, output_bytes=size
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/p100_v3_doe_matched")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-repetitions", type=int, choices=range(0, 3), default=1)
    args = parser.parse_args()
    root, reference, output = (
        value.resolve() for value in (args.root, args.reference, args.output)
    )
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=reference, text=True
        ).strip()
        != REFERENCE
    ):
        raise ValueError("Reference checkout is not at the pre-speedup SHA")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=reference,
        text=True,
    ).strip():
        raise ValueError("Reference checkout has modifications")
    output.mkdir(parents=True, exist_ok=False)
    manifest = load(root / "design.json")
    cases = select_cases(manifest)
    results = []
    for index, (cid, j, u, mix, kind) in enumerate(cases):
        rows = list(csv_rows(plan_path(root, cid, "A")))
        row = rows[18 + index]
        for rep in range(
            1 + (args.extra_repetitions if j in (2, 6, 8) and mix != "inference" else 0)
        ):
            label = f"p100doe_parity_{output.name}_{index}_{rep}"
            outputs = {}
            for fast in (False, True) if (index + rep) % 2 == 0 else (True, False):
                outputs[fast] = run_one(
                    ROOT if fast else reference,
                    root,
                    manifest,
                    row,
                    output,
                    label + ("_fast" if fast else "_ref"),
                    fast,
                )
            ref, fast = outputs[False], outputs[True]
            passed = (
                ref["signatures"] == fast["signatures"]
                and ref["compact"] == fast["compact"]
            )
            record = dict(
                context=cid,
                j=j,
                u=u,
                mix=mix,
                variant=kind,
                repetition=rep,
                plan_row_id=row["plan_row_id"],
                fast_seconds=fast["seconds"],
                reference_seconds=ref["seconds"],
                speedup=ref["seconds"] / fast["seconds"],
                scientific_exact=passed,
                fast_hashes=fast["signatures"],
                reference_hashes=ref["signatures"],
                compact_exact=ref["compact"] == fast["compact"],
                fast_output_bytes=fast["output_bytes"],
            )
            results.append(record)
            json_write(
                output / "parity.json",
                {
                    "reference": REFERENCE,
                    "pairs": results,
                    "median_speedup": statistics.median(
                        result["speedup"] for result in results
                    ),
                },
                immutable=False,
            )
            print(cid, rep, passed, f"{record['speedup']:.2f}x", flush=True)
            if not passed:
                raise ValueError(f"Scientific parity mismatch: {cid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
