"""Linux pilot resource sampler. Runs the supplied command with unchanged CLI arguments."""

import argparse
import os
import sqlite3
import subprocess
import time
from pathlib import Path

from .collection import PILOT_CONTEXT
from .io import ROOT, json_write


def workers_at_depth_two(lines, launcher):
    """The DOE CLI's children are combo runners, whose children are wizard workers."""
    parsed = []
    for line in lines:
        pid, parent, rss, cpu = line.split()
        parsed.append((int(pid), int(parent), int(rss), float(cpu)))
    runners = {pid for pid, parent, _, _ in parsed if parent == launcher}
    return [(pid, rss, cpu) for pid, parent, rss, cpu in parsed if parent in runners]


def allocated_bytes(root):
    paths = [root]
    for folder in (
        "src/peacsim/output/optimization",
        ".flexdc_plan_chunks",
        ".flexdc_worker_logs",
    ):
        paths.extend(
            (ROOT / folder).glob("*p100doe_*")
            if folder.endswith("worker_logs")
            else (ROOT / folder).glob("p100doe_*")
        )
    size = 0
    for path in paths:
        for directory, _, files in os.walk(path):
            for name in files:
                try:
                    size += (Path(directory) / name).stat().st_blocks * 512
                except FileNotFoundError:
                    pass
    return size


def valid_rows(root, stage):
    database = root / "observations.sqlite"
    if not database.exists():
        return 0
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return connection.execute(
            "SELECT count(*) FROM observations WHERE context_id=? AND stage=?",
            (PILOT_CONTEXT, stage),
        ).fetchone()[0]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/p100_v3_doe_matched")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--valid-rows",
        type=int,
        choices=(150, 50),
        required=True,
        help="Expected pilot size (150 for A, 50 for B).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not (Path("/proc").is_dir() and os.name == "posix"):
        parser.error("A command and a Linux SCC execution host are required")
    if args.output.exists():
        parser.error("Choose a new telemetry path for each attempt")
    initial = allocated_bytes(args.root)
    stage = "A" if args.valid_rows == 150 else "B"
    before = valid_rows(args.root, stage)
    started = time.monotonic()
    worker_peaks = {}
    concurrent_peak = samples = 0
    storage_peak = initial
    next_storage = started
    env = {**os.environ, "LC_ALL": "C"}
    with subprocess.Popen(command, cwd=ROOT) as process:
        while process.poll() is None:
            lines = subprocess.check_output(
                ["ps", "-A", "-o", "pid=,ppid=,rss=,pcpu="], text=True, env=env
            ).splitlines()
            workers = workers_at_depth_two(lines, process.pid)
            concurrent_peak = max(concurrent_peak, len(workers))
            for pid, rss, cpu in workers:
                peak = worker_peaks.setdefault(pid, {"rss_kib": 0, "cpu_percent": 0.0})
                peak["rss_kib"] = max(peak["rss_kib"], rss)
                peak["cpu_percent"] = max(peak["cpu_percent"], cpu)
            now = time.monotonic()
            if now >= next_storage:
                storage_peak = max(storage_peak, allocated_bytes(args.root))
                next_storage = now + 5
            samples += 1
            time.sleep(0.5)
        code = process.returncode
    elapsed = time.monotonic() - started
    final = allocated_bytes(args.root)
    after = valid_rows(args.root, stage)
    if not 0 <= before <= after <= args.valid_rows:
        raise ValueError("Unexpected pilot ledger counts before and after collection")
    report = dict(
        command=command,
        exit_code=code,
        elapsed_seconds=elapsed,
        samples=samples,
        expected_valid_rows=args.valid_rows,
        previous_valid_rows=before,
        total_valid_rows=after,
        new_valid_rows=after - before,
        completed_rows_per_hour=(after - before) * 3600 / elapsed,
        peak_concurrent_workers=concurrent_peak,
        worker_peaks=worker_peaks,
        storage_initial_bytes=initial,
        storage_peak_sampled_bytes=max(storage_peak, final),
        storage_final_bytes=final,
        resource_sample_interval_seconds=0.5,
        storage_sample_interval_seconds=5,
        cpu_measurement="ps sampled process lifetime average percent",
        scope="dedicated checkout DOE root and worker directories",
    )
    json_write(args.output, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
