"""DOE orchestration using the unchanged exact-plan runner and wizard."""

import csv
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .audit import audit_design, audit_plan, require
from .design import A_COUNT, B_COUNT, CLUSTER, generate
from .io import (
    ROOT,
    csv_rows,
    csv_write,
    digest,
    encode,
    json_write,
    load,
    protected,
    sha,
)
from .ledger import measured, reconcile
from .sampling import FIELDS, adaptive, broad, domain_for, plan_rows

RUNNER = ROOT / "am_run_flexdc_dataset_parallel_combo.py"
OUTPUT = ROOT / "src/peacsim/output/optimization"
TRANSIENTS = (
    "power_trace.csv",
    "job_table.csv",
    "PRtable.csv",
    "base_weights.csv",
    "waiting_jobs_by_job_type.csv",
    "probabilistic_results.csv",
)
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
RUNTIME_PATHS = [
    RUNNER,
    ROOT / "am_generate_flexdc_exact_sweep_plan.py",
    ROOT / "am_generate_flexdc_j2_pair_workloads.py",
    ROOT / "am_vet_flexdc_sweep_dataset_v3.py",
    *sorted((ROOT / "src/peacsim").glob("*.py")),
    *sorted((ROOT / "src/peacsim/parsing").glob("*.py")),
    *sorted((ROOT / "p100_v3_doe").glob("*.py")),
]


@contextmanager
def exclusive(root):
    lock = root / "operation.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"DOE operation locked: {lock}. Verify no surviving workers before removing a stale lock."
        ) from exc
    interrupted = False
    try:
        import socket

        json_write(
            lock / "owner.json", {"host": socket.gethostname(), "pid": os.getpid()}
        )
        yield
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if not interrupted:
            (lock / "owner.json").unlink(missing_ok=True)
            lock.rmdir()


def plan_path(root, cid, stage):
    return root / "plans" / stage / (cid + ".csv")


def seal_plan(root, cid, stage, rows, source=None, diagnostics=None, design_hash=None):
    path = plan_path(root, cid, stage)
    csv_write(path, rows, FIELDS)
    json_write(
        path.with_suffix(".json"),
        dict(
            context_id=cid,
            stage=stage,
            rows=A_COUNT if stage == "A" else B_COUNT,
            plan_sha256=sha(path),
            source=source,
            diagnostics=diagnostics,
            design_sha256=design_hash or sha(root / "design.json"),
        ),
    )
    return path


def read_plan(root, manifest, cid, stage, design_hash=None):
    path = plan_path(root, cid, stage)
    rows = list(csv_rows(path))
    receipt = load(path.with_suffix(".json"))
    require(
        receipt["context_id"] == cid
        and receipt["stage"] == stage
        and receipt["rows"] == len(rows)
        and receipt["plan_sha256"] == sha(path)
        and receipt["design_sha256"] == (design_hash or sha(root / "design.json")),
        f"Plan seal mismatch {cid}/{stage}",
    )
    return rows, receipt


def generate_all(root):
    protected()
    manifest = generate(root)
    report = audit_design(root, manifest)
    design_hash = sha(root / "design.json")
    seen = set()
    for index, (cid, context) in enumerate(sorted(manifest["contexts"].items()), 1):
        domain = domain_for(manifest, context)
        seal_plan(
            root,
            cid,
            "A",
            plan_rows(root, manifest, cid, "A", broad(domain, cid)),
            design_hash=design_hash,
        )
        rows, _ = read_plan(root, manifest, cid, "A", design_hash)
        audit_plan(root, manifest, cid, "A", rows)
        ids = {row["plan_row_id"] for row in rows}
        require(not seen & ids, f"Cross-context Plan_Row_ID collision {cid}")
        seen.update(ids)
        if index % 200 == 0:
            print(
                f"Stage A planned and audited: {index}/{len(manifest['contexts'])}",
                flush=True,
            )
    require(len(seen) == 244800, "Stage A budget mismatch")
    aggregate = root / "stage_a.csv"
    if not aggregate.exists():
        temporary = root / "stage_a.pending"
        with temporary.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
            writer.writeheader()
            for cid in sorted(manifest["contexts"]):
                writer.writerows(csv_rows(plan_path(root, cid, "A")))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, aggregate)
    # The aggregate is portable evidence. Execution uses independently sealed context files.
    count = 0
    aggregate_rows = csv_rows(aggregate)
    for cid in sorted(manifest["contexts"]):
        for expected in csv_rows(plan_path(root, cid, "A")):
            row = next(aggregate_rows, None)
            require(
                row == expected and row["plan_row_id"] in seen,
                "Aggregate plan differs from sealed context plan",
            )
            seen.remove(row["plan_row_id"])
            count += 1
    require(next(aggregate_rows, None) is None, "Unexpected extra aggregate rows")
    require(count == 244800 and not seen, "Incomplete aggregate Stage A plan")
    report["stage_a_sha256"] = sha(aggregate)
    report["design_sha256"] = design_hash
    report["protected_sha256"] = protected()
    json_write(root / "preflight.json", report)
    return report


def choose_contexts(root, manifest, selectors, model, partition):
    universe = set(manifest["contexts"])
    if model is not None:
        require(model in manifest["models"], f"Unknown model {model}")
        split = load(root / "splits" / (model + ".json"))
        require(
            sha(root / "splits" / (model + ".json")) == manifest["split_hashes"][model],
            "Split seal mismatch",
        )
        if partition in ("train", "validation"):
            universe = set(split[partition])
        else:
            universe = set().union(*(set(value) for value in split["test"].values()))
    if selectors:
        selected = set()
        for item in selectors:
            found = {
                cid
                for cid, context in manifest["contexts"].items()
                if cid == item or context["workload"] == item
            }
            require(bool(found), f"Unknown workload/context {item}")
            selected.update(found)
        universe &= selected
    require(bool(universe), "No contexts selected")
    return sorted(universe)


def prepare_b(root, manifest, connection, contexts):
    ready, incomplete = [], []
    design_hash = sha(root / "design.json")
    started = time.monotonic()
    for index, cid in enumerate(contexts):
        if index % 50 == 0:
            print(
                f"Stage B planning: {index}/{len(contexts)} examined, "
                f"{len(ready)} ready, elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        a_rows, receipt = read_plan(root, manifest, cid, "A", design_hash)
        observed = measured(connection, cid, "A")
        if len(observed) != A_COUNT:
            incomplete.append(
                dict(
                    context_id=cid, valid=len(observed), missing=A_COUNT - len(observed)
                )
            )
            continue
        require(
            {r["plan_row_id"] for r in a_rows} == observed.keys(),
            f"Stage A evidence ID mismatch {cid}",
        )
        require(
            all(observed[row["plan_row_id"]]["planned"] == row for row in a_rows),
            f"Stage A observation inputs differ {cid}",
        )
        source = dict(
            a_plan_sha256=receipt["plan_sha256"], a_observation_sha256=digest(observed)
        )
        path = plan_path(root, cid, "B")
        if path.exists() and path.with_suffix(".json").exists():
            rows, frozen = read_plan(root, manifest, cid, "B", design_hash)
            require(
                frozen["source"] == source,
                f"Frozen Stage B Stage A lineage differs {cid}",
            )
        else:
            domain = domain_for(manifest, manifest["contexts"][cid])
            points, diagnostics = adaptive(domain, cid, a_rows, observed)
            seal_plan(
                root,
                cid,
                "B",
                plan_rows(root, manifest, cid, "B", points),
                source,
                diagnostics,
                design_hash,
            )
            rows, _ = read_plan(root, manifest, cid, "B", design_hash)
        audit_plan(root, manifest, cid, "B", rows, a_rows)
        ready.append(cid)
    report = dict(
        ready_contexts=len(ready), plan_rows=len(ready) * B_COUNT, incomplete=incomplete
    )
    json_write(root / "stage_b_progress.json", report, immutable=False)
    return report


def runtime_identity():
    from importlib.metadata import version

    return dict(
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_PATHS},
        python=list(sys.version_info[:3]),
        packages={p: version(p) for p in ("numpy", "pandas", "scipy", "tqdm")},
    )


def worker_env():
    environment = dict(
        os.environ, FLEXDC_P100_DOE_EVIDENCE="1", PYTHONDONTWRITEBYTECODE="1"
    )
    for key in (*THREAD_KEYS, "FOUNDATION_V1_SIMPLE_AREA"):
        if key == "FOUNDATION_V1_SIMPLE_AREA":
            environment.pop(key, None)
        else:
            environment[key] = "1"
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    return environment


def attempts(root):
    return sorted((root / "attempts").glob("*/receipt.json"))


def collect_attempts(root, manifest, connection):
    errors, counters = [], Counter()
    execution_source = root / "execution_source.json"
    for path in attempts(root):
        attempt = load(path)
        if attempt["dry_run"] or (path.parent / "ingested.json").exists():
            continue
        filename = path.parent / "plan.csv"
        design_root = root.parent if root.name == "smoke" else root
        require(
            sha(filename) == attempt["plan_sha256"]
            and attempt["design_sha256"] == sha(design_root / "design.json")
            and digest(load(execution_source)) == attempt["runtime_sha256"],
            f"Attempt seal mismatch {path}",
        )
        by_context = {}
        attempt_counters, attempt_errors = Counter(), []
        for row in csv_rows(filename):
            by_context.setdefault(row["context_id"], []).append(row)
        prefix = attempt["run_id"] + "_AQA_"
        for cid, rows in by_context.items():
            # Same grouping, sort and split as the combo runner.
            rows.sort(key=lambda row: row["plan_row_id"])
            n = min(len(rows), attempt["chunks"])
            offset = 0
            for index in range(n):
                size = len(rows) // n + (index < len(rows) % n)
                expected = {
                    row["plan_row_id"]: row for row in rows[offset : offset + size]
                }
                offset += size
                label = f"{prefix}{cid}_prchunk{index + 1:02d}of{n:02d}"
                matches = list(OUTPUT.glob(label + "_*"))
                require(len(matches) <= 1, f"Ambiguous worker output for {label}")
                if not matches:
                    continue
                one = reconcile(
                    connection,
                    matches[0],
                    expected,
                    manifest,
                    {
                        "attempt": attempt["run_id"],
                        "git": attempt["git"],
                        "plan_sha256": attempt["plan_sha256"],
                        "design_sha256": attempt["design_sha256"],
                        "runtime_sha256": attempt["runtime_sha256"],
                        "worker_output_path": matches[0].relative_to(ROOT).as_posix(),
                        "output_mtime_ns": matches[0].stat().st_mtime_ns,
                    },
                )
                attempt_errors.extend(one.pop("invalid"))
                attempt_counters.update(one)
                if root.name != "smoke" and one["committed"] == len(expected):
                    require(
                        matches[0].resolve().parent == OUTPUT.resolve()
                        and not matches[0].is_symlink(),
                        "Unexpected worker directory for compact retention",
                    )
                    for name in TRANSIENTS:
                        (matches[0] / name).unlink(missing_ok=True)
        json_write(
            path.parent / "ingested.json",
            dict(counters=dict(attempt_counters), errors=attempt_errors),
        )
        errors.extend(attempt_errors)
        counters.update(attempt_counters)
    return dict(counters=dict(counters), errors=errors)


def run(
    root, manifest, connection, stage, contexts, workers, chunks, dry_run, smoke=False
):
    protected()
    execution = root / "smoke" if smoke else root
    runtime = runtime_identity()
    json_write(execution / "execution_source.json", runtime)
    if not dry_run:
        collect_attempts(execution, manifest, connection)
    pending = []
    design_hash = sha(root / "design.json")
    for cid in contexts:
        path = plan_path(root, cid, stage)
        if stage == "B" and not path.with_suffix(".json").exists():
            continue
        rows, receipt = read_plan(root, manifest, cid, stage, design_hash)
        previous = (
            read_plan(root, manifest, cid, "A", design_hash)[0] if stage == "B" else []
        )
        audit_plan(root, manifest, cid, stage, rows, previous)
        if stage == "B":
            require(
                receipt["source"]
                == dict(
                    a_plan_sha256=sha(plan_path(root, cid, "A")),
                    a_observation_sha256=digest(measured(connection, cid, "A")),
                ),
                f"Stage B predecessor differs {cid}",
            )
        completed = measured(connection, cid, stage)
        pending.extend(
            row
            for row in (rows[:2] if smoke else rows)
            if row["plan_row_id"] not in completed
        )
    if not pending:
        return dict(status="NO_PENDING_ROWS", scheduled=0)
    require(0 < workers <= 128 and 0 < chunks <= 150, "Invalid worker/chunk limit")
    # Leave a distinct launch namespace per invocation, especially after interruption.
    run_id = (
        "p100doe_"
        + digest([root.relative_to(ROOT).as_posix(), design_hash])[:10]
        + "_"
        + stage.lower()
        + "_"
        + uuid.uuid4().hex[:10]
    )
    location = execution / "attempts" / run_id
    plan = location / "plan.csv"
    csv_write(plan, pending, FIELDS)
    receipt = dict(
        run_id=run_id,
        chunks=chunks,
        workers=workers,
        dry_run=dry_run,
        stage=stage,
        rows=len(pending),
        plan_sha256=sha(plan),
        design_sha256=design_hash,
        runtime_sha256=digest(runtime),
        git=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    json_write(location / "receipt.json", receipt)
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--sweep-plan-file",
        str(plan),
        "--plan-num-chunks-per-context",
        str(chunks),
        "--plan-max-workers",
        str(workers),
        "--plan-run-id",
        run_id,
        "--output-prefix",
        run_id,
        "--gradient-config",
        str(root / manifest["gradient"]),
        "--cluster-config",
        str(CLUSTER),
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
    if dry_run:
        command.append("--plan-dry-run")
    print(f"{run_id}: {len(pending)} planned rows, {workers} workers", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=worker_env())
    json_write(location / "runner_exit.json", {"returncode": result.returncode})
    harvested = collect_attempts(execution, manifest, connection) if not dry_run else {}
    report = dict(
        status="DRY_RUN" if dry_run else "COMPLETE",
        scheduled=len(pending),
        harvested=harvested,
        returncode=result.returncode,
        run_id=run_id,
    )
    if result.returncode or (
        not dry_run
        and (
            harvested["errors"]
            or harvested["counters"].get("committed", 0) != len(pending)
        )
    ):
        report["status"] = "INCOMPLETE_OR_INVALID"
    json_write(location / "summary.json", report)
    return report


def export(root, manifest, connection, complete):
    counts = Counter()
    near = Counter()
    coverage = {}
    missing = []
    design_hash = sha(root / "design.json")
    destination = root / "observations.csv"
    with tempfile.NamedTemporaryFile(
        dir=root, delete=False, mode="w", newline="", encoding="utf8"
    ) as stream:
        temporary = Path(stream.name)
        columns = (
            "Plan_Row_ID",
            "Context_ID",
            "workload_id",
            "parent_group",
            "N",
            "U",
            "J",
            "stage",
            "Pbar_kw_per_server",
            "R_kw_per_server",
            "weights",
            "mean_tracking",
            "p90_tracking",
            "qos_delay_probabilities",
            "feasible",
            "violation_distance",
            "objective",
            "power_cost",
            "tracking_cost",
            "tracking_constraint_cost",
            "qos_cost",
            "arrival_sha256",
            "job_names",
            "source_profiles",
            "per_job_physical_qos",
            "qos_observation_counts",
            "candidate_source",
            "source_provenance",
        )
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for cid, context in manifest["contexts"].items():
            w = manifest["workloads"][context["workload"]]
            for stage, expected in (("A", A_COUNT), ("B", B_COUNT)):
                observations = measured(connection, cid, stage)
                if len(observations) != expected:
                    missing.append(
                        dict(
                            context=cid,
                            stage=stage,
                            missing=expected - len(observations),
                        )
                    )
                if not observations:
                    continue
                rows, receipt = read_plan(root, manifest, cid, stage, design_hash)
                planned = {r["plan_row_id"]: r for r in rows}
                require(
                    observations.keys() <= planned.keys(),
                    f"Foreign observation IDs for {cid}/{stage}",
                )
                if stage == "B":
                    require(
                        receipt["source"]
                        == dict(
                            a_plan_sha256=sha(plan_path(root, cid, "A")),
                            a_observation_sha256=digest(measured(connection, cid, "A")),
                        ),
                        f"Stage B lineage mismatch {cid}",
                    )
                context_report = coverage.setdefault(cid, {})
                context_report[stage] = dict(
                    rows=len(observations),
                    near_boundary=sum(
                        abs(v["margin"]) <= 0.25 for v in observations.values()
                    ),
                    feasible=sum(v["feasible"] for v in observations.values()),
                )
                for rid, v in observations.items():
                    row = planned[rid]
                    require(
                        v["planned"] == row,
                        f"Observation row does not match sealed plan {rid}",
                    )
                    counts[stage] += 1
                    near[stage] += abs(v["margin"]) <= 0.25
                    writer.writerow(
                        (
                            rid,
                            cid,
                            context["workload"],
                            w["parent"],
                            context["n"],
                            context["u"],
                            context["j"],
                            stage,
                            row["Pbar_kw_per_server"],
                            row["R_kw_per_server"],
                            row["weights"],
                            v["mean_tracking"],
                            v["p90"],
                            encode(v["qos_probabilities"]),
                            v["feasible"],
                            v["violation_distance"],
                            v["objective"],
                            v["power_cost"],
                            v["tracking_cost"],
                            v["tracking_constraint_cost"],
                            v["qos_cost"],
                            v["evidence"]["arrival_sha256"],
                            encode(w["jobs"]),
                            encode(w["sources"]),
                            v["simulator_results"]["workload_mix"],
                            encode(v["evidence"]["job_counts"]),
                            row["weight_family"],
                            encode(v["provenance"]),
                        )
                    )
        stream.flush()
        os.fsync(stream.fileno())
    require(
        connection.execute("SELECT count(*) FROM observations").fetchone()[0]
        == sum(counts.values()),
        "Unexpected context or stage in observation ledger",
    )
    report = dict(
        status="PASS" if not missing else "INCOMPLETE",
        collected=dict(counts),
        near_boundary=dict(near),
        context_coverage=coverage,
        missing=missing,
        expected=326400,
    )
    json_write(root / "collection_audit.json", report, immutable=False)
    if missing and complete:
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return report
