# P100 Model V3 DOE SCC runbook

This branch collects the revised 326,400-row P100 dataset. Start from the
supervisor handoff copy of this runbook, which supplies the reviewed commit.
Use Bash in a dedicated checkout and an SCC-approved
allocation for all simulator commands. Choose the project filesystem,
Python module, worker count and job request using your site's approved
configuration. These commands do not submit scheduler jobs.

## Verify and prepare

~~~bash
set -euo pipefail
git clone --branch p100-v3-generalization-doe https://github.com/amenon871/FlexDC.git FlexDC-P100-DOE
cd FlexDC-P100-DOE
# For an existing dedicated checkout, fetch the same branch from origin.
REVIEWED_SHA='@REVIEWED_SHA@'
EXPECTED_STAGE_A='0748532090fecde23162127386c9883979e0a4a7c5553c5ba3c975620c6461f0'
EXPECTED_PILOT_A='f9df88bcf1aff69ff6f9a1486b63bfbd3d67045082afc8920db80d2a16cefebc'
git fetch origin p100-v3-generalization-doe
git switch --detach "$REVIEWED_SHA"
git rev-parse HEAD
test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"

python3.12 -m venv .p100-doe-venv
source .p100-doe-venv/bin/activate
python -m pip install -r requirements-p100-v3-doe.txt
export PYTHONPATH="$PWD:$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

DOE=runs/p100_v3_doe_matched
PILOT=P100D_J2_011_V0_U6
python -u -B -m p100_v3_doe generate
python -B -m p100_v3_doe audit-design
python -B -m p100_v3_doe audit-matched
printf '%s  %s\n' "$EXPECTED_STAGE_A" "$DOE/stage_a.csv" | sha256sum -c -
printf '%s  %s\n' "$EXPECTED_PILOT_A" "$DOE/plans/A/$PILOT.csv" | sha256sum -c -
test "$(wc -l < "$DOE/stage_a.csv")" -eq 244801
test "$(wc -l < "$DOE/plans/A/$PILOT.csv")" -eq 151
python -B -c 'from peacsim.aqa_runtimepolicy import AQARuntimePolicy as A; assert A.__dict__.get("supports_deferred_realtime_qos") is True'
~~~

The default root uses the revised scientific identity. Keep the old
runs/p100_v3_doe_release directory separate. Its observations cannot be imported
into this plan. Run generation on the same default root for byte-level
reproducibility. The aggregate includes root-relative config paths and its
checksum assumes the default root. Require 741 matched workloads, 1,041
utilization pairs, 156,150 matched bid tuples, and zero difference or mismatch.
Stop on any preflight failure.

## Mandatory 150+50 pilot

Run inside the approved pilot allocation. Set WORKERS to a bounded count
appropriate for the slot and memory limits (1–4 workers for the four pilot
chunks). Run only this exact context.
The pilot uses the production ledger so its 200 valid rows can be reused.

~~~bash
: "${WORKERS:?Set WORKERS to the measured allocation-safe worker count}"
test "$WORKERS" -ge 1
test "$WORKERS" -le 4
mkdir -p "$DOE/pilot_evidence"
PILOT_EVIDENCE="$(mktemp -d "$DOE/pilot_evidence/attempt.XXXXXXXX")"
printf 'Pilot evidence directory: %s\n' "$PILOT_EVIDENCE"
RUN_A=(python -u -B -m p100_v3_doe run --stage A --case "$PILOT" --workers "$WORKERS" --chunks 4 --batch-contexts 1 --execute)
RUN_B=(python -u -B -m p100_v3_doe run --stage B --case "$PILOT" --workers "$WORKERS" --chunks 4 --batch-contexts 1 --execute)

python -B -m p100_v3_doe run --stage A --case "$PILOT" --workers "$WORKERS" --chunks 4 --dry-run
python -u -B -m p100_v3_doe.telemetry --output "$PILOT_EVIDENCE/stage_a_resources.json" --valid-rows 150 -- "${RUN_A[@]}" \
    > "$PILOT_EVIDENCE/stage_a.log" 2>&1
python -B -m p100_v3_doe audit-stage --stage A --case "$PILOT" \
    | tee "$PILOT_EVIDENCE/stage_a_audit.json"
python -u -B -m p100_v3_doe stage-b --case "$PILOT" \
    | tee "$PILOT_EVIDENCE/stage_b_planning.log"
python -B -c 'import json,sys; r=json.load(open(sys.argv[1])); assert (r["ready_contexts"],r["plan_rows"],r["incomplete"])==(1,50,[])' \
    "$DOE/stage_b_progress.json"
cp "$DOE/plans/B/$PILOT.json" "$PILOT_EVIDENCE/stage_b_plan_receipt.json"
python -u -B -m p100_v3_doe.telemetry --output "$PILOT_EVIDENCE/stage_b_resources.json" --valid-rows 50 -- "${RUN_B[@]}" \
    > "$PILOT_EVIDENCE/stage_b.log" 2>&1
python -B -m p100_v3_doe pilot-report \
    | tee "$PILOT_EVIDENCE/pilot_scientific_report.json"
~~~

The Stage A audit must show 150 unique, legal, fully reconciled rows and a
single frozen arrival digest. The Stage B receipt seals the Stage A plan and
observation digests and records measured-anchor versus fallback provenance.
The pilot report re-audits both stages, including correctness of applied
coordinates, mixed QoS where applicable, tracking, cost and the fast-path
marker. Its scientific PASS requires 150+50, a common digest across all 200
bids, and at least one Stage B result satisfying abs(m) ≤ .25, where
m=max(p90/.30,max(Pj)/.10)−1 uses measured output. The report includes both
stage near-boundary counts, feasible counts and closest absolute margins.

If the scientific report fails or any simulator/output audit fails, preserve
the logs and stop. In particular, zero observed Stage B boundary hits is a
failed pilot, even with 200 valid rows. Do not change the threshold, seed or
bids. Review the evidence before deciding on another release.

## Verify no-op resume and resources

~~~bash
python -B -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("select stage,count(*) from observations group by stage order by stage").fetchall()); c.close()' \
    "$DOE/observations.sqlite" > "$PILOT_EVIDENCE/ledger_before.txt"
find "$DOE/attempts" -name receipt.json -print | sort > "$PILOT_EVIDENCE/attempts_before.txt"
"${RUN_A[@]}" > "$PILOT_EVIDENCE/resume_a.json"
"${RUN_B[@]}" > "$PILOT_EVIDENCE/resume_b.json"
python -B -c 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute("select stage,count(*) from observations group by stage order by stage").fetchall()); c.close()' \
    "$DOE/observations.sqlite" > "$PILOT_EVIDENCE/ledger_after.txt"
find "$DOE/attempts" -name receipt.json -print | sort > "$PILOT_EVIDENCE/attempts_after.txt"
cmp "$PILOT_EVIDENCE/ledger_before.txt" "$PILOT_EVIDENCE/ledger_after.txt"
cmp "$PILOT_EVIDENCE/attempts_before.txt" "$PILOT_EVIDENCE/attempts_after.txt"
python -B -c 'import json,sys; reports=[json.load(open(p)) for p in sys.argv[1:]]; assert all(r["status"]=="NO_PENDING_ROWS" and r["scheduled"]==0 for r in reports)' \
    "$PILOT_EVIDENCE/resume_a.json" "$PILOT_EVIDENCE/resume_b.json"
python -B -c 'import json,sys; reports=[json.load(open(p)) for p in sys.argv[1:]]; assert all(r["exit_code"]==0 and r["worker_peaks"] and r["peak_concurrent_workers"]>0 for r in reports); print(json.dumps(reports,indent=2))' \
    "$PILOT_EVIDENCE/stage_a_resources.json" "$PILOT_EVIDENCE/stage_b_resources.json"
~~~

Save the measured wall time, audited rows/hour, requested workers, per-worker
sampled peak RSS, sampled CPU and sampled peak/final allocated storage from
these JSON files. The sampler checks Linux wizard descendants every 0.5 s and
storage every 5 s. Shorter peaks can be missed. Rows/hour uses newly accepted
pilot rows from the stage ledger divided by elapsed time. Keep earlier
measurements if you had to resume a partially completed pilot stage. Retain all runner
logs, launch manifests, attempt receipts, summaries and ingestion records.
The operator records failed and retried chunks in the pilot notes
using the launch manifests (status and return code) and attempt plans
(rescheduled IDs). Inspect worker logs for errors/warnings and record any
interrupted attempts. Work interrupted before a terminal record may have an
unknown executed-row count. Raw artifact storage is scoped to the dedicated
DOE root and worker directories. It includes the frozen plans.

## Explicit operator promotion and remaining collection

The mandatory gate requires correct reviewed Git SHA and plan hashes, design
and matched audits, 150+50 valid rows, one common arrival digest, at least one
measured Stage B boundary hit, zero scheduled work on both resumes, and actual
SCC resource/throughput evidence. Review and record all ten results before
manually running the following commands. The pilot does not launch production.
In a new allocation, re-enter the same checkout, reactivate its Python
environment and restore the setup variables. Set PILOT_EVIDENCE to the
verified pilot attempt directory printed by your shell, and WORKERS to the
production allocation's approved count.

~~~bash
test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"
cmp "$PILOT_EVIDENCE/ledger_before.txt" "$PILOT_EVIDENCE/ledger_after.txt"
cmp "$PILOT_EVIDENCE/attempts_before.txt" "$PILOT_EVIDENCE/attempts_after.txt"
python -B - "$PILOT_EVIDENCE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
def read(name):
    return json.loads((p / name).read_text())
r = read("pilot_scientific_report.json")
assert r["status"] == "PASS" and r["total_rows"] == 200
assert r["common_arrivals"] and r["stage_b_boundary_pass"]
for stage in ("a", "b"):
    resume = read(f"resume_{stage}.json")
    assert resume["status"] == "NO_PENDING_ROWS" and resume["scheduled"] == 0
    resources = read(f"stage_{stage}_resources.json")
    assert resources["exit_code"] == 0 and resources["worker_peaks"]
PY
# Set WORKERS to the separately approved production allocation's worker count.
python -u -B -m p100_v3_doe run --stage A --workers "$WORKERS" --chunks 4 --batch-contexts 24 --execute
python -u -B -m p100_v3_doe audit-stage --stage A > "$DOE/full_stage_a_audit.log"
python -u -B -m p100_v3_doe stage-b
python -B -c 'import json,sys; r=json.load(open(sys.argv[1])); assert (r["ready_contexts"],r["plan_rows"],r["incomplete"])==(1632,81600,[])' \
    "$DOE/stage_b_progress.json"
python -u -B -m p100_v3_doe run --stage B --workers "$WORKERS" --chunks 4 --batch-contexts 24 --execute
python -u -B -m p100_v3_doe audit-stage --stage B > "$DOE/full_stage_b_audit.log"
python -B -m p100_v3_doe audit-results --require-complete > "$DOE/final_audit.log"
~~~

Resume with the same command, root and runtime. Complete pilot A/B rows are
skipped and its immutable B plan is validated against its A lineage. This
leaves 244,650 Stage A and 81,550 Stage B rows if the pilot is the only completed
collection. A hard termination may leave operation.lock. Verify your SCC
allocation and all of its runner and wizard workers have exited before moving
only that stale lock directory aside inside the DOE root. Leave output CSVs,
receipts, evidence JSONL, plans and SQLite files intact. Genuine missing rows
can be rescheduled from the same immutable plan.

Final output is observations.csv (326,400 audited rows), collection_audit.json,
design.json, preflight.json, matched_stage_a.json, the 22 split files and
evaluation_reporting.json. Join the split context IDs to Context_ID. Stage A
utilization matching uses workload_id and stage_a_sample_index. See
EVALUATION.md for the leave-one, leave-two and utilization reporting contract.
