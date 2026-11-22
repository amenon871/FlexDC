# P100 Model V3 Generalization DOE

This package prepares and collects a 326,400-row FlexDC dataset. It builds the
104-profile library, 741 workload definitions, 1,632 contexts and 22 frozen
model-data splits. Stage A has 244,800 rows. Stage B adds 81,600 rows, using only
the audited Stage A observations for each context.

## SCC operator runbook

Use a dedicated checkout of the reviewed commit from the supervisor report.
Commands below assume Bash from the repository root, a site-approved Python
3.12 interpreter, and a writable checkout on the selected project filesystem.
Run the smoke and production commands inside your approved SCC job allocation.
Set WORKERS to the worker count permitted by that allocation and memory budget.
The workflow does not submit scheduler jobs.

~~~bash
git fetch origin p100-v3-generalization-doe
git switch --detach <reviewed-commit-sha>
git rev-parse HEAD
python3.12 -m venv .p100-doe-venv
source .p100-doe-venv/bin/activate
python -m pip install -r requirements-p100-v3-doe.txt
export PYTHONPATH="$PWD:$PWD/src"
python -B -m p100_v3_doe generate
python -B -m p100_v3_doe audit-design
~~~

The default root is runs/p100_v3_doe_release. For an isolated second campaign use the
global flag --root runs/p100_v3_doe_NAME before the action on every invocation.
Directory names enter the plan's relative config paths, so keep the chosen
relative root fixed through execution and resume. Run generation twice if you
want to verify all existing frozen artifact bytes by regeneration. A mismatch
stops generation. The reference plan's checksums assume the default root.

~~~bash
# Show the exact runner invocations for a bounded subset. This runs zero simulations.
python -B -m p100_v3_doe smoke --case P100D_J2_011_V0_U6 --chunks 2 --workers 2 --dry-run
# Two one-hour simulations with distinct bids and a shared arrival oracle.
python -B -m p100_v3_doe smoke --case P100D_J2_011_V0_U6 --chunks 2 --workers 2
# Add repeated --case selectors or omit them for the default one context per J.
# An immediate rerun of the successful smoke returns NO_PENDING_ROWS.
~~~

The smoke ledger lives in runs/p100_v3_doe_release/smoke. A smoke PASS checks the
lower-case AQA path, order of job types, QoS evidence, applied bid/weights,
objective reconstruction and common-arrival digest. Preserve its compact
artifacts, logs and per-attempt summary.

~~~bash
# Set WORKERS to the tested worker count on the allocated node.
: "${WORKERS:?Set WORKERS to the approved worker count}"
python -u -B -m p100_v3_doe run --stage A --workers "$WORKERS" --chunks 4 --batch-contexts 24 --execute
python -B -m p100_v3_doe audit-results
python -u -B -m p100_v3_doe stage-b
python -u -B -m p100_v3_doe run --stage B --workers "$WORKERS" --chunks 4 --batch-contexts 24 --execute
python -B -m p100_v3_doe audit-results --require-complete
~~~

A run without --execute is a dry run. For staged collection, pass repeated
--case context/workload IDs or --model MODEL --partition train|validation|test
to run and stage-b. Stage B only creates plans for contexts with 150 audited
Stage A observations. Require stage_b_progress.json to report 1,632 ready
contexts, 81,600 rows and no incomplete contexts for full collection. Review
run summaries after each invocation. A preparation run can report partial
readiness without creating plans for missing contexts.

Resume invokes the same run command. The adapter reconciles audited
result/diagnostic/evidence triplets from interrupted attempts, including
successful rows from incomplete chunks. The immutable context plan is reused.
After a hard termination a stale operation.lock can remain. Confirm your
scheduler job and its combo-runner and wizard processes have exited on the
execution host before moving only that lock directory aside within the DOE
root. Run the same command. Keep every attempt receipt, stage receipt, output
CSV, evidence JSONL and SQLite ledger. A changed runtime seal requires a
separate reviewed run or an explicit migration review.

The runner prints progress by chunk. Check the current attempt's receipt,
runner exit status and ingestion summary under attempts/. Raw worker
diagnostics appear under src/peacsim/output/optimization/p100doe_* and runner
logs under .flexdc_worker_logs. Completed, reconciled production chunks have
their allowlisted transient tables removed. Each worker retains its current
and terminal traces during a batch. Retired logs, output CSVs and SQLite
provenance are part of the record. Stop and inspect if a context repeatedly
lacks QoS evidence or reports an arrival mismatch.

Final products:

- observations.csv and collection_audit.json.
- design.json, preflight.json, profiles/, workloads/, experiments/.
- plans/A and plans/B with per-context receipts.
- splits/<model>.json. Join the explicit context IDs to the exported Context_ID
  column and retain both A/B strata. u07_matched contains the three utilization
  cells for each of the same 30 held-out workloads within a baseline fold.
- The original result and diagnostic CSVs retain the historical FlexDC labels.
  The consolidated table names the equivalent canonical quantities and links
  back to the original records through attempted-run provenance.

See the supervisor evidence for local runtimes and storage measurements.
Size the first SCC allocation from its own smoke throughput, storage quota
and peak worker memory. Rolling batches limit large traces to the present
batch plus failed attempts, while compact records and the ledger grow with
the dataset. Keep stage plans and result provenance together when moving
artifacts. The raw budget tracks distinct plan rows. Genuine execution retries
are separately logged and can make attempted process work exceed that count.
