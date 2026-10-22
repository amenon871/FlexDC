# Foundation V1 per-row SCC storage-safety patch

Result: PASS. No Foundation scientific simulations were run during this patch.

Baseline: dd4320ebbd8213570cd630a360cd37a515a69cae.
Branch: foundation-dataset-v1-prep. This is an execution/storage-only revision.

## Exact lifecycle and preservation

The Foundation guards inside objective_function call begin_row BEFORE creating
base_weights or running FlexDC. The existing simulator, metric extraction,
results writer and diagnostics writer retain their historical code. Immediately
after those writers close: sync_compact_csvs flushes/fsyncs both CSVs;
write_evidence writes the original atomic/checksummed Foundation receipt;
finalize_row verifies the receipt checksum, exactly one matching Plan_Row_ID row
in EACH CSV and exact serialized values, syncs the compact files/directories,
then unlinks the explicit transient files and verifies they are absent.
Only after CLEANUP_COMPLETE can that worker start its next candidate.

TRANSIENT ALLOWLIST (exact names, no wildcards or recursive removal):
- power_trace.csv
- job_table.csv
- PRtable.csv
- base_weights.csv
- waiting_jobs_by_job_type.csv
- probabilistic_results.csv

The last two are optional per-second QoS arrays written by Simulator.final_output
under availability_qos. Existing QoS-cost consumers run before this hook;
Foundation ingest, resume, adaptive planning and combo output validation use
compact CSVs/receipts, not these arrays. Normal Foundation AQA may not create them.
Small qos_summary.csv, tracking_error_90th.csv, cost_power_rsr.csv, QoSDegAvg.csv
are retained, as are all logs, compact CSVs, evidence, plans, chunk metadata,
ledgers, manifests, receipts and source/config hashes. Nonregular/symlink or
escaping deletion paths are refused. No broad deletion is implemented.

## Failure and resume contract

Scientific and cleanup statuses are independent. EXECUTION_COMPLETE is terminal;
CLEANUP_REQUIRED/CLEANUP_COMPLETE describe only storage. The cleanup finalizer
has three attempts per file, with 0.05s and 0.10s backoff. Failure records exact
path/error/attempt in foundation_cleanup/F1_*.json, signals the bounded combo
pool and stops queued launches/next rows. Already running rows can finish and
persist/clean before stopping. No global worker-serialization lock is added.
A fallback Foundation worker-failure signal also prevents queued launches.

Campaign resume ingests scientific receipts FIRST and only then recovers cleanup.
Terminal row IDs never return to missing scientific work. Recovery also handles
a crash after evidence persistence but before its cleanup-state update. A
cleanup-pending row cannot advance the stage, even if every scientific receipt
is already present. Full campaign status becomes TRANSIENT_CLEANUP_REQUIRED.
True unreceipted simulator/extraction failures preserve transient debugging
files and remain technically retryable in a fresh worker directory. Direct
chunk reuse refuses to overwrite existing Foundation compact evidence; use
campaign --resume. Evidence-insufficient terminal rows are cleaned identically,
remain excluded from adaptive anchors/training labels, and retain the original
scientific shortfall stop. No replacement scientific coordinates are invented.

## Telemetry and scaling

Per-row cleanup receipts track files/bytes deleted (stat immediately before
unlink), deletion failures and retries. Scoped transient_cleanup_status.json,
stage status and full completion metrics expose cumulative counters and pending
cleanup. No per-row recursive size scan. Power loss between unlink and the
telemetry update can undercount bytes; it cannot trigger scientific reexecution.
The parallel mock completed 20 rows using four workers: peak power_trace count
was four, then zero, rather than growing with completed rows.

## Verification

90 tests passed; 0 failed; 0 skipped; 0 warnings. This retains all 70 previously
approved cases plus 19 storage tests and one full-campaign cleanup-stop test.
Coverage includes all requested A-I cases, the actual exact-plan two-row loop,
real evidence serialization, exact CSV integrity failures, checksum failures,
cleanup-only resume with zero simulator calls for the completed row, fallback
pool stop, non-Foundation isolation and historical wizard AST equality after
removing only the three opt-in Foundation guards.

Full CLI dry runs: ALL, 32 workers and 96 workers, each 14 steps / 500000 rows.
Every stage retains --plan-max-workers. Combo dry run: 150 EXISTING master rows,
six chunks; no workers launched. Wizard: first/last 25-row chunks validated with
--validate-sweep-plan-only. No optimization or scientific simulation was run.

Master Phase-A before AND after:
bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f

All frozen scientific/config files in scientific_identity.json match the
approved baseline release hashes. Counts remain 400 profiles, 500 workloads,
2000 contexts, 300000 static +160000 expected adaptive +40000 expected repeat
rows =500000. TRAIN_DEV450000 then isolated sealed LOCKED50000 remain unchanged.
No profile/workload/context, coordinates, seeds, IDs, counts, partitions,
acquisition/repeat logic, cost, ISO, idle power or model-data semantics changed.

One SCC command remains:

```sh
python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 96 --execute --resume
```

96 is illustrative; any valid user worker count propagates unchanged.

The updated preexecution_release.json preserves every old guarded path and adds
new storage source/tests. Strict byte hashes remain; SCC must deploy these
reviewed bytes. Do not reseal modified SCC files to bypass a mismatch.

Validation artifacts: storage_safety_validation/tests.json,
cli_and_identity_gates.json, scientific_identity.json, full_dry_run_workers32.json,
full_dry_run_workers96.json. Local ignored *_log.txt files retain complete logs.
changed_files.json lists every intended file; Git index hashes are checked before
commit. Old review ZIPs, worker outputs, scratch files, SQLite ledgers and logs
are intentionally excluded from the commit.

NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS WERE RUN DURING THIS PATCH TASK.

FOUNDATION TRANSIENT FILES ARE DELETED AFTER EACH SUCCESSFULLY PERSISTED ROW,
BEFORE THAT WORKER PROCEEDS TO ITS NEXT ROW.
