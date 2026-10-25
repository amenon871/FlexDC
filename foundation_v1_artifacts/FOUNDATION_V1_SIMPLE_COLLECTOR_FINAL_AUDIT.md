# Foundation V1 simple collector: final audit

Starting commit: `7eae6d5ac893f14ad41fca6bf829f9f86411db14`.
Branch: `foundation-dataset-v1-prep`. No main merge or history rewrite.
Release source identity and final test/soak numbers are recorded in
`simple_collector_validation/final_gates.json` and `source_guard_changes.json`.
The commit containing this report is the release commit; its SHA is available
from `git log -1` and the final handoff, avoiding a self-referential Git hash.

## Incident: verified defect versus unavailable SCC row

**VERIFIED:** all 300000 original master rows reconstruct correctly from their
original CSV strings. The plans are scientifically intact. Default pandas
C-engine CSV float conversion changes 22 P/R values enough to change their
`round(value,12)` identity-hash inputs. The old combo runner parsed those doubles,
serialized them into chunk CSVs, and ReceiptBackend passed those converted chunk
rows to audit.observation(). Its make_row() check reconstructed different IDs and
raised Geometry/row content identity mismatch. This was a transport/audit
assumption defect, not evidence of an invalid physical candidate.

Example immutable row: `F1_e72484ba822d2dc2993b7f85a945f814`, task
`F1_TRAIN_P100_J4_045__N250__U080`.
Original R=`0.06556568344550005`; default pandas conversion R=`0.0655656834455`.
Original round12=`0.065565683446`; converted round12=`0.065565683445`.
Original Base_Geometry_ID=`f6b08d77860ee4c44a0fe942c30c3d4b476dd5ffbd512d7d53085b9010c2401a`;
reconstructed=`accdccc14ad3a1b612048d624e33683f3314edf1798606b4048497e4c80df700`.
Reconstructed row ID=`F1_7cbf9659c3d4d06847d19ca7ed387779`.
The legacy hot-path exception was reproduced using the existing production audit
function, unchanged scientific plan, and finite synthetic evidence.

**UNKNOWN:** the exact row that triggered SCC attempt0005. Its failure log/receipt
is unavailable; the researcher confirmed there is no copy. We do not invent an
exact incident row. The earliest locally affected TRAIN_DEV context in the old
sorted launch order begins around row 40200:
`F1_TRAIN_A100_J2_334__N1000__U060`, affected row
`F1_403dd336bef4ecac9a5e7322b97f9528`. This is consistent with roughly 41k total
progress, but does not establish which worker first reported the SCC exception.
`geometry_incident.json` records every affected row, original/reconstructed IDs,
field values, source/wave/seed and the explicit limitation.

## Execution architecture and scientific boundary

Production entry: am_run_flexdc_foundation_campaign_v1.py -> campaign.main() ->
simple_campaign.dispatch()/run_full() -> simple_collector.collect() -> real wizard
and FlexDC -> collector_receipts.publish(). Old ledger-led execution and systemic
technical circuit-breaker APIs remain available only for historical regression
coverage; the new production CLI never dispatches to them.

The collector reads original plan strings with csv.DictReader, groups rows into
up-to-25-row same-context chunks, and writes those same values to transport CSVs.
Foundation-v2 wizard parsing uses pandas float_precision=round_trip; historical
non-Foundation parsing is unchanged. Chunking amortizes interpreter/config setup.
Completion is always row-local, never a chunk transaction. No make_row(), geometry
hash regeneration, objective certification or SQLite is used by collect/publish.
The wizard still performs its existing legal-input checks and extracts the
canonical measured objective; it does not redefine simulator or science logic.

At launch/setup, verify the immutable master/source/config seal and pinned stack.
During a row, minimally check successful extraction, exact original ID, readable
compact files, receipt checksum and requested/applied N/U/J/seed/workload/P/R and
weights. The existing1e-9 weight serialization tolerance is retained, with both
requested/applied vectors and deltas recorded. Input-file hash provenance is
cached once per worker context instead of rereading ISO/workload bytes every row.
No normalization, candidate coordinates or allocation policy changes.

At stage boundary, certify all collected receipts using the existing strong
scientific observation audit: original/reconstructed identity, legal domain,
requested/applied and serialized values, grid/workload identities, evidence,
finite behavior vector and canonical objective. Verify compact CSV row identity,
full release hashes, task/source/seed/geometry quotas and duplicates. Only a
scientifically clean stage can generate the next measured-only adaptive wave.

## Primary completion and derived data

`foundation_v1_runs/<scope>/collector_v2/<stage>/receipts/<id-prefix>/<Plan_Row_ID>.json`
is authoritative. Schema: FOUNDATION_V1_COMPLETION_V2. Minimum fields include
original Plan_Row_ID/Base_Geometry_ID/Task_ID/Workload_ID/context, seed, requested
P/R/weights, actual applied N/U/J/seed, absolute results/diagnostics/evidence paths,
EXECUTION_COMPLETE, timestamp, embedded evidence, and strict-JSON SHA256 checksum.
No NaN/Infinity serialization is allowed.

Durability order: close/fsync both CSV rows -> atomic evidence receipt -> verify
basic request correspondence and sync output/evidence directories -> atomic
completion marker -> append/fsync completion event -> cleanup. Temporary writes
are fsynced and atomically renamed, with directory fsync on POSIX. Missing/corrupt
completion means retry. A hard death after durable completion does not rerun that
row. CSV-only or active/no-receipt attempts do not establish completion.

observations.sqlite is rebuilt/certified from these receipts after collection.
It is used only for audited adaptive selection and exports, not pending-row
scheduling. Deleting this derived DB cannot lose execution progress. Status has
its own disposable SQLite offset cache; that cache is never execution authority.
New format is explicitly marked. Old attempt0001-0005 SQLite state is not migrated:
use the confirmed fresh-start operation before a brand-new campaign.

## Fail-forward retries, repair, cleanup and interruption

A worker crash, SIGSEGV, missing/malformed output or row-local receipt exception
never cancels other healthy workers. Decode negative return codes in structured
attempt records. Three launch attempts are allowed in one collection pass.
Remaining rows from a failed chunk split into individual immediate retries;
receipted rows never retry. Exhausted IDs quarantine while the rest of the stage
finishes. At stage end, permit two additional bounded repair/audit cycles
(maximum9 technical launch attempts per persistently bad row across three passes).

Audit discrepancies archive only the affected completion marker with exact
expected/observed row IDs, geometry IDs, P/R, weights, source, wave, seed,
plan/receipt/output paths and applied values. Clean rows stay complete. Repair
reruns the same original scientific row, then certifies again. Deterministic
unrepairable discrepancies stop only AFTER unrelated stage collection finishes
and BEFORE any downstream adaptive generation. Insufficient-evidence receipts
remain terminal, excluded from usable labels/anchors, and stop advancement for
explicit policy review; they are not silently relabeled or identically rerun.

Transient allowlist remains power_trace.csv, job_table.csv, PRtable.csv,
base_weights.csv, waiting_jobs_by_job_type.csv, probabilistic_results.csv.
After completion, delete these with existing bounded retries. If unlink fails,
retain compact science, record CLEANUP_DEBT and stop ONLY that local physical
chunk to avoid overwriting its raw debugging files. Remaining IDs move to fresh
worker directories while other workers continue. Retry debt at stage audit.
No unlink failure creates a global storage-stop marker in the new path.
Raw outputs remain physically partition-scoped under
`src/peacsim/output/optimization/foundation_v1_simple/train_dev/collector_v2/`
and `foundation_v1_simple/locked_synthetic_test/collector_v2/`; isolated test
outputs use only the separate `foundation_v1_simple/validation/` namespace.

Actual available filesystem capacity is checked independently. Default emergency
watermark=10GiB, configurable with --storage-emergency-gib. Falling below that
threshold stops launches and drains current workers; it does not erase durable
completion. Local derived progress/log metadata failure probes shared writability
and cannot veto safely persisted rows. Genuine shared output filesystem failure
can stop collection. No broad transient wildcard deletion is used.

SIGINT/SIGTERM stop assignments, allow current atomic work to finish, and persist
interruption status. Drain is bounded: terminate after30s, kill after35s where
needed. Parent death is detected before another row; completed markers survive
allocation loss. No completion is inferred from a dead process. Resume scans plan
and completion receipts once, skips complete IDs, and retries incomplete IDs.
The same command works with --resume. There is no migration requirement for old
SCC attempt data.

## Remaining global stops

| Condition | Why stopping is justified |
|---|---|
| Master/source release hash changed; global plan unreadable or duplicate original IDs | Cannot trust the shared immutable execution input |
| Required shared config/executable or pinned numerical environment missing | Exact intended execution is unavailable |
| Shared output filesystem genuinely unwritable | Durable scientific completion cannot be saved |
| Actual free space below configured emergency watermark | Further launches could exhaust physical storage |
| User/scheduler signal or parent loss | Explicit termination / allocation ending |
| Final bounded stage audit/evidence shortfall | Collection already finished; scientific advancement must wait |

There is NO new-path global circuit breaker based only on repeated worker exit
codes, one bad scientific row, or one unlink failure with healthy storage.

## Fresh start and status

Fresh-start dry-run prints exact generated execution roots/manifests only.
Confirmed reset removes foundation_v1_runs, the two explicitly owned raw output
subtrees, narrowly identified Foundation chunk/log roots and launch manifests.
Paths are resolved and checked within the repository; symlinks/junctions/reparse
subtrees are refused. Both current and legacy campaign locks are checked.
Source/configs/profiles/workloads/master and approved plan artifacts are excluded.
No real SCC/local production run tree was reset during development; only temporary
fixtures were deleted. Run-owned adaptive plans under the new execution root are
regenerated later from new audited evidence using unchanged acquisition code;
canonical approved scientific artifacts remain untouched.

--status reads bounded per-slot event journals incrementally plus pending pointers
that close the marker-to-event crash window. A parent completion journal also
closes worker-event failure gaps. No recursive corpus scan is required per status
call. It reports stage, planned/complete/pending/running, live known worker PIDs,
retries/quarantine/audit failures, partition/overall totals, last-hour completions
and last-minute throughput. Known dead PIDs correct stale active-worker estimates.
The 50000-marker/journal fixture benchmarks cold and warm status and cache loss.
A supplementary full-schema fixture verified24844 fake completion receipts:
cold0.493s, warm0.031s, cache-loss rebuild0.480s; counts remained24844 complete,
35156 pending,3 quarantined and4 technical retries. Its generator was deliberately
bounded after more than20000 markers; the separate50000-entry suite test passed.
These fixtures are synthetic metadata only, not scientific observations.

## Validation and limitations

See final_gates.json, final test logs, source_guard_changes.json and measured soak
summaries for exact final counts/timing/source identity. The reduced 750-row
synthetic-simulator campaign uses REAL production orchestration, original Phase-A
rows for TRAIN/DEVELOPMENT/LOCKED tasks, REAL adaptive.wave/repeats generation,
strong audits and exports. It completes all 12 stages, resumes interruption in
Wave2, and rebuilds deleted derived DBs without rerunning clean rows. Simulator
labels in this fixture are explicitly synthetic, not scientific observations.

Real isolated tests cover A100/P100,J2/J4/J6,N250/N1000,U0.6/U0.8 through production
wizard/FlexDC/results/strict evidence/completion/audit/cleanup/resume. Earlier extended
soaks used32 logical chunks,256 original rows and8 real concurrent workers. The
final sealed-source soaks use32 logical chunks and128 original rows with8 real
concurrent workers, plus one same-row repair simulation in the fault case. The
fault soak safely injects one boundary -11, a bad geometry receipt and an unlink
failure; repair runs only the affected exact row. It never intentionally crashes
native FlexDC. Scheduler32-worker fill is separately tested synthetically.

Early development probes exposed an overly strict bitwise weight comparison and
relative receipt paths. They were corrected to the existing1e-9 serialization
contract and absolute paths, with regression tests; failed probes remain isolated
and are not part of any corpus. They are not hidden or counted as final passes.

All prior 287 cases remain. Retired stage-CLI fixtures now invoke the historical
entry fixture against the retained current stage APIs; they are not claimed as
coverage of the new production CLI. The historical AST comparison strips only
the Foundation-specific parser branch and verifies non-Foundation behavior remains
unchanged. New tests cover the actual new CLI and collector. Scientific sampling,
adaptive/repeat methods, domains, costs, ISO and thresholds remain unchanged.

Local hardware has 16 logical CPUs and approximately 32GiB RAM;8 simulator processes
are supported. These laptop measurements cannot certify the SCC 32-worker target
of roughly 2.76 rows/s. Report measured worker-seconds, slot occupancy and replacement
gaps without claiming a cross-hardware improvement. No 4-minute-per-row orchestration
barrier is intentionally introduced; collection contains no global simulation lock
or dataset-wide hot-path audit. Stage certification remains intentionally strong.

Master before/after:
`bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f`.
400 profiles,500 workloads,2000 contexts,300000 static+160000 adaptive+40000 repeat
rows=500000; TRAIN_DEV450000,LOCKED50000. No scientific coordinates changed.

## Final measured gates (before commit)

The complete Foundation suite passed **315 tests, 0 failures, 0 skips and 0
warnings** in 378.21 seconds, on the final source identity below. All 287 previous
cases were retained and 28 new cases added; case names are in test_summary.json.
The 750-row reduced full campaign, real adaptive/repeat generation, all 12 scoped
stages/exports, hard interruption, derived-database loss, safe fresh reset,
32-worker scheduler fill, exact-plan authority and targeted audit repair passed.

Final release identity:
`75109d5f0953dcf60a6e2761db1f838d97dda4554d9937a06436a281fe25f0d5`.
All 621 guarded files verify; all previous guarded paths are retained. The 531
previously guarded non-Python scientific/configuration artifacts are byte-identical.
Exact changed source hashes are in source_guard_changes.json.

| Final sealed test | Workers | Original rows | Collection wall seconds | Rows/s | Occupied worker-s/row | Provisioned worker-s/row | Slot occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Normal real soak | 8 | 128 | 223.234 | 0.573389 | 11.581539 | 13.952125 | 83.009% |
| Injected-fault real soak | 8 | 128 | 234.218 | 0.546499 | 11.956313 | 14.638625 | 81.676% |

Occupied worker-seconds sum active process-slot durations. Provisioned worker-
seconds = workers * collection wall time / original rows, and are the metric
comparable in definition to the earlier SCC estimate of 11.6. Do not substitute
occupied time for that SCC metric. The normal local provisioned value is 13.952;
this laptop result does not establish improved or equivalent SCC throughput.
Final actual process intervals confirm 8 overlapping native worker launches in
both tests. Median replacement gaps are 0.109s (normal) and 0.109s (fault); maximum gaps are 0.218s and 0.140s.

Fault collection includes one safe worker-boundary -11 injection and one real
worker with mocked unlink failure. Unrelated workers continue. The later audit
identifies only the deliberately corrupted geometry receipt and runs exactly
one additional real simulation of its original row. Thus the fault test has 129
real simulations total; its timing table describes 128-row collection and excludes
subsequent offline certification/one-row repair. Final audits and cleanup pass;
both tests then prove zero simulation calls on resume. No native SIGSEGV was
intentionally induced, and this does not establish the old native crash's cause.

Across ALL isolated development probes and real validation runs, including
unsuccessful early post-simulation plumbing probes, 911 real test rows were
persisted. They remain outside every production ledger/export. Early 7+3 probes
failed their then-current completion/audit gates; they are disclosed, preserved
and not counted as final success. Later 3 integration rows, extended 256-row normal
and 257-row fault tests,128-row intermediate soak and the final 128+129 rows passed.
real_validation_summary.json preserves the measured history and exact namespace
counts. No new scientific observations are accepted into the 500000-row corpus.

The full production CLI dry-run shows all 14 collection/export steps, 32 requested
workers, 500000 expected rows and 0 simulations. The real production root status
is NOT_STARTED with 0/500000; fresh-start was only previewed there. Confirmed reset
was tested ONLY against temporary fixtures, preserving scientific hashes.

THE COMPLETE MINI-CAMPAIGN, REAL FLEXDC PARALLEL SOAK, FAILURE INJECTION,
INTERRUPTION/RESUME, AND FRESH-START TESTS ALL PASSED BEFORE COMMITTING.

## SCC stop, upgrade, reset, start and status

Stop OLD collection using Ctrl+C once in its original foreground/tmux pane. Wait
for that run's workers to exit. Inspect owned processes; do not kill unrelated
experiments or modify code under active workers:

```sh
pgrep -u "$(id -u)" -af '[a]m_run_flexdc_foundation_campaign_v1.py|[a]m_run_flexdc_dataset_parallel_combo.py|[a]m_data_extraction_wizard.py'
```

In the existing SCC checkout, same pinned Python3.10/NumPy2.2.6/pandas2.3.3/SciPy1.15.3
environment active:

```sh
test "$(git branch --show-current)" = "foundation-dataset-v1-prep" || exit 1
git status --short
git pull --ff-only origin foundation-dataset-v1-prep
git rev-parse HEAD
python -B -c "from foundation_v1.environment import versions; print(versions(require=True))"
python -B -c "from foundation_v1.full_campaign import verify_release; verify_release(); print('SOURCE_GUARD_PASS')"
python -B am_run_flexdc_foundation_campaign_v1.py --fresh-start --dry-run
```

Review the printed targets. Only then deliberately discard generated execution
state; this is a NEW run, not migration of old observations:

```sh
python -B am_run_flexdc_foundation_campaign_v1.py --fresh-start --confirm-fresh-start
python -B am_run_flexdc_foundation_campaign_v1.py --status
python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 32 --execute
```

After reset, status must show 0/500000 and NOT_STARTED. In another terminal:

```sh
python -B am_run_flexdc_foundation_campaign_v1.py --status
```

For future allocations, repeat the full command with --resume and WITHOUT another
fresh-start. Do not reseal SCC source to bypass CRLF/LF/hash disagreement: reviewed
byte hashes and .gitattributes remain strict. No main merge, force push or science
regeneration is needed.

THE NEW COLLECTOR EXECUTES THE IMMUTABLE PLAN DIRECTLY AND DOES NOT
RECONSTRUCT ROW IDENTITY IN THE HOT PATH.

ONE ROW OR WORKER FAILURE CANNOT HALT UNRELATED COLLECTION.

SCIENTIFIC AUDITING OCCURS AT STAGE BOUNDARIES AND ONLY BAD/MISSING ROWS ARE
REPAIRED.

SQLITE IS DERIVED STATE; DURABLE PER-ROW RECEIPTS ARE THE PRIMARY COMPLETION
RECORD.

THE NEW CAMPAIGN CAN RESUME ACROSS FUTURE SCC ALLOCATIONS, BUT NO COMPATIBILITY
WITH THE OLD ATTEMPT0001-0005 RUN DATA IS REQUIRED.

A SAFE FRESH-START COMMAND IS PROVIDED SO THE USER CAN DELETE ALL OLD
FOUNDATION EXECUTION OUTPUTS AND BEGIN FROM ZERO.

THE 500,000-ROW SCIENTIFIC PLAN AND MASTER PHASE-A SHA ARE UNCHANGED.

NO REAL FOUNDATION 500,000-ROW CAMPAIGN WAS STARTED DURING VALIDATION.

## Exact release file inventory

Modified existing files:

- `.gitignore`
- `FILE_HASHES.sha256`
- `foundation_v1/campaign.py`
- `foundation_v1/release.py`
- `foundation_v1/scope.py`
- `foundation_v1_artifacts/FULL_CAMPAIGN_GUIDE.md`
- `foundation_v1_artifacts/README.md`
- `foundation_v1_artifacts/SCC_COMMANDS.md`
- `foundation_v1_artifacts/preexecution_release.json`
- `src/peacsim/am_data_extraction_wizard.py`
- `src/peacsim/foundation_v1_cleanup.py`
- `src/peacsim/foundation_v1_evidence.py`
- `tests/test_foundation_v1.py`
- `tests/test_foundation_v1_cleanup.py`
- `tests/test_foundation_v1_nan.py`
- `tests/test_foundation_v1_patch.py`
- `tests/test_foundation_v1_resilience.py`

New files:

- `foundation_v1/collector_receipts.py`
- `foundation_v1/simple_campaign.py`
- `foundation_v1/simple_collector.py`
- `foundation_v1/simple_integration.py`
- `foundation_v1_artifacts/FOUNDATION_V1_SIMPLE_COLLECTOR_FINAL_AUDIT.md`
- `foundation_v1_artifacts/simple_collector_validation/additional_safety_gates.json`
- `foundation_v1_artifacts/simple_collector_validation/changed_files.json`
- `foundation_v1_artifacts/simple_collector_validation/final_gates.json`
- `foundation_v1_artifacts/simple_collector_validation/fresh_start_preview.json`
- `foundation_v1_artifacts/simple_collector_validation/full_campaign_preview.json`
- `foundation_v1_artifacts/simple_collector_validation/geometry_incident.json`
- `foundation_v1_artifacts/simple_collector_validation/real_validation_summary.json`
- `foundation_v1_artifacts/simple_collector_validation/scientific_identity.json`
- `foundation_v1_artifacts/simple_collector_validation/source_guard_changes.json`
- `foundation_v1_artifacts/simple_collector_validation/status_no_campaign.json`
- `foundation_v1_artifacts/simple_collector_validation/status_receipt_benchmark.json`
- `foundation_v1_artifacts/simple_collector_validation/stop_path_audit.json`
- `foundation_v1_artifacts/simple_collector_validation/test_summary.json`
- `tests/test_foundation_v1_legacy_entry_fixture.py`
- `tests/test_foundation_v1_simple_collector.py`

Excluded: worker/raw outputs, runtime plans/receipts, SQLite state, local logs/XML, review ZIPs, and all `.env` files. No scientific profile, workload, context or plan file changed.
