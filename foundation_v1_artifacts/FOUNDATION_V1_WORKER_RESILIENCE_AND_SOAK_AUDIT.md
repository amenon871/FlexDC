# Foundation V1 worker resilience and real parallel-soak audit

Result: PASS. Pre-fix commit: eb308a8ff6fd6f90fbb7f1d2fff2a52ad7e04c1c.
Branch: foundation-dataset-v1-prep. The fix revision is the commit containing this report.
No Foundation scientific dataset campaign was started during validation.

## Exact SCC failure: verified controller cause versus unknown native cause

Reported root worker:
foundation_v1_runs/train_dev/foundation_v1_82678bd28c06_AQA_F1_DEVELOPMENT_A100_J3_047__N250__U080_prchunk02of06
returned -11 after configuration initialization. On Linux -11 is signal11,
SIGSEGV. The underlying native crash cause is UNKNOWN. No native stack or
reproduction establishes whether simulator/native library/environment caused it.
SCC diagnostics (about2 threads/worker,99% CPU,368GiB available,OMP1) do not
establish oversubscription or memory exhaustion. Thread defaults are defensive,
NOT a claimed root-cause fix.

VERIFIED source cause of the cascade: the old exact-plan parent mapped ANY
nonzero child exit to failed_process and wrote foundation_storage_stop.json.
The Foundation worker check_stop then raised CleanupRequired in every healthy
worker at its next row boundary. This converted a local native crash into a
campaign-wide cleanup failure. Before changing source, a mocked concurrent
-11 test reproduced that exact marker and loss of pending healthy launches.
The captured failing output and source SHA are retained in pre_fix.log/pre_fix.json.

## Failure taxonomy and scope

|Class|Meaning/action|
|---|---|
|SUCCESS|Worker exited normally; completion additionally requires audited terminal receipts.|
|SCIENTIFIC_TERMINAL|Valid or insufficient-evidence terminal execution; never repeat completed IDs because of a later process error.|
|TECHNICAL_RETRY_REQUIRED|Isolated SIGSEGV/SIGKILL/nonzero exit, crash, incomplete or malformed temporary output: retry only nonterminal rows; unrelated work continues.|
|TECHNICAL_RETRY_EXHAUSTED|Three total attempts exhausted; quarantine exact incomplete IDs and continue other chunks.|
|STAGE_COMPLETE_WITH_TECHNICAL_SHORTFALL|Queue drained as much safe work as possible; stage cannot advance or claim full scientific completion.|
|SYSTEMIC_TECHNICAL_FAILURE|Same failure signature across enough independent workers; stop new launches, drain existing work, preserve state.|
|TRANSIENT_CLEANUP_REQUIRED|Actual unresolved deletion/storage invariant after scientific persistence; global storage stop, cleanup-only recovery, no resimulation.|
|GLOBAL_FATAL_INVARIANT|Immutable source/plan/request identity or ledger integrity is untrustworthy; stop for review.|
|INTERRUPTED|Explicit shutdown or parent loss; stop new work, bounded drain, retain only authoritative terminal completion.|

Failure grouping is POSIX signal name for negative codes, or exit/incomplete-
receipt signature where no finer cause is known. It is an operational signature,
not proof that all positive exit-code failures have the same internal cause.

Foundation exact-plan requests dispatch to foundation_v1.resilient before the
historical scheduler branch. Legacy non-Foundation behavior remains unchanged;
its old failed_process vocabulary is not the Foundation execution path.

## Retry, quarantine and systemic-breaker policy

Default:3 TOTAL technical attempts per logical chunk (initial+2 retries), exposed
as --foundation-max-technical-attempts. Each retry has a fresh mutable directory
and log, copies the SAME immutable rows/seed/P/R/weights/context and excludes
all audited terminal IDs. CSV-only rows do not count as completion. Retries are
queued behind eligible unrelated work, so a bad chunk does not monopolize slots.
All attempts retain return codes, decoded signals, host, command, log, timestamps,
terminal count and exact incomplete Plan_Row_IDs in per-attempt JSON receipts
and an append-only manifest. Exhaustion quarantines work; it never labels the
context infeasible or silently reduces the scientific dataset.

Circuit breaker:at least5 DISTINCT logical chunks with the same signature AND
at least20% of the most recent50 completion events retained within300 seconds.
Equality at20% trips; repeated crashes of one logical chunk cannot satisfy the
independence requirement. A five-minute,50-event window limits stale incidents;
five independent failures avoids a single-crash cascade. Positive exit codes
are intentionally conservative signatures requiring review, not root-cause
attribution. Controlled interruptions are not counted as technical failure rate.
Tests exercise5/25 equality,5/26 non-trigger, time expiry, independence, single
-11 success on retry and persistent -11 exhaustion.

Within one queue identity retry counts persist across direct resume. After a
stage quarantines work, it STOPS and requires operator review. An explicit new
campaign resume/allocation creates a new outer attempt and may retry only still
nonterminal work; it is not an automatic unbounded retry loop.

## Storage, scientific terminality and old SCC upgrade

Real deletion failure remains a separate foundation_cleanup_stop.json, never
created for a generic process error. The existing per-row durable CSV/evidence
barrier, explicit six-file allowlist and bounded deletion retries remain.
Scientific receipt is recorded before cleanup recovery; cleanup-pending rows
are not simulated again. Insufficient evidence remains terminal but excluded
from usable training/adaptive observations; the original scientific quota gate
still stops advancement if required usable rows are missing.

The old foundation_storage_stop.json is preserved as historical evidence; new
attempts use separate cleanup/systemic/integrity/interruption marker identities.
No old marker/ledger/output deletion is required. Queue completion is execution
completion; the campaign still separately audits usable evidence quotas before
adaptive or repeat generation. Adaptive acquisition, domains, seeds and repeat
selection were NOT redesigned.

The four-state upgrade fixture includes:valid terminal receipt, CSV-only old
NaN failure, active/no receipt, and valid receipt with cleanup pending. It models
attempt0001 NaN,0002 technical trouble,0003 root-11 plus collateral stop and0004
partial successful work. Resume cleans the pending terminal row without a
simulation, skips completed science and retries only the two nonterminal rows
in attempt0005. Canonical ledger has four unique observations; a second resume
launches zero calls. Failed old debugging artifacts and stale markers remain.

## Shutdown and parent-loss behavior

The new full-campaign launcher forwards SIGTERM/SIGINT to the combo queue.
The queue stops launching new chunks, writes an interruption marker, allows
current rows to persist/clean and gives30 seconds to drain. At30 seconds it
requests termination; at35 seconds it escalates kill for remaining processes.
This is shutdown handling, NOT a scientific row deadline. POSIX workers use
separate sessions so the parent can mediate the drain. Parent-loss checks stop
orphaned queues/workers from starting additional work. Unreceipted interrupted
rows remain retryable; status caches never invent scientific completion.
SIGKILL/node loss cannot write a graceful receipt; authoritative row receipts,
immutable plans and transactional ledgers remain the recovery source.

Windows venv python.exe is a redirector, so getppid alone is not reliable there.
An early injected-soak validation exposed a false parent-loss check before ANY
scientific simulation started; the systemic breaker stopped it with a technical
reason, not a cleanup reason. This failed development run is preserved. Windows
now checks the recorded parent's process liveness; POSIX uses parent identity.
Live-parent and parent-loss tests pass. Actual SGE termination was NOT exercised
on this Windows host; shutdown behavior was tested with controlled fixtures.

## Thread settings and diagnostics

Foundation children default the following ONLY when not explicitly supplied:
OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,NUMEXPR_NUM_THREADS,
VECLIB_MAXIMUM_THREADS,BLIS_NUM_THREADS,PYTHONFAULTHANDLER all1.
Effective values are recorded in launch logs/status. Explicit user values remain.
Negative return codes decode as -11 SIGSEGV,-9 SIGKILL,-15 SIGTERM. Diagnostics
include logical chunk,attempt,host,argv,log,time,terminal count and remaining IDs.
No native crash was deliberately triggered; real injected testing reports one
-11 at the subprocess boundary while the other workers are real FlexDC processes.

## Throughput audit and measured real tests

No simulation/global worker lock was added. The existing campaign lock prevents
duplicate orchestrators; workers do not acquire it. Only one parent writes the
queue-local ledger, so simulator workers do not contend for SQLite writes.
The worker cap is passed unchanged to every stage. Slots refill on a0.2-second
poll rather than waiting for a slow sibling chunk. Attempt manifests append
one row instead of rewriting the entire history each completion. Worker output
pointers avoid directory-wide discovery scans in the normal path. Plans are
read at chunk setup, not per row. Receipt validation is per finished chunk;
no dataset-wide audit is performed in simulator row loops. Cleanup fsyncs only
compact files and their immediate directories, not recursive trees. Redundant
writes of an unchanged current-row cleanup pointer were removed.

The scientific receipts and canonical observations remain compact but durable;
a queue-local receipt ledger adds recovery metadata. Its persistence is not a
second training corpus. Old scientific/debug output trees are never deleted.

|Real test|Workers|Logical chunks|Rows|Wall seconds|Rows/sec|Effective seconds/row|Live-slot occupancy|
|---|---:|---:|---:|---:|---:|---:|---:|
|integrations|1|3|3|42.063|0.0713|14.021|91.0%|
|soak|4|24|144|402.968|0.3573|2.798|87.3%|
|fault-final|4|24|144|401.015|0.3591|2.785|89.7%|
|nested|2|3|3|30.000|0.1000|10.000|81.0%|

Real integration contexts:A100 J2 N250/U0.6, A100 J4 N1000/U0.8 and P100 J6
N1000/U0.8. Soaks use24 contexts,6 EXISTING immutable Phase-A rows per context,
4 workers and144 real rows each. The injected run has25 physical attempts for24
logical chunks:one deliberate -11 and its successful unchanged-input retry.
Healthy workers continued launching and completed their work. Each soak ended
with144 strict receipts,144 valid terminal ledger rows, no false cleanup stop,
all six transient filenames absent, and an identical-queue resume with ZERO
new simulator calls. Nested production-style directory plumbing also passed
under an isolated test root that the production campaign cannot ingest.

Normal observed completion-to-replacement launch:median0.092s,max0.223s.
Injected:median0.087s,max0.100s. These use parent-observed completion timestamps,
not an OS scheduler trace. Slot occupancy is process-lifetime occupancy, NOT
CPU utilization. Worker-lifetime seconds per row averaged9.80 normal and10.02
injected, including startup. Within-worker completion intervals had medians
7.81/8.12 seconds, with maxima36.22/36.93 seconds for heterogeneous rows. The
slowest cases exceed the historical roughly10-second reference; no claim is
made that every profile/row has that runtime. There is no measured four-minute
per-row executor behavior or global serialization in these tests. Raw simulator-
only time was not separately instrumented. Timings are local smoke measurements,
not an SCC/GPU benchmark or a demonstrated improvement over the old implementation;
background unit validation overlapped part of the normal run.

Total successful isolated real TEST rows:294 =3 representative integrations
+144 normal soak +144 injected soak +3 nested-layout integrations. All stayed
outside the production campaign output roots and ledgers. The failed early
Windows parent-check probe started zero simulations. Raw test outputs, local
SQLite indexes and worker logs are ignored and NOT committed as dataset outputs.

## Tests, source guard and scientific identity

Final complete suite:287 passed,0 failed,0 skipped,0 relevant warnings.
All269 prior test cases remain; the obsolete case that expected generic worker
failure to stop everyone now asserts the intended TRUE-cleanup-stop behavior.
New cases cover the pre-fix cascade, partial receipt-aware retry, bounded
quarantine, exact breaker thresholds, true cleanup stops, malformed local state,
immutable-plan corruption, parent-loss/forwarded signals, live Windows parents,
slot refill and the actual four-state/attempt-history upgrade.
Full test log is retained locally in resilience_validation/full_tests_final.log.
The machine-readable stop_path_audit.json records258 reviewed raise sites across
the controller, wizard, evidence/schema/cleanup, ledger, adaptive and repeat paths.
Scientific/source/ledger corruption is globally unsafe. Local process/output
failures are retryable. Adaptive/repeat quota/evidence gates intentionally block
future science after the execution stage has collected all safe unrelated work.

Master Phase-A SHA BEFORE AND AFTER:

```
bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f
```

400 synthetic profiles,500 workload definitions,2000 contexts,300000 static,
160000 expected adaptive and40000 expected repeat rows remain byte-identical
scientifically; eventual corpus remains500000. No P/R/weight coordinate, seed,
QoS, profile, workload, ISO, cost, idle power, hardware or model-data meaning
changed. The source guard retains every previous path and now guards615 files.
source_guard_changes.json records exact changed hashes and scientific_identity.json
records unchanged scientific/config hashes. Reviewed source seal for the final
injected and nested tests:

```
a94cb501c6fb2e2f251ee64a321c6df2040f6942686fcef939ad6631ec701cba
```

## Safe SCC stop, upgrade, preflight and resume

For the CURRENT OLD eb308a8 run, use Ctrl+C once in its ORIGINAL foreground/tmux
pane, which signals the old foreground process group. Wait for its workers to
exit. Do NOT kill only the old parent and assume grandchildren stopped. Inspect
owned processes from another pane:

```sh
pgrep -u "$(id -u)" -af '[a]m_run_flexdc_foundation_campaign_v1.py|[a]m_run_flexdc_dataset_parallel_combo.py|[a]m_data_extraction_wizard.py'
```

Confirm no processes belonging to THIS Foundation run remain before pulling;
do not kill unrelated experiments matching these script names. If the old run
is not accessible in its original terminal, inspect its exact PID/process group
before signaling; do not issue a blanket pkill. After installing this patch,
future SIGTERM to the verified campaign parent is forwarded for bounded drain.
Do not modify source under active old workers.

From the existing SCC checkout, with the SAME pinned numerical environment
active (Python3.10.x,NumPy2.2.6,pandas2.3.3,SciPy1.15.3):

```sh
test "$(git branch --show-current)" = "foundation-dataset-v1-prep" || exit 1
git status --short
git pull --ff-only origin foundation-dataset-v1-prep
git rev-parse HEAD
python -B -c "from foundation_v1.environment import versions; print(versions(require=True))"
python -B -c "from foundation_v1.full_campaign import verify_release; verify_release(); print('SOURCE_GUARD_PASS')"
python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 24 --execute --resume
```

Do NOT delete observations,receipts,ledgers,plans,worker outputs or prior logs.
No manual stale-marker removal is required. Old valid terminal observations
are reused; old CSV-only, active/no-receipt, root-crash and collateral-stop rows
are retried with unchanged scientific identity. Cleanup-pending terminal rows
receive cleanup only. Original evidence-shortfall policy remains unchanged.
No reset/force push/main merge is required. If SCC byte hashes disagree, stop
for review instead of resealing to bypass the guard.

ONE ISOLATED WORKER FAILURE NO LONGER HALTS HEALTHY FOUNDATION WORKERS.

SIGSEGV AND OTHER TECHNICAL PROCESS FAILURES ARE RETRIED LOCALLY AND ARE NOT
MISCLASSIFIED AS TRANSIENT CLEANUP FAILURES.

TRUE TRANSIENT CLEANUP FAILURE STILL STOPS NEW WORK TO PROTECT STORAGE SAFETY.

THE EXISTING SCC FOUNDATION CAMPAIGN STATE CAN BE RESUMED WITHOUT DELETING
SCIENTIFIC PROGRESS.

REAL FLEXDC PARALLEL EXECUTION WAS TESTED BEFORE THIS PATCH WAS COMMITTED.

NO FOUNDATION SCIENTIFIC DATASET CAMPAIGN WAS STARTED DURING PATCH VALIDATION.
