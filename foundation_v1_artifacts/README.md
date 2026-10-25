> **Current execution release:** use [FOUNDATION_V1_SIMPLE_COLLECTOR_FINAL_AUDIT.md](FOUNDATION_V1_SIMPLE_COLLECTOR_FINAL_AUDIT.md) for deployment, fresh reset, 32-worker collection, receipt-based resume and status. The older implementation notes below describe retained historical APIs; they do not override the new collector instructions.

# Foundation V1 approved full campaign

Start with FULL_CAMPAIGN_GUIDE.md and FULL_CAMPAIGN_VALIDATION.md. Scientific definitions, profiles and all300000 master Phase-A rows are unchanged. The locked synthetic outcomes are precomputed in the same collection but sealed from fitting and selection.

# Foundation V1: one command on one SCC fat node

Use the committed foundation-dataset-v1-prep branch. Do not merge into main to deploy this campaign. Run from the FlexDC repository root.

## Environment (entire campaign and every resume)

Python3.10.x, NumPy2.2.6, pandas2.3.3, SciPy1.15.3. Use a Python3.10 environment; install the local simulator editable and requirements-foundation-v1.txt. The full orchestrator and adaptive plan generator reject numerical stack mismatches. Do not change versions between stages or resumes.

```
python -m pip install -e .
python -m pip install -r requirements-foundation-v1.txt
```

## One invocation

Dry-run the entire sequence first (no generated adaptive coordinates, no SQLite ledger, no FlexDC):
```
python -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 96 --dry-run
```

After allocating the ONE FAT NODE and separately authorizing scientific collection:
```
python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 96 --execute --resume
```
96 is an example, not a hard-coded limit; choose a worker count appropriate for the allocated node. That exact value reaches EVERY stage's bounded parallel-combo-runner. No nested pools, multi-node orchestration, network workers, MPI or distributed DB are added. A global local lock prevents a second orchestrator, including stage-only commands.

TRAIN_DEV: Phase A -> audit -> measured Wave1 -> audit -> Wave2 -> audit -> Wave3 -> audit -> Wave4 -> audit -> measured repeat selection -> repeats -> audit/export TRAIN, DEVELOPMENT, TRAIN_DEV. Then LOCKED: the same six stages -> isolated locked export. Finally a counts/provenance completion manifest/report. Normal successful execution represents500000 observations; hard technical/scientific gates can safely halt it.

TRAIN_DEV270000 Phase-A +144000 adaptive +36000 repeats =450000. LOCKED30000 +16000 +4000 =50000. Eventual total300000 +160000 +40000 =500000;250 per task across2000 tasks. Master Phase-A SHA256 remains bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f.

## Sealed test, not post-freeze generation

The researcher explicitly chose to precompute LOCKED_SYNTHETIC_TEST in this same collection campaign before Meta V4/FlexPFN development. It is predeclared, operationally isolated and SEALED. Never use it for training, normalization, architecture/hyperparameter selection or other development choices. Do not claim outcomes were generated after model freeze. Future completely new real profiles remain the strongest untouched external holdout. --unlock-locked-test authorizes collection only.

Ledgers/status/exports: foundation_v1_runs/train_dev/ and foundation_v1_runs/locked_synthetic_test/. Raw worker trees: src/peacsim/output/optimization/foundation_v1_runs/<group>/. TRAIN/DEVELOPMENT/TRAIN_DEV exports never contain locked rows; no mixed model-training CSV is created. Root full-campaign reports contain counts/provenance, not detailed locked QoS metrics.

## Recovery / safety

Rerun the SAME full command with --resume after interruption. Progress is rebuilt from immutable plans/receipts and checksum-validated partition ledgers; cached COMPLETE flags are not trusted. Previously generated adaptive coordinates are reused rather than regenerated from later observations. Every stage re-ingests/audits, skips all terminal row IDs and submits only nonterminal rows. Missing atomic plan receipts can be reconstructed only by deterministic generation from the unchanged prerequisite evidence. A fully complete rerun validates ledgers/exports and returns FOUNDATION_V1_COMPLETE with simulation_calls_this_invocation=0.

Technically complete insufficient-evidence rows stay terminal and are never automatically rerun, never scored feasible/infeasible, never adaptive anchors or valid training labels. STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL / EVIDENCE_SHORTFALL_REQUIRES_POLICY stop automatic progression. No replacement coordinates or silent quota changes. Genuine missing/corrupt/crashed executions remain same-ID retryable. Scientific input/output mismatches stop for review.

Machine progress: foundation_v1_runs/full_campaign_status__ALL.json. Final manifest/report: foundation_v1_runs/foundation_v1_completion__ALL.json and .md. They include exact stage/count/source/seed/retry/evidence/provenance/version/Git/timestamp fields. Export receipts seal the isolated output files. Incomplete/unreceipted technical executions cannot be claimed as valid simulations; submitted rows and audited newly completed rows are reported separately.

## Byte hashes: Windows to SCC

The hash guard remains byte-exact. .gitattributes preserves the reviewed source/config/plan bytes across checkouts; two canonical ISO templates retain their existing approved CRLF bytes in the committed branch. No values or scientific hashes are changed. Before commit all sealed Git-index blobs are checked against the reviewed seal. SCC must use the exact approved source/config bytes; editor or Git newline conversion is NOT normalized at runtime. A mismatch must stop or be addressed by a newly reviewed release seal, never silently accepted. Do not run the release-sealing preparation utility on SCC to bypass a disagreement.

Review ZIPs, runtime ledgers, worker outputs, temporary logs and scientific simulator results are excluded from source commits. Frozen pre-execution scientific manifests/INIs/plans are intentional versioned inputs. No actual Foundation simulations were run during implementation.


## Foundation-only per-row SCC storage safety

After EACH row the worker closes and fsyncs both compact CSVs, atomically writes
its Foundation evidence receipt, verifies its checksum and exact CSV row identity,
and only then unlinks the explicit transient allowlist. It verifies absence
(in particular power_trace.csv) before the next candidate starts.

Allowlist: power_trace.csv, job_table.csv, PRtable.csv, base_weights.csv,
waiting_jobs_by_job_type.csv, probabilistic_results.csv. The last two are optional
per-second QoS arrays, created by Simulator.final_output only when availability_qos
is enabled; all active extraction consumers have finished before finalization.
Small QoS/cost/tracking summaries, results/diagnostics, evidence, logs, plans,
ledgers and provenance remain. No wildcard or recursive deletion is used.

Science and cleanup have separate states. EXECUTION_COMPLETE remains terminal
whether evidence is sufficient or insufficient. CLEANUP_REQUIRED receives at most
three deletion attempts per file (0.05s and 0.10s backoff). Unresolved deletion
records its exact path/error and raises TRANSIENT_CLEANUP_REQUIRED. A shared local
stop signal prevents subsequent rows/chunks; already running rows may finish and
persist/clean normally. No global cleanup lock serializes the worker pool.

Resume the SAME full campaign command after resolving permissions/storage issues.
It ingests scientific receipts first, retries ONLY pending cleanup, and excludes
all terminal row IDs from scientific scheduling. A crash immediately after the
evidence receipt but before the cleanup-state update is also recovered. True
unreceipted execution failures retain raw debugging files and remain retryable.
Use campaign resume, not manual reexecution/overwriting of a partially used chunk.

Cleanup receipts: each worker's foundation_cleanup/F1_*.json and current.json.
Scoped storage gate: foundation_v1_runs/<group>/transient_cleanup_status.json.
Stage/full-campaign summaries include files/bytes deleted, deletion failures,
retries, and pending cleanup counts. Telemetry uses cheap per-file stat calls,
never per-row recursive size scans. A power loss between unlink and its telemetry
write can undercount deleted bytes; it cannot cause scientific reexecution.

This storage policy is opt-in for FOUNDATION_V1 rows only. Historical simulator
artifact retention is unchanged. Reviewed source byte hashes remain strict; use
the NEW reviewed release bytes on SCC rather than resealing modified files there.
