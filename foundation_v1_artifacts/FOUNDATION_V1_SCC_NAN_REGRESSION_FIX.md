# Foundation V1 SCC NaN production regression fix

Status: PASS. Approved pre-fix source: `24ed05c3845ade024f63214a6ab00dd08ad84a67`.
Branch: `foundation-dataset-v1-prep`. Fix commit is the commit containing this report.

## Observed SCC failure and pre-fix reproduction

The reported SCC traceback terminated at:

```text
src/peacsim/foundation_v1_evidence.py, write_evidence(...)
    json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False, ...)
ValueError: Out of range float values are not JSON compliant
```

This is the exact supplied traceback suffix; a full original SCC stack was not
provided in the request. The full LOCAL reproduction stack is retained in the
ignored `scc_nan_validation/pre_fix_pytest.log`.

All 24 SCC workers reportedly completed their first simulator execution, then
failed in receipt serialization. Their offending provenance field was
`Replicate_Of_Plan_Row_ID = nan`, not an observed nonfinite scientific label.
Ordinary non-repeat rows intentionally have `replicate_of_plan_row_id = ""`:
there is no parent repeat. pandas.read_csv converts that empty CSV cell to NaN.
The historical provenance function forwarded it into BOTH CSV record lists;
strict JSON correctly rejected that float. Existing metadata cleaning did not
sanitize the separately reconstructed results/diagnostics records.

BEFORE changing production source, the new exact failure-shape test was run
against commit24ed05c twice. Both runs failed in the real write_evidence strict
json.dumps call with the exact exception above. Scientific outputs in the
fixture were finite. The result is recorded in pre_fix_reproduction.json.
The SAME test now passes.

## Minimal execution/schema correction

`plan_provenance_values` has one Foundation-only normalization guard. Historical
non-Foundation behavior is retained. The shared foundation_v1_schema module
covers all22 provenance fields:15 optional,7 required. Optional missing scalar
representations None/NaN/pandas.NA/NaT/blank/nan/none become canonical CSV blanks.
A valid repeat parent is preserved exactly. Required IDs, flags, seed and
scientific numeric inputs are checked rather than silently blanked.

The evidence writer independently normalizes metadata and validates required
labels BEFORE serialization. The canonical audit reuses this same required-field
validator, including complete J-wide QoS and weight vectors. Required NaN/Inf
values cause a controlled InvalidFoundationObservation (ValueError) identifying
the exact field/path, and no valid receipt is emitted. Optional nested metadata
uses strict-JSON null where appropriate. Unknown nonfinite scientific record
values are rejected with their path rather than hidden. No thresholds, cost
formulas, evidence counts or feasible/infeasible semantics changed.

Both final JSON dumps retain allow_nan=False. Real receipts were parsed with
parse_constant rejecting NaN/Infinity/-Infinity. No non-standard JSON constants
were accepted. Receipt body checksums and exact compact CSV checks still pass.

## Real FlexDC integration (measured local TEST results)

These used immutable existing Phase-A rows and canonical one-hour configs,
normal AQA, fixed grid and existing seeds. They were real simulator executions
through production combo runner -> wizard -> FlexDC -> both CSVs -> evidence ->
per-row cleanup -> successful worker exit. Numerical infeasibility is NOT a
technical failure; these checks validate execution, not candidate quality.

| Test | Workload / N / U | J | p90 | Pj | Canonical objective | Result |
|---|---|---:|---:|---|---:|---|
|individual_A|F1_DEVELOPMENT_A100_J2_039__N250__U060|2|0.394|[0.0038748425845200885, 0.0]|45.22038667580008|PASS: valid receipt, strict JSON, cleanup, exit0|
|individual_B|F1_DEVELOPMENT_A100_J2_039__N1000__U080|2|0.008|[0.9745937161430119, 0.9404491916014339]|97.89580684295785|PASS: valid receipt, strict JSON, cleanup, exit0|
|individual_C|F1_TRAIN_P100_J6_009__N1000__U080|6|8.451|[0.0, 0.0, 0.0, 0.10941405165231877, 0.0, 0.25870275850203583]|241.37223528314098|PASS: valid receipt, strict JSON, cleanup, exit0|

The exact failure FAMILY F1_DEVELOPMENT_A100_J2_039/N250/U0.6 was covered. Its
candidate is the first existing immutable master row selected for that context;
the SCC message did not supply an exact failed Plan_Row_ID, so this report does
not claim the local geometry is identical to the particular SCC example.

The six-row mini plan covered J2 A100 N250/U0.6, J2 A100 N1000/U0.8,
J6 P100 N1000/U0.8, J3 P100, J4 A100 and J5 P100 at N250/U0.6. Three workers ran
in parallel. ALL six workers exited0; both compact CSVs, valid evidence and
cleanup passed. No failed_process or global storage-stop marker occurred.
The mini was repeated after the final defensive required-field validation;
that FINAL six-row run also passed under the final release source seal.

A real seed21 test repeat based on a valid existing geometry passed. The parent
ID survived exactly in results, diagnostics and evidence; Base_Geometry_ID was
unchanged, seed21 had a distinct Plan_Row_ID and repeat grouping was retained.
The FINAL real repeat under the final source seal also passed.

Total real TEST executions:17 =3 individual +6 first mini +1 first repeat
+6 final mini +1 final repeat. There were NO automatic technical retries or
candidate searches. Both final mini/repeat reports identify the exact final
reviewed source seal. All17 observations passed canonical audit with sufficient
job-type evidence. Worker warnings:0 across17 worker logs.

Raw/compact simulator test outputs remain ISOLATED under
src/peacsim/output/optimization/foundation_nan_test_only_20261023_*;
plans are under .flexdc_plan_chunks/foundation_nan_test_only_20261023/.
They never entered foundation_v1_runs, any production ledger, or the500000-row
corpus. Those simulator output artifacts and *_real_results.json local summaries
are ignored and are NOT committed. This report records only review measurements.

## Exact SCC partial-attempt recovery

The regression fixture constructs the precise deployment state: a RUNNING row,
raw debug files, results and diagnostics with nan parent provenance, NO evidence
receipt, failed prior attempt and a stale attempt-level stop marker.
Ingest recognizes ZERO terminal scientific rows. CSV existence alone is not
completion. Resume schedules the same immutable Plan_Row_ID in a NEW attempt
(attempt0002 in the fixture), then accepts only the new valid receipt. Exactly
one canonical observation and one terminal row exist. A second resume launches
zero more simulations. The old failed CSV/raw attempt is not ingested as valid.
Canonical observation primary-key uniqueness and the existing scoped export
suite preserve one canonical row per Plan_Row_ID; no real campaign export was
run during this patch.

No campaign/orchestrator change was needed: attempt counters already create new
run IDs/output prefixes and a separate .flexdc_worker_logs/plan_<run_id>/ stop
marker. An OLD attempt's stop marker does not affect a new attempt. Manual
marker deletion is NOT required. Preserve old CSVs, manifests, plans, receipts,
ledgers and debug artifacts. Do not delete the existing SCC run tree.

## Storage and complete test gates

All six allowlisted transients (power_trace, job_table, PRtable, base_weights,
waiting_jobs_by_job_type, probabilistic_results CSVs) were absent after every
successful real test row. Compact CSVs and evidence remained. The prior90 tests
remain, including before-next-row cleanup, technical failure debugging,
cleanup-only recovery without resimulation and evidence-shortfall safety.

Full final suite:269 passed,0 failed,0 skipped,0 warnings. This includes179 new
NaN/provenance/strict-JSON/repeat/SCC-resume regressions. Full test output is in
the ignored full_unit_tests.log; counts are in unit_tests.json. The final
24-worker full-campaign DRY RUN still shows14 sequential stage/export steps and
500000 expected observations, with zero simulation calls.

## Scientific identity and source guard

Master Phase-A SHA BEFORE AND AFTER:

```
bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f
```

400 synthetic profiles,500 workloads,2000 contexts,300000 static rows,
160000 expected adaptive rows,40000 expected repeat rows and500000 eventual
rows remain unchanged. Every frozen scientific/config file matches the approved
pre-fix release hash. No scientific coordinate, profile value, workload, seed,
ID, QoS assignment, adaptive/repeat logic, count, partition, hardware, cost,
idle-power, ISO or model-data semantics changed.

preexecution_release.json retains EVERY prior guarded path and now guards612
files. release_guard_changes.json lists exact old/new hashes and all8 changed or
added guarded paths. Changes are provenance/evidence/schema validation, its
shared audit call, test/integration support and explicit release revision.
No hash guard was weakened. Final real mini/repeat code seal:

```
5a7c060a9535c74bedf0e76e3f5ba2ac701d5cba3f2c168815e21e46cae5954a
```

## SCC deployment and resume

Run from the EXISTING SCC repository checkout, with the SAME pinned environment
active (Python3.10.x, NumPy2.2.6, pandas2.3.3, SciPy1.15.3). The environment's
activation path is installation-specific and was not supplied; do not replace
it with a new numerical stack. Inspect local changes; do NOT reset or delete
run artifacts. The expected branch is foundation-dataset-v1-prep.

```sh
git status --short
git branch --show-current
git pull --ff-only origin foundation-dataset-v1-prep
git rev-parse HEAD
python -B -c "from foundation_v1.environment import versions; print(versions(require=True))"
python -B -c "from foundation_v1.full_campaign import verify_release; verify_release(); print('SOURCE_GUARD_PASS')"
python -u -B am_run_flexdc_foundation_campaign_v1.py --full-campaign --partition ALL --unlock-locked-test --max-workers 24 --execute --resume
```

24 matches the reported SCC worker count; use the node's approved allocation if
different. Do not run multiple orchestrators. No manual stale-marker removal or
ledger/plan/tree deletion is needed. Rows whose old simulation completed but
whose evidence receipt is missing are correctly retried; the old CSV alone
cannot supply a complete audited evidence label. Valid terminal receipts are
reused. Existing technical, storage and scientific evidence-shortfall safety
stops remain in force. Do not reseal SCC files to bypass byte-hash disagreement.

THE EXACT SCC FAILURE WAS REPRODUCED BEFORE THE FIX.

REAL FLEXDC FOUNDATION ROWS WERE EXECUTED SUCCESSFULLY THROUGH THE COMPLETE
POST-FIX EVIDENCE AND CLEANUP PATH BEFORE COMMITTING.

THE POST-FIX MINI PARALLEL COMBO RUN COMPLETED WITHOUT failed_process OR
TRANSIENT_CLEANUP_REQUIRED.

NO FOUNDATION SCIENTIFIC DATASET CAMPAIGN WAS STARTED DURING PATCH VALIDATION.
