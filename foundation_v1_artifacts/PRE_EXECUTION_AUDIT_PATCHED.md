# Patched pre-execution gates

**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**

|Gate|Result|
|---|---|
|Terminal insufficient evidence never automatically retried|PASS|
|Technical failure same-ID recovery|PASS|
|No incomplete adaptive/model-training labels|PASS|
|Explicit scientific shortfall gate|PASS|
|Default TRAIN_DEV and locked interlock|PASS|
|Scoped adaptive parents and isolated exports|PASS|
|All partition counts/row identities|PASS|
|400 profiles/500 INIs/2000 contexts/master300000|PASS|
|All original distributions/domains/cost/ISO checks|PASS|
|Complete deterministic regeneration and child-view reproducibility|PASS|
|A100 attribution; no cross-hardware interpolation|PASS|
|Unchanged one-fat-node bounded combo runner|PASS|
|Actual FlexDC/SCC simulation|NOT RUN (forbidden)|

See PATCH_RELEASE_NOTES.md for exact semantics and retained raw evidence. See patch_v1/*.json and test_results.txt for machine-readable verification.

# Foundation V1 narrow patch release

**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**

## Blocker 1: execution completion is not scientific label validity

Previously, ingest discarded a technically complete insufficient-evidence receipt. Resume checked only the valid-observations table, saw its row ID missing, and could repeat that identical deterministic simulation forever.

Now terminal_executions records every audited technically complete receipt with EXECUTION_COMPLETE, independently of whether observation_status is VALID_OBSERVATION or INSUFFICIENT_JOB_TYPE_EVIDENCE. observations contains only complete evidence-valid labels. Resume subtracts the union of these terminal IDs and legacy valid IDs, so an identical insufficient-evidence execution is never automatically scheduled again. Source receipt, raw result/diagnostic fields, counts, applied inputs, row identity, body checksum and plan provenance remain retained. Conflicting terminal receipts stop for review rather than replacing evidence.

Row execution states: PLANNED (implicit for untouched rows), RUNNING, TECHNICAL_RETRY_REQUIRED, EXECUTION_COMPLETE. Observation states: VALID_OBSERVATION, INSUFFICIENT_JOB_TYPE_EVIDENCE; INVALID_OBSERVATION is recorded for bad technical receipts. Feasibility states: NUMERICALLY_FEASIBLE, NUMERICALLY_INFEASIBLE, NOT_EVALUABLE_DUE_TO_INSUFFICIENT_EVIDENCE. Missing evidence never receives a feasible/infeasible scientific label. Vmax/Vsum are null for incomplete evidence; raw simulator metrics remain available as provenance, not adaptive labels.

Subprocess launch/crash, corrupt/truncated output, missing receipts or incomplete atomic writes remain nonterminal and may retry the SAME scientific row identity. Each launch has a separate attempt identity. ExecutionComplete requires successful input/hash/schema/objective/evidence validation, not merely a zero subprocess return code. A technical failure is not infeasibility.

Every stage summarizes terminal/valid/insufficient counts, affected tasks and job types, and all available evidence denominators. After all planned executions terminate with any evidence shortfall, status is STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL. Next-wave/repeat generation checks usable unique quota for every selected task before generating any coordinate and stops with EVIDENCE_SHORTFALL_REQUIRES_POLICY. Final scoped export requires all planned valid labels. No replacement geometry, fresh seed, longer duration or masked training policy is invented here. Technical failures can coexist with shortfalls and remain resumable; the shortfall report is still explicit.

## Blocker 2: collect development before unlocking held-out data

Default scope is TRAIN_DEV (alias TRAIN+DEVELOPMENT), not ALL. TRAIN and DEVELOPMENT can also run independently. Each adaptive task reads only its own evidence-valid observations in its selected scope. Locked observations are never required for train/dev wave generation.

Phase-A child files preserve EVERY master field and original Foundation_Row_ID, Plan_Row_ID, Base_Geometry_ID and Task_ID. Names are foundation_v1_phaseA__TRAIN.csv.gz, __DEVELOPMENT.csv.gz, __TRAIN_DEV.csv.gz and __LOCKED_SYNTHETIC_TEST.csv.gz. The full master remains immutable. Adaptive names use foundation_v1_wave01__TRAIN_DEV.csv.gz etc.; repeat plans are likewise scoped. Optional per-task plans also contain their task identity in the filename.

Without --unlock-locked-test, locked scientific execution, adaptive/repeat generation, ingestion and export refuse without explicit sealed-test collection authorization. Metadata audits/dry runs may count and validate locked definitions. The flag is an operational deliberate-action interlock; it does not prove a model freeze or enforce filesystem access control against a determined operator. ALL requires unlock for outcome operations and dispatches isolated TRAIN_DEV and LOCKED stages sequentially; there is no mixed training export.

Ledger/status/execution-child plans/exports: foundation_v1_runs/train_dev/ versus foundation_v1_runs/locked_synthetic_test/. Raw worker trees: src/peacsim/output/optimization/foundation_v1_runs/train_dev/ versus locked_synthetic_test/. Chunk/log/launch identities include scope. Database contamination checks inspect task-ID metadata before outcome bodies. TRAIN, DEVELOPMENT and TRAIN_DEV exports filter selected tasks and never include locked rows. Synthetic integration tests verify actual scoped exports and adaptive generation without any held-out outcomes in the train/dev ledger.

The approved full campaign collects TRAIN+DEVELOPMENT Phase A -> waves1..4 -> repeats -> export, then deliberately unlocked LOCKED Phase A -> waves1..4 -> repeats -> export. Locked outcomes are precomputed but sealed from all fitting and model selection. See SCC_COMMANDS.md for future templates only.

|Scope|Workloads|Contexts|Phase A|Adaptive|Repeats|Final expected|
|---|---:|---:|---:|---:|---:|---:|
|TRAIN|400|1600|240000|128000|32000|400000|
|DEVELOPMENT|50|200|30000|16000|4000|50000|
|TRAIN_DEV|450|1800|270000|144000|36000|450000|
|LOCKED_SYNTHETIC_TEST|50|200|30000|16000|4000|50000|
|Eventual ALL|500|2000|300000|160000|40000|500000|

TRAIN_DEV overlaps TRAIN/DEVELOPMENT by identity; views are alternative execution scopes, not additional scientific rows. Terminal IDs avoid double collection when switching between those scopes.

## Preserved design and A100 provenance

All400 synthetic profile coordinates/IDs,500 workload INIs/compositions/QoS assignments,2000 contexts,variable-J domains,cost/ISO constants,idle120W,master seed,candidate allocations,adaptive radii and measured-only rules remain unchanged. The master Phase-A compressed bytes are identical before/after. No P100/A100 cross-hardware interpolation was added. All5 MoE anchors are A100 for Foundation V1. Only real-anchor attribution in profiles.json and the corresponding profile_partition_freeze seal changed; synthetic profile manifest bytes did not. Version-1 real-anchor profile_hash values are intentionally preserved: the generator computes the existing legacy identity hash, then exposes confirmed A100 operational attribution. The legacy token exists only in that hash-compatibility expression and historical patch evidence, not operational attribution fields.

ONE FAT SCC NODE remains the approved architecture: one global campaign lock, one orchestrator, unchanged parallel-combo-runner, bounded local workers. No multi-node/distributed orchestration was added. Simulator/AQA/estimator source is unchanged by this patch. No commit or push.

## Validation and limits

All original tests remain, with the old insufficient-evidence numerical assertion updated to the required non-evaluable semantics and the helper mock updated for scoped arguments. Additional tests cover valid/insufficient terminal resume, missing/truncated/corrupt receipts, mixed scheduling, full fake crash/retry/success, explicit stage shortfall, adaptive quota stopping, locked interlocks, scope-only adaptive parents, contamination refusal, isolated exports and count algebra. All test observations are synthetic. Existing full master preflight rechecks all rows/domains/source quotas and all500 real-parser INIs. Deterministic regeneration compares complete generator outputs in temporary/in-memory space without overwriting originals. Child views are regenerated byte-identically and compared row-for-row to the master.

Real simulator integration and SCC execution are NOT RUN. This patch does not certify future hardware/power-loss behavior, and no automatic evidence-replacement policy is approved.

Master Phase-A SHA256 before: `bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f`

Master Phase-A SHA256 after: `bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f`

Deterministic regenerated files: 517. All match.

Final unit/synthetic log:
```
.....................................................                    [100%]
53 passed in 27.94s
```

Public CLI validation: PASS for TRAIN_DEV270000,TRAIN240000,DEVELOPMENT30000,LOCKED30000 metadata dry runs. Locked and ALL execution without unlock refuse before launch; locked adaptive generation also refuses. No scientific database/output was created.

Unchanged combo-runner integration:150 existing planned rows,6 worker command chunks,--plan-dry-run only; no FlexDC workers launched.
