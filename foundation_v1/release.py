"""Explicit pre-execution release sealing; never run silently on SCC."""
from pathlib import Path
import argparse
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,read_json,write_json,file_hash,digest
from foundation_v1.environment import versions
from foundation_v1.full_campaign import MASTER_SHA256

def seal():
    old=read_json(ARTIFACTS/'preexecution_release.json')
    if file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')!=MASTER_SHA256:raise ValueError('STOP: master scientific coordinates changed')
    # Retain EVERY previously guarded path; only add reviewed implementation files.
    names=set(old['source_files'])
    for pattern in ('foundation_v1/*.py','am_*foundation*v1*.py','tests/test_foundation_v1*.py','src/peacsim/foundation_v1*.py'):
        names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))
    names.add('requirements-foundation-v1.txt');names.add('.gitattributes');names.add('configs/foundation_v1/approved_scientific_function_hashes.json')
    provenance={name:file_hash(ROOT/name) for name in sorted(names)}
    result={**old,'status':'PREPARED_NOT_RUN','revision':'FOUNDATION_V1_SIMPLE_COLLECTOR_V2',
        'source_files':provenance,'uncommitted_source_sha256':digest(provenance),
        'versions':versions(require=True)['actual'],'sealed_test_policy':'outcomes precomputed in isolated collection; never model fitting/selection',
        'scientific_simulations_started':False}
    write_json(ARTIFACTS/'preexecution_release.json',result,False);return result

def write_operational_docs():
    for p in ARTIFACTS.glob('*.md'):
        body=p.read_text(encoding='utf8')
        replacements={
          'locked outcomes require explicit unlock after model/protocol freeze':'locked outcomes require explicit unlock and remain sealed from model fitting/tuning',
          '450000 TRAIN_DEV before freeze and50000 LOCKED only after explicit unlock':'450000 TRAIN_DEV then50000 isolated LOCKED in the same precomputation campaign',
          'refuse with an explanation that model/protocol development must first be frozen':'refuse without explicit sealed-test collection authorization',
          'Collect TRAIN+DEVELOPMENT Phase A -> waves1..4 -> repeats -> export; train/select Meta V4 and FlexPFN using only these partitions; freeze model/protocol; then deliberately unlock LOCKED Phase A -> waves1..4 -> repeats -> export.':'The approved full campaign collects TRAIN+DEVELOPMENT Phase A -> waves1..4 -> repeats -> export, then deliberately unlocked LOCKED Phase A -> waves1..4 -> repeats -> export. Locked outcomes are precomputed but sealed from all fitting and model selection.',
          'ONLY THEN':'Separately and without using held-out outcomes for development',
          'TRAIN+DEVELOPMENT collection precedes model/protocol freeze; locked scientific collection requires an explicit unlock afterward.':'The one-command collection precomputes both isolated corpora; locked outcomes remain sealed from fitting and model selection.'}
        for a,b in replacements.items():body=body.replace(a,b)
        p.write_text(body,encoding='utf8')
    (ARTIFACTS/'SCC_COMMANDS.md').write_text(DOC,encoding='utf8')
    (ARTIFACTS/'FULL_CAMPAIGN_GUIDE.md').write_text(DOC,encoding='utf8')
    (ARTIFACTS/'README.md').write_text('# Foundation V1 approved full campaign\n\nStart with FULL_CAMPAIGN_GUIDE.md and FULL_CAMPAIGN_VALIDATION.md. Scientific definitions, profiles and all300000 master Phase-A rows are unchanged. The locked synthetic outcomes are precomputed in the same collection but sealed from fitting and selection.\n\n'+DOC,encoding='utf8')
    (ARTIFACTS/'LOCKED_TEST_ISOLATION_AUDIT.md').write_text('# Sealed synthetic-test policy\n\nLOCKED_SYNTHETIC_TEST is predeclared and operationally isolated, but simulator outcomes are precomputed in the same campaign before model development. Unlock explicitly authorizes collection, not fitting/tuning. Never use locked rows for Meta V4/FlexPFN fitting, architecture/hyperparameter selection, normalization or ordinary train/dev reports. Separate ledgers/raw roots/exports persist. Future completely new real-family profiles remain the strongest untouched external holdout. This is operational isolation, not adversarial filesystem access control.\n',encoding='utf8')

DOC="""# Foundation V1: one command on one SCC fat node

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
"""

DOC += """

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
"""

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',action='store_true');p.add_argument('--docs',action='store_true');a=p.parse_args()
    if a.docs:write_operational_docs()
    if a.seal:print(seal()['uncommitted_source_sha256'])
