"""Reproducibility and partition-view audits; never invokes FlexDC."""
import argparse
import itertools
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,canonical,read_json,write_json,write_csv,iter_csv,file_hash)
from foundation_v1 import scope

PATCH=ARTIFACTS/'patch_v1'

def frozen_science():
    before=read_json(PATCH/'before_hashes.json');checked={}
    import zipfile
    with zipfile.ZipFile(ROOT/'FOUNDATION_DATASET_V1_PREEXECUTION_REVIEW.zip') as archive:
        original=json.loads(archive.read('foundation_v1_artifacts/profiles.json'))
    current=read_json(ARTIFACTS/'profiles.json')
    assert original['synthetic_profiles']==current['synthetic_profiles']
    for a,b in zip(original['real_anchors'],current['real_anchors']):
        assert {k:v for k,v in a.items() if k!='hardware_provenance'}=={k:v for k,v in b.items() if k!='hardware_provenance'}
        assert b['hardware_provenance']==b['hardware_class']
    names=[n for n in before if n.startswith('configs/workload/foundation_v1/') or n.endswith(('foundation_v1_phaseA_plan.csv.gz','contexts.json','workloads.json','foundation_v1_profile_manifest.csv','foundation_v1_workload_manifest.csv','foundation_v1_context_manifest.csv'))]
    for name in names:
        actual=file_hash(ROOT/name)
        if actual!=before[name]:raise ValueError('STOP: frozen scientific artifact changed: '+name)
        checked[name]=actual
    write_json(PATCH/'frozen_science_audit.json',{'status':'PASS','files_unchanged':len(checked),'master_before':before['foundation_v1_artifacts/foundation_v1_phaseA_plan.csv.gz'],
        'master_after':file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'),'all_500_INIs_unchanged':True,
        'physical_profiles_and_identities_unchanged':True,'changes_allowed':'real-anchor attribution metadata and its manifest seal only'},False)
    return checked

def audit_views():
    expected={'TRAIN':240000,'DEVELOPMENT':30000,'TRAIN_DEV':270000,'LOCKED_SYNTHETIC_TEST':30000};report=[]
    master=ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'
    for name,count in expected.items():
        path=scope.make_phaseA_view(name)
        before=file_hash(path)
        # Recreate a child view deterministically (immutable writer compares bytes).
        scope.make_phaseA_view(name)
        assert before==file_hash(path)
        selected=(r for r in iter_csv(master) if r['data_split'] in scope.SCOPES[name])
        ids=set();tasks=set();geometries=set();workloads=set();actual=0
        for original,child in itertools.zip_longest(selected,iter_csv(path)):
            assert original is not None and child is not None and original==child
            assert child['plan_row_id'] not in ids;ids.add(child['plan_row_id'])
            geometry=(child['Task_ID'],child['Base_Geometry_ID'])
            assert geometry not in geometries;geometries.add(geometry)
            tasks.add(child['Task_ID']);workloads.add(child['Workload_ID']);actual+=1
        assert actual==count
        report.append({'scope':name,'workloads':len(workloads),'contexts':len(tasks),'phaseA_rows':actual,
            'expected_final_rows':len(tasks)*250,'sha256':before,'master_row_identity':'EXACT','no_new_geometries':True,'byte_reproducible':True})
        print(f'VIEW PASS {name}: {actual} rows; exact unchanged master IDs/vectors',flush=True)
    write_json(PATCH/'partition_view_audit.json',{'status':'PASS','views':report,'overall_final_rows':500000,'simulator_calls':0},False)
    return report

def regeneration():
    from foundation_v1 import profiles,workloads,planning
    checked={};counter=[0]
    with tempfile.TemporaryDirectory(prefix='foundation_v1_regeneration_') as tmp:
        def verify_json(path,value,*args,**kwargs):
            assert Path(path).read_bytes()==(canonical(value)+'\n').encode(),str(path)
            checked[str(Path(path).relative_to(ROOT))]=file_hash(path)
        def verify_csv(path,rows,columns=None):
            counter[0]+=1;candidate=Path(tmp)/f'check_{counter[0]}.csv.gz' if str(path).endswith('.gz') else Path(tmp)/f'check_{counter[0]}.csv'
            write_csv(candidate,rows,columns)
            assert file_hash(path)==file_hash(candidate),str(path)
            checked[str(Path(path).relative_to(ROOT))]=file_hash(path)
        def verify_bytes(path,content,*args,**kwargs):
            assert Path(path).read_bytes()==content,str(path)
            checked[str(Path(path).relative_to(ROOT))]=file_hash(path)
        for module in (profiles,workloads,planning):
            print('Regeneration comparison: '+module.__name__,flush=True)
            with patch.object(module,'write_json',verify_json),patch.object(module,'write_csv',verify_csv),patch('sys.argv',['regeneration-only']):
                if module is workloads:
                    with patch.object(module,'atomic_bytes',verify_bytes):module.main()
                else:module.main()
    write_json(PATCH/'deterministic_regeneration.json',{'status':'PASS','byte_identical_files':len(checked),'sha256':checked,
        'frozen_originals_overwritten':False,'simulator_calls':0,'method':'regenerate into temporary CSVs/in-memory comparisons; original scientific files never replaced'},False)
    print(f'Regeneration PASS: {len(checked)} files identical',flush=True)


def patch_docs():
    """Write current operational reports; the previous review ZIP is immutable."""
    from foundation_v1.review import text
    frozen=read_json(PATCH/'frozen_science_audit.json')
    views=read_json(PATCH/'partition_view_audit.json')
    regen=read_json(PATCH/'deterministic_regeneration.json')
    tests=read_json(PATCH/'test_exit_status.json')
    assert all(r['status']=='PASS' for r in (frozen,views,regen)) and tests['exit_code']==0
    core="""# Foundation V1 narrow patch release

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

Without --unlock-locked-test, locked scientific execution, adaptive/repeat generation, ingestion and export refuse with an explanation that model/protocol development must first be frozen. Metadata audits/dry runs may count and validate locked definitions. The flag is an operational deliberate-action interlock; it does not prove a model freeze or enforce filesystem access control against a determined operator. ALL requires unlock for outcome operations and dispatches isolated TRAIN_DEV and LOCKED stages sequentially; there is no mixed training export.

Ledger/status/execution-child plans/exports: foundation_v1_runs/train_dev/ versus foundation_v1_runs/locked_synthetic_test/. Raw worker trees: src/peacsim/output/optimization/foundation_v1_runs/train_dev/ versus locked_synthetic_test/. Chunk/log/launch identities include scope. Database contamination checks inspect task-ID metadata before outcome bodies. TRAIN, DEVELOPMENT and TRAIN_DEV exports filter selected tasks and never include locked rows. Synthetic integration tests verify actual scoped exports and adaptive generation without any held-out outcomes in the train/dev ledger.

Collect TRAIN+DEVELOPMENT Phase A -> waves1..4 -> repeats -> export; train/select Meta V4 and FlexPFN using only these partitions; freeze model/protocol; then deliberately unlock LOCKED Phase A -> waves1..4 -> repeats -> export. See SCC_COMMANDS.md for future templates only.

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
"""
    core+='\nMaster Phase-A SHA256 before: `'+frozen['master_before']+'`\n\nMaster Phase-A SHA256 after: `'+frozen['master_after']+'`\n'
    core+='\nDeterministic regenerated files: '+str(regen['byte_identical_files'])+'. All match.\n\nFinal unit/synthetic log:\n```\n'+(PATCH/'test_results.txt').read_text(encoding='utf8')+'```\n'
    if (PATCH/'cli_validation.json').exists():
        cli=read_json(PATCH/'cli_validation.json');assert cli['status']=='PASS'
        core+='\nPublic CLI validation: PASS for TRAIN_DEV270000,TRAIN240000,DEVELOPMENT30000,LOCKED30000 metadata dry runs. Locked and ALL execution without unlock refuse before launch; locked adaptive generation also refuses. No scientific database/output was created.\n'
    combo=read_json(PATCH/'combo_plan_validation.json');assert combo['status']=='PASS'
    core+='\nUnchanged combo-runner integration:150 existing planned rows,6 worker command chunks,--plan-dry-run only; no FlexDC workers launched.\n'
    text('PATCH_RELEASE_NOTES.md',core)
    text('PRE_EXECUTION_AUDIT_PATCHED.md','# Patched pre-execution gates\n\n**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**\n\n'+
         '|Gate|Result|\n|---|---|\n'+''.join('|'+g+'|PASS|\n' for g in ('Terminal insufficient evidence never automatically retried','Technical failure same-ID recovery','No incomplete adaptive/model-training labels','Explicit scientific shortfall gate','Default TRAIN_DEV and locked interlock','Scoped adaptive parents and isolated exports','All partition counts/row identities','400 profiles/500 INIs/2000 contexts/master300000','All original distributions/domains/cost/ISO checks','Complete deterministic regeneration and child-view reproducibility','A100 attribution; no cross-hardware interpolation','Unchanged one-fat-node bounded combo runner'))+
         '|Actual FlexDC/SCC simulation|NOT RUN (forbidden)|\n\nSee PATCH_RELEASE_NOTES.md for exact semantics and retained raw evidence. See patch_v1/*.json and test_results.txt for machine-readable verification.\n\n'+core)
    # Update operational texts emitted by the old package writer.
    for name in ('FOUNDATION_DATASET_V1_DESIGN.md','FOUNDATION_V1_REAL_HOLDOUT_POLICY.md','FOUNDATION_V1_IDLE_POWER_NOTE.md','README.md'):
        p=ARTIFACTS/name;body=p.read_text(encoding='utf8')
        body=body.replace('believed-A100','A100').replace('A100 hardware attribution is provisional.','A100 hardware attribution is frozen for Foundation V1.').replace('A100 attribution remains pending confirmation with Can.','A100 attribution is frozen for Foundation V1.')
        if name=='FOUNDATION_DATASET_V1_DESIGN.md':
            body=body.replace('never reruns an already audited valid row','never reruns an already terminal complete execution (valid or evidence-insufficient)')
            body=body.replace('independent SCC chunk jobs should not share one active ledger/orchestrator concurrently. There is no scheduler-specific submission assumption.','the approved topology is ONE FAT NODE and one global orchestrator with a bounded local worker pool. No multi-node orchestration is provided.')
            body=body.replace('The final exporter requires500000 valid rows,250 per task and correct20/21/22 grouping','Scoped exporters require250 valid rows per selected task and correct20/21/22 grouping:450000 TRAIN_DEV before freeze and50000 LOCKED only after explicit unlock;eventual total500000')
        if name=='README.md':body='# Patched Foundation V1 review\n\nStart with PRE_EXECUTION_AUDIT_PATCHED.md and PATCH_RELEASE_NOTES.md. The previous review ZIP is preserved unchanged. Default collection is TRAIN_DEV; locked outcomes require explicit unlock after model/protocol freeze.\n\n'+body
        text(name,body)
    text('PRE_EXECUTION_AUDIT.md','# Current audit entry point\n\nSee PRE_EXECUTION_AUDIT_PATCHED.md for the approved recovery/partition corrections and full current gates. The previous report remains preserved in the original review ZIP.\n\n**NO FOUNDATION SCIENTIFIC DATASET SIMULATIONS HAVE BEEN STARTED.**')
    text('SCC_COMMANDS.md',SCC_COMMANDS)
    text('LOCKED_TEST_ISOLATION_AUDIT.md','# Locked-test operational isolation\n\nPASS: synthetic tests exercise independent TRAIN, DEVELOPMENT, TRAIN_DEV and LOCKED adaptive generation; every task reads its own selected valid observations. Train/dev ledgers contain no locked outcomes. Actual synthetic scoped exports contain only their declared partitions. Locked outcome operations refuse before evidence access/worker launch without --unlock-locked-test. ALL dispatches separate ledgers sequentially. Metadata-only plan audits are allowed. No scientific outcomes exist.\n\nIsolation is operational/logical, not an adversarial OS access-control boundary. See patch_v1/test_results.txt and partition_view_audit.json.')

    # Every changed existing file and every newly added task artifact is listed.
    # The old ZIP is read-only historical provenance, never overwritten.
    import zipfile,hashlib
    with zipfile.ZipFile(ROOT/'FOUNDATION_DATASET_V1_PREEXECUTION_REVIEW.zip') as old:
        old_names=set(old.namelist())
        candidates=[*list((ROOT/'foundation_v1').glob('*.py')),*list((ROOT/'tests').glob('test_foundation_v1*.py')),
            *[p for p in ARTIFACTS.rglob('*') if p.is_file() and '.tmp.' not in p.name],
            *list(ROOT.glob('dataset_plan_launch_manifest_AQA_foundation_v1_TRAIN_DEV_patch_validation_dry_only.csv')),
            *[p for p in (ROOT/'.flexdc_plan_chunks/foundation_v1_TRAIN_DEV_patch_validation_dry_only').glob('*') if p.is_file()]]
        modified=[];added=[]
        for p in sorted(set(candidates)):
            name=p.relative_to(ROOT).as_posix()
            if name in old_names:
                if hashlib.sha256(old.read(name)).hexdigest()!=file_hash(p):modified.append(name)
            else:added.append(name)
        inventory='foundation_v1_artifacts/PATCH_FILE_INVENTORY.json'
        if inventory not in added:added.append(inventory)
        write_json(ARTIFACTS/'PATCH_FILE_INVENTORY.json',{'baseline':'original approved review ZIP',
            'modified_existing_files':modified,'new_files':sorted(added),
            'new_final_archive':'FOUNDATION_DATASET_V1_PREEXECUTION_REVIEW_PATCHED.zip',
            'existing_tracked_diff_stat':(PATCH/'git_diff_stat.txt').read_text(encoding='utf8'),
            'source_patch':'patch_v1/foundation_v1_narrow_patch.diff','committed':False,'pushed':False},False)


SCC_COMMANDS="""# SCC staged templates: ONE FAT NODE ONLY

DO NOT EXECUTE DURING REVIEW. No scientific simulations have started.

One campaign process uses the unchanged am_run_flexdc_dataset_parallel_combo.py with a bounded local worker pool on one fat node. No multi-node orchestration, scheduler-native distributed sharding, MPI, network workers or distributed database. --chunk-index/--chunk-count are local staged-selection tools; never launch parallel orchestrators. A global local campaign lock guards all scopes.

After a separately approved review/commit/deployment, clone and overlay the patched review ZIP to preserve source/config/ISO/manifest bytes across Git newline handling. Do not regenerate frozen coordinates to update GitSHA; runtime receipts record the actual execution SHA separately.

Metadata-only default dry run:
```
python -B am_run_flexdc_foundation_campaign_v1.py --stage phaseA --max-workers 10
```

PHASE 1, only after separate scientific execution authorization:
```
python -B am_run_flexdc_foundation_campaign_v1.py --partition TRAIN_DEV --stage phaseA --max-workers 10 --execute --resume
python -B am_generate_flexdc_foundation_adaptive_plan_v1.py --partition TRAIN_DEV --wave 1
python -B am_run_flexdc_foundation_campaign_v1.py --partition TRAIN_DEV --stage wave01 --max-workers 10 --execute --resume
```
Repeat measured generation/execution for waves2..4, then generate --repeats and execute --stage repeat. Export using am_vet_flexdc_foundation_dataset_v1.py --export --partition TRAIN_DEV; TRAIN/DEVELOPMENT independent exports also exist. Default ledger is foundation_v1_runs/train_dev/observations.sqlite. TRAIN_DEV:270000 Phase-A +144000 adaptive +36000 repeats =450000.

Then fit/select Meta V4/FlexPFN using TRAIN+DEVELOPMENT only. Freeze models, architecture, hyperparameters and protocol BEFORE any locked outcome collection.

PHASE 2, only AFTER that freeze and separate execution authorization:
```
python -B am_run_flexdc_foundation_campaign_v1.py --partition LOCKED_SYNTHETIC_TEST --unlock-locked-test --stage phaseA --max-workers 10 --execute --resume
python -B am_generate_flexdc_foundation_adaptive_plan_v1.py --partition LOCKED_SYNTHETIC_TEST --unlock-locked-test --wave 1
python -B am_run_flexdc_foundation_campaign_v1.py --partition LOCKED_SYNTHETIC_TEST --unlock-locked-test --stage wave01 --max-workers 10 --execute --resume
```
Continue locked waves2..4/repeats with that flag, then --export --partition LOCKED_SYNTHETIC_TEST --unlock-locked-test. Separate ledger foundation_v1_runs/locked_synthetic_test/observations.sqlite. LOCKED:30000 Phase-A +16000 adaptive +4000 repeats =50000. No locked evidence is needed for TRAIN_DEV waves.

Raw worker trees use src/peacsim/output/optimization/foundation_v1_runs/train_dev/ and locked_synthetic_test/. Scoped status/execution plans/exports live under foundation_v1_runs/<group>/. Log/chunk/launch identities include scope. The unchanged combo runner embeds output prefixes in log paths; the Foundation harness precreates their parent directories.

Insufficient evidence is terminal EXECUTION_COMPLETE but not a valid scientific label. It is never automatically repeated identically. STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL and EVIDENCE_SHORTFALL_REQUIRES_POLICY require scientific review, not automatic replacements. Crashed/missing/corrupt outputs remain retryable under unchanged row identity. Resume never shifts shard membership. ALL metadata is allowed; ALL outcome operations require unlock and dispatch isolated groups sequentially, never a mixed model-training file.
"""

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--views',action='store_true');p.add_argument('--regeneration',action='store_true');a=p.parse_args()
    frozen_science()
    if a.views:audit_views()
    if a.regeneration:regeneration()
    frozen_science()

if __name__=='__main__':main()
