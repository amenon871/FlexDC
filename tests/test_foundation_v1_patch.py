from tests.test_foundation_v1_legacy_entry_fixture import invoke as legacy_stage_entry
"""Narrow release-blocker regression tests; all outcomes are synthetic."""
import copy
import json
import sqlite3
from pathlib import Path
import pytest

from am_flexdc_foundation_v1_common import (contexts,Domain,make_row,write_csv,write_json,read_json,read_csv,digest,file_hash)
from foundation_v1 import audit,campaign,adaptive,scope
from tests.test_foundation_v1 import fake_receipt,fake_observations

@pytest.fixture
def task():return copy.deepcopy(contexts()[0])

def reseal(receipt):
    receipt.pop('body_sha256',None);receipt['body_sha256']=digest(receipt);return receipt

def insufficient(receipt):
    receipt=copy.deepcopy(receipt);receipt['evidence_counts'][0]['observations']=0
    receipt['evidence_status']='INSUFFICIENT_JOB_TYPE_EVIDENCE';return reseal(receipt)

def receipt_folder(tmp_path,row,receipt):
    path=tmp_path/'raw'/'foundation_evidence';path.mkdir(parents=True,exist_ok=True)
    write_json(path/(row['plan_row_id']+'.json'),receipt)
    return path.parent

def test_complete_and_insufficient_are_both_terminal(task,tmp_path):
    row,r=fake_receipt(task);plan=tmp_path/'p.csv.gz';write_csv(plan,[row]);db=tmp_path/'valid.sqlite'
    out=receipt_folder(tmp_path,row,r);report=audit.ingest(plan,[out],db)
    assert report['status']=='VALID_COMPLETE' and not campaign.pending_rows(plan,db)
    # A distinct isolated ledger for the same deterministic insufficient execution.
    bad=tmp_path/'insufficient.sqlite';(out/'foundation_evidence'/(row['plan_row_id']+'.json')).write_text(json.dumps(insufficient(r)))
    report=audit.ingest(plan,[out],bad)
    assert report['status']=='STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL'
    assert report['counts']['EXECUTION_COMPLETE']==1 and report['counts']['VALID_COMPLETE']==0
    assert report['counts']['INSUFFICIENT_JOB_TYPE_EVIDENCE']==1 and report['counts']['missing']==0
    assert not report['scientifically_complete'] and report['task_shortfalls'][0]['valid_row_shortfall']==1
    assert report['evidence_shortfalls'][0]['job_types'][0]['job_type_id']==0
    assert not campaign.pending_rows(plan,bad)
    assert audit.read_task_observations(bad,task['Task_ID'],True)==[]
    with sqlite3.connect(bad) as conn:
        body=json.loads(conn.execute('SELECT body FROM terminal_executions').fetchone()[0])
        assert body['execution_status']=='EXECUTION_COMPLETE'
        assert body['observation_status']=='INSUFFICIENT_JOB_TYPE_EVIDENCE'
        assert body['numerical_status']=='NOT_EVALUABLE_DUE_TO_INSUFFICIENT_EVIDENCE'
        assert body['Vmax'] is None and body['Vsum'] is None and not body['evidence_qualified_feasible']
        assert body['raw_results']==r['results'] and body['evidence_counts'][0]==0
    # Reingestion/restart is idempotent and cannot turn terminal shortfall into missing.
    for _ in range(3):
        assert audit.ingest(plan,[out],bad)['counts']['missing']==0
        assert not campaign.pending_rows(plan,bad)
    with pytest.raises(ValueError,match='Incomplete scientific labels'):audit.store(audit.connection(tmp_path/'reject.sqlite'),body,{})
    with pytest.raises(ValueError,match='insufficient'):adaptive.usable([body],len(body['Pj']))

@pytest.mark.parametrize('kind',['missing','truncated','corrupt'])
def test_technical_failures_remain_retryable(task,tmp_path,kind):
    row,r=fake_receipt(task);plan=tmp_path/'p.csv.gz';write_csv(plan,[row]);db=tmp_path/'x.sqlite'
    out=tmp_path/'raw';(out/'foundation_evidence').mkdir(parents=True)
    p=out/'foundation_evidence'/(row['plan_row_id']+'.json')
    if kind=='truncated':p.write_text('{')
    elif kind=='corrupt':r['body_sha256']='wrong';p.write_text(json.dumps(r))
    report=audit.ingest(plan,[out],db)
    assert report['status']=='TECHNICAL_RETRY_REQUIRED' and report['counts']['missing']==1
    assert [x['plan_row_id'] for x in campaign.pending_rows(plan,db)]==[row['plan_row_id']]
    # Same immutable identity is accepted when a later identical scientific attempt completes.
    row,r=fake_receipt(task);p.write_text(json.dumps(r))
    assert audit.ingest(plan,[out],db)['status']=='VALID_COMPLETE'
    assert campaign.pending_rows(plan,db)==[]

def test_mixed_resume_schedules_only_technical_and_never_started(task,tmp_path):
    rows=[];db=tmp_path/'mix.sqlite';conn=audit.connection(db)
    for i in range(4):
        t=copy.deepcopy(task);t['Task_ID']=task['Task_ID']+'_SYNTHETIC_'+str(i)
        row,r=fake_receipt(t);rows.append(row)
        if i<2:
            o=audit.observation(row,r if i==0 else insufficient(r),t)
            with conn:audit.store_execution(conn,o,{})
        elif i==2:
            with conn:audit.execution_state(conn,row['plan_row_id'],t['Task_ID'],'TECHNICAL_RETRY_REQUIRED',{'reason':'fake subprocess crash'})
    conn.close();plan=tmp_path/'mixed.csv.gz';write_csv(plan,rows)
    assert [r['plan_row_id'] for r in campaign.pending_rows(plan,db)]==[r['plan_row_id'] for r in rows[2:]]

def prepare_fake_campaign(task,tmp_path,monkeypatch,row):
    rows=row if isinstance(row,list) else [row]
    plan=tmp_path/'p.csv.gz';write_csv(plan,rows);db=tmp_path/'db.sqlite'
    monkeypatch.setattr(campaign,'ARTIFACTS',tmp_path)
    monkeypatch.setattr(campaign,'stage_path',lambda *args:plan)
    monkeypatch.setattr(campaign,'read_json',lambda p:{'source_files':{},'phaseA_sha256':file_hash(plan)})
    # Frozen source/master guards still execute against this synthetic plan only.
    write_csv(tmp_path/'foundation_v1_phaseA_plan.csv.gz',rows)
    monkeypatch.setattr(scope,'run_root',lambda s:tmp_path/'runs'/scope.group(s))
    monkeypatch.setattr(scope,'raw_root',lambda s:tmp_path/'raw'/scope.group(s))
    monkeypatch.setattr(scope,'assert_database_scope',lambda *args:None)
    return plan,db

def test_campaign_insufficient_resume_launches_zero_more_processes(task,tmp_path,monkeypatch):
    row,r=fake_receipt(task);plan,db=prepare_fake_campaign(task,tmp_path,monkeypatch,row)
    out=receipt_folder(tmp_path,row,insufficient(r));audit.ingest(plan,[out],db)
    def forbidden(*args,**kwargs):raise AssertionError('Terminal insufficient row reran')
    monkeypatch.setattr(campaign,'call_combo',forbidden)
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(db),'--task',task['Task_ID']])
    for _ in range(2):
        with pytest.raises(RuntimeError,match='EVIDENCE_SHORTFALL'):legacy_stage_entry()
    report=read_json(tmp_path/'runs/train_dev/phaseA__TRAIN_DEV_status.json')
    assert report['counts']['missing']==0 and not report['scientifically_complete']

def test_campaign_successful_resume_launches_zero_more_processes(task,tmp_path,monkeypatch):
    row,r=fake_receipt(task);plan,db=prepare_fake_campaign(task,tmp_path,monkeypatch,row)
    out=receipt_folder(tmp_path,row,r);audit.ingest(plan,[out],db)
    monkeypatch.setattr(campaign,'call_combo',lambda *a,**k:pytest.fail('Valid row reran'))
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(db),'--task',task['Task_ID']])
    assert legacy_stage_entry()['status']=='VALID_COMPLETE'

@pytest.mark.parametrize('partition',['LOCKED_SYNTHETIC_TEST','ALL'])
def test_locked_execution_refuses_before_launch(partition,monkeypatch):
    monkeypatch.setattr(campaign,'call_combo',lambda *a,**k:pytest.fail('Locked run launched'))
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--partition',partition])
    with pytest.raises(ValueError,match='sealed synthetic test partition'):legacy_stage_entry()
    scope.require_unlock(partition,True)

def test_locked_explicit_unlock_dispatches_to_isolated_scope(monkeypatch):
    calls=[];monkeypatch.setattr(campaign,'execute_scope',lambda a,p:calls.append((p,a.unlock_locked_test)))
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--partition','ALL','--unlock-locked-test'])
    legacy_stage_entry()
    assert calls==[('TRAIN_DEV',True),('LOCKED_SYNTHETIC_TEST',True)]
    assert scope.database_path('TRAIN')==scope.database_path('DEVELOPMENT')
    assert scope.database_path('TRAIN')!=scope.database_path('LOCKED_SYNTHETIC_TEST')
    assert scope.raw_root('TRAIN_DEV')!=scope.raw_root('LOCKED_SYNTHETIC_TEST')

def test_default_scope_is_train_dev(monkeypatch):
    calls=[];monkeypatch.setattr(campaign,'execute_scope',lambda a,p:calls.append((p,a.execute)))
    monkeypatch.setattr('sys.argv',['foundation']);legacy_stage_entry()
    assert calls==[('TRAIN_DEV',False)] and scope.normalize('TRAIN+DEVELOPMENT')=='TRAIN_DEV'

@pytest.mark.parametrize('partition',['TRAIN','DEVELOPMENT','TRAIN_DEV','LOCKED_SYNTHETIC_TEST'])
def test_adaptive_scope_uses_only_its_partitions(task,tmp_path,monkeypatch,partition):
    tasks=[];ledgers={}
    for name in ('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST'):
        t=copy.deepcopy(task);t['Task_ID']='SYNTHETIC_'+name;t['split']=name;tasks.append(t)
        db=tmp_path/(scope.group(name)+'.sqlite');ledgers[name]=db
        conn=audit.connection(db)
        for o in fake_observations(t,lambda x,i:(.2,[.1]*Domain(t['domain']).j)):
            o.update(Task_ID=t['Task_ID'],simulation_seed=20)
            with conn:audit.store(conn,o,{})
        conn.close()
    monkeypatch.setattr(scope,'contexts',lambda:tasks)
    monkeypatch.setattr(scope,'ARTIFACTS',tmp_path)
    monkeypatch.setattr(scope,'run_root',lambda s:tmp_path/'runs'/scope.group(s))
    # Train/dev ledger has NO held-out outcomes; locked ledger has NO train/dev.
    path=adaptive.generate_stage(ledgers['TRAIN'] if partition=='TRAIN_DEV' else ledgers[partition],partition,wave_number=1,unlock=partition=='LOCKED_SYNTHETIC_TEST')
    rows=read_csv(path);expected=[t for t in tasks if t['split'] in scope.SCOPES[partition]]
    assert {r['Task_ID'] for r in rows}=={t['Task_ID'] for t in expected}
    assert len(rows)==20*len(expected)
    receipt=read_json(path.with_suffix('.receipt.json'))
    assert set(receipt['source_partitions'])==set(scope.SCOPES[partition])
    for r in rows:
        provenance=json.loads(r['provenance']) if 'provenance' in r else {}
        if provenance.get('parent_row_ids'):
            assert all(pid in {o['Plan_Row_ID'] for o in audit.read_task_observations(ledgers[expected[0]['split']],r['Task_ID'])} for pid in provenance['parent_row_ids'])

def test_adaptive_locked_gate_and_shortfall_produce_no_coordinates(task,tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='sealed synthetic test partition'):adaptive.generate_stage('does-not-exist','LOCKED_SYNTHETIC_TEST',wave_number=1)
    monkeypatch.setattr(scope,'contexts',lambda:[task]);monkeypatch.setattr(scope,'ARTIFACTS',tmp_path)
    monkeypatch.setattr(scope,'run_root',lambda s:tmp_path/'runs')
    db=tmp_path/'x.sqlite';conn=audit.connection(db)
    obs=fake_observations(task,lambda x,i:(.2,[.1]*Domain(task['domain']).j))
    for o in obs[:-1]:
        o.update(Task_ID=task['Task_ID'],simulation_seed=20)
        with conn:audit.store(conn,o,{})
    conn.close()
    with pytest.raises(ValueError,match='EVIDENCE_SHORTFALL_REQUIRES_POLICY'):adaptive.generate_stage(db,'TRAIN_DEV',wave_number=1)
    assert not list(tmp_path.glob('*wave*.csv.gz'))
    assert read_json(tmp_path/'runs/adaptive_shortfall.json')['tasks'][0]['usable_unique']==149

def test_train_dev_rejects_contaminated_locked_database(task,tmp_path):
    locked=next(t for t in contexts() if t['split']=='LOCKED_SYNTHETIC_TEST')
    conn=audit.connection(tmp_path/'bad.sqlite')
    with conn:audit.store(conn,{'Plan_Row_ID':'F1_SYNTHETIC','Task_ID':locked['Task_ID'],'simulation_seed':20},{})
    conn.close()
    with pytest.raises(ValueError,match='Partition contamination'):scope.assert_database_scope(tmp_path/'bad.sqlite','TRAIN_DEV')

def test_partition_count_algebra_and_same_ids():
    counts={s:len(scope.selected_tasks(s)) for s in scope.SCOPES}
    assert counts=={'TRAIN':1600,'DEVELOPMENT':200,'TRAIN_DEV':1800,'LOCKED_SYNTHETIC_TEST':200,'ALL':2000}
    assert counts['TRAIN_DEV']*150==270000 and counts['LOCKED_SYNTHETIC_TEST']*150==30000
    assert counts['TRAIN_DEV']*250==450000 and counts['LOCKED_SYNTHETIC_TEST']*250==50000
    assert counts['ALL']*250==500000
    # Child-plan row and geometry identity audit is performed over ALL master rows
    # by patch_review.audit_views(), not by sampling a few example row IDs.

def test_a100_metadata_identity_and_frozen_science():
    from foundation_v1.profiles import generate
    from am_flexdc_foundation_v1_common import ARTIFACTS
    data=read_json(ARTIFACTS/'profiles.json');real,synthetic=generate()
    assert data=={'real_anchors':real,'synthetic_profiles':synthetic}
    assert all(a['hardware_provenance']=='A100' for a in real if a['hardware_class']=='A100')
    before=read_json(ARTIFACTS/'patch_v1/before_hashes.json')
    for name,sha in before.items():
        if name.endswith('.ini') or name.endswith('foundation_v1_phaseA_plan.csv.gz') or name.endswith('foundation_v1_profile_manifest.csv') or name.endswith('contexts.json') or name.endswith('workloads.json'):
            assert file_hash(ARTIFACTS.parent/name)==sha,name


def test_fake_crash_then_success_retries_same_identity(task,tmp_path,monkeypatch):
    row,r=fake_receipt(task);plan,db=prepare_fake_campaign(task,tmp_path,monkeypatch,row)
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    calls=[]
    def launch(command,**kwargs):
        ids=[x['plan_row_id'] for x in read_csv(command[command.index('--sweep-plan-file')+1])];calls.append(ids)
        if len(calls)==1:raise OSError('SYNTHETIC subprocess launch crash')
        out=scope.raw_root('TRAIN_DEV')/'foundation_v1_FAKE'/'foundation_evidence';out.mkdir(parents=True)
        write_json(out/(row['plan_row_id']+'.json'),r);return 0
    monkeypatch.setattr(campaign,'call_combo',launch)
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(db),'--task',task['Task_ID']])
    with pytest.raises(RuntimeError,match='Partial outputs'):legacy_stage_entry()
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT status FROM execution_states').fetchone()[0]=='TECHNICAL_RETRY_REQUIRED'
    assert legacy_stage_entry()['status']=='VALID_COMPLETE'
    assert calls==[[row['plan_row_id']],[row['plan_row_id']]]
    legacy_stage_entry();assert len(calls)==2


def test_scoped_exports_never_mix_locked_outcomes(task,tmp_path,monkeypatch):
    tasks=[]
    for part in ('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST'):
        t=copy.deepcopy(task);t.update(Task_ID='FAKE_EXPORT_'+part,split=part);tasks.append(t)
        conn=audit.connection(tmp_path/(scope.group(part)+'.sqlite'))
        row,r=fake_receipt(t);template=audit.observation(row,r,t)
        for i in range(230):
            o={**template,'Plan_Row_ID':f'F1_FAKE_{part}_{i}', 'Base_Geometry_ID':f'B_{part}_{i}'}
            with conn:audit.store(conn,o,{})
        for i in range(10):
            for seed in (21,22):
                o={**template,'Plan_Row_ID':f'F1_FAKE_{part}_{i}_{seed}','Base_Geometry_ID':f'B_{part}_{i}','simulation_seed':seed}
                with conn:audit.store(conn,o,{})
        conn.close()
    monkeypatch.setattr(scope,'contexts',lambda:tasks)
    monkeypatch.setattr(scope,'run_root',lambda p:tmp_path/'runs'/scope.group(p))
    db=tmp_path/'train_dev.sqlite'
    for part,n in [('TRAIN_DEV',500),('TRAIN',250),('DEVELOPMENT',250)]:
        audit.export(db,part)
        rows=read_csv(tmp_path/'runs/train_dev/exports'/part/'foundation_v1_master_observations.csv.gz')
        assert len(rows)==n and {r['split'] for r in rows}==set(scope.SCOPES[part])
        assert all(r['split']!='LOCKED_SYNTHETIC_TEST' for r in rows)
    with pytest.raises(ValueError,match='sealed synthetic test partition'):audit.export(tmp_path/'locked_synthetic_test.sqlite','LOCKED_SYNTHETIC_TEST')
    audit.export(tmp_path/'locked_synthetic_test.sqlite','LOCKED_SYNTHETIC_TEST',True)
    rows=read_csv(tmp_path/'runs/locked_synthetic_test/exports/LOCKED_SYNTHETIC_TEST/foundation_v1_master_observations.csv.gz')
    assert len(rows)==250 and {r['split'] for r in rows}=={'LOCKED_SYNTHETIC_TEST'}


def test_completed_chunk_is_not_misreported_as_technical_failure(task,tmp_path,monkeypatch):
    row,r=fake_receipt(task);other,_=fake_receipt(contexts()[1])
    plan,db=prepare_fake_campaign(task,tmp_path,monkeypatch,[row,other])
    out=receipt_folder(tmp_path,row,r);audit.ingest(plan,[out],db)
    monkeypatch.setattr(campaign,'call_combo',lambda *a,**k:pytest.fail('Completed chunk relaunched'))
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(db),'--chunk-count','2','--chunk-index','0'])
    report=legacy_stage_entry()
    assert report['status']=='VALID_COMPLETE' and report['counts']['planned']==1
    assert report['whole_scope_progress']['counts']['missing']==1
    assert not report['whole_scope_progress']['scientifically_complete']
