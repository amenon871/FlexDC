"""Campaign-boundary regressions; all state is disposable and local."""
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import numpy as np

from am_flexdc_foundation_v1_common import ARTIFACTS,Domain,contexts,iter_csv,write_csv,write_json,file_hash,digest
from foundation_v1 import adaptive,audit,scope,simple_campaign as campaign,full_campaign,stage_certification as certificates,boundary_snapshot
from tests.test_foundation_v1_simple_collector import FakeLaunch


def test_adaptive_query_uses_task_seed_row_index(tmp_path):
    database=tmp_path/'observations.sqlite'
    db=sqlite3.connect(database)
    db.execute('CREATE TABLE observations(row_id TEXT PRIMARY KEY,task_id TEXT,seed INTEGER,body TEXT,body_hash TEXT,source TEXT)')
    body=json.dumps({'Plan_Row_ID':'r','Task_ID':'t'})
    db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?)',('r','t',20,body,digest(json.loads(body)),'{}'));db.commit()
    before=' '.join(str(x) for x in db.execute("EXPLAIN QUERY PLAN SELECT body FROM observations WHERE task_id='t' AND seed=20 ORDER BY row_id").fetchone())
    db.close();audit.ensure_planning_indexes(database)
    ro=audit.readonly_connection(database)
    after=' '.join(str(x) for x in ro.execute("EXPLAIN QUERY PLAN SELECT body FROM observations WHERE task_id='t' AND seed=20 ORDER BY row_id").fetchone());ro.close()
    assert 'SCAN observations' in before
    assert 'idx_observations_task_seed_row' in after and 'SEARCH observations' in after


def test_read_only_snapshot_preserves_scientific_inventory(tmp_path):
    source=tmp_path/'source.sqlite';db=audit.connection(source)
    body={'Plan_Row_ID':'r','Task_ID':'t','simulation_seed':20};db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?)',('r','t',20,json.dumps(body),digest(body),'{}'));db.commit();db.close()
    receipt=boundary_snapshot.snapshot(source,tmp_path/'snapshot.sqlite')
    assert receipt['source_was_opened_read_only']
    assert receipt['source_inventory']['database_content_sha256']==receipt['destination_inventory']['database_content_sha256']


def test_default_protocol_preserves_exact_wave_bytes():
    assert adaptive.PRODUCTION_PROTOCOL['initial']==150
    assert adaptive.PRODUCTION_PROTOCOL['wave_quotas']==(4,6,6,4)
    assert sum(adaptive.PRODUCTION_PROTOCOL['wave_quotas'])==20
    assert adaptive.PRODUCTION_PROTOCOL['repeat_bases']==10
    assert adaptive.PRODUCTION_PROTOCOL['repeat_seeds']==(21,22)


def test_vectorized_global_pool_is_bit_exact_for_all_job_counts_and_servers():
    seen=set()
    for task in contexts():
        key=(task['domain']['job_type_count'],task['domain']['server_count'])
        if key in seen:continue
        seen.add(key);domain=Domain(task['domain'])
        legacy=np.asarray(domain.global_pool(257,'boundary-parity',*key))
        fast=np.asarray(adaptive.exact_global_pool(domain,257,'boundary-parity',*key))
        assert np.array_equal(legacy,fast),key
    assert seen=={(j,n) for j in range(2,7) for n in (250,1000)}


def test_scaled_real_state_machine_crosses_every_boundary_and_resumes(tmp_path,monkeypatch):
    selected=[next(t for t in contexts() if t['split']==split and t['domain']['job_type_count']==2)
        for split in ('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST')]
    ids={t['Task_ID'] for t in selected};master={t['Task_ID']:[] for t in selected}
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if row['Task_ID'] in ids and len(master[row['Task_ID']])<4:master[row['Task_ID']].append(row)
        if all(len(v)==4 for v in master.values()):break
    def tasks(part=None,task_id=None):
        return [t for t in selected if t['split'] in scope.SCOPES[scope.normalize(part)] and (task_id is None or t['Task_ID']==task_id)]
    monkeypatch.setattr(scope,'ROOT',tmp_path);monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign.collector,'ROOT',tmp_path)
    (tmp_path/'src/peacsim/output/optimization').mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(scope,'selected_tasks',tasks)
    release={'uncommitted_source_sha256':'BOUNDARY_FIXTURE','phaseA_sha256':'FIXTURE'}
    monkeypatch.setattr(campaign,'verify_release',lambda:release)
    monkeypatch.setattr(campaign,'versions',lambda require=False:{'status':'FIXTURE'})
    monkeypatch.setattr('foundation_v1.environment.versions',lambda require=False:{'status':'FIXTURE'})
    monkeypatch.setattr('foundation_v1.operational_release.require_transition',lambda *a,**k:[])
    protocol={'initial':4,'wave_quotas':(1,1,1,1),'radii':tuple(adaptive.RADII),'global_pool':128,
        'repeat_bases':2,'repeat_seeds':(21,),'name':'LOCAL_BOUNDARY_FIXTURE'}
    def view(part):
        rows=[row for task in tasks(part) for row in master[task['Task_ID']]];path=scope.plan_path('phaseA',part)
        write_csv(path,rows);write_json(path.with_suffix('.receipt.json'),{'scope':part,'rows':len(rows),'sha256':file_hash(path)})
        return path
    monkeypatch.setattr(scope,'make_phaseA_view',view)
    launcher=FakeLaunch(tmp_path/'simulator_outputs',selected)
    original=campaign.collector.collect
    monkeypatch.setattr(campaign.collector,'collect',lambda *a,**k:original(*a,**dict(k,launch=launcher,chunk_rows=2)))
    monkeypatch.setattr(full_campaign,'export_scope',lambda *a,**k:{'status':'FIXTURE_EXPORT_PASS'})
    def metrics(part):
        db=sqlite3.connect(scope.database_path(part));count=db.execute('SELECT count(*) FROM observations').fetchone()[0];db.close()
        return {'valid_rows':count}
    monkeypatch.setattr(full_campaign,'ledger_metrics',metrics)
    args=SimpleNamespace(max_workers=3,execute=True,resume=True,unlock_locked_test=True,storage_emergency_gib=0,
        adaptive_progress_seconds=.01,adaptive_stall_seconds=30.,test_protocol=protocol)
    state=campaign.run_full(args,'ALL')
    assert state['status']=='FOUNDATION_V1_COMPLETE'
    assert set(state['stages'])=={p+':'+s for p in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') for s in full_campaign.STAGES}
    first_calls=len(launcher.calls)
    again=campaign.run_full(args,'ALL')
    assert again['status']=='FOUNDATION_V1_COMPLETE' and len(launcher.calls)==first_calls
    assert all(v['simulation_calls_this_invocation']==0 for v in again['stages'].values())


def test_planning_verifier_rejects_consumed_body_tampering(tmp_path):
    db=audit.connection(tmp_path/'o.sqlite')
    body={'Plan_Row_ID':'row','Task_ID':'task','simulation_seed':20}
    db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?)',('row','task',20,json.dumps(body),digest(body),'{}'));db.commit();db.close()
    ro=audit.readonly_connection(tmp_path/'o.sqlite')
    assert audit.read_task_observations(tmp_path/'o.sqlite','task',True,ro,True)==[body];ro.close()
    db=sqlite3.connect(tmp_path/'o.sqlite');db.execute("UPDATE observations SET body='{}' WHERE row_id='row'");db.commit();db.close()
    ro=audit.readonly_connection(tmp_path/'o.sqlite')
    try:
        import pytest
        with pytest.raises(ValueError,match='checksum mismatch'):audit.read_task_observations(tmp_path/'o.sqlite','task',True,ro,True)
    finally:ro.close()


def test_exact_recovery_state_prepares_missing_wave1_without_simulation(tmp_path,monkeypatch):
    task=next(t for t in contexts() if t['split']=='TRAIN' and t['domain']['job_type_count']==2)
    rows=[]
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if row['Task_ID']==task['Task_ID']:rows.append(row)
        if len(rows)==4:break
    protocol={'initial':4,'wave_quotas':(1,1,1,1),'radii':tuple(adaptive.RADII),'global_pool':128,
        'repeat_bases':2,'repeat_seeds':(21,),'name':'LOCAL_BOUNDARY_FIXTURE'}
    def tasks(part=None,task_id=None):return [task] if task_id in (None,task['Task_ID']) else []
    monkeypatch.setattr(scope,'ROOT',tmp_path);monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign.collector,'ROOT',tmp_path)
    (tmp_path/'src/peacsim/output/optimization').mkdir(parents=True)
    monkeypatch.setattr(scope,'selected_tasks',tasks)
    release={'uncommitted_source_sha256':'BOUNDARY_FIXTURE','phaseA_sha256':'FIXTURE'}
    monkeypatch.setattr(campaign,'verify_release',lambda:release);monkeypatch.setattr(campaign,'versions',lambda require=False:{'status':'FIXTURE'})
    monkeypatch.setattr('foundation_v1.environment.versions',lambda require=False:{'status':'FIXTURE'})
    transition_checks=[]
    monkeypatch.setattr('foundation_v1.operational_release.require_transition',lambda *a,**k:transition_checks.append((a,k)) or [])
    root=tmp_path/'foundation_v1_runs';scope.PLAN_ROOT=root/'scientific_stage_plans'
    plan=scope.plan_path('phaseA','TRAIN_DEV');write_csv(plan,rows);write_json(plan.with_suffix('.receipt.json'),{'scope':'TRAIN_DEV','rows':4,'sha256':file_hash(plan)})
    area=campaign.stage_area('TRAIN_DEV','phaseA');launcher=FakeLaunch(tmp_path/'outputs',[task])
    campaign.collector.collect(plan,area,2,0,launcher,2)
    database=scope.database_path('TRAIN_DEV');assert campaign.certify(plan,area,database)['status']=='PASS'
    certificates.create(plan,area,database,'TRAIN_DEV','phaseA',4,release);scope.PLAN_ROOT=None
    args=SimpleNamespace(max_workers=2,unlock_locked_test=False,execute=False,adaptive_progress_seconds=.01,
        adaptive_stall_seconds=30.,test_protocol=protocol)
    result=campaign.prepare_next_stage(args,'TRAIN_DEV')
    assert result['status']=='NEXT_STAGE_READY' and result['stage']=='wave01' and result['plan_rows']==4
    assert result['simulation_calls']==0 and result['SAFE_TO_REQUEST_COMPUTE_ALLOCATION']=='YES'
    assert transition_checks
