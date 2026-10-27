"""Final operational corrections. All simulator launches are fake and temporary."""
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from am_flexdc_foundation_v1_common import digest,write_json,write_csv,file_hash,iter_csv
from foundation_v1 import simple_campaign as full,simple_collector as collector,collector_receipts as receipts,scope
from foundation_v1 import stage_certification as cert,operational_release as op
from tests.test_foundation_v1_simple_collector import setup_queue,FakeLaunch,fixture_rows

RELEASE={'uncommitted_source_sha256':'SYNTHETIC_FINAL_REVIEW','phaseA_sha256':cert.MASTER}


def certified_stage(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3);launch=FakeLaunch(tmp_path,[task]);database=tmp_path/'audited.sqlite'
    collector.collect(plan,area,2,0,launch,1)
    assert full.certify(plan,area,database)['status']=='PASS'
    body=cert.create(plan,area,database,'TRAIN_DEV','phaseA',3,RELEASE)
    return rows,plan,area,database,body,launch


@pytest.mark.parametrize('damage',['missing','corrupt','fields','plan','audit','database_missing','database_corrupt','generation','journal','source','receipt_deleted'])
def test_certificate_rejects_changes(tmp_path,monkeypatch,damage):
    rows,plan,area,database,body,launch=certified_stage(tmp_path,monkeypatch)
    path=area/'stage_certification.json';release=RELEASE
    if damage=='missing':path.unlink()
    elif damage=='corrupt':path.write_text('corrupt')
    elif damage in ('fields','plan'):
        if damage=='fields':body.pop('certificate_id')
        else:body['plan_sha256']='wrong'
        body.pop('body_sha256');body['body_sha256']=digest(body);write_json(path,body,False)
    elif damage=='audit':write_json(area/'stage_audit.json',{'status':'CHANGED'},False)
    elif damage=='database_missing':database.unlink()
    elif damage=='database_corrupt':database.write_bytes(b'corrupt database')
    elif damage=='generation':
        with sqlite3.connect(area/'recovery_index.sqlite') as db:db.execute("UPDATE metadata SET value='new' WHERE key='generation'")
        db.close()
    elif damage=='journal':
        with (area/'events/parent.jsonl').open('a') as f:f.write(json.dumps({'id':rows[0]['plan_row_id'],'time':time.time()})+'\n')
    elif damage=='source':release={**RELEASE,'uncommitted_source_sha256':'OTHER'}
    else:receipts.receipt_path(area,rows[0]['plan_row_id']).unlink()
    cached,reason=cert.validate(plan,area,database,'TRAIN_DEV','phaseA',3,release)
    assert cached is None and reason


def test_certificate_created_only_after_strong_pass(tmp_path):
    area=tmp_path/'area';area.mkdir();write_json(area/'stage_audit.json',{'status':'REPAIR_REQUIRED','PASS':2,'planned':3},False)
    with pytest.raises(ValueError,match='before complete PASS'):cert.create(tmp_path/'unused',area,tmp_path/'unused_db','TRAIN_DEV','phaseA',3,RELEASE)
    assert not (area/'stage_certification.json').exists()


def test_new_invalidation_and_repair_recertifies_only_affected_row(tmp_path,monkeypatch):
    rows,plan,area,database,old,launch=certified_stage(tmp_path,monkeypatch)
    key=rows[0]['plan_row_id'];receipts.archive(area,key,{'reason':'TEST_ONLY'})
    assert not (area/'stage_certification.json').exists()
    actual=collector.collect
    monkeypatch.setattr(collector,'collect',lambda p,a,w,s:actual(p,a,w,s,launch,1))
    monkeypatch.setattr(full,'stage_area',lambda part,stage:area)
    monkeypatch.setattr(scope,'plan_path',lambda stage,part:plan)
    monkeypatch.setattr(scope,'database_path',lambda part:database)
    monkeypatch.setattr(full,'verify_release',lambda:RELEASE)
    before=len(launch.calls)
    result=full.execute_stage(SimpleNamespace(stage='phaseA',max_workers=2,storage_emergency_gib=0),'TRAIN_DEV')
    assert result['status']=='VALID_COMPLETE' and len(launch.calls)==before+1
    assert launch.calls[-1][0]['plan_row_id']==key
    new=cert.create(plan,area,database,'TRAIN_DEV','phaseA',3,RELEASE)
    assert new['certificate_id']!=old['certificate_id'] and cert.validate(plan,area,database,'TRAIN_DEV','phaseA',3,RELEASE)[0]


def multi_stage_fixture(tmp_path,monkeypatch,n=3):
    from foundation_v1 import full_campaign as historical
    monkeypatch.setattr(full,'ROOT',tmp_path);monkeypatch.setattr(scope,'ROOT',tmp_path)
    monkeypatch.setattr(collector.time,'sleep',lambda s:None)
    monkeypatch.setattr(full,'verify_release',lambda:RELEASE)
    monkeypatch.setattr(full,'versions',lambda require=False:{'status':'SYNTHETIC'})
    monkeypatch.setattr(full,'counts_for',lambda part:{'tasks':1,'total':3*n})
    monkeypatch.setattr(full,'STAGES',('phaseA','wave01','wave02'))
    monkeypatch.setattr(historical,'validate_plan',lambda *a:None)
    monkeypatch.setattr(historical,'export_scope',lambda *a:{})
    monkeypatch.setattr(historical,'ledger_metrics',lambda p:{'valid_rows':3*n})
    task,rows=fixture_rows(3*n);plans={};launch=FakeLaunch(tmp_path/'fake_outputs',[task])
    plan_root=tmp_path/'foundation_v1_runs/scientific_stage_plans';plan_root.mkdir(parents=True)
    write_json(tmp_path/'foundation_v1_runs/execution_format.json',{'schema':'FOUNDATION_V1_COMPLETION_V2','scientific_master':'SYNTHETIC_FIXTURE_ONLY'},False)
    db=scope.database_path('TRAIN_DEV')
    for i,stage in enumerate(full.STAGES):
        selected=rows[i*n:(i+1)*n];plan=plan_root/('foundation_v1_'+stage+'__TRAIN_DEV.csv.gz');write_csv(plan,selected)
        write_json(plan.with_suffix('.receipt.json'),{'scope':'TRAIN_DEV','rows':n,'sha256':file_hash(plan)})
        plans[stage]=plan;area=full.stage_area('TRAIN_DEV',stage)
        collector.collect(plan,area,1,0,launch,1,stop_after=1 if stage=='wave02' else None)
        if stage!='wave02':
            assert full.certify(plan,area,db)['status']=='PASS'
            cert.create(plan,area,db,'TRAIN_DEV',stage,n,RELEASE)
    original=collector.collect
    monkeypatch.setattr(collector,'collect',lambda p,a,w,s:original(p,a,w,s,launch,1))
    return plans,launch,db


@pytest.mark.parametrize('damage',[None,'missing','corrupt','plan','audit','invalidation'])
def test_run_full_skips_certified_prior_stages_and_resumes_active(tmp_path,monkeypatch,damage):
    plans,launch,db=multi_stage_fixture(tmp_path,monkeypatch)
    if damage:
        area=full.stage_area('TRAIN_DEV','phaseA');path=area/'stage_certification.json'
        if damage=='missing':path.unlink()
        elif damage=='corrupt':path.write_text('invalid')
        elif damage=='plan':
            body=json.loads(path.read_text());body['plan_sha256']='wrong';body.pop('body_sha256');body['body_sha256']=digest(body);write_json(path,body,False)
        elif damage=='audit':write_json(area/'stage_audit.json',{'status':'CHANGED'},False)
        else:
            row=next(iter_csv(plans['phaseA']));receipts.archive(area,row['plan_row_id'],{'reason':'SYNTHETIC after certificate'})
    calls=[];opened=[];actual_stage=full.execute_stage;actual_load=receipts.load
    def traced_stage(a,p):calls.append(a.stage);return actual_stage(a,p)
    def traced_load(area,key):opened.append(Path(area).name);return actual_load(area,key)
    monkeypatch.setattr(full,'execute_stage',traced_stage);monkeypatch.setattr(receipts,'load',traced_load)
    result=full.run_full(SimpleNamespace(max_workers=2,execute=True,resume=True,unlock_locked_test=False,storage_emergency_gib=0),'TRAIN_DEV')
    assert result['status']=='FOUNDATION_SCOPE_COMPLETE'
    assert calls==(['wave02'] if damage is None else ['phaseA','wave02'])
    if damage is None:assert not any(s in ('phaseA','wave01') for s in opened)
    assert result['resume_telemetry']['stages_skipped']==(2 if damage is None else 1)


def event(root,name,key,t,invalid=False):
    area=root/'foundation_v1_runs/train_dev/collector_v2/phaseA';events=area/'events';events.mkdir(parents=True,exist_ok=True)
    with (events/name).open('a') as f:f.write(json.dumps({'id':key,'time':t,'invalidated':invalid})+'\n')
    write_json(area/'progress.json',{'planned':3,'worker_pids':[]},False)


def test_status_completion_invalidation_repair_stale_and_rebuild(tmp_path,monkeypatch):
    monkeypatch.setattr(full.time,'time',lambda:40.)
    event(tmp_path,'a.jsonl','F1_TEST',10)
    assert full.live_status(tmp_path)['overall_complete']==1
    event(tmp_path,'invalidations.jsonl','F1_TEST',20,True)
    status=full.live_status(tmp_path)
    assert status['overall_complete']==status['rows_completed_last_hour']==0 and status['rows_per_second_last_minute']==0
    assert status['stages'][0]['pending']==3
    event(tmp_path,'z_stale.jsonl','F1_TEST',10)
    assert full.live_status(tmp_path)['overall_complete']==0
    event(tmp_path,'repair.jsonl','F1_TEST',30)
    repaired=full.live_status(tmp_path)
    assert repaired['overall_complete']==repaired['rows_completed_last_hour']==1
    assert repaired['stages'][0]['pending']==2 and repaired['rows_per_second_last_minute']==1/60
    (tmp_path/'foundation_v1_runs/status_cache.sqlite').unlink()
    assert full.live_status(tmp_path)==repaired


def test_legacy_status_cache_rebuilt_not_trusted(tmp_path):
    event(tmp_path,'success.jsonl','F1_TEST',10);event(tmp_path,'invalidations.jsonl','F1_TEST',20,True)
    db=sqlite3.connect(tmp_path/'foundation_v1_runs/status_cache.sqlite')
    db.execute('CREATE TABLE events(stage TEXT,id TEXT,t REAL,PRIMARY KEY(stage,id))')
    db.execute('CREATE TABLE offsets(path TEXT PRIMARY KEY,position INTEGER)')
    db.execute("INSERT INTO events VALUES('train_dev:phaseA','F1_TEST',20)");db.commit();db.close()
    assert full.live_status(tmp_path)['overall_complete']==0


def operational_fixture(tmp_path):
    artifacts=tmp_path/'foundation_v1_artifacts';artifacts.mkdir()
    source=tmp_path/'source.py';source.write_text('old')
    old_files={'source.py':file_hash(source)};old=dict(uncommitted_source_sha256=digest(old_files),source_files=old_files,phaseA_sha256=cert.MASTER,scientific_simulations_started=False)
    write_json(artifacts/'preexecution_release.json',old)
    old_bytes=(artifacts/'preexecution_release.json').read_bytes();source.write_text('operational bookkeeping only')
    files={'source.py':file_hash(source)}
    record=dict(previous_source_identity=old['uncommitted_source_sha256'],historical_release_sha256=file_hash(artifacts/'preexecution_release.json'),master_phaseA_sha256=cert.MASTER,scientific_master_changed=False,operational_patch_after_campaign_start=True,scientific_simulations_started=True,source_files=files,new_source_identity=digest(files),deployment_timestamp=None,completed_rows_at_transition=None)
    record['body_sha256']=digest(record);write_json(artifacts/op.NAME,record)
    return artifacts,old,old_bytes


def test_operational_guard_preserves_original_release_and_checks_patch(tmp_path):
    artifacts,old,before=operational_fixture(tmp_path)
    current=op.verify(tmp_path,artifacts,old)
    assert current['collection_initial_source_identity']==old['uncommitted_source_sha256']!=current['uncommitted_source_sha256']
    assert (artifacts/'preexecution_release.json').read_bytes()==before
    (tmp_path/'source.py').write_text('unreviewed')
    with pytest.raises(ValueError,match='Frozen operational'):op.verify(tmp_path,artifacts,old)


def test_transition_is_explicit_immutable_and_required_on_existing_campaign(tmp_path):
    artifacts,old,before=operational_fixture(tmp_path);current=op.verify(tmp_path,artifacts,old)
    status=tmp_path/'captured_status.json';write_json(status,dict(test_fixture_only=True,source_identity=old['uncommitted_source_sha256'],overall_complete=117000,active={'partition':'TRAIN_DEV','stage':'phaseA'}))
    write_json(tmp_path/'foundation_v1_runs/simple_campaign_status.json',{'source_identity':old['uncommitted_source_sha256']},False)
    with pytest.raises(ValueError,match='SOURCE_TRANSITION_REQUIRED'):op.require_transition(tmp_path,current)
    path=op.record_transition(tmp_path,current,status,old['uncommitted_source_sha256'],'TEST_OPERATOR','TEST_RELEASE','TEST_ONLY')
    body=read_json(path);assert body['test_fixture_only'] and body['overall_completed_rows_at_transition']==117000
    assert op.require_transition(tmp_path,current)==[str(path)]
    with pytest.raises(FileExistsError):op.record_transition(tmp_path,current,status,old['uncommitted_source_sha256'],transition_id='TEST_ONLY')
    with pytest.raises(ValueError,match='old source'):op.record_transition(tmp_path,current,status,'WRONG')
    assert read_json(path)==body and (artifacts/'preexecution_release.json').read_bytes()==before


from am_flexdc_foundation_v1_common import read_json
