"""Clean-slate production collector tests. Simulator boundary is fake unless integration harness."""
import copy
import csv
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,contexts,make_row,Domain,digest,write_json,write_csv,iter_csv,read_json,file_hash)
from foundation_v1 import simple_collector as c,simple_campaign as full,collector_receipts as rec,scope
from peacsim import foundation_v1_cleanup as cleanup
from tests.test_foundation_v1 import fake_receipt
from tests.test_foundation_v1_cleanup import persist,transients


def emit(area,row,output,task):
    output.mkdir(parents=True,exist_ok=True)
    _,body=fake_receipt(task);body.pop('body_sha256');key=row['plan_row_id'];j=int(row['job_type_count'])
    weights=json.loads(row['weights']) if isinstance(row['weights'],str) else row['weights']
    p=float(row['Pbar_kw_per_server']);r=float(row['R_kw_per_server']);n=int(row['server_count']);u=float(row['utilization']);seed=int(row['simulation_seed'])
    body['Plan_Row_ID']=key
    for record in (body['results'],body['diagnostics']):
        record.update(Plan_Row_ID=key,Pbar_kw_per_server=p,R_kw_per_server=r,server_count=n,utilization=u,Simulation_Seed=seed)
    for i,v in enumerate(weights):body['results']['Weight_'+str(i)]=v
    body['applied'].update(weights=weights,N=n,U=u,J=j,seed=seed,P_watts=round(p*1000*n),R_watts=round(r*1000*n))
    body['body_sha256']=digest(body)
    with patch.dict(os.environ,{'FOUNDATION_V1_SIMPLE_AREA':str(area),'FOUNDATION_V1_SIMPLE_WORKER':'fake_slot'}):
        cleanup.begin_row(output,row);persist(output,row,body)
        rec.publish(output,row)
        cleanup.finalize_row(output,key)
    return rec.load(area,key)


def fixture_rows(n=5):
    task=contexts()[0];d=Domain(task['domain'])
    return task,[make_row(task,x,'GLOBAL_ESCAPE') for x in d.global_pool(n,'simple_fake_rows')]


class FakeLaunch:
    def __init__(self,root,tasks,fail=None,partial=False):
        self.root=root;self.tasks={t['Task_ID']:t for t in tasks};self.calls=[];self.fail=fail or (lambda rows,number:False);self.partial=partial
    def __call__(self,cmd,**kwargs):
        rows=list(iter_csv(cmd[cmd.index('--sweep-plan-file')+1]));number=len(self.calls)
        self.calls.append(rows);failed=self.fail(rows,number)
        todo=rows[:1] if failed and self.partial else [] if failed else rows
        for row in todo:emit(Path(kwargs['env']['FOUNDATION_V1_SIMPLE_AREA']),row,self.root/('out'+str(number))/row['plan_row_id'],self.tasks[row['Task_ID']])
        class Process:
            def poll(self):return -11 if failed else 0
            def terminate(self):pass
            def wait(self,**kwargs):return self.poll()
        return Process()


def setup_queue(tmp_path,monkeypatch,n=7):
    task,rows=fixture_rows(n);plan=tmp_path/'plan.csv.gz';write_csv(plan,rows);area=tmp_path/'area'
    monkeypatch.setattr(c.time,'sleep',lambda delay:None)
    return task,rows,plan,area


def test_original_ids_are_authority_reconstruction_is_offline(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3)
    rows[0].update(plan_row_id='F1_ORIGINAL_AUTHORITY',Plan_Row_ID='F1_ORIGINAL_AUTHORITY',Foundation_Row_ID='F1_ORIGINAL_AUTHORITY')
    plan=tmp_path/'different.csv.gz';write_csv(plan,rows)
    launch=FakeLaunch(tmp_path,[task]);result=c.collect(plan,area,2,0,launch,1)
    assert result['completed']==3 and rec.load(area,'F1_ORIGINAL_AUTHORITY')['Plan_Row_ID']=='F1_ORIGINAL_AUTHORITY'
    report=full.certify(plan,area,tmp_path/'derived.sqlite')
    assert set(report['QUARANTINED'])=={'F1_ORIGINAL_AUTHORITY'} and report['PASS']==2
    detail=report['QUARANTINED']['F1_ORIGINAL_AUTHORITY']
    assert detail['original_plan_row_id']!=detail['reconstructed_plan_row_id']
    assert detail['P'] and detail['R'] and detail['source_plan']


def test_sigsegv_partial_chunk_skips_completed_and_finishes_others(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch)
    launch=FakeLaunch(tmp_path,[task],lambda rows,n:n==0,True)
    result=c.collect(plan,area,3,0,launch,3)
    assert result['completed']==7
    first=rows[0]['plan_row_id'];assert sum(any(r['plan_row_id']==first for r in call) for call in launch.calls)==1
    assert len(list((area/'workers').glob('*.attempt.json')))>3
    assert not list(area.glob('*stop*'))


def test_repeated_bad_row_quarantines_but_rest_finishes_and_repair_is_bounded(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch)
    bad=rows[0]['plan_row_id'];launch=FakeLaunch(tmp_path,[task],lambda rs,n:any(r['plan_row_id']==bad for r in rs))
    result=c.collect(plan,area,2,0,launch,3)
    assert result['completed']==6 and result['quarantined_ids']==[bad]
    assert sum(any(r['plan_row_id']==bad for r in rs) for rs in launch.calls)==3
    before=len(launch.calls);again=c.collect(plan,area,2,0,launch,3)
    assert again['new_completions']==0 and len(launch.calls)==before+3


@pytest.mark.parametrize('kind',['geometry','weights','objective','receipt'])
def test_stage_audit_names_only_bad_row_and_repair_skips_clean(tmp_path,monkeypatch,kind):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3)
    launch=FakeLaunch(tmp_path,[task]);c.collect(plan,area,2,0,launch,1)
    key=rows[0]['plan_row_id'];body=rec.load(area,key)
    if kind=='geometry':body['Base_Geometry_ID']='WRONG'
    elif kind=='weights':body['evidence']['applied']['weights'][0]+=.01
    elif kind=='objective':body['evidence']['diagnostics']['Diagnostic_FullPaperObjective_Cost']+=1
    else:body['checksum']='WRONG'
    if kind!='receipt':
        ev=body['evidence'];ev.pop('body_sha256');ev['body_sha256']=digest(ev);write_json(Path(body['evidence_file']),ev,False)
        body.pop('checksum');body['checksum']=digest(body)
    write_json(rec.receipt_path(area,key),body,False)
    report=full.certify(plan,area,tmp_path/'derived.sqlite')
    assert report['PASS']==2 and set(report['QUARANTINED'])=={key}
    rec.archive(area,key,report['QUARANTINED'][key]);before=len(launch.calls)
    c.collect(plan,area,2,0,launch,1)
    assert len(launch.calls)==before+1 and launch.calls[-1][0]['plan_row_id']==key
    assert full.certify(plan,area,tmp_path/'derived.sqlite')['status']=='PASS'


def test_cleanup_debt_is_local_and_low_space_is_real_global_gate(tmp_path,monkeypatch):
    task,row=fixture_rows(1);row=row[0];area=tmp_path/'area';output=tmp_path/'output'
    # Publish completion first, then simulate actual unlink failure.
    body=emit(area,row,output,task);transients(output)
    monkeypatch.setenv('FOUNDATION_V1_SIMPLE_AREA',str(area))
    monkeypatch.setattr(cleanup,'_unlink',lambda path:(_ for _ in ()).throw(PermissionError('FAKE locked file')))
    with pytest.raises(cleanup.CleanupDebt):cleanup.finalize_row(output,row['plan_row_id'])
    assert rec.load(area,row['plan_row_id'])['execution_status']=='EXECUTION_COMPLETE'
    assert cleanup._current(output)['cleanup_status']=='CLEANUP_DEBT'
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    assert c.check_storage(tmp_path,10)>0
    monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=1))
    with pytest.raises(c.StorageEmergency):c.check_storage(tmp_path,10)


def test_sqlite_loss_rebuilds_without_completed_resimulation(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,5);launch=FakeLaunch(tmp_path,[task])
    c.collect(plan,area,2,0,launch,2);db=tmp_path/'derived.sqlite'
    assert full.certify(plan,area,db)['status']=='PASS';db.unlink()
    before=len(launch.calls);assert c.collect(plan,area,2,0,launch)['new_completions']==0
    assert len(launch.calls)==before and full.certify(plan,area,db)['PASS']==5
    connection=sqlite3.connect(db);assert connection.execute('select count(*) from observations').fetchone()[0]==5;connection.close()


def test_csv_without_completion_never_counts_as_done(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3);launch=FakeLaunch(tmp_path,[task])
    (tmp_path/'abandoned.csv').write_text('Plan_Row_ID\n'+rows[0]['plan_row_id'])
    assert c.collect(plan,area,2,0,launch,1)['new_completions']==3


def test_fresh_start_deletes_only_exact_generated_targets(tmp_path,monkeypatch):
    keep=['foundation_v1_artifacts/foundation_v1_phaseA_plan.csv.gz','configs/workload/science.ini','foundation_v1/source.py','profiles.json','.flexdc_plan_chunks/other_experiment/keep.txt']
    for name in keep:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'IMMUTABLE')
    remove=['foundation_v1_runs/train_dev/observations.sqlite','src/peacsim/output/optimization/foundation_v1_runs/old/power_trace.csv','.flexdc_plan_chunks/foundation_v1_old/plan.csv','.flexdc_worker_logs/plan_foundation_v1_old/log.txt']
    for name in remove:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('OLD EXECUTION')
    monkeypatch.setattr(full,'ROOT',tmp_path)
    a=SimpleNamespace(confirm_fresh_start=False,dry_run=True)
    listing=full.fresh_start(a);assert listing['targets'] and all((tmp_path/n).exists() for n in remove)
    a.confirm_fresh_start=True;a.dry_run=False;full.fresh_start(a)
    assert all(not (tmp_path/n).exists() for n in remove)
    assert all((tmp_path/n).read_bytes()==b'IMMUTABLE' for n in keep)


def test_unconfirmed_fresh_start_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='requires'):full.fresh_start(SimpleNamespace(confirm_fresh_start=False,dry_run=False))


def test_status_50000_events_fast_cache_and_pending_crash_window(tmp_path):
    area=tmp_path/'foundation_v1_runs/train_dev/collector_v2/phaseA';events=area/'events';events.mkdir(parents=True)
    count=50000
    with (events/'slot_0.jsonl').open('w') as stream:
        for i in range(count):stream.write(json.dumps({'id':'F1_'+str(i),'time':time.time()})+'\n')
    # Tens of thousands of actual fake durable markers accompany the journal.
    for i in range(count):
        key='F1_'+str(i);body={'schema':rec.SCHEMA,'Plan_Row_ID':key,'timestamp':time.time()};body['checksum']=digest(body)
        path=rec.receipt_path(area,key);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(body))
    write_json(area/'progress.json',{'planned':60000,'workers_active':32,'technical_retries':4,'quarantined':3},False)
    start=time.monotonic();status=full.live_status(tmp_path);first=time.monotonic()-start
    start=time.monotonic();again=full.live_status(tmp_path);second=time.monotonic()-start
    assert status['overall_complete']==count and status['stages'][0]['pending']==10000
    assert again==status and first<10 and second<2
    (tmp_path/'foundation_v1_runs/status_cache.sqlite').unlink()
    assert full.live_status(tmp_path)['overall_complete']==count


@pytest.mark.parametrize('workers',[8,32])
def test_scheduler_fills_requested_worker_slots(tmp_path,monkeypatch,workers):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,workers+8)
    launch=FakeLaunch(tmp_path,[task]);result=c.collect(plan,area,workers,0,launch,1)
    assert result['peak_workers']==workers and result['completed']==workers+8


def test_float_round_trip_original_plan_is_valid(tmp_path):
    import pandas as pd
    # Real immutable row at the documented 12-decimal identity boundary.
    row=next(r for r in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz') if r['plan_row_id']=='F1_e72484ba822d2dc2993b7f85a945f814')
    path=tmp_path/'row.csv';write_csv(path,[row]);task=next(t for t in contexts() if t['Task_ID']==row['Task_ID'])
    legacy=pd.read_csv(path).iloc[0].to_dict();new=pd.read_csv(path,float_precision='round_trip').iloc[0].to_dict()
    assert make_row(task,[float(legacy['Pbar_kw_per_server']),float(legacy['R_kw_per_server']),*json.loads(row['weights'])],row['Candidate_Source'])['plan_row_id']!=row['plan_row_id']
    assert make_row(task,[float(new['Pbar_kw_per_server']),float(new['R_kw_per_server']),*json.loads(row['weights'])],row['Candidate_Source'])['plan_row_id']==row['plan_row_id']


def test_real_parent_hard_kill_preserves_atomic_completions(tmp_path,monkeypatch):
    import subprocess,sys
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3)
    script=tmp_path/'parent.py'
    script.write_text("""import time,sys
from pathlib import Path
from foundation_v1 import simple_collector as c
from am_flexdc_foundation_v1_common import contexts,iter_csv
from tests.test_foundation_v1_simple_collector import emit
plan=Path(sys.argv[1]);area=Path(sys.argv[2]);ready=Path(sys.argv[3]);rows=list(iter_csv(plan));task=contexts()[0]
class Blocked:
 def poll(self):return None
 def terminate(self):pass
 def wait(self,**kwargs):return 0
def launch(cmd,**kw):
 if not ready.exists():
  emit(area,rows[0],area/'finished_output',task)
  (area/'csv_without_completion.csv').write_text('Plan_Row_ID\\n'+rows[1]['plan_row_id'])
  ready.write_text('READY')
 return Blocked()
c.collect(plan,area,2,0,launch,1)
""",encoding='utf8')
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'))
    ready=tmp_path/'ready';proc=subprocess.Popen([sys.executable,'-B',str(script),str(plan),str(area),str(ready)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+25
        while not ready.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.1)
        assert ready.exists(),proc.stderr.read().decode() if proc.poll() is not None else 'parent readiness timeout'
        proc.kill();proc.wait(timeout=10)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    launch=FakeLaunch(tmp_path,[task]);result=c.collect(plan,area,2,0,launch,1)
    assert result['completed']==3 and result['new_completions']==2
    assert rows[0]['plan_row_id'] not in {r['plan_row_id'] for call in launch.calls for r in call}


def test_full_production_state_machine_real_planners_synthetic_simulator(tmp_path,monkeypatch):
    from foundation_v1 import full_campaign as historical
    all_tasks=contexts()
    selected=[next(t for t in all_tasks if t['split']==split and t['domain']['job_type_count']==2) for split in ('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST')]
    ids={t['Task_ID'] for t in selected};master={t['Task_ID']:[] for t in selected}
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if row['Task_ID'] in ids:master[row['Task_ID']].append(row)
        if all(len(v)==150 for v in master.values()):break
    def tasks(part=None,task_id=None):return [t for t in selected if t['split'] in scope.SCOPES[scope.normalize(part)] and (task_id is None or t['Task_ID']==task_id)]
    monkeypatch.setattr(scope,'ROOT',tmp_path);monkeypatch.setattr(full,'ROOT',tmp_path)
    monkeypatch.setattr(full,'versions',lambda require=False:{'status':'SYNTHETIC'})
    monkeypatch.setattr('foundation_v1.environment.versions',lambda require=False:{'status':'SYNTHETIC'})
    monkeypatch.setattr(scope,'selected_tasks',tasks)
    monkeypatch.setattr(full,'verify_release',lambda:{'uncommitted_source_sha256':'SYNTHETIC_EXECUTION_ONLY_REAL_SCIENTIFIC_PLANNERS'})
    def view(part):
        rows=[row for t in tasks(part) for row in master[t['Task_ID']]];path=scope.plan_path('phaseA',part)
        write_csv(path,rows);write_json(path.with_suffix('.receipt.json'),{'scope':part,'rows':len(rows),'sha256':file_hash(path)})
        return path
    monkeypatch.setattr(scope,'make_phaseA_view',view)
    launcher=FakeLaunch(tmp_path/'mock_outputs',selected)
    original_collect=c.collect
    def collect(*args,**kwargs):kwargs['launch']=launcher;return original_collect(*args,**kwargs)
    monkeypatch.setattr(c,'collect',collect)
    # Actual production parent, stage certification, real adaptive/repeat methods,
    # canonical database/export pipeline; only FlexDC boundary is synthetic.
    a=SimpleNamespace(max_workers=4,execute=True,resume=True,unlock_locked_test=True,storage_emergency_gib=0)
    state=full.run_full(a,'ALL')
    assert state['status']=='FOUNDATION_V1_COMPLETE' and state['valid_rows']==750
    assert len(state['stages'])==12
    completed_calls=len(launcher.calls)
    # Interrupt a real production state transition in Wave2, then resume only
    # its missing row. All clean stages/rows remain durable.
    wave_area=full.stage_area('TRAIN_DEV','wave02')
    key=next(iter(rec.completed(wave_area)));rec.archive(wave_area,key,{'reason':'SYNTHETIC_INTERRUPT_TARGET'})
    before_collect=full.execute_stage;once=[]
    def interrupt_stage(args,part):
        if args.stage=='wave02' and part=='TRAIN_DEV' and not once:
            once.append(True);raise KeyboardInterrupt('SYNTHETIC allocation ending')
        return before_collect(args,part)
    with patch.object(full,'execute_stage',interrupt_stage):
        with pytest.raises(KeyboardInterrupt):full.run_full(a,'ALL')
    resumed=full.run_full(a,'ALL');assert resumed['valid_rows']==750
    assert len(launcher.calls)==completed_calls+1
    completed_calls=len(launcher.calls)
    # Remove derived state entirely, then rebuild all stages from receipts.
    for database in tmp_path.glob('foundation_v1_runs/*/observations.sqlite'):database.unlink()
    again=full.run_full(a,'ALL')
    assert again['valid_rows']==750 and len(launcher.calls)==completed_calls
    assert all(stage['simulation_calls_this_invocation']==0 for stage in again['stages'].values())


@pytest.mark.parametrize('delta,accepted',[(1e-16,True),(1e-5,False)])
def test_existing_weight_serialization_tolerance_recorded(tmp_path,monkeypatch,delta,accepted):
    task,rows=fixture_rows(1);row=rows[0];area=tmp_path/'area';output=tmp_path/'output'
    body=emit(area,row,output,task);ev=body['evidence'];ev.pop('body_sha256');ev['applied']['weights'][0]+=delta;ev['body_sha256']=digest(ev)
    write_json(output/'foundation_evidence'/(row['plan_row_id']+'.json'),ev,False)
    monkeypatch.setenv('FOUNDATION_V1_SIMPLE_AREA',str(area));monkeypatch.setenv('FOUNDATION_V1_SIMPLE_WORKER','tolerance_test')
    if accepted:
        rec.publish(output,row);new=rec.load(area,row['plan_row_id']);assert new['weight_serialization_deltas'][0]!=0
    else:
        with pytest.raises(ValueError,match='serialization tolerance'):rec.publish(output,row)



def test_completion_paths_are_absolute_across_worker_and_parent_cwd(tmp_path,monkeypatch):
    task,rows=fixture_rows(1);row=rows[0];area=tmp_path/'area';output=tmp_path/'worker'/'output'
    output.parent.mkdir();monkeypatch.chdir(output.parent)
    body=emit(area,row,Path('output'),task)
    assert Path(body['result_file']).is_absolute() and Path(body['evidence_file']).is_absolute()
    monkeypatch.chdir(tmp_path)
    assert Path(body['evidence_file']).exists()



def test_new_production_cli_routes_to_simple_full_campaign(monkeypatch):
    from foundation_v1 import campaign
    calls=[]
    monkeypatch.setattr(full,'run_full',lambda a,part:calls.append((part,a.max_workers,a.resume)))
    monkeypatch.setattr('sys.argv',['foundation','--full-campaign','--partition','ALL','--unlock-locked-test','--max-workers','32','--execute'])
    campaign.main();assert calls==[('ALL',32,False)]


def test_new_status_cli_and_nonexecution_default(monkeypatch):
    from foundation_v1 import campaign,full_campaign
    monkeypatch.setattr(full,'live_status',lambda:{'overall_complete':42})
    monkeypatch.setattr('sys.argv',['foundation','--status']);assert campaign.main()['overall_complete']==42
    calls=[];monkeypatch.setattr(full,'preview',lambda a,p:calls.append((p,a.execute)))
    monkeypatch.setattr('sys.argv',['foundation']);campaign.main();assert calls==[('TRAIN_DEV',False)]



def test_local_parent_metadata_write_error_does_not_stop_science(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,4);launch=FakeLaunch(tmp_path,[task]);real_write=c.write_json;failed=[]
    def write(path,body,immutable=True):
        if Path(path).name=='progress.json' and not failed:
            failed.append(True);raise PermissionError('SYNTHETIC one locked progress cache')
        return real_write(path,body,immutable)
    monkeypatch.setattr(c,'write_json',write)
    assert c.collect(plan,area,2,0,launch,1)['completed']==4 and failed


def test_local_transport_write_error_is_row_retried(tmp_path,monkeypatch):
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,4);launch=FakeLaunch(tmp_path,[task]);real_write=c.write_csv;failed=[]
    def write(path,values,*args,**kwargs):
        if not failed:failed.append(True);raise PermissionError('SYNTHETIC one locked transport file')
        return real_write(path,values,*args,**kwargs)
    monkeypatch.setattr(c,'write_csv',write)
    assert c.collect(plan,area,2,0,launch,1)['completed']==4 and failed



@pytest.mark.parametrize('part,split',[('train_dev','TRAIN'),('locked_synthetic_test','LOCKED_SYNTHETIC_TEST')])
def test_raw_worker_output_prefix_is_physically_partition_scoped(tmp_path,monkeypatch,part,split):
    task=next(t for t in contexts() if t['split']==split)
    row=make_row(task,Domain(task['domain']).global_pool(1,'partition_path_only_test')[0],'GLOBAL_ESCAPE')
    plan=tmp_path/(part+'.csv.gz');write_csv(plan,[row]);area=tmp_path/part/'collector_v2'/'phaseA'
    native=FakeLaunch(tmp_path/'outputs',[task]);labels=[]
    def launch(cmd,**kwargs):
        labels.append(cmd[cmd.index('--output-dir')+1]);return native(cmd,**kwargs)
    assert c.collect(plan,area,1,0,launch)['completed']==1
    assert labels[0].startswith('foundation_v1_simple/'+part+'/collector_v2/phaseA/')



def test_preview_is_new_path_and_creates_no_runtime_state(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'ROOT',tmp_path);monkeypatch.setattr(full,'verify_release',lambda:{'uncommitted_source_sha256':'SYNTHETIC'})
    before=list(tmp_path.iterdir());result=full.preview(SimpleNamespace(max_workers=32),'ALL')
    assert result['expected_rows']==500000 and result['simulation_calls']==0 and len(result['steps'])==14
    assert result['mode']=='SIMPLE_COLLECTOR_V2' and all('command_template' not in step for step in result['steps'])
    assert list(tmp_path.iterdir())==before
