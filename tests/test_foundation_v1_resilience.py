from tests.test_foundation_v1_legacy_entry_fixture import invoke as legacy_stage_entry
"""Resilient local queue tests: no real simulator processes in this module."""
import json
import signal
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
import pytest
from foundation_v1 import resilient
from peacsim.foundation_v1_cleanup import CleanupRequired
from am_flexdc_foundation_v1_common import write_csv,read_json


class Clock:
    def __init__(self):self.value=0.;self.callback=None
    def __call__(self):return self.value
    def sleep(self,delay):
        self.value+=delay
        if self.callback:self.callback()


def queue_fixture(tmp_path,monkeypatch,failures=None,n=7,rows=2,cleanup_failure=False,partial=False):
    chunks=tmp_path/'chunks';logs=tmp_path/'logs';chunks.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    invocations=[];launched=[];clock=Clock();done=set();failures=failures or {}
    for i in range(n):
        path=chunks/(str(i)+'.csv')
        data=[{'plan_row_id':f'F1_FAKE_{i}_{j}','seed':'20','geometry':str(j)} for j in range(rows)]
        write_csv(path,data)
        invocations.append({'chunk_file':str(path),'expected_plan_ids':[r['plan_row_id'] for r in data],
            'output_label':'fake_'+str(i),'command':['MOCK','--sweep-plan-file',str(path),'--output-dir','fake_'+str(i)]})
    class Process:
        def __init__(self,logical,attempt,code,ids):self.logical=logical;self.attempt=attempt;self.code=code;self.ids=ids;self.polls=0
        def poll(self):self.polls+=1;return self.code if self.polls>=2 else None
        def terminate(self):self.code=-15
        def kill(self):self.code=-9
    def launch(command,**kwargs):
        label=command[command.index('--output-dir')+1];logical=int(label.split('_')[1]);attempt=int(label.rsplit('attempt',1)[1])
        path=command[command.index('--sweep-plan-file')+1];ids=[r['plan_row_id'] for r in resilient.read_rows(path)]
        code=failures.get((logical,attempt),0)
        proc=Process(logical,attempt,code,ids)
        launched.append({'logical':logical,'attempt':attempt,'ids':ids,'env':kwargs['env'],'proc':proc,'command':command})
        return proc
    class Backend:
        def terminal(self,ids):return set(ids)&done
        def collect(self,item):
            if cleanup_failure and item['logical']=='0':
                done.update(item['record']['expected_plan_ids']);raise CleanupRequired('ACTUAL FAKE UNLINK FAILURE')
            proc=item.get('proc')
            if proc is None:return None
            if proc.code==0:done.update(proc.ids)
            elif partial and proc.ids:done.add(proc.ids[0])
            return tmp_path
        def close(self):pass
    api=SimpleNamespace(REPO_ROOT=tmp_path,PEACSIM_DIR=tmp_path,build_env=lambda:{})
    args=SimpleNamespace(plan_max_workers=2,policy_name='AQA',foundation_max_technical_attempts=3)
    backend=Backend()
    def run():return resilient.run(args,invocations,chunks,logs,api,backend,launch,clock,clock.sleep)
    return SimpleNamespace(run=run,launched=launched,logs=logs,clock=clock,done=done,args=args,invocations=invocations,api=api,backend=backend)


def test_prefixed_single_sigsegv_never_requests_cleanup_stop(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,failures={(0,1):-11})
    result=f.run()
    assert result['status']=='VALID_COMPLETE' and len(f.done)==14
    assert len(f.launched)==8 and f.launched[-1]['logical']==0 and f.launched[-1]['attempt']==2
    assert not (f.logs/'foundation_cleanup_stop.json').exists()
    assert not (f.logs/'foundation_storage_stop.json').exists()
    records=[read_json(p) for p in (f.logs/'attempt_receipts').glob('*.json')]
    failure=next(r for r in records if r.get('return_code')==-11)
    assert failure['signal_name']=='SIGSEGV' and failure['status']=='TECHNICAL_RETRY_REQUIRED'
    assert failure['remaining_row_ids']==['F1_FAKE_0_0','F1_FAKE_0_1']


def test_partial_chunk_retry_skips_terminal_same_input_rows(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,failures={(0,1):-11},partial=True)
    assert f.run()['status']=='VALID_COMPLETE'
    retry=next(r for r in f.launched if r['logical']==0 and r['attempt']==2)
    assert retry['ids']==['F1_FAKE_0_1']
    assert resilient.read_rows(retry['command'][retry['command'].index('--sweep-plan-file')+1])==[{'plan_row_id':'F1_FAKE_0_1','seed':'20','geometry':'1'}]


def test_repeated_sigsegv_quarantines_after_three_attempts(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,failures={(0,i):-11 for i in (1,2,3)})
    with pytest.raises(SystemExit,match='STAGE_COMPLETE_WITH_TECHNICAL_SHORTFALL'):f.run()
    state=read_json(f.logs/'scheduler_summary.json')
    assert state['quarantine'][0]['status']=='TECHNICAL_RETRY_EXHAUSTED'
    assert len([r for r in f.launched if r['logical']==0])==3
    assert len(f.done)==12 and not (f.logs/'foundation_cleanup_stop.json').exists()
    assert all(r['ids']==['F1_FAKE_0_0','F1_FAKE_0_1'] for r in f.launched if r['logical']==0)


def test_systemic_five_independent_same_signal_stops_new_launches(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,failures={(i,1):-11 for i in range(7)},n=12)
    with pytest.raises(SystemExit,match='SYSTEMIC_TECHNICAL_FAILURE'):f.run()
    assert (f.logs/'foundation_systemic_technical_stop.json').exists()
    assert not (f.logs/'foundation_cleanup_stop.json').exists()
    assert len(f.launched)<12


def test_breaker_exact_threshold_rate_window_and_independence():
    b=resilient.CircuitBreaker()
    for i in range(20):assert not b.add('s'+str(i),None,0)
    for i in range(4):assert not b.add('f'+str(i),'SIGSEGV',0)
    assert b.add('f4','SIGSEGV',0) # 5/25 ==20%, equality trips
    c=resilient.CircuitBreaker()
    for i in range(21):c.add('s'+str(i),None,0)
    for i in range(5):assert not c.add('f'+str(i),'SIGSEGV',0) #5/26 <20%
    for _ in range(20):assert not resilient.CircuitBreaker().add('same','SIGSEGV',0)
    d=resilient.CircuitBreaker()
    for i in range(4):d.add(str(i),'SIGSEGV',0)
    assert not d.add('new','SIGSEGV',301)


def test_true_cleanup_stop_retains_terminal_and_stops_new_chunks(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,cleanup_failure=True)
    with pytest.raises(SystemExit,match='TRANSIENT_CLEANUP_REQUIRED'):f.run()
    assert len(f.launched)==2 and 'F1_FAKE_0_0' in f.done
    assert (f.logs/'foundation_cleanup_stop.json').exists()


def test_interruption_drains_then_same_identity_resume(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch)
    @contextmanager
    def handler(callback):
        fired=[]
        def interrupt_once():
            if not fired:fired.append(True);callback(signal.SIGTERM,None)
        f.clock.callback=interrupt_once
        yield
        f.clock.callback=None
    monkeypatch.setattr(resilient,'interrupts',handler)
    with pytest.raises(SystemExit,match='INTERRUPTED'):f.run()
    assert len(f.launched)==2 and len(f.done)==4
    @contextmanager
    def no_interrupt(callback):yield
    monkeypatch.setattr(resilient,'interrupts',no_interrupt)
    result=f.run()
    assert result['status']=='VALID_COMPLETE' and len(f.done)==14
    assert len(f.launched)==7


def test_environment_defaults_are_foundation_only_and_explicit_values_preserved():
    env=resilient.worker_env({'MKL_NUM_THREADS':'2'})
    assert env['MKL_NUM_THREADS']=='2'
    assert all(env[k]=='1' for k in resilient.THREAD_DEFAULTS if k!='MKL_NUM_THREADS')
    assert resilient.decode_returncode(-11)['signal_name']=='SIGSEGV'
    assert resilient.decode_returncode(-9)['signal_name']=='SIGKILL'
    assert resilient.decode_returncode(-15)['signal_name']=='SIGTERM'


def test_csv_only_and_old_attempts_0001_through_0004_resume(tmp_path,monkeypatch):
    # Existing NaN/collateral-stop fixtures exercise real receipt ingestion;
    # this queue fixture verifies old attempt outcomes never imply completion.
    f=queue_fixture(tmp_path,monkeypatch,failures={(0,1):-11})
    f.done.update({'F1_FAKE_1_0','F1_FAKE_1_1'})
    (f.logs/'foundation_storage_stop.json').write_text('{"status":"FOUNDATION_WORKER_STOP","attempt":"0003"}')
    assert f.run()['status']=='VALID_COMPLETE'
    assert not any(r['logical']==1 for r in f.launched)
    assert len(f.done)==14
    count=len(f.launched)
    assert f.run()['status']=='VALID_COMPLETE' and len(f.launched)==count


def test_launch_slot_replenished_without_waiting_for_other_chunk(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch,n=10)
    assert f.run()['peak_workers']==2
    assert len(f.launched)==10


def test_malformed_worker_local_state_retries_without_global_stop(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch);old=f.backend.collect;first=[]
    def collect(item):
        if item['logical']=='0' and not first:
            first.append(True);raise ValueError('malformed worker-local temporary JSON')
        return old(item)
    f.backend.collect=collect
    assert f.run()['status']=='VALID_COMPLETE'
    assert len(f.launched)==8 and not (f.logs/'foundation_global_integrity_stop.json').exists()


def test_launched_plan_mutation_is_global_integrity_stop(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch)
    def change():
        if f.launched:
            cmd=f.launched[0]['command'];p=Path(cmd[cmd.index('--sweep-plan-file')+1]);p.write_text('CORRUPTED SCIENTIFIC PLAN')
            f.clock.callback=None
    f.clock.callback=change
    with pytest.raises(SystemExit,match='GLOBAL_FATAL_INVARIANT'):f.run()
    assert (f.logs/'foundation_global_integrity_stop.json').exists()
    assert not (f.logs/'foundation_cleanup_stop.json').exists()


def test_repeated_one_chunk_cannot_trip_independent_worker_breaker():
    breaker=resilient.CircuitBreaker()
    for i in range(100):assert not breaker.add('only-one-logical-chunk','SIGSEGV',i)


def test_actual_four_state_resume_and_attempt0001_to0004_history(tmp_path,monkeypatch):
    import copy,sqlite3
    from foundation_v1 import campaign,audit,scope
    from peacsim import foundation_v1_cleanup as cleanup
    from am_flexdc_foundation_v1_common import contexts,write_json,read_csv
    from tests.test_foundation_v1 import fake_receipt
    from tests.test_foundation_v1_patch import prepare_fake_campaign
    from tests.test_foundation_v1_cleanup import persist,transients,begin
    tasks=[];pairs=[]
    for i in range(4):
        task=copy.deepcopy(contexts()[0]);task['Task_ID']+='_RESILIENCE_FAKE_'+str(i)
        tasks.append(task);pairs.append(fake_receipt(task))
    rows=[pair[0] for pair in pairs]
    plan,database=prepare_fake_campaign(tasks[0],tmp_path,monkeypatch,rows)
    monkeypatch.setattr(audit,'contexts',lambda:tasks)
    monkeypatch.setattr(scope,'selected_tasks',lambda *args:tasks)
    oldroot=scope.raw_root('TRAIN_DEV')
    for i,(row,receipt) in enumerate(pairs):
        output=oldroot/('foundation_v1_old_attempt'+str(i+1));output.mkdir(parents=True)
        begin(output,row)
        if i in (0,3):persist(output,row,receipt)
        if i==0:cleanup.finalize_row(output,row['plan_row_id'])
        else:transients(output) # CSV-only, active, or cleanup-pending
        if i==1:
            for name in ('grid_search_results.csv','grid_search_diagnostics.csv'):(output/name).write_text('Plan_Row_ID,Replicate_Of_Plan_Row_ID\n'+row['plan_row_id']+',nan\n')
    conn=audit.connection(database)
    with conn:
        for i in range(1,5):
            conn.execute('INSERT INTO attempts VALUES(?,?,?)',('attempt'+str(i).zfill(4),'TECHNICAL_RETRY_REQUIRED' if i<4 else 'INTERRUPTED',json.dumps({'history': 'NaN' if i==1 else 'SIGSEGV -11 plus collateral stop' if i==3 else 'partial prior execution'})))
    conn.close()
    marker=tmp_path/'old_attempt0003/foundation_storage_stop.json';write_json(marker,{'status':'FOUNDATION_WORKER_STOP','return_code':-11})
    calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        subset=read_csv(command[command.index('--sweep-plan-file')+1])
        assert {r['plan_row_id'] for r in subset}=={rows[1]['plan_row_id'],rows[2]['plan_row_id']}
        assert command[command.index('--plan-run-id')+1].endswith('attempt0005')
        assert not (oldroot/'foundation_v1_old_attempt4/power_trace.csv').exists() # cleanup ONLY already done
        for i in (1,2):
            row,receipt=pairs[i];output=oldroot/('foundation_v1_new_'+str(i));persist(output,row,receipt)
        return 0
    monkeypatch.setattr(campaign,'call_combo',execute)
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(database)])
    result=legacy_stage_entry();assert result['status']=='VALID_COMPLETE' and len(calls)==1
    with sqlite3.connect(database) as conn:assert conn.execute('SELECT count(*) FROM observations').fetchone()[0]==4
    assert legacy_stage_entry()['simulation_calls_this_invocation']==0 and len(calls)==1
    assert marker.exists() and (oldroot/'foundation_v1_old_attempt2/power_trace.csv').exists()


def test_worker_parent_loss_is_interruption_not_cleanup(monkeypatch,tmp_path):
    from peacsim import foundation_v1_cleanup as cleanup
    monkeypatch.setenv('FOUNDATION_V1_PARENT_PID','12345')
    monkeypatch.setattr(cleanup,'parent_changed',lambda expected:True)
    with pytest.raises(cleanup.FoundationInterrupted,match='queue parent exited'):cleanup.check_stop()
    assert not list(tmp_path.glob('*stop*'))


def test_queue_parent_loss_drains_without_cleanup_stop(tmp_path,monkeypatch):
    f=queue_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('FOUNDATION_V1_CAMPAIGN_PARENT_PID','12345')
    monkeypatch.setattr(resilient,'parent_changed',lambda expected:True)
    with pytest.raises(SystemExit,match='INTERRUPTED'):f.run()
    assert not f.launched and not (f.logs/'foundation_cleanup_stop.json').exists()


def test_managed_parent_forwards_termination(monkeypatch):
    calls=[]
    class Child:
        def poll(self):return None
        def send_signal(self,sig):calls.append(sig)
        def wait(self):return 143
    @contextmanager
    def signal_context(callback):
        callback(signal.SIGTERM,None);yield
    monkeypatch.setattr(resilient.subprocess,'Popen',lambda *args,**kwargs:Child())
    monkeypatch.setattr(resilient,'interrupts',signal_context)
    assert resilient.call_combo(['FAKE_ONLY'],'.')==143
    assert calls==[signal.SIGTERM]


def test_live_actual_parent_is_not_falsely_rejected():
    import os
    from peacsim.foundation_v1_cleanup import parent_changed
    assert not parent_changed(str(os.getppid()))
    assert not parent_changed(None)
