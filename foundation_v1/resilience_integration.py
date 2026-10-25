"""Isolated real Foundation resilience/throughput TESTS; never production data."""
import argparse
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,contexts,iter_csv,write_csv,read_json,write_json,file_hash
from foundation_v1.full_campaign import verify_release,MASTER_SHA256
from peacsim.foundation_v1_cleanup import TRANSIENT_FILES

OUT=ARTIFACTS/'resilience_validation'
SCRATCH=ROOT/'.flexdc_plan_chunks/resilience_test_20261024'


def plans():
    tasks=contexts()
    chosen=[next(t for t in tasks if t['Workload_ID']=='F1_DEVELOPMENT_A100_J2_039' and t['domain']['server_count']==250 and t['domain']['utilization']==.6),
        next(t for t in tasks if t['hardware_class']=='A100' and t['domain']['job_type_count']==4 and t['domain']['server_count']==1000 and t['domain']['utilization']==.8),
        next(t for t in tasks if t['hardware_class']=='P100' and t['domain']['job_type_count']==6 and t['domain']['server_count']==1000 and t['domain']['utilization']==.8)]
    #24 independent contexts; six original Phase-A rows each, no new geometries.
    for t in tasks:
        if t['Task_ID'] not in {v['Task_ID'] for v in chosen}:chosen.append(t)
        if len(chosen)==24:break
    rows={t['Task_ID']:[] for t in chosen}
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if row['Task_ID'] in rows and len(rows[row['Task_ID']])<6:rows[row['Task_ID']].append(row)
        if all(len(v)==6 for v in rows.values()):break
    assert sum(map(len,rows.values()))==144
    return chosen,rows


def inspect(run_id,elapsed,expected,release):
    logroot=ROOT/'.flexdc_worker_logs'/('plan_'+run_id)
    state=read_json(logroot/'scheduler_summary.json');assert state['status']=='VALID_COMPLETE'
    attempts=[read_json(p) for p in sorted((logroot/'attempt_receipts').glob('*.json'))]
    assert state['terminal_rows']==len(expected)
    db=sqlite3.connect(logroot/'terminal_receipts.sqlite')
    entries=db.execute('SELECT row_id,body FROM terminal_executions').fetchall();db.close()
    assert {r[0] for r in entries}=={r['plan_row_id'] for r in expected}
    assert len(entries)==len(expected)
    outputs={r['output_dir'] for r in attempts if r.get('output_dir')}
    strict=0
    for output in outputs:
        directory=Path(output)
        assert all(not (directory/name).exists() for name in TRANSIENT_FILES)
        for p in (directory/'foundation_evidence').glob('F1_*.json'):
            def reject(token):raise ValueError('Invalid strict JSON '+token)
            body=json.loads(p.read_text(encoding='utf8'),parse_constant=reject)
            assert body['runtime_provenance']['reviewed_source_identity']==release['uncommitted_source_sha256'];strict+=1
        assert (directory/'grid_search_results.csv').exists() and (directory/'grid_search_diagnostics.csv').exists()
    assert strict==len(expected)
    assert not (logroot/'foundation_cleanup_stop.json').exists()
    assert not (logroot/'foundation_systemic_technical_stop.json').exists()
    assert all(r['status'] in ('SUCCESS','SCIENTIFIC_TERMINAL','TECHNICAL_RETRY_REQUIRED') for r in attempts)
    measured={'status':'PASS','test_only':True,'real_rows':len(expected),'logical_chunks':len({r['logical_chunk'] for r in attempts}),
        'worker_attempts':len(attempts),'workers':state['max_workers'],'wall_seconds':elapsed,
        'rows_per_second':len(expected)/elapsed,'effective_seconds_per_row':elapsed/len(expected),
        'live_process_slot_utilization':state['worker_slot_seconds']/(elapsed*state['max_workers']),
        'peak_workers':state['peak_workers'],'strict_receipts':strict,'terminal_ledger_rows':len(entries),
        'cleanup':'PASS','source_identity':release['uncommitted_source_sha256'],
        'technical_incidents':[{k:r.get(k) for k in ('logical_chunk','attempt','return_code','signal_number','signal_name','remaining_row_ids')} for r in attempts if r.get('return_code')],
        'production_ledger_used':False,'output_root':str(logroot),
        'individual_row_observations':[json.loads(body) for _,body in entries] if len(expected)==3 else []}
    return measured


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['integrations','soak','fault','fault-final'],required=True)
    a=p.parse_args();release=verify_release();chosen,bytask=plans();OUT.mkdir(exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
    expected=[bytask[t['Task_ID']][0] for t in chosen[:3]] if a.case=='integrations' else [r for t in chosen for r in bytask[t['Task_ID']]]
    plan=SCRATCH/(a.case+'.csv.gz');write_csv(plan,expected)
    run_id='foundation_resilience_test_20261024_'+a.case
    if (ROOT/'.flexdc_worker_logs'/('plan_'+run_id)/'scheduler_summary.json').exists():raise FileExistsError('Do not rerun a completed/partial real TEST automatically: '+run_id)
    from foundation_v1.campaign import commands
    workers=1 if a.case=='integrations' else 4
    command=commands(plan,workers,run_id,'TRAIN_DEV')
    command[command.index('--plan-num-chunks-per-context')+1]='1'
    command[command.index('--output-prefix')+1]='foundation_resilience_test_20261024_'+a.case
    start=time.monotonic();log=OUT/(a.case+'_real.log')
    print('REAL TEST '+a.case+' rows='+str(len(expected))+' workers='+str(workers),flush=True)
    if a.case not in ('fault','fault-final'):
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8')
        with log.open('w',encoding='utf8') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Real test failed; inspect '+str(log)
    else:
        # Safe orchestration-boundary injection: report one -11 without crashing
        # native FlexDC. ALL other worker invocations are real subprocesses.
        import am_run_flexdc_dataset_parallel_combo as combo
        from foundation_v1 import resilient
        args=combo.build_parser().parse_args(command[4:]);real=subprocess.Popen;injected=[]
        class Failed:
            def poll(self):return -11
            def terminate(self):pass
        def launch(cmd,**kwargs):
            if not injected:
                injected.append(True);kwargs['stdout'].write('INJECTED_BOUNDARY_SIGSEGV_NOT_NATIVE_CRASH\n');kwargs['stdout'].flush();return Failed()
            return real(cmd,**kwargs)
        with log.open('w',encoding='utf8') as stream,contextlib.redirect_stdout(stream),patch.object(resilient.subprocess,'Popen',launch):
            combo.run_exact_plan(args)
        assert len(injected)==1
    elapsed=time.monotonic()-start
    result=inspect(run_id,elapsed,expected,release)
    if a.case in ('fault','fault-final'):
        assert len(result['technical_incidents'])==1 and result['technical_incidents'][0]['return_code']==-11
    # Replay identical queue identity: receipt/ledger reconstruction, zero calls.
    import am_run_flexdc_dataset_parallel_combo as combo
    from foundation_v1 import resilient
    args=combo.build_parser().parse_args(command[4:])
    def forbidden(*args,**kwargs):raise AssertionError('Completed TEST row was rerun on resume')
    with patch.object(resilient.subprocess,'Popen',forbidden):combo.run_exact_plan(args)
    result['idempotent_queue_resume']='PASS_ZERO_SIMULATIONS'
    assert file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')==MASTER_SHA256
    write_json(OUT/(a.case+'_real_results.json'),result,False)
    print(json.dumps({k:v for k,v in result.items() if k!='individual_row_observations'},indent=2),flush=True)


if __name__=='__main__':main()
