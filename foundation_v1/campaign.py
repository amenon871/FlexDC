"""Exact-plan combo-runner orchestration; dry-run unless explicitly executed."""
import argparse
from contextlib import nullcontext
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,contexts,read_csv,write_csv,
    write_json,file_hash,digest,read_json,iter_csv)
from foundation_v1.audit import connection,ingest,verify_database,terminal_ids,stage_summary,execution_state


class CampaignLock:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):
        self.f=self.path.open('a+b');self.f.seek(0)
        if self.path.stat().st_size==0:self.f.write(b'0');self.f.flush();self.f.seek(0)
        if sys.platform=='win32':
            import msvcrt
            try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:self.f.close();raise RuntimeError('Foundation campaign already running')
        else:
            import fcntl
            fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return self
    def __exit__(self,*args):self.f.close()


def stage_path(stage,partition='TRAIN_DEV',task=None):
    from foundation_v1.scope import plan_path
    candidate=plan_path(stage,partition,task)
    return candidate if task and candidate.exists() else plan_path(stage,partition)


def commands(plan,workers,run_id,partition='TRAIN_DEV'):
    from foundation_v1.scope import group
    return [sys.executable,'-u','-B',str(ROOT/'am_run_flexdc_dataset_parallel_combo.py'),
      '--sweep-plan-file',str(plan),'--plan-num-chunks-per-context','6','--plan-max-workers',str(workers),
      '--plan-run-id',run_id,'--output-prefix','foundation_v1_runs/'+group(partition)+'/foundation_v1_'+digest(run_id)[:12],'--policy-name','AQA',
      '--node-count-control','true','--r-over-p-max','0.6',
      '--gradient-config',str(ROOT/'configs/foundation_v1/gradient_cost_rsr.ini'),
      '--cluster-config',str(ROOT/'configs/cluster/cluster.ini')]


def pending_rows(plan,database,partition=None,task=None):
    from foundation_v1.scope import normalize,SCOPES
    if Path(database).exists():verify_database(database)
    db=connection(database);complete=terminal_ids(db);db.close()
    allowed=SCOPES[normalize(partition)]
    return [r for r in iter_csv(plan) if r['plan_row_id'] not in complete and r['data_split'] in allowed and (task is None or r['Task_ID']==task)]


def execute_scope(a,partition):
    from foundation_v1.scope import (SCOPES,selected_tasks,run_root,raw_root,database_path,
        assert_database_scope,require_unlock,make_phaseA_view)
    if a.execute:require_unlock(partition,a.unlock_locked_test)
    selected_tasks(partition,a.task)  # Unknown/out-of-scope tasks fail loudly.
    plan=stage_path(a.stage,partition,a.task)
    if not plan.exists() and a.stage=='phaseA':plan=make_phaseA_view(partition)
    if not plan.exists():raise ValueError('Stage plan does not exist: generate it only from audited previous-stage results')
    receipt_path=plan.with_suffix('.receipt.json')
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt.get('plan_sha256',receipt.get('sha256'))!=file_hash(plan):raise ValueError('Scoped plan/receipt hash mismatch')
    release=read_json(ARTIFACTS/'preexecution_release.json')
    for name,value in release['source_files'].items():
        if file_hash(ROOT/name)!=value:raise ValueError('Frozen source changed;create a reviewed new release: '+name)
    if file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')!=release['phaseA_sha256']:raise ValueError('Phase-A plan identity mismatch')
    def members():
        index=0
        for row in iter_csv(plan):
            if row['data_split'] not in SCOPES[partition]:raise ValueError('Plan contains rows outside its declared partition scope')
            if a.task is not None and row['Task_ID']!=a.task:continue
            if index%a.chunk_count==a.chunk_index:yield row
            index+=1
    if not a.execute:
        count=sum(1 for _ in members())
        report={'status':'PLANNED_NOT_RUN','partition':partition,'stage':a.stage,'rows':count,'max_workers':a.max_workers,
            'plan_sha256':file_hash(plan),'command_template':commands(plan,a.max_workers,'foundation_v1_'+partition+'_'+a.stage,partition)+['--plan-dry-run'],
            'command_template_scope':'dry-only; scientific execution always uses selected scoped missing-row child plan',
            'execution_requires_explicit_execute':True,'locked_test_unlocked':a.unlock_locked_test,'simulation_calls':0}
        print(json.dumps(report,indent=2));return report
    if not a.resume:raise ValueError('Execution requires --resume for deterministic recovery semantics')
    directory=run_root(partition);database=a.database or database_path(partition)
    assert_database_scope(database,partition)
    directory.mkdir(parents=True,exist_ok=True)
    with (nullcontext() if getattr(a,'_campaign_lock_held',False) else CampaignLock(directory.parent/'campaign.lock')):
        outputs=sorted(raw_root(partition).glob('foundation_v1_*'))
        recovered=ingest(plan,outputs,database)
        if any('mismatch' in r['reason'].lower() or 'conflicting' in r['reason'].lower() or 'differs' in r['reason'].lower() for r in recovered['failures']):
            write_json(directory/'recovery_stop.json',recovered,False);raise ValueError('Scientific input/output disagreement:stop for review')
        # Stable shard membership is assigned before removing completed executions.
        allowed={r['plan_row_id'] for r in members()}
        missing=[r for r in pending_rows(plan,database,partition,a.task) if r['plan_row_id'] in allowed]
        if not allowed:raise ValueError('No rows selected')
        if not missing:
            summary=stage_summary(plan,database,allowed)
            summary['whole_scope_progress']=recovered
            summary['selected_scope']={'task':a.task,'chunk_index':a.chunk_index,'chunk_count':a.chunk_count}
            summary['simulation_calls_this_invocation']=0
            summary['simulation_rows_submitted_this_invocation']=0
            write_json(directory/f'{a.stage}__{partition}_status.json',summary,False)
            print(json.dumps(summary,indent=2))
            if summary['counts']['INSUFFICIENT_JOB_TYPE_EVIDENCE']:raise RuntimeError('STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL: terminal rows are not rerun; scientific policy review required')
            return summary
        selection=digest([partition,a.stage,[r['plan_row_id'] for r in missing]])[:16]
        chunks=directory/'execution_plans';chunks.mkdir(exist_ok=True)
        subset=chunks/f'{a.stage}_{selection}.csv.gz';write_csv(subset,missing)
        db=connection(database);attempt=db.execute('SELECT count(*) FROM attempts').fetchone()[0]+1
        run_id=f'foundation_v1_{partition}_{a.stage}_{selection}_attempt{attempt:04d}'
        command=commands(subset,a.max_workers,run_id,partition)
        with db:db.execute('INSERT INTO attempts VALUES(?,?,?)',(run_id,'RUNNING',json.dumps({'Plan_Row_IDs':[r['plan_row_id'] for r in missing],'command':command,'subset_sha256':file_hash(subset)})))
        with db:
            for row in missing:execution_state(db,row['plan_row_id'],row['Task_ID'],'RUNNING',{'attempt':run_id})
        db.close()
        # The unchanged combo runner nests the output prefix in its log label.
        # Pre-create only that parent; no architecture/worker behavior changes.
        prefix=command[command.index('--output-prefix')+1]
        (ROOT/'.flexdc_worker_logs'/f'plan_{run_id}'/Path(prefix).parent).mkdir(parents=True,exist_ok=True)
        status_path=directory/f'{a.stage}__{partition}_status.json'
        write_json(status_path,{'status':'RUNNING','stage':a.stage,'partition':partition,'attempt':run_id,'rows':len(missing)},False)
        try:code=subprocess.call(command,cwd=ROOT)
        except OSError as error:code=-1;write_json(directory/f'{run_id}_launch_error.json',{'error':str(error),'status':'TECHNICAL_RETRY_REQUIRED'},False)
        verified=ingest(plan,sorted(raw_root(partition).glob('foundation_v1_*')),database)
        whole=verified
        verified=stage_summary(plan,database,allowed)
        verified.update(whole_scope_progress=whole,selected_scope={'task':a.task,'chunk_index':a.chunk_index,'chunk_count':a.chunk_count},
            simulation_calls_this_invocation=max(0,verified['counts']['EXECUTION_COMPLETE']-recovered['counts']['EXECUTION_COMPLETE']),
            simulation_rows_submitted_this_invocation=len(missing))
        status=verified['status']
        db=connection(database)
        with db:
            db.execute('UPDATE attempts SET status=? WHERE attempt_id=?',(status,run_id))
            completed=terminal_ids(db)
            for row in missing:
                if row['plan_row_id'] not in completed:execution_state(db,row['plan_row_id'],row['Task_ID'],'TECHNICAL_RETRY_REQUIRED',{'attempt':run_id,'returncode':code})
        db.close()
        write_json(status_path,{'status':status,'returncode':code,'stage':a.stage,'partition':partition,'attempt':run_id,'audit':verified},False)
        if status=='STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL':raise RuntimeError('Terminal evidence shortfall requires scientific policy; identical rows will NOT rerun')
        if status!='VALID_COMPLETE':raise RuntimeError('Partial outputs preserved;explicit same-identity resume schedules only nonterminal rows')
        print(f'{status}: advance next wave only through measured-result planner')
        return verified


def main():
    from foundation_v1.scope import CHOICES,normalize,require_unlock
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--stage',choices=['phaseA','wave01','wave02','wave03','wave04','repeat'])
    mode.add_argument('--full-campaign',action='store_true',help='Automatically audit/generate/run all stages and isolated final exports')
    p.add_argument('--partition',choices=CHOICES,default='TRAIN_DEV',help='Default TRAIN+DEVELOPMENT; ALL dispatches isolated groups sequentially')
    p.add_argument('--unlock-locked-test',action='store_true',help='Explicitly authorize isolated sealed-test precomputation; never use those outcomes for fitting/tuning')
    p.add_argument('--task');p.add_argument('--chunk-index',type=int,default=0);p.add_argument('--chunk-count',type=int,default=1)
    p.add_argument('--max-workers',type=int,default=10)
    execution=p.add_mutually_exclusive_group();execution.add_argument('--execute',action='store_true');execution.add_argument('--dry-run',action='store_true',help='Explicit no-execution plan preview (also the default)')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--database',type=Path,help='Override ledger path; partition contamination checks still apply')
    a=p.parse_args();partition=normalize(a.partition)
    if a.max_workers<1 or a.chunk_count<1 or not 0<=a.chunk_index<a.chunk_count:p.error('Invalid worker/chunk bounds')
    if a.execute:require_unlock(partition,a.unlock_locked_test)
    if a.full_campaign:
        if a.task or a.database or a.chunk_count!=1 or a.chunk_index!=0:p.error('--full-campaign owns whole isolated scopes; do not combine with task/database/chunk overrides')
        from foundation_v1.full_campaign import run_full
        return run_full(a,partition)
    a.stage=a.stage or 'phaseA'
    if partition=='ALL':
        if a.database or a.task:p.error('ALL uses separate default ledgers; select a single scope for --database/--task')
        return [execute_scope(a,s) for s in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST')]
    return execute_scope(a,partition)
