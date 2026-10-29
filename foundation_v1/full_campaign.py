"""One-node sequential-scope campaign state machine; no acquisition logic lives here."""
import copy
import json
import sqlite3
import subprocess
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,PHASE_A,Domain,contexts,
    read_json,write_json,iter_csv,file_hash,make_row,row_geometry,digest)
from foundation_v1 import campaign,adaptive,audit,scope
from foundation_v1.environment import versions
from peacsim.foundation_v1_cleanup import telemetry

STAGES=('phaseA','wave01','wave02','wave03','wave04','repeat')
MASTER_SHA256='bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f'

def now():return datetime.now(timezone.utc).isoformat()

def counts_for(partition):
    n=len(scope.selected_tasks(partition))
    return {'tasks':n,'phaseA':150*n,'adaptive':80*n,'repeat':20*n,'total':250*n}

def verify_release():
    release=read_json(ARTIFACTS/'preexecution_release.json')
    if file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')!=MASTER_SHA256:raise ValueError('STOP: approved master Phase-A bytes changed')
    if release['phaseA_sha256']!=MASTER_SHA256:raise ValueError('Release master identity changed')
    from foundation_v1 import operational_release
    if (ARTIFACTS/operational_release.NAME).exists():return operational_release.verify(ROOT,ARTIFACTS,release)
    for name,expected in release['source_files'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Frozen source/config changed; a newly reviewed release is required: '+name)
    return release

def scopes_for(partition):return ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if partition=='ALL' else (partition,)

def dry_run(a,partition):
    verify_release();steps=[]
    for part in scopes_for(partition):
        n=counts_for(part)['tasks']
        for stage in STAGES:
            path=scope.plan_path(stage,part)
            steps.append({'partition':part,'stage':stage,'rows':n*(150 if stage=='phaseA' else 20),
                'plan_status':'EXISTING_PLAN' if path.exists() else 'WILL_GENERATE_AFTER_PREVIOUS_STAGE',
                'plan':str(path),'max_workers':a.max_workers,
                'command_template':campaign.commands(path,a.max_workers,'full_preview_'+part+'_'+stage,part)+['--plan-dry-run']})
        steps.append({'partition':part,'stage':'export','status':'WILL_AUDIT_AND_EXPORT_SCOPED_VALID_LABELS',
            'export_partitions':['TRAIN','DEVELOPMENT','TRAIN_DEV'] if part=='TRAIN_DEV' else [part]})
    result={'status':'PLANNED_NOT_RUN','mode':'FULL_CAMPAIGN','partition':partition,'steps':steps,
        'expected':{part:counts_for(part) for part in scopes_for(partition)},
        'expected_total_rows':sum(counts_for(p)['total'] for p in scopes_for(partition)),
        'locked_policy':'PRECOMPUTED_SEALED_SYNTHETIC_TEST; never fitting or model selection',
        'simulation_calls_this_invocation':0,'adaptive_coordinates_generated':False,'environment':versions()}
    print(json.dumps(result,indent=2));return result

def validate_plan(path,part,stage,protocol=None):
    receipt_path=path.with_suffix('.receipt.json')
    if not receipt_path.exists():raise ValueError('Missing atomic stage receipt: '+str(receipt_path))
    receipt=read_json(receipt_path);sha=file_hash(path)
    if receipt.get('plan_sha256',receipt.get('sha256'))!=sha:raise ValueError('Stage plan/receipt hash mismatch')
    if receipt.get('scope')!=part:raise ValueError('Stage receipt partition mismatch')
    if stage!='phaseA' and receipt.get('adaptive_protocol_sha256')!=file_hash(ARTIFACTS/'foundation_v1_adaptive_protocol.json'):raise ValueError('Adaptive protocol identity mismatch')
    tasks={t['Task_ID']:t for t in scope.selected_tasks(part)};counts=Counter();sources=defaultdict(Counter);seen=set();groups=defaultdict(lambda:defaultdict(set));domains={}
    wave=0 if stage=='phaseA' else 5 if stage=='repeat' else int(stage[-2:])
    for row in iter_csv(path):
        task=tasks.get(row['Task_ID'])
        if task is None or row['data_split']!=task['split']:raise ValueError('Stage plan task/partition mismatch')
        key=row['plan_row_id']
        if key in seen:raise ValueError('Duplicate stage scientific row identity')
        seen.add(key);counts[task['Task_ID']]+=1;sources[task['Task_ID']][row['Candidate_Source']]+=1
        seed=int(row['simulation_seed'])
        if int(row['Adaptive_Wave'])!=wave or (stage=='repeat' and seed not in (21,22)) or (stage!='repeat' and seed!=20):raise ValueError('Stage/seed contract mismatch')
        if stage!='phaseA':
            d=domains.setdefault(task['Task_ID'],Domain(task['domain']));x=d.validate(row_geometry(row))
            expected=make_row(task,x,row['Candidate_Source'],wave,simulation_seed=seed)
            if any(row[k]!=expected[k] for k in ('plan_row_id','Plan_Row_ID','Foundation_Row_ID','Base_Geometry_ID')):raise ValueError('Stage scientific row content identity mismatch')
        groups[task['Task_ID']][row['Base_Geometry_ID']].add(seed)
    required=(protocol['initial'] if stage=='phaseA' else
        protocol['repeat_bases']*len(protocol['repeat_seeds']) if stage=='repeat' else
        sum(protocol['wave_quotas'])) if protocol else (150 if stage=='phaseA' else 20)
    if set(counts)!=set(tasks) or set(counts.values())!={required} or receipt['rows']!=len(seen):raise ValueError('Stage candidate/task count quota mismatch')
    if stage=='phaseA' and protocol is None and any(dict(v)!=PHASE_A for v in sources.values()):raise ValueError('Phase-A source quota mismatch')
    if stage.startswith('wave') and any(v['GLOBAL_ESCAPE']<(protocol['wave_quotas'][0] if protocol else 4) for v in sources.values()):raise ValueError('Missing protected global-escape quota')
    if stage=='repeat':
        repeat_bases=protocol['repeat_bases'] if protocol else 10;repeat_seeds=set(protocol['repeat_seeds']) if protocol else {21,22}
        if any(len(v)!=repeat_bases or any(s!=repeat_seeds for s in v.values()) for v in groups.values()):raise ValueError('Repeat plan groups/seeds mismatch')
    elif any(len(v)!=required for v in groups.values()):raise ValueError('Stage unique-geometry quota mismatch')
    return {'plan_sha256':sha,'rows':len(seen),'tasks':len(tasks),'status':'VALID_PLAN'}

def inspect_stage(part,stage):
    path=scope.plan_path(stage,part);db=scope.database_path(part)
    if not path.exists():return {'status':'PENDING' if stage=='phaseA' else 'NOT_GENERATED'}
    # A CSV without a receipt can occur between the two atomic writes. It cannot
    # execute; deterministic generation below may complete the receipt safely.
    if not path.with_suffix('.receipt.json').exists():return {'status':'PLAN_RECEIPT_PENDING'}
    validate_plan(path,part,stage)
    if not db.exists():return {'status':'PENDING'}
    scope.assert_database_scope(db,part);audit.verify_database(db)
    summary=audit.stage_summary(path,db)
    return {'status':'COMPLETE' if summary['status']=='VALID_COMPLETE' else summary['status'],
            'counts':summary['counts'],'scientifically_complete':summary['scientifically_complete']}

def export_scope(part,unlock):
    db=scope.database_path(part);audit.verify_database(db)
    outputs=['TRAIN','DEVELOPMENT','TRAIN_DEV'] if part=='TRAIN_DEV' else [part]
    receipt_path=scope.run_root(part)/'full_export_receipt.json';seal=audit.verify_database(db)
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt['database_content_sha256']!=seal:raise ValueError('Export ledger identity changed')
        for name,sha in receipt['files'].items():
            if file_hash(scope.run_root(part)/name)!=sha:raise ValueError('Corrupt final export; stop for review: '+name)
        return receipt
    for selected in outputs:audit.export(db,selected,unlock)
    paths=sorted(p for p in (scope.run_root(part)/'exports').rglob('*') if p.is_file())
    receipt={'status':'COMPLETE','partitions':outputs,'database_content_sha256':seal,
        'files':{p.relative_to(scope.run_root(part)).as_posix():file_hash(p) for p in paths}}
    write_json(receipt_path,receipt);return receipt

def ledger_metrics(part):
    dbpath=scope.database_path(part);scope.assert_database_scope(dbpath,part);audit.verify_database(dbpath)
    db=sqlite3.connect(dbpath);splits=Counter();sources=Counter();seed_counts=Counter()
    for (body,) in db.execute('SELECT body FROM observations'):
        row=json.loads(body)
        if row['observation_status']!='VALID_OBSERVATION' or row['evidence_status']!='SUFFICIENT_JOB_TYPE_EVIDENCE':raise ValueError('Incomplete label in final observations')
        splits[row['split']]+=1;sources[row['Candidate_Source']]+=1;seed_counts[str(row['simulation_seed'])]+=1
    expected=counts_for(part)
    if sum(splits.values())!=expected['total'] or seed_counts!={'20':expected['tasks']*230,'21':expected['tasks']*10,'22':expected['tasks']*10}:raise ValueError('Final corpus/seed count mismatch')
    insufficient=db.execute("SELECT count(*) FROM terminal_executions WHERE observation_status='INSUFFICIENT_JOB_TYPE_EVIDENCE'").fetchone()[0]
    technical_attempts=0;submitted=0;unique=set()
    for status,details in db.execute('SELECT status,details FROM attempts'):
        technical_attempts+=status!='VALID_COMPLETE'
        ids=json.loads(details).get('Plan_Row_IDs',[]);submitted+=len(ids);unique.update(ids)
    db.close();invalid=set()
    for stage in STAGES:
        status_file=scope.run_root(part)/f'{stage}__{part}_status.json'
        if not status_file.exists():continue
        state=read_json(status_file);state=state.get('audit',state);state=state.get('whole_scope_progress',state)
        invalid.update(r['receipt'] for r in state.get('failures',[]) if 'receipt' in r)
    return {'valid_rows':sum(splits.values()),'partition_counts':dict(splits),'candidate_source_counts':dict(sources),
        'simulation_seed_counts':dict(seed_counts),'repeat_rows':seed_counts['21']+seed_counts['22'],
        'evidence_shortfall_rows':insufficient,'technical_or_interrupted_attempts':technical_attempts,
        'technical_retry_row_submissions':max(0,submitted-len(unique)),
        'invalid_receipt_count':len(invalid),'invalid_execution_count_definition':'unique invalid technical receipt paths retained in per-stage audits; not scientific infeasibility',
        'database_content_sha256':audit.verify_database(dbpath),
        'transient_cleanup': telemetry(sorted(scope.raw_root(part).glob('foundation_v1_*')))}

def run_full(a,partition):
    partition=scope.normalize(partition)
    if not a.execute:return dry_run(a,partition)
    scope.require_unlock(partition,a.unlock_locked_test)
    if not a.resume:raise ValueError('Full execution requires --resume for crash-safe idempotency')
    release=verify_release();environment=versions(require=True)
    root=ROOT/'foundation_v1_runs';root.mkdir(parents=True,exist_ok=True)
    path=root/f'full_campaign_status__{partition}.json'
    with campaign.CampaignLock(root/'campaign.lock'):
        started=now()
        if path.exists():
            try:started=read_json(path).get('campaign_started_at',started)
            except (ValueError,OSError):pass # status cache is never scientific truth
        state={'status':'RUNNING','mode':'FULL_CAMPAIGN','partition':partition,'campaign_started_at':started,
            'invocation_started_at':now(),'max_workers':a.max_workers,'simulation_calls_this_invocation':0,
            'simulation_rows_submitted_this_invocation':0,'reviewed_source_identity':release['uncommitted_source_sha256'],
            'environment':environment,'scopes':{}}
        def save():
            state['updated_at']=now();write_json(path,state,False)
        active=None
        try:
            # Rebuild progress from immutable plans and authoritative ledgers.
            # No old status COMPLETE flag can suppress an audit or simulation.
            for part in scopes_for(partition):
                scope.assert_database_scope(scope.database_path(part),part)
                state['scopes'][part]={stage:inspect_stage(part,stage) for stage in STAGES}
                state['scopes'][part]['export']={'status':'PENDING'}
            save()
            for part in scopes_for(partition):
                for i,stage in enumerate(STAGES):
                    active=(part,stage);state['active']={'partition':part,'stage':stage};save()
                    stage_path=scope.plan_path(stage,part)
                    if not stage_path.exists() or not stage_path.with_suffix('.receipt.json').exists():
                        if stage=='phaseA':scope.make_phaseA_view(part)
                        else:
                            # The previous stage has passed all gates in THIS
                            # invocation, including recovery after a crash.
                            if state['scopes'][part][STAGES[i-1]]['status']!='COMPLETE':raise ValueError('Previous stage is not scientifically complete')
                            adaptive.generate_stage(scope.database_path(part),part,
                                wave_number=None if stage=='repeat' else int(stage[-2:]),
                                repeat=stage=='repeat',unlock=a.unlock_locked_test)
                    plan_audit=validate_plan(stage_path,part,stage)
                    state['scopes'][part][stage]={'status':'RUNNING','plan':plan_audit};save()
                    stage_args=copy.copy(a);stage_args.stage=stage;stage_args._campaign_lock_held=True
                    result=campaign.execute_scope(stage_args,part)
                    failures=result.get('whole_scope_progress',result).get('failures',[])
                    if any(any(word in f['reason'].lower() for word in ('mismatch','conflicting','differs')) for f in failures):raise ValueError('Stage scientific input/output disagreement: stop for review')
                    expected=counts_for(part)['tasks']*(150 if stage=='phaseA' else 20)
                    if result['status']!='VALID_COMPLETE' or not result['scientifically_complete'] or result['counts']['VALID_COMPLETE']!=expected or result['counts']['planned']!=expected:
                        raise RuntimeError('Stage scientific gate failed: '+result['status'])
                    state['simulation_calls_this_invocation']+=result.get('simulation_calls_this_invocation',0)
                    state['simulation_rows_submitted_this_invocation']+=result.get('simulation_rows_submitted_this_invocation',0)
                    state['scopes'][part][stage]={'status':'COMPLETE','counts':result['counts'],'plan':plan_audit,'transient_cleanup':result.get('transient_cleanup',{})};save()
                active=(part,'export');state['active']={'partition':part,'stage':'export'};save()
                receipt=export_scope(part,a.unlock_locked_test)
                state['scopes'][part]['export']={'status':'COMPLETE','receipt':str(scope.run_root(part)/'full_export_receipt.json')};save()
            metrics={part:ledger_metrics(part) for part in scopes_for(partition)}
            total=sum(m['valid_rows'] for m in metrics.values())
            state['transient_cleanup']={name:sum(m.get('transient_cleanup',{}).get(name,0) for m in metrics.values()) for name in ('transient_cleanup_files_deleted','transient_cleanup_bytes_deleted','transient_cleanup_failures','transient_cleanup_retries','transient_cleanup_pending')}
            if total!=sum(counts_for(p)['total'] for p in scopes_for(partition)):raise ValueError('Final full corpus total differs')
            state.update(status='FOUNDATION_V1_COMPLETE' if partition=='ALL' else 'FOUNDATION_SCOPE_COMPLETE',
                ended_at=now(),valid_rows=total,expected={p:counts_for(p) for p in scopes_for(partition)},metrics=metrics,
                git_SHA=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                release=release,sealed_test_policy='precomputed in this campaign; isolated from model fitting and selection; no mixed training CSV',
                counting_note='simulation_calls counts newly audited terminal executions; submitted row count includes technical attempts; unreceipted crashes cannot be counted as scientifically complete')
            state.pop('active',None);save()
            manifest=root/f'foundation_v1_completion__{partition}.json';write_json(manifest,state,False)
            report=f"# {state['status']}\n\nValidated rows: {total}. No mixed locked/train model-training file was created.\n\n"
            report+='\n'.join(f"- {p}: {m['valid_rows']} valid rows; {m['repeat_rows']} repeats; {m['evidence_shortfall_rows']} evidence shortfalls; {m['technical_retry_row_submissions']} technical retry submissions." for p,m in metrics.items())
            report+='\n\nLocked metrics are sealed; this campaign summary reports counts only. Future completely new real profiles remain the strongest untouched external holdout.\n'
            (root/f'foundation_v1_completion__{partition}.md').write_text(report,encoding='utf8')
            print(json.dumps({'status':state['status'],'valid_rows':total,'simulation_calls_this_invocation':state['simulation_calls_this_invocation'],'manifest':str(manifest)},indent=2));return state
        except BaseException as error:
            text=str(error)
            state['status']='TRANSIENT_CLEANUP_REQUIRED' if 'TRANSIENT_CLEANUP_REQUIRED' in text else 'EVIDENCE_SHORTFALL_REQUIRES_POLICY' if 'SHORTFALL' in text else 'STOPPED_REQUIRES_REVIEW_OR_RESUME'
            if isinstance(error,(ValueError,sqlite3.DatabaseError)):state['status']='GLOBAL_FATAL_INVARIANT'
            if isinstance(error,KeyboardInterrupt):state['status']='INTERRUPTED'
            for category in ('GLOBAL_FATAL_INVARIANT','SYSTEMIC_TECHNICAL_FAILURE','INTERRUPTED','STAGE_COMPLETE_WITH_TECHNICAL_SHORTFALL'):
                if category in text:state['status']=category
            state['error']={'type':type(error).__name__,'message':text}
            if active:state['scopes'][active[0]][active[1]]={'status':state['status']}
            save();raise
