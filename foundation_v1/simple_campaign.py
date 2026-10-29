"""Clean-slate Foundation campaign: collect -> bounded repair -> certify -> advance."""
import copy
import json
import os
import shutil
import subprocess
import sqlite3
import stat
import time
from pathlib import Path
from collections import Counter
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,contexts,iter_csv,read_json,write_json,file_hash,make_row,row_geometry,digest,canonical
from foundation_v1 import scope,audit,adaptive,collector_receipts as receipts,simple_collector as collector
from foundation_v1 import stage_certification as certificates
from foundation_v1.full_campaign import STAGES,verify_release,counts_for
from foundation_v1.campaign import CampaignLock
from foundation_v1.environment import versions


def stage_area(part,stage):return scope.run_root(part)/'collector_v2'/stage


def mismatch(row,plan,body,error):
    reconstructed={}
    try:
        task=next(t for t in contexts() if t['Task_ID']==row['Task_ID'])
        reconstructed=make_row(task,row_geometry(row),row['Candidate_Source'],int(row['Adaptive_Wave']),simulation_seed=int(row['simulation_seed']))
    except (ValueError,KeyError,StopIteration) as reconstruction_error:reconstructed={'error':str(reconstruction_error)}
    return {'Plan_Row_ID':row['plan_row_id'],'Task_ID':row['Task_ID'],'reason':str(error),
        'original_plan_row_id':row['plan_row_id'],'reconstructed_plan_row_id':reconstructed.get('plan_row_id'),
        'original_Base_Geometry_ID':row['Base_Geometry_ID'],'reconstructed_Base_Geometry_ID':reconstructed.get('Base_Geometry_ID'),
        'P':row['Pbar_kw_per_server'],'R':row['R_kw_per_server'],'weights':row['weights'],
        'Candidate_Source':row['Candidate_Source'],'Adaptive_Wave':row['Adaptive_Wave'],'seed':row['simulation_seed'],
        'source_plan':str(plan),'completion_receipt':str(receipts.receipt_path(body.get('_area',''),row['plan_row_id'])),
        'evidence_file':body.get('evidence_file'),'output':body.get('result_file'),
        'observed_requested':body.get('requested'),'observed_applied':body.get('applied'),
        'observed_evidence_metadata':body.get('evidence',{}).get('plan_metadata'),
        'observed_results':body.get('evidence',{}).get('results'),
        'observed_diagnostics':body.get('evidence',{}).get('diagnostics')}


def certify(plan,area,database):
    """Strong scientific checks happen only after ALL current-stage collection."""
    certificates.invalidate(area,'strong audit started')
    tasks={t['Task_ID']:t for t in contexts()};db=audit.connection(database)
    bad={};missing=[];short=[];cleanup_debts=[];passed=0;seen=set()
    with db:
        for row in iter_csv(plan):
            key=row['plan_row_id']
            if key in seen:raise ValueError('Duplicate immutable stage ID: '+key)
            seen.add(key)
            # Index is derived: replace this stage's old audited labels only.
            for table in ('observations','terminal_executions','execution_states'):
                db.execute('DELETE FROM '+table+' WHERE row_id=?',(key,))
            body={}
            try:
                body=receipts.load(area,key)
                for field in ('Base_Geometry_ID','Task_ID','Workload_ID'):
                    if body[field]!=row[field]:raise ValueError('Completion '+field+' mismatch: expected='+str(row[field])+' observed='+str(body[field]))
                evidence=read_json(body['evidence_file'])
                if evidence!=body['evidence']:raise ValueError('Completion/evidence receipt mismatch')
                # CSV row integrity is a stage check, not a prerequisite to launch.
                from peacsim.foundation_v1_cleanup import verify_compact
                verify_compact(Path(body['result_file']).parent,key)
                obs=audit.observation(row,evidence,tasks[row['Task_ID']])
                audit.store_execution(db,obs,{'receipt':str(receipts.receipt_path(area,key)),'plan':str(plan)})
                if obs['observation_status']=='VALID_OBSERVATION':passed+=1
                else:short.append({'Plan_Row_ID':key,'Task_ID':row['Task_ID'],'evidence_counts':obs['job_type_evidence']})
                from peacsim.foundation_v1_cleanup import _state_path,finalize_row,CleanupDebt
                cleanup_file=_state_path(Path(body['result_file']).parent,key)
                if cleanup_file.exists() and read_json(cleanup_file)['cleanup_status']!='CLEANUP_COMPLETE':
                    try:finalize_row(Path(body['result_file']).parent,key,best_effort=True)
                    except CleanupDebt as cleanup_error:cleanup_debts.append({'Plan_Row_ID':key,'error':str(cleanup_error),'output':str(Path(body['result_file']).parent)})
            except FileNotFoundError as error:
                if not body:missing.append(key)
                else:bad[key]=mismatch(row,plan,dict(body,_area=str(area)),error)
            except (ValueError,KeyError,TypeError,OSError) as error:
                bad[key]=mismatch(row,plan,dict(body,_area=str(area)),error)
    db.close()
    status='PASS' if not bad and not missing and not short else 'EVIDENCE_SHORTFALL_REQUIRES_POLICY' if short else 'REPAIR_REQUIRED'
    result={'status':status,'planned':len(seen),'PASS':passed,'QUARANTINED':bad,'MISSING':missing,'evidence_shortfalls':short,'cleanup_debts':cleanup_debts,'audited_at':time.time()}
    write_json(Path(area)/'stage_audit.json',result,False)
    return result


def execute_stage(a,part):
    area=stage_area(part,a.stage);plan=scope.plan_path(a.stage,part)
    database=scope.database_path(part)
    total_calls=0
    for cycle in range(collector.REPAIR_CYCLES+1):
        result=collector.collect(plan,area,a.max_workers,getattr(a,'storage_emergency_gib',10.0))
        total_calls+=result['new_completions']
        if result['status']=='INTERRUPTED':raise RuntimeError('INTERRUPTED: resume exact plan and durable receipts')
        verify_release()  # Full source/scientific certification at stage boundary.
        report=certify(plan,area,database)
        if report['status']=='PASS':
            return {'status':'VALID_COMPLETE','scientifically_complete':True,
                'counts':{'planned':report['planned'],'VALID_COMPLETE':report['PASS'],'EXECUTION_COMPLETE':report['PASS']},
                'simulation_calls_this_invocation':total_calls,'simulation_rows_submitted_this_invocation':total_calls}
        if report['evidence_shortfalls']:raise RuntimeError('EVIDENCE_SHORTFALL_REQUIRES_POLICY: all unrelated collection completed; no deterministic insufficient-evidence retries')
        # Only invalid rows lose their success marker. Clean IDs never rerun.
        for key,reason in report['QUARANTINED'].items():receipts.archive(area,key,reason)
        from foundation_v1.recovery_index import invalidate
        for key in report['MISSING']:invalidate(area,key)
        if cycle==collector.REPAIR_CYCLES:
            raise RuntimeError('STAGE_AUDIT_SHORTFALL: all unrelated collection completed; see '+str(area/'stage_audit.json'))
    raise AssertionError('unreachable')


def run_full(a,part):
    release=verify_release();environment=versions(require=True);scope.require_unlock(part,a.unlock_locked_test)
    root=ROOT/'foundation_v1_runs';root.mkdir(exist_ok=True)
    scope.PLAN_ROOT=root/'scientific_stage_plans'
    from foundation_v1.operational_release import require_transition
    transitions=require_transition(ROOT,release)
    state={'operational_patch_release':release.get('operational_patch_release'),'operational_patch_release_sha256':release.get('operational_patch_release_sha256'),'source_transition_receipts':transitions,'collection_initial_source_identity':release.get('collection_initial_source_identity',release['uncommitted_source_sha256']),'resume_telemetry':{'certificates_checked':0,'stages_skipped':0},'schema':'FOUNDATION_V1_SIMPLE_CAMPAIGN_V2','status':'RUNNING','started':time.time(),
        'max_workers':a.max_workers,'environment':environment,'pid':os.getpid(),'active':None,'stages':{},'source_identity':release['uncommitted_source_sha256'],'expected':{p:counts_for(p) for p in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,))}}
    state_file=root/'simple_campaign_status.json'
    from foundation_v1.full_campaign import export_scope,ledger_metrics,validate_plan
    with CampaignLock(ROOT/'foundation_v1_campaign.lock'):
        format_file=root/'execution_format.json'
        if not format_file.exists() and any(root.glob('*/observations.sqlite')):
            raise ValueError('FRESH_START_REQUIRED: legacy execution state is not migrated; preview and confirm the safe fresh-start command')
        if format_file.exists() and read_json(format_file).get('schema')!='FOUNDATION_V1_COMPLETION_V2':
            raise ValueError('Unrecognized execution format; review before collecting')
        write_json(format_file,{'schema':'FOUNDATION_V1_COMPLETION_V2','scientific_master':release['phaseA_sha256'] if 'phaseA_sha256' in release else 'SYNTHETIC_FIXTURE'},False)
        try:
            for selected in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,)):
                for i,stage in enumerate(STAGES):
                    state['active']={'partition':selected,'stage':stage};write_json(state_file,state,False)
                    path=scope.plan_path(stage,selected)
                    if not path.exists():
                        if stage=='phaseA':scope.make_phaseA_view(selected)
                        else:adaptive.generate_stage(scope.database_path(selected),selected,
                            wave_number=None if stage=='repeat' else int(stage[-2:]),repeat=stage=='repeat',unlock=a.unlock_locked_test,
                            progress_seconds=getattr(a,'adaptive_progress_seconds',10.),stall_seconds=getattr(a,'adaptive_stall_seconds',300.),
                            protocol=getattr(a,'test_protocol',None))
                    # Byte integrity before execution, geometry certification after.
                    plan_receipt=read_json(path.with_suffix('.receipt.json'))
                    if plan_receipt.get('plan_sha256',plan_receipt.get('sha256'))!=file_hash(path):raise ValueError('Immutable stage plan hash changed')
                    expected_rows=int(plan_receipt.get('rows',state['expected'][selected]['tasks']*(150 if stage=='phaseA' else 20)))
                    area=stage_area(selected,stage);database=scope.database_path(selected)
                    certificate,reason=certificates.validate(path,area,database,selected,stage,expected_rows,release)
                    state['resume_telemetry']['certificates_checked']+=1
                    if certificate:
                        print('STAGE CERTIFIED '+selected+':'+stage+' rows='+str(expected_rows)+'/'+str(expected_rows)+'; previously PASS; historical receipt re-scan skipped',flush=True)
                        report={'status':'VALID_COMPLETE','scientifically_complete':True,'certified_stage_skipped':True,
                            'counts':{'planned':expected_rows,'VALID_COMPLETE':expected_rows,'EXECUTION_COMPLETE':expected_rows},
                            'simulation_calls_this_invocation':0,'simulation_rows_submitted_this_invocation':0}
                        state['resume_telemetry']['stages_skipped']+=1
                    else:
                        certificate_path=area/'stage_certification.json'
                        prefix='STAGE CERTIFICATE ABSENT: performing certified recovery/audit before resume' if not certificate_path.exists() else 'STAGE CERTIFICATE STALE/CORRUPT: performing strong re-audit before resume'
                        print(prefix+' '+selected+':'+stage+'; '+reason,flush=True)
                        call=copy.copy(a);call.stage=stage
                        report=execute_stage(call,selected)
                        validate_plan(path,selected,stage,getattr(a,'test_protocol',None))
                        certificates.create(path,area,database,selected,stage,expected_rows,release)
                    state['stages'][selected+':'+stage]=report;write_json(state_file,state,False)
                export_scope(selected,a.unlock_locked_test)
            metrics={p:ledger_metrics(p) for p in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,))}
            for selected,metric in metrics.items():
                worker_records=(record for stage in STAGES for record in (stage_area(selected,stage)/'workers').glob('*.attempt.json'))
                attempt_count=failed_count=retry_count=0
                for path in worker_records:
                    record=read_json(path);attempt_count+=1;failed_count+=record.get('return_code',0)!=0;retry_count+=record.get('attempt',1)>1
                metric.update(worker_attempts=attempt_count,failed_worker_attempts=failed_count,technical_retry_worker_attempts=retry_count,
                    repaired_receipt_count=sum(len(list((stage_area(selected,stage)/'quarantined_receipts').glob('*.json'))) for stage in STAGES))
            from am_flexdc_foundation_v1_common import ROOT as code_root
            state['git_SHA']=subprocess.check_output(['git','-C',str(code_root),'rev-parse','HEAD'],text=True).strip()
            state.update(status='FOUNDATION_V1_COMPLETE' if part=='ALL' else 'FOUNDATION_SCOPE_COMPLETE',
                ended=time.time(),valid_rows=sum(m['valid_rows'] for m in metrics.values()),metrics=metrics,active=None)
            write_json(state_file,state,False)
            write_json(root/'simple_campaign_completion.json',state,False)
            (root/'simple_campaign_completion.md').write_text('# '+state['status']+'\n\nValid rows: '+str(state['valid_rows'])+'\n\nTRAIN/DEVELOPMENT and LOCKED exports remain physically separate. Locked outcomes are sealed from fitting and selection.\n',encoding='utf8')
            return state
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) or 'INTERRUPTED' in str(error) else 'STOPPED_AT_STAGE_BOUNDARY_OR_GLOBAL_CONDITION',error=str(error))
            write_json(state_file,state,False);raise
        finally:scope.PLAN_ROOT=None


def prepare_next_stage(a,part):
    """Prepare exactly the next unprepared simulator stage and launch zero workers."""
    release=verify_release();versions(require=True);scope.require_unlock(part,a.unlock_locked_test)
    root=ROOT/'foundation_v1_runs';root.mkdir(exist_ok=True);scope.PLAN_ROOT=root/'scientific_stage_plans'
    from foundation_v1.full_campaign import validate_plan
    try:
        for selected in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,)):
            for index,stage in enumerate(STAGES):
                path=scope.plan_path(stage,selected)
                if not path.exists():
                    if stage=='phaseA':path=scope.make_phaseA_view(selected)
                    else:
                        previous=STAGES[index-1];previous_path=scope.plan_path(previous,selected)
                        if not previous_path.exists():raise RuntimeError('PREVIOUS_STAGE_PLAN_MISSING: '+selected+':'+previous)
                        previous_receipt=read_json(previous_path.with_suffix('.receipt.json'))
                        previous_rows=int(previous_receipt.get('rows',counts_for(selected)['tasks']*(150 if previous=='phaseA' else 20)))
                        previous_area=stage_area(selected,previous);database=scope.database_path(selected)
                        certificate,reason=certificates.validate(previous_path,previous_area,database,selected,previous,previous_rows,release)
                        if not certificate:
                            label='ABSENT' if not (previous_area/'stage_certification.json').exists() else 'STALE_OR_CORRUPT'
                            print('STAGE CERTIFICATE '+label+': performing certified recovery/audit before planning '+selected+':'+stage,flush=True)
                            report=certify(previous_path,previous_area,database)
                            if report['status']!='PASS':raise RuntimeError('PREVIOUS_STAGE_NOT_CERTIFIABLE: '+canonical(report))
                            certificates.create(previous_path,previous_area,database,selected,previous,previous_rows,release)
                        path=adaptive.generate_stage(database,selected,wave_number=None if stage=='repeat' else int(stage[-2:]),
                            repeat=stage=='repeat',unlock=a.unlock_locked_test,progress_seconds=a.adaptive_progress_seconds,
                            stall_seconds=a.adaptive_stall_seconds,protocol=getattr(a,'test_protocol',None))
                    validate_plan(path,selected,stage,getattr(a,'test_protocol',None))
                    receipt=read_json(path.with_suffix('.receipt.json'))
                    result={'status':'NEXT_STAGE_READY','partition':selected,'stage':stage,'plan':str(path),
                        'plan_rows':receipt.get('rows'),'plan_sha256':file_hash(path),'pending_rows':receipt.get('rows'),
                        'simulation_calls':0,'SAFE_TO_REQUEST_COMPUTE_ALLOCATION':'YES'}
                    print(json.dumps(result,indent=2));return result
                receipt=read_json(path.with_suffix('.receipt.json'))
                expected=int(receipt.get('rows',counts_for(selected)['tasks']*(150 if stage=='phaseA' else 20)))
                area=stage_area(selected,stage);database=scope.database_path(selected)
                certificate,reason=certificates.validate(path,area,database,selected,stage,expected,release)
                if not certificate:
                    next_missing=index+1<len(STAGES) and not scope.plan_path(STAGES[index+1],selected).exists()
                    if next_missing:
                        label='ABSENT' if not (area/'stage_certification.json').exists() else 'STALE_OR_CORRUPT'
                        print('STAGE CERTIFICATE '+label+': performing certified recovery/audit before planning '+selected+':'+STAGES[index+1],flush=True)
                        report=certify(path,area,database)
                        if report['status']!='PASS':raise RuntimeError('CURRENT_STAGE_NOT_CERTIFIABLE: '+canonical(report))
                        certificates.create(path,area,database,selected,stage,expected,release)
                        continue
                    result={'status':'NEXT_STAGE_ALREADY_PLANNED','partition':selected,'stage':stage,'plan':str(path),
                        'plan_rows':expected,'plan_sha256':file_hash(path),'pending_rows':expected,'simulation_calls':0,
                        'certificate_status':'STALE_OR_MISSING','certificate_reason':reason,'SAFE_TO_REQUEST_COMPUTE_ALLOCATION':'YES'}
                    print(json.dumps(result,indent=2));return result
        result={'status':'FOUNDATION_V1_COMPLETE','simulation_calls':0,'SAFE_TO_REQUEST_COMPUTE_ALLOCATION':'NO_WORK_REMAINING'}
        print(json.dumps(result,indent=2));return result
    finally:scope.PLAN_ROOT=None


def reset_targets(root):
    root=Path(root).resolve()
    targets=[root/'foundation_v1_runs',root/'src/peacsim/output/optimization/foundation_v1_runs',root/'src/peacsim/output/optimization/foundation_v1_simple']
    for folder,pattern in (('.flexdc_plan_chunks','foundation_v1_*'),('.flexdc_worker_logs','plan_foundation_v1_*')):
        targets.extend((root/folder).glob(pattern))
    targets.extend(p for p in root.glob('dataset_plan_launch_manifest_*.csv') if 'foundation_v1_' in p.name)
    result=[]
    for path in targets:
        if not path.exists():continue
        resolved=path.resolve()
        if root not in resolved.parents or path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&0x400:raise ValueError('Unsafe reset target: '+str(path))
        if any(part in ('configs','foundation_v1_artifacts','foundation_v1','src') for part in path.relative_to(root).parts[:1]) and path not in targets[:3]:raise ValueError('Non-execution reset target')
        # Refuse symlinks/junctions anywhere in a deletion tree.
        if path.is_dir():
            for base,dirs,files in os.walk(path,followlinks=False):
                for name in dirs+files:
                    child=Path(base)/name
                    if child.is_symlink() or getattr(child.lstat(),'st_file_attributes',0)&0x400:raise ValueError('Linked reset subtree: '+str(child))
        result.append(path)
    return result


def fresh_start(a):
    # No production deletion occurs implicitly: separate command, double opt-in.
    from contextlib import ExitStack
    with ExitStack() as locks:
        locks.enter_context(CampaignLock(ROOT/'foundation_v1_campaign.lock'))
        old_lock=ROOT/'foundation_v1_runs/campaign.lock'
        if old_lock.exists():locks.enter_context(CampaignLock(old_lock))
        targets=reset_targets(ROOT)
        report={'status':'FRESH_START_DRY_RUN','targets':[str(p) for p in targets],'scientific_artifacts_deleted':False}
        if a.confirm_fresh_start:
            if a.dry_run:raise ValueError('Choose confirmed reset OR dry-run')
            for path in targets:
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink()
            report['status']='FRESH_START_COMPLETE'
        elif not a.dry_run:raise ValueError('Fresh start requires --dry-run or --confirm-fresh-start')
        print(json.dumps(report,indent=2));return report


def process_alive(pid):
    if not pid:return False
    if os.name=='nt':
        from peacsim.foundation_v1_cleanup import parent_changed
        return not parent_changed(str(pid))
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


def _status_event(db,stage,key,t,valid=True):
    # Status is a projection of the newest durable event, not completion authority.
    db.execute('''INSERT INTO events(stage,id,t,valid_complete) VALUES(?,?,?,?)
        ON CONFLICT(stage,id) DO UPDATE SET t=excluded.t,valid_complete=excluded.valid_complete
        WHERE excluded.t>=events.t''',(stage,key,float(t),int(valid)))


def live_status(root=ROOT):
    """Disposable, timestamp-ordered status projection; collection never uses it."""
    root=Path(root);base=root/'foundation_v1_runs'
    if not base.exists():return {'overall_complete':0,'overall_expected':500000,'status':'NOT_STARTED'}
    db=sqlite3.connect(base/'status_cache.sqlite')
    columns=[r[1] for r in db.execute('PRAGMA table_info(events)')]
    if columns and columns!=['stage','id','t','valid_complete']:
        # Legacy cache may already have consumed invalidations as successes.
        with db:db.execute('DROP TABLE events');db.execute('DROP TABLE IF EXISTS offsets')
    db.execute('CREATE TABLE IF NOT EXISTS events(stage TEXT,id TEXT,t REAL,valid_complete INTEGER NOT NULL,PRIMARY KEY(stage,id))')
    db.execute('CREATE TABLE IF NOT EXISTS offsets(path TEXT PRIMARY KEY,position INTEGER)')
    stages=[]
    for part in ('train_dev','locked_synthetic_test'):
        for area in (base/part/'collector_v2').glob('*'):
            if not area.is_dir():continue
            stage=part+':'+area.name
            for path in sorted((area/'events').glob('*.jsonl')):
                old=db.execute('SELECT position FROM offsets WHERE path=?',(str(path),)).fetchone();offset=old[0] if old else 0
                if path.stat().st_size<offset:offset=0
                with path.open('r',encoding='utf8') as stream:
                    stream.seek(offset)
                    while True:
                        line=stream.readline()
                        if not line or not line.endswith('\n'):break
                        try:event=json.loads(line)
                        except ValueError:break
                        _status_event(db,stage,event['id'],event['time'],not event.get('invalidated',False));offset=stream.tell()
                db.execute('INSERT OR REPLACE INTO offsets VALUES(?,?)',(str(path),offset))
            for path in (area/'events').glob('*.pending.json'):
                try:
                    pending=read_json(path);body=receipts.load(area,pending['Plan_Row_ID'])
                    _status_event(db,stage,body['Plan_Row_ID'],body['timestamp'])
                except (ValueError,OSError,KeyError):pass
            progress=read_json(area/'progress.json') if (area/'progress.json').exists() else {}
            audited=read_json(area/'stage_audit.json') if (area/'stage_audit.json').exists() else {}
            for key in [*audited.get('QUARANTINED',{}),*audited.get('MISSING',[])]:
                _status_event(db,stage,key,audited.get('audited_at',0),False)
            complete=db.execute('SELECT count(*) FROM events WHERE stage=? AND valid_complete=1',(stage,)).fetchone()[0]
            live=sum(process_alive(pid) for pid in progress.get('worker_pids',[]))
            stages.append(dict(progress,stage=stage,completed=complete,pending=max(0,progress.get('planned',complete)-complete),
                estimated_workers_active=live,audit_failures=len(audited.get('QUARANTINED',{})),cleanup_debts=len(audited.get('cleanup_debts',[]))))
    db.commit();now=time.time();total=db.execute('SELECT count(*) FROM events WHERE valid_complete=1').fetchone()[0]
    hour=db.execute('SELECT count(*) FROM events WHERE valid_complete=1 AND t>=?',(now-3600,)).fetchone()[0]
    minute=db.execute('SELECT count(*) FROM events WHERE valid_complete=1 AND t>=?',(now-60,)).fetchone()[0]
    counts={part:db.execute('SELECT count(*) FROM events WHERE valid_complete=1 AND stage LIKE ?',(part+':%',)).fetchone()[0] for part in ('train_dev','locked_synthetic_test')};db.close()
    state=read_json(base/'simple_campaign_status.json') if (base/'simple_campaign_status.json').exists() else {}
    return {'status':('INTERRUPTED_RECOVERABLE' if state.get('status')=='RUNNING' and not process_alive(state.get('pid')) else state.get('status','COLLECTING')),'active':state.get('active'),'source_identity':state.get('source_identity'),'overall_complete':total,'overall_expected':500000,
        'partition_complete':counts,'rows_completed_last_hour':hour,'rows_per_second_last_minute':minute/60,'stages':stages}


def preview(a,part):
    """Accurate no-execution preview of the new collector, not legacy combo commands."""
    release=verify_release();steps=[]
    for selected in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,)):
        for stage in STAGES:
            path=ROOT/'foundation_v1_runs/scientific_stage_plans'/('foundation_v1_'+stage+'__'+selected+'.csv.gz')
            steps.append({'partition':selected,'stage':stage,'rows':counts_for(selected)['tasks']*(150 if stage=='phaseA' else 20),
                'plan':str(path),'plan_status':'EXISTING_PLAN' if path.exists() else 'WILL_COPY_IMMUTABLE_MASTER_VIEW' if stage=='phaseA' else 'WILL_GENERATE_AFTER_PREVIOUS_STAGE_AUDIT',
                'collector':'simple_collector.collect -> direct exact-plan wizard workers',
                'max_workers':a.max_workers,'chunk_rows':collector.CHUNK_ROWS,'after_collection':'bounded repair -> strong stage audit'})
        steps.append({'partition':selected,'stage':'export','partitions':['TRAIN','DEVELOPMENT','TRAIN_DEV'] if selected=='TRAIN_DEV' else [selected]})
    result={'status':'PLANNED_NOT_RUN','mode':'SIMPLE_COLLECTOR_V2','partition':part,'steps':steps,
        'expected_rows':sum(counts_for(p)['total'] for p in (('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') if part=='ALL' else (part,))),
        'source_identity':release['uncommitted_source_sha256'],'simulation_calls':0,'adaptive_coordinates_generated':False,
        'completion_authority':'durable per-row receipts; SQLite derived only'}
    print(json.dumps(result,indent=2));return result


def dispatch(a,parser):
    if a.fresh_start:
        if a.execute:parser.error('Fresh-start is a separate explicit operation; do not combine with --execute')
        return fresh_start(a)
    if a.confirm_fresh_start:parser.error('--confirm-fresh-start requires --fresh-start')
    if a.status:
        result=live_status();print(json.dumps(result,indent=2));return result
    if a.max_workers<1 or a.storage_emergency_gib<0:parser.error('Invalid workers/storage threshold')
    if a.adaptive_progress_seconds<=0 or a.adaptive_stall_seconds<=0:parser.error('Adaptive progress/stall intervals must be positive')
    part=scope.normalize(a.partition)
    if (a.full_campaign or a.prepare_next_stage) and (a.task or a.database or a.chunk_count!=1 or a.chunk_index!=0):
        parser.error('Full campaign owns whole isolated scopes; task/database/chunk overrides are retired')
    if a.stage:parser.error('Stage-only launching is retired for the new collector; use --full-campaign')
    if a.prepare_next_stage:
        if a.execute:parser.error('--prepare-next-stage launches zero simulator workers; omit --execute')
        return prepare_next_stage(a,part)
    if not a.execute:return preview(a,part)
    if not a.full_campaign:parser.error('The new collector executes the full campaign; use --full-campaign')
    return run_full(a,part)
