"""Strict row input/evidence audits and crash-safe canonical observation index."""
import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
from peacsim.foundation_v1_schema import validate_required_labels

from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,Domain,canonical,contexts,
    read_json,read_csv,row_geometry,make_row,digest,file_hash,write_json,write_csv,violations,parser)


def observation(row,receipt,task):
    if receipt.get('body_sha256')!=digest({k:v for k,v in receipt.items() if k!='body_sha256'}):raise ValueError('Corrupt evidence receipt')
    if receipt.get('schema')!='FOUNDATION_V1_EVIDENCE' or receipt['Plan_Row_ID']!=row['plan_row_id']:raise ValueError('Foreign/historical row identity')
    d=Domain(task['domain']);x=d.validate(row_geometry(row));a=receipt['applied'];r=receipt['results'];diag=receipt['diagnostics']
    expected=make_row(task,x,row['Candidate_Source'],int(row['Adaptive_Wave']),simulation_seed=int(row['simulation_seed']))
    if expected['plan_row_id']!=row['plan_row_id'] or expected['Base_Geometry_ID']!=row['Base_Geometry_ID']:raise ValueError('Geometry/row content identity mismatch')
    if a['N']!=d.context.server_count or a['U']!=d.context.utilization or a['seed']!=int(row['simulation_seed']) or a['J']!=d.j:raise ValueError('Applied N/U/seed/J mismatch')
    if a['duration']!=3600 or a['idle_watts']!=120. or a['policy_module']!='peacsim.aqa_runtimepolicy' or not a['node_count_control']:raise ValueError('Canonical execution contract mismatch')
    if not np.allclose(a['weights'],x[2:],rtol=0,atol=1e-9):raise ValueError('Applied weights differ')
    if abs(a['P_watts']-round(x[0]*1000*a['N']))>0 or abs(a['R_watts']-round(x[1]*1000*a['N']))>0:raise ValueError('Applied power quantization mismatch')
    if Path(a['workload_path']).name!=Path(row['workload_config']).name or a['workload_sha256']!=task['Workload_SHA256']:raise ValueError('Applied workload mismatch')
    if a['iso_sha256']!=task['ISO_SHA256'] or a['randomize_iso_start'] or not a['normalize_iso_signal'] or a['iso_signal_start_hour']!=d.context.iso_signal_start_hour or a['iso_signal_granularity']!=d.context.iso_signal_granularity:raise ValueError('Grid identity mismatch')
    for record in (r,diag):
        for key,expected_value in (('Pbar_kw_per_server',x[0]),('R_kw_per_server',x[1]),('server_count',a['N']),('utilization',a['U']),('Simulation_Seed',a['seed'])):
            if not math.isclose(float(record[key]),expected_value,rel_tol=0,abs_tol=1e-9):raise ValueError('Serialized request differs:'+key)
        if str(record['Plan_Row_ID'])!=row['plan_row_id']:raise ValueError('CSV/receipt row differs')
    if not np.allclose([float(r[f'Weight_{i}']) for i in range(d.j)],x[2:],rtol=0,atol=1e-9):raise ValueError('Serialized weights differ')
    validate_required_labels(r,diag,d.j)
    pj=json.loads(r['QoS_Delay_Probabilities'])
    p90=float(diag['Ctrack_Epsilon_90th']);mean=float(r['Mtrack_Error_MeanAbs_Normalized'])
    if len(pj)!=d.j or not np.isfinite([mean,p90,*pj]).all() or mean<0:raise ValueError('Missing/nonfinite behavior vector')
    vmax,vsum=violations(p90,pj)
    counts=receipt['evidence_counts']
    if len(counts)!=d.j or [c['job_type_id'] for c in counts]!=list(range(d.j)):raise ValueError('Wrong evidence vector')
    sufficient=all(c['observations']>0 for c in counts)
    evidence='SUFFICIENT_JOB_TYPE_EVIDENCE' if sufficient else 'INSUFFICIENT_JOB_TYPE_EVIDENCE'
    if receipt['evidence_status']!=evidence:raise ValueError('Evidence status mismatch')
    money=float(r['Simulator_RSR_Total_Cost'])
    objective=money+float(np.logaddexp(0,10*(p90-.3)))+20*float(np.logaddexp(0,2*(np.array(pj)-.1)).sum())
    if not math.isclose(objective,float(diag['Diagnostic_FullPaperObjective_Cost']),rel_tol=0,abs_tol=1e-7):raise ValueError('Canonical objective mismatch')
    return {'Plan_Row_ID':row['plan_row_id'],'Foundation_Row_ID':row['Foundation_Row_ID'],'Base_Geometry_ID':row['Base_Geometry_ID'],
      'Task_ID':task['Task_ID'],'Workload_ID':task['Workload_ID'],'split':task['split'],'hardware_class':task['hardware_class'],
      'Profile_IDs':task['Profile_IDs'],'J':d.j,'QoS_pattern':task['QoS_pattern'],'N':a['N'],'U':a['U'],
      'x':x.tolist(),'simulation_seed':a['seed'],'Candidate_Source':row['Candidate_Source'],'Adaptive_Wave':int(row['Adaptive_Wave']),
      'execution_provenance':receipt.get('runtime_provenance',{}),
      'mean_tracking':mean,'p90':p90,'Pj':pj,'objective':objective,'objective_M_RSR':money,
      'objective_tracking':float(np.logaddexp(0,10*(p90-.3))),'objective_QoS':20*float(np.logaddexp(0,2*(np.array(pj)-.1)).sum()),
      'evidence_counts':[c['observations'] for c in counts],'evidence_status':evidence,'input_status':'LEGAL_INPUT',
      'execution_status':'EXECUTION_COMPLETE',
      'observation_status':'VALID_OBSERVATION' if sufficient else 'INSUFFICIENT_JOB_TYPE_EVIDENCE',
      'numerical_status':('NUMERICALLY_FEASIBLE' if vmax==0 else 'NUMERICALLY_INFEASIBLE') if sufficient else 'NOT_EVALUABLE_DUE_TO_INSUFFICIENT_EVIDENCE',
      'evidence_qualified_feasible':sufficient and vmax==0,'Vmax':vmax if sufficient else None,'Vsum':vsum if sufficient else None,
      'raw_results':r,'raw_diagnostics':diag,'job_type_evidence':counts,'row_plan':row,'applied':a}


def connection(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS observations(row_id TEXT PRIMARY KEY,task_id TEXT,seed INTEGER,body TEXT,body_hash TEXT,source TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS terminal_executions(row_id TEXT PRIMARY KEY,task_id TEXT,seed INTEGER,observation_status TEXT,body TEXT,body_hash TEXT,source TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS execution_states(row_id TEXT PRIMARY KEY,task_id TEXT,status TEXT,details TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS attempts(attempt_id TEXT PRIMARY KEY,status TEXT,details TEXT)')
    return db


def store(db,obs,source):
    if obs.get('observation_status','VALID_OBSERVATION')!='VALID_OBSERVATION' or obs.get('evidence_status','SUFFICIENT_JOB_TYPE_EVIDENCE')!='SUFFICIENT_JOB_TYPE_EVIDENCE':raise ValueError('Incomplete scientific labels cannot enter valid observations')
    body=canonical(obs);old=db.execute('SELECT body_hash FROM observations WHERE row_id=?',(obs['Plan_Row_ID'],)).fetchone()
    if old and old[0]!=digest(obs):raise ValueError('Conflicting duplicate scientific row')
    db.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?)',(obs['Plan_Row_ID'],obs['Task_ID'],obs['simulation_seed'],body,digest(obs),canonical(source)))


def verify_database(path):
    if not Path(path).exists():raise FileNotFoundError(path)
    db=sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro',uri=True)
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Corrupt Foundation observation database')
        h=[]
        for body,expected in db.execute('SELECT body,body_hash FROM observations ORDER BY row_id'):
            if digest(json.loads(body))!=expected:raise ValueError('Observation body checksum mismatch')
            h.append(expected)
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='terminal_executions'").fetchone():
            for body,expected in db.execute('SELECT body,body_hash FROM terminal_executions ORDER BY row_id'):
                if digest(json.loads(body))!=expected:raise ValueError('Terminal execution body checksum mismatch')
                h.append(expected)
        return digest(h)
    finally:db.close()


def read_task_observations(path,task_id,primary_only=False):
    db=sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro',uri=True)
    try:return [json.loads(r[0]) for r in db.execute('SELECT body FROM observations WHERE task_id=?'+(' AND seed=20' if primary_only else '')+' ORDER BY row_id',(task_id,))]
    finally:db.close()


def execution_state(db,row_id,task_id,status,details=None):
    if status not in ('PLANNED','RUNNING','TECHNICAL_RETRY_REQUIRED','EXECUTION_COMPLETE'):raise ValueError('Unknown execution status')
    if status!='EXECUTION_COMPLETE' and db.execute('SELECT 1 FROM terminal_executions WHERE row_id=?',(row_id,)).fetchone():return
    db.execute('INSERT OR REPLACE INTO execution_states VALUES(?,?,?,?)',(row_id,task_id,status,canonical(details or {})))


def store_execution(db,obs,source):
    if obs['execution_status']!='EXECUTION_COMPLETE' or obs['observation_status'] not in ('VALID_OBSERVATION','INSUFFICIENT_JOB_TYPE_EVIDENCE'):
        raise ValueError('Only technically audited complete executions are terminal')
    old=db.execute('SELECT body_hash FROM terminal_executions WHERE row_id=?',(obs['Plan_Row_ID'],)).fetchone()
    if old and old[0]!=digest(obs):raise ValueError('Conflicting duplicate terminal execution')
    db.execute('INSERT OR IGNORE INTO terminal_executions VALUES(?,?,?,?,?,?,?)',(obs['Plan_Row_ID'],obs['Task_ID'],obs['simulation_seed'],obs['observation_status'],canonical(obs),digest(obs),canonical(source)))
    execution_state(db,obs['Plan_Row_ID'],obs['Task_ID'],'EXECUTION_COMPLETE',{'observation_status':obs['observation_status']})
    if obs['observation_status']=='VALID_OBSERVATION':store(db,obs,source)


def terminal_ids(db):
    # Legacy valid observations remain terminal without requiring data rewrite.
    return {r[0] for r in db.execute('SELECT row_id FROM observations UNION SELECT row_id FROM terminal_executions')}


def stage_summary(plan_path,database,row_ids=None):
    from collections import Counter
    db=connection(database);done=terminal_ids(db)
    valid={r[0] for r in db.execute('SELECT row_id FROM observations')}
    insufficient={row_id:json.loads(body) for row_id,body in db.execute("SELECT row_id,body FROM terminal_executions WHERE observation_status='INSUFFICIENT_JOB_TYPE_EVIDENCE'")}
    planned=Counter();usable=Counter();executed=Counter();shortfalls=[];total=0
    for row in __import__('am_flexdc_foundation_v1_common').iter_csv(plan_path):
        key=row['plan_row_id']
        if row_ids is not None and key not in row_ids:continue
        task=row['Task_ID'];planned[task]+=1;total+=1
        usable[task]+=key in valid;executed[task]+=key in done
        if key in insufficient:
            o=insufficient[key]
            shortfalls.append({'Plan_Row_ID':key,'Task_ID':task,'job_types':[e for e in o['job_type_evidence'] if e['observations']<=0],
                'all_job_type_evidence':o['job_type_evidence'],'evidence_counts':o['evidence_counts']})
    db.close()
    missing=total-sum(executed.values());nshort=len(shortfalls)
    status='TECHNICAL_RETRY_REQUIRED' if missing else 'STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL' if nshort else 'VALID_COMPLETE'
    return {'status':status,'counts':{'planned':total,'EXECUTION_COMPLETE':sum(executed.values()),
        'VALID_COMPLETE':sum(usable.values()),'INSUFFICIENT_JOB_TYPE_EVIDENCE':nshort,
        'INVALID_COMPLETE':0,'missing':missing,'TECHNICAL_RETRY_REQUIRED':missing},
        'scientifically_complete':not missing and not nshort,'evidence_shortfalls':shortfalls,
        'task_shortfalls':[{'Task_ID':t,'planned':n,'terminal_executed':executed[t],'usable_valid':usable[t],
            'valid_row_shortfall':n-usable[t]} for t,n in sorted(planned.items()) if usable[t]!=n]}


def ingest(plan_path,output_directories,database):
    plan=read_csv(plan_path);by_id={r['plan_row_id']:r for r in plan}
    if len(by_id)!=len(plan) or any(not x.startswith('F1_') for x in by_id):raise ValueError('Foreign/duplicate plan identity')
    tasks={t['Task_ID']:t for t in contexts()};db=connection(database);failures=[]
    plan_hash=file_hash(plan_path)
    with db:
        for folder in output_directories:
            for p in Path(folder).glob('foundation_evidence/F1_*.json'):
                if p.stem not in by_id:continue
                try:
                    receipt=read_json(p);key=receipt['Plan_Row_ID']
                    if key!=p.stem:raise ValueError('Receipt filename/row identity mismatch')
                    o=observation(by_id[key],receipt,tasks[by_id[key]['Task_ID']])
                    store_execution(db,o,{'receipt':str(p),'receipt_sha256':file_hash(p),'plan_sha256':plan_hash,'new_foundation_only':True})
                except (ValueError,KeyError,OSError,TypeError) as e:
                    execution_state(db,p.stem,by_id[p.stem]['Task_ID'],'TECHNICAL_RETRY_REQUIRED',{'reason':str(e)})
                    failures.append({'receipt':str(p),'reason':str(e),'execution_status':'TECHNICAL_RETRY_REQUIRED','observation_status':'INVALID_OBSERVATION'})
    db.close()
    report=stage_summary(plan_path,database);report.update(failures=failures,database_content_sha256=verify_database(database),input_plan_sha256=plan_hash)
    return report


def export(database,partition='TRAIN_DEV',unlock=False):
    from foundation_v1.scope import normalize,require_unlock,selected_tasks,SCOPES,assert_database_scope,run_root
    partition=normalize(partition);require_unlock(partition,unlock)
    if partition=='ALL':raise ValueError('Export isolated partitions separately; no mixed train/dev/locked model-training file')
    assert_database_scope(database,partition)
    selected={t['Task_ID']:t for t in selected_tasks(partition)}
    # Independent TRAIN/DEVELOPMENT exports filter metadata before reading bodies.
    verify_database(database);db=sqlite3.connect(database)
    db.execute('CREATE TEMP TABLE selected_tasks(task_id TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO selected_tasks VALUES(?)',[(t,) for t in selected])
    clause=' WHERE task_id IN (SELECT task_id FROM selected_tasks)'
    expected=250*len(selected)
    output=run_root(partition)/'exports'/partition;output.mkdir(parents=True,exist_ok=True)
    count=db.execute('SELECT count(*) FROM observations'+clause).fetchone()[0]
    if count!=expected:raise ValueError(f'Final scoped export requires exactly {expected} valid rows, got {count}')
    tasks=selected
    for t in tasks:
        rows=read_task_observations(database,t)
        if len(rows)!=250 or sum(r['simulation_seed']==20 for r in rows)!=230:raise ValueError('Final task count mismatch')
        groups={r['Base_Geometry_ID'] for r in rows if r['simulation_seed']!=20}
        if len(groups)!=10 or any({r['simulation_seed'] for r in rows if r['Base_Geometry_ID']==g}!={20,21,22} for g in groups):raise ValueError('Repeat grouping mismatch')
    def rows():
        for (body,) in db.execute('SELECT body FROM observations'+clause+' ORDER BY task_id,seed,row_id'):yield json.loads(body)
    write_csv(output/'foundation_v1_master_observations.csv.gz',rows())
    def diagnostics():
        for row in rows():yield {k:row[k] for k in ('Plan_Row_ID','Task_ID','Base_Geometry_ID','input_status','observation_status','evidence_status','evidence_counts','applied','objective_M_RSR','objective_tracking','objective_QoS','Vmax','Vsum')}
    write_csv(output/'foundation_v1_master_diagnostics.csv.gz',diagnostics())
    write_csv(output/'foundation_v1_row_manifest.csv.gz',({k:r[k] for k in ('Foundation_Row_ID','Plan_Row_ID','Base_Geometry_ID','Task_ID','Workload_ID','Profile_IDs','split','hardware_class','simulation_seed','Candidate_Source','Adaptive_Wave')} for r in rows()))
    repeated={json.loads(body)['Base_Geometry_ID'] for (body,) in db.execute('SELECT body FROM observations'+clause+' AND seed!=20')}
    write_csv(output/'foundation_v1_repeat_group_manifest.csv.gz',({'Base_Geometry_ID':r['Base_Geometry_ID'],'Plan_Row_ID':r['Plan_Row_ID'],'simulation_seed':r['simulation_seed'],'Task_ID':r['Task_ID']} for r in rows() if r['Base_Geometry_ID'] in repeated))
    from collections import Counter,defaultdict
    splits=Counter();sampling=Counter();feasibility=defaultdict(lambda:[0,0])
    for row in rows():
        splits[row['split']]+=1;sampling[(row['split'],row['Candidate_Source'])]+=1
        f=feasibility[row['Task_ID']];f[0]+=1;f[1]+=int(row['evidence_qualified_feasible'])
    if dict(splits)!={s:250*sum(t['split']==s for t in selected.values()) for s in SCOPES[partition]}:raise ValueError('Final partition row counts differ')
    write_csv(output/'foundation_v1_split_summary.csv',[{'split':s,'rows':n} for s,n in sorted(splits.items())])
    write_csv(output/'foundation_v1_sampling_summary.csv',[{'split':s,'Candidate_Source':family,'rows':n} for (s,family),n in sorted(sampling.items())])
    write_csv(output/'foundation_v1_feasibility_summary.csv',[{'Task_ID':t,'rows':v[0],'evidence_qualified_feasible':v[1]} for t,v in sorted(feasibility.items())])
    source=read_json(ARTIFACTS/'preexecution_release.json')
    write_json(output/'foundation_v1_provenance.json',{'release':source,'database_content_sha256':verify_database(database),'observation_sources':'ONLY audited Foundation evidence sidecars with matching immutable planned row IDs','historical_V3_rows':0})
    write_json(output/'foundation_v1_dataset_manifest.json',{'version':'FOUNDATION_V1','partition':partition,'rows':expected,'workloads':len(selected)//4,'tasks':len(selected),'split_usage':{'TRAIN':'model fitting','DEVELOPMENT':'architecture/hyperparameter selection','LOCKED_SYNTHETIC_TEST':'frozen synthetic-profile evaluation only'},'corpus_shared_by':['Meta V4','FlexPFN'],'repeat_grouping':'Base_Geometry_ID: seed20 base plus seed21/22 repeats; never split across context/query silently','allowed_future_K':[0,1,2,4,8,16,32],'model_specific_tensors_created':False})
    db.close()


def main():
    from foundation_v1.scope import CHOICES,normalize,SCOPES,database_path,run_root,require_unlock,assert_database_scope
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--plan');p.add_argument('--outputs',nargs='*');p.add_argument('--database');p.add_argument('--export',action='store_true')
    p.add_argument('--partition',choices=CHOICES,default='TRAIN_DEV');p.add_argument('--unlock-locked-test',action='store_true')
    a=p.parse_args();scope=normalize(a.partition)
    if a.preflight:
        from foundation_v1.review import preflight
        preflight()
    elif a.export:
        require_unlock(scope,a.unlock_locked_test)
        if scope=='ALL':
            if a.database:p.error('ALL must use separate partition ledgers')
            for part in ('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST'):export(database_path(part),part,a.unlock_locked_test)
        else:export(a.database or database_path(scope),scope,a.unlock_locked_test)
    elif a.plan and a.outputs:
        require_unlock(scope,a.unlock_locked_test)
        if scope=='ALL':p.error('Ingest each isolated partition plan separately')
        if any(r['data_split'] not in SCOPES[scope] for r in __import__('am_flexdc_foundation_v1_common').iter_csv(a.plan)):p.error('Plan includes rows outside selected partition')
        database=a.database or database_path(scope);assert_database_scope(database,scope)
        report=ingest(a.plan,a.outputs,database);write_json(run_root(scope)/'last_ingest.json',report,False);print(canonical(report))
    else:p.error('Choose --preflight, --export or --plan plus --outputs')
