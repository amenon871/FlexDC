"""Row-local durable completion. No scientific identity/objective reconstruction."""
import csv
import json
import math
import os
import time
from pathlib import Path
from am_flexdc_foundation_v1_common import digest, read_json, write_json

SCHEMA = 'FOUNDATION_V1_COMPLETION_V2'


def receipt_path(area, key):
    if not key.startswith('F1_') or not key.replace('_', '').isalnum():
        raise ValueError('Invalid Plan_Row_ID: '+str(key))
    return Path(area)/'receipts'/key[3:5]/(key+'.json')


def load(area, key):
    body=read_json(receipt_path(area,key))
    if body.get('schema')!=SCHEMA or body.get('Plan_Row_ID')!=key:
        raise ValueError('Completion schema/ID mismatch: '+key)
    required=('Base_Geometry_ID','Task_ID','Workload_ID','context','simulation_seed','requested','applied','result_file','diagnostic_file','evidence_file','timestamp','evidence')
    if body.get('execution_status')!='EXECUTION_COMPLETE' or any(k not in body for k in required):
        raise ValueError('Incomplete completion record: '+key)
    if body.get('checksum')!=digest({k:v for k,v in body.items() if k!='checksum'}):
        raise ValueError('Completion checksum mismatch: '+key)
    return body


def completed(area, wanted=None):
    result={}
    for bucket in (Path(area)/'receipts').glob('*'):
        if not bucket.is_dir(): continue
        for path in bucket.glob('F1_*.json'):
            if wanted is not None and path.stem not in wanted: continue
            try:body=load(area,path.stem);result[path.stem]={'timestamp':body['timestamp']}
            except (ValueError,KeyError,OSError,TypeError):pass
    return result


def publish(output, metadata):
    area=os.environ.get('FOUNDATION_V1_SIMPLE_AREA')
    if not area: return
    key=str(metadata['plan_row_id']); output=Path(output).resolve()
    evidence=read_json(output/'foundation_evidence'/(key+'.json'))
    if evidence.get('body_sha256')!=digest({k:v for k,v in evidence.items() if k!='body_sha256'}):
        raise ValueError('Evidence checksum mismatch: '+key)
    applied=evidence['applied']
    # Minimal execution check against the exact request, never make_row().
    for field,value in (('N',int(metadata['server_count'])),('U',float(metadata['utilization'])),
                        ('J',int(metadata['job_type_count'])),('seed',int(metadata['simulation_seed']))):
        if applied[field]!=value:raise ValueError('Applied request mismatch: '+key+' '+field)
    if Path(applied['workload_path']).name!=Path(metadata['workload_config']).name:
        raise ValueError('Applied workload mismatch: '+key)
    weights=metadata['weights']
    if isinstance(weights,str):weights=json.loads(weights)
    if len(applied['weights'])!=len(weights) or any(not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-9) for a,b in zip(applied['weights'],weights)):
        raise ValueError('Applied weights mismatch beyond existing 1e-9 serialization tolerance: '+key)
    for field,col in (('P_watts','Pbar_kw_per_server'),('R_watts','R_kw_per_server')):
        if applied[field]!=round(float(metadata[col])*1000*applied['N']):
            raise ValueError('Applied power mismatch: '+key+' '+field)
    for filename,field in (('grid_search_results.csv','results'),('grid_search_diagnostics.csv','diagnostics')):
        if evidence[field].get('Plan_Row_ID')!=key:raise ValueError('Persisted record ID mismatch: '+key)
        with (output/filename).open('rb') as stream:
            if not stream.read(1):raise ValueError('Empty compact output: '+filename)
    from peacsim.foundation_v1_cleanup import _sync_directory
    _sync_directory(output/'foundation_evidence');_sync_directory(output)
    body={'schema':SCHEMA,'Plan_Row_ID':key,'Base_Geometry_ID':metadata['Base_Geometry_ID'],
          'Task_ID':metadata['Task_ID'],'Workload_ID':metadata['Workload_ID'],
          'context':metadata['context_id'],'simulation_seed':int(metadata['simulation_seed']),
          'requested':{'P':float(metadata['Pbar_kw_per_server']),'R':float(metadata['R_kw_per_server']),'weights':weights},
          'applied':applied,'weight_serialization_tolerance':1e-9,'weight_serialization_deltas':[float(a)-float(b) for a,b in zip(applied['weights'],weights)],'result_file':str(output/'grid_search_results.csv'),
          'diagnostic_file':str(output/'grid_search_diagnostics.csv'),
          'evidence_file':str(output/'foundation_evidence'/(key+'.json')),
          'execution_status':'EXECUTION_COMPLETE','timestamp':time.time(),'evidence':evidence}
    body['checksum']=digest(body)
    worker=os.environ['FOUNDATION_V1_SIMPLE_WORKER']; events=Path(area)/'events';events.mkdir(exist_ok=True,parents=True)
    # Pending pointer closes receipt->journal crash window for live status.
    pending=events/(worker+'.pending.json')
    write_json(pending,{'Plan_Row_ID':key,'timestamp':body['timestamp']},False)
    write_json(receipt_path(area,key),body,False)
    with (events/(worker+'.jsonl')).open('a',encoding='utf8') as stream:
        stream.write(json.dumps({'id':key,'time':body['timestamp']})+'\n');stream.flush();os.fsync(stream.fileno())
    # Pointer may remain: status deduplicates the already journaled ID.


def archive(area,key,reason):
    path=receipt_path(area,key)
    if path.exists():
        target=Path(area)/'quarantined_receipts'/(str(time.time_ns())+'_'+key+'.json')
        target.parent.mkdir(exist_ok=True,parents=True);os.replace(path,target)
    write_json(Path(area)/'audit_failures'/(key+'.json'),reason,False)
