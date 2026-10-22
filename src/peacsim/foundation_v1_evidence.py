"""Foundation-only compact observational receipt; never changes simulator state."""
import csv
import hashlib
import json
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def runtime_provenance():
    root=Path(__file__).resolve().parents[2]
    release=json.loads((root/'foundation_v1_artifacts/preexecution_release.json').read_text(encoding='utf8'))
    return {'execution_git_SHA':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'planning_base_git_SHA':release['base_git_SHA'],'reviewed_source_identity':release['uncommitted_source_sha256']}


def write_evidence(output_dir, metadata, simulator, experiment, job, params,
                   raw_results, diagnostic_results):
    if metadata.get('design_version') != 'FOUNDATION_V1':
        raise ValueError('Foundation evidence hook requires Foundation plan identity')
    row_id=str(metadata['plan_row_id'])
    if not row_id.startswith('F1_') or not row_id.replace('_','').isalnum():raise ValueError('Invalid Foundation row ID')
    directory=Path(output_dir)
    records=[]
    for name,values in (('grid_search_results.csv',raw_results),('grid_search_diagnostics.csv',diagnostic_results)):
        with (directory/name).open(newline='') as f: columns=next(csv.reader(f))
        if len(columns)!=len(values):raise ValueError('Extraction schema width mismatch')
        records.append(dict(zip(columns,values)))
    counts=[]
    # Count precisely the population used by the existing one-hour delay estimator.
    # Pj is copied from that estimator's output, never recomputed here.
    table=simulator._job_table
    for i in range(job.job_type_count):
        typed=table[table['job_type_id']==i]
        finished=typed[(typed['end_time']!=-1)&(typed['arrival_time']>0)]
        unfinished=typed[(typed['end_time']==-1)&(typed['arrival_time']>0)&
            (typed['arrival_time']<=3600)&(typed['arrival_time']+job.all_min_execution_time[i]<3600)]
        counts.append({'job_type_id':i,'job_name':job.all_jobs[i],'generated':len(typed),
            'eligible_completed':len(finished),'eligible_unfinished':len(unfinished),
            'observations':len(finished)+len(unfinished)})
    policy=simulator._runtime_policy
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if hasattr(value,'item'):value=value.item()
        return None if isinstance(value,float) and not math.isfinite(value) else value
    body={'schema':'FOUNDATION_V1_EVIDENCE','Plan_Row_ID':row_id,'plan_metadata':clean(metadata),
        'results':records[0],'diagnostics':records[1],'evidence_counts':counts,
        'runtime_provenance':runtime_provenance(),
        'applied':{'P_watts':policy.P,'R_watts':policy.R,'weights':[policy.weights[i] for i in range(job.job_type_count)],
            'N':int(experiment.server_count),'U':float(experiment.utilization),'seed':int(experiment.random_seed),
            'J':int(job.job_type_count),'duration':int(experiment.simulation_duration),'idle_watts':float(experiment.idle_power),
            'workload_path':str(job.workload_mix_name),'policy_module':type(policy).__module__,
            'node_count_control':bool(policy._policy_parameters['node_count_control']),
            'iso_file_path':experiment.iso_file_path,'iso_signal_start_hour':experiment.iso_signal_start_hour,
            'iso_signal_granularity':experiment.iso_signal_granularity,'normalize_iso_signal':experiment.normalize_iso_signal,
            'randomize_iso_start':experiment.randomize_iso_start,
            'iso_sha256':hashlib.sha256(Path(experiment.iso_file_path).read_bytes()).hexdigest(),
            'workload_sha256':hashlib.sha256(Path(job.workload_mix_name).read_bytes()).hexdigest()},
        'power_serialization':'int(round(kW_per_server*1000*N));at most0.5W cluster rounding',
        'power_quantization_delta_watts':[float(policy.P)-float(params[0])*1000*experiment.server_count,
                                          float(policy.R)-float(params[1])*1000*experiment.server_count],
        'evidence_status':'SUFFICIENT_JOB_TYPE_EVIDENCE' if all(c['observations']>0 for c in counts) else 'INSUFFICIENT_JOB_TYPE_EVIDENCE'}
    # Normalize NumPy scalars and integer mapping keys using JSON-compatible values.
    def scalar(x):
        if hasattr(x,'item'):return x.item()
        raise TypeError(type(x).__name__)
    content=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False,default=scalar)
    body=json.loads(content);body['body_sha256']=hashlib.sha256(content.encode()).hexdigest()
    encoded=(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    folder=directory/'foundation_evidence';folder.mkdir(exist_ok=True)
    path=folder/(row_id+'.json')
    if path.exists():
        if path.read_bytes()!=encoded:raise ValueError('Existing Foundation row receipt differs')
        return
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
