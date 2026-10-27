"""Canonical export -> one disposable tensor bundle; no database or simulator calls."""
import csv,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .features import job_features,global_features,fit_normalization,validate_domain,canonical_coordinates
from .objective import MEAN_FLOOR,P90_FLOOR,reconstruct,CONSTANTS,TRACKING_LIMIT,QOS_LIMIT,P90_BOUNDARY,QOS_BOUNDARY

MASTER_PHASE_A_SHA256='bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f'
PHASE_A_QUOTAS=dict(JOINT_SOBOL_INTERIOR=48,STRUCTURED_PR=30,PHYSICAL_PR_BOUNDARY=24,WEIGHT_SIMPLEX_BOUNDARY=24,CROSS_EXTREME=16,EQUAL_WEIGHT_REFERENCE=8)
ADAPTIVE_SOURCES={'ADAPTIVE_LOCAL','ADAPTIVE_BOUNDARY','BRIDGE','GLOBAL_ESCAPE'}
PARTITIONS=('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST')
REQUIRED=('Plan_Row_ID','Base_Geometry_ID','Task_ID','Workload_ID','split','hardware_class','Profile_IDs','J','N','U','x','simulation_seed','Candidate_Source','Adaptive_Wave','mean_tracking','p90','Pj','objective','evidence_counts','input_status','observation_status','evidence_status','raw_results')
TENSOR_KEYS=('jobs','globals','mask','tracking_logs','qos','mean_tracking','p90','objective','power_cost','r_actual_watts','price_coefficient','hour_seconds','geometry','evidence_counts')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def unpack(value):
    return json.loads(value) if isinstance(value,str) else value


def write_json(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(body,indent=2,allow_nan=False)+'\n',encoding='utf8')


def discover(input_path):
    p=Path(input_path)
    if p.is_file():return [p]
    # Prefer individual exports, never double-count TRAIN_DEV and its children.
    found=list(p.rglob('foundation_v1_master_observations.csv.gz'))
    paths=[x for x in found if x.parent.name in PARTITIONS]
    if Counter(x.parent.name for x in paths)!=Counter({s:1 for s in PARTITIONS}):
        raise ValueError('Expected exactly one canonical TRAIN, DEVELOPMENT and LOCKED export; provide their common root')
    return sorted(paths)


def read_rows(paths):
    for path in paths:
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',newline='',encoding='utf8') as f:
            reader=csv.DictReader(f)
            missing=set(REQUIRED)-set(reader.fieldnames or [])
            if missing:raise ValueError('Missing Foundation columns: '+str(sorted(missing)))
            yield from reader


def prepare(input_path,output,profiles_path,workloads_path,fixture=False):
    paths=discover(input_path);profiles_doc=json.loads(Path(profiles_path).read_text());workloads=json.loads(Path(workloads_path).read_text())
    profiles={x['profile_id']:x for x in profiles_doc['real_anchors']+profiles_doc['synthetic_profiles']};definitions={x['Workload_ID']:x for x in workloads}
    max_j=max(w['J'] for w in workloads)
    if max_j>8:raise ValueError('Foundation schema unexpectedly exceeds J=8; review rather than truncate')
    mixes={}
    for w in workloads:
        mix=[]
        for pid,qos in zip(w['Profile_IDs'],w['QoS_values']):
            s=profiles[pid];mix.append([s['min_job_power_watts'],s['max_job_power_watts'],s['min_time_seconds'],s['max_time_seconds'],qos,s['job_size']])
            if s['hardware_class']!=w['hardware_class']:raise ValueError('Cross-hardware workload')
            if pid.startswith('SYN_') and s['split']!=w['split']:raise ValueError('Synthetic-profile partition leakage')
        physical=np.asarray(mix,dtype=np.float64)
        if physical.shape!=(w['J'],6) or not np.isfinite(physical).all() or (physical<=0).any() or (physical[:,1]<physical[:,0]).any() or (physical[:,3]<physical[:,2]).any():raise ValueError('Invalid physical profile features: '+w['Workload_ID'])
        mixes[w['Workload_ID']]=physical
    tensors=defaultdict(list);meta=[];seen=set();groups=defaultdict(dict);counts=Counter();task_counts=Counter();group_geometry={}
    for row in read_rows(paths):
        missing=[k for k in REQUIRED if k not in row or row[k]=='']
        if missing:raise ValueError('Missing required field(s): '+str(missing))
        key=row['Plan_Row_ID'];wid=row['Workload_ID'];w=definitions.get(wid)
        if w is None:raise ValueError('Unknown Workload_ID: '+wid)
        if key in seen:raise ValueError('Duplicate Plan_Row_ID: '+key)
        seen.add(key);split=row['split'];hw=row['hardware_class'];j=int(row['J']);n=int(row['N']);u=float(row['U']);seed=int(row['simulation_seed'])
        task=f'{wid}__N{n}__U{round(u*100):03d}'
        if split not in PARTITIONS or split!=w['split'] or row['Task_ID']!=task or hw!=w['hardware_class'] or j!=w['J'] or unpack(row['Profile_IDs'])!=w['Profile_IDs']:
            raise ValueError('Foundation split/Task_ID/profile identity mismatch: '+key)
        if n not in (250,1000) or u not in (.6,.8) or seed not in (20,21,22):raise ValueError('Unknown context/realization')
        if row['input_status']!='LEGAL_INPUT' or row['observation_status']!='VALID_OBSERVATION' or row['evidence_status']!='SUFFICIENT_JOB_TYPE_EVIDENCE':raise ValueError('Incomplete or invalid scientific label: '+key)
        mix=mixes[wid];x=np.asarray(unpack(row['x']),dtype=np.float64);pmin,pmax,lo,hi=validate_domain(x,mix,n)
        pj=np.asarray(unpack(row['Pj']),dtype=np.float64);evidence=np.asarray(unpack(row['evidence_counts']),dtype=np.float64)
        if len(pj)!=j or len(evidence)!=j or not np.isfinite(pj).all() or np.any((pj<0)|(pj>1)) or not np.isfinite(evidence).all() or np.any(evidence<=0):raise ValueError('Wrong/missing Pj or job evidence: '+key)
        mean=float(row['mean_tracking']);p90=float(row['p90']);objective=float(row['objective']);raw=unpack(row['raw_results'])
        money=[float(raw[c]) for c in ['Simulator_Power_Cost','R_actual_watts','Mtrack_Price_Coefficient_piE','Mtrack_Hour_Seconds','Pbar_ratio','R_ratio']]
        if not np.isfinite([mean,p90,objective,*money]).all() or min(mean,p90)<0:raise ValueError('Required nonfinite/negative behavior label: '+key)
        if money[1]!=round(x[1]*n*1000) or money[3]!=3600:raise ValueError('Canonical applied reserve/duration mismatch: '+key)
        expected=money[0]+money[1]*money[2]*money[3]*mean+CONSTANTS['psi1']*np.logaddexp(0,CONSTANTS['psi2']*(p90-TRACKING_LIMIT))+CONSTANTS['beta']*np.logaddexp(0,CONSTANTS['rho']*(pj-QOS_LIMIT)).sum()
        if not np.isclose(expected,objective,rtol=0,atol=1e-6):raise ValueError('Canonical objective mismatch: '+key)
        group=(task,row['Base_Geometry_ID'])
        if seed in groups[group]:raise ValueError('Duplicate realization in Base_Geometry group')
        groups[group][seed]=key
        if group in group_geometry and not np.allclose(group_geometry[group],x,rtol=0,atol=1e-12):raise ValueError('Base_Geometry contains different candidate vectors')
        group_geometry[group]=x.copy();counts[split]+=1;task_counts[task]+=1
        tokens=np.zeros((max_j,13));tokens[:j]=job_features(mix,x[2:],u,n);mask=np.arange(max_j)<j;qos=np.zeros(max_j);qos[:j]=pj
        geom=np.zeros(max_j+1);geom[:j+1]=canonical_coordinates(x,mix,n)[0]
        padded_evidence=np.zeros(max_j);padded_evidence[:j]=evidence;tensors['evidence_counts'].append(padded_evidence)
        tensors['jobs'].append(tokens);tensors['globals'].append(global_features(*x[:2],n,u,j,hw,*money[4:]));tensors['mask'].append(mask);tensors['qos'].append(qos)
        tensors['tracking_logs'].append(np.log([mean+MEAN_FLOOR,p90+P90_FLOOR]));tensors['mean_tracking'].append(mean);tensors['p90'].append(p90);tensors['objective'].append(objective)
        for name,val in zip(['power_cost','r_actual_watts','price_coefficient','hour_seconds'],money[:4]):tensors[name].append(val)
        tensors['geometry'].append(geom)
        meta.append({k:row[k] for k in ['Plan_Row_ID','Base_Geometry_ID','Task_ID','Workload_ID','split','hardware_class','Candidate_Source'] }|{'J':j,'N':n,'U':u,'simulation_seed':seed,'Adaptive_Wave':int(row['Adaptive_Wave']),'Profile_IDs':w['Profile_IDs']})
    if not meta:raise ValueError('No Foundation observations')
    for group,seeds in groups.items():
        if 20 not in seeds or set(seeds) not in ({20},{20,21,22}):raise ValueError('Repeat groups need one seed20 base and paired21/22')
    if not fixture:
        if dict(counts)!={'TRAIN':400000,'DEVELOPMENT':50000,'LOCKED_SYNTHETIC_TEST':50000} or len(task_counts)!=2000 or set(task_counts.values())!={250} or len({m['Workload_ID'] for m in meta})!=500:raise ValueError('Final Foundation corpus is incomplete: '+str(dict(counts)))
        per_task=defaultdict(list)
        for m in meta:per_task[m['Task_ID']].append(m)
        for task,rows in per_task.items():
            c=Counter(m['simulation_seed'] for m in rows)
            waves=Counter(m['Adaptive_Wave'] for m in rows if m['simulation_seed']==20)
            if c!={20:230,21:10,22:10} or waves!={0:150,1:20,2:20,3:20,4:20}:raise ValueError('Task source/wave/realization quotas differ: '+task)
            validate_task_sources(rows,task)
        if Counter(w['split'] for w in workloads)!={'TRAIN':400,'DEVELOPMENT':50,'LOCKED_SYNTHETIC_TEST':50} or len(profiles_doc['synthetic_profiles'])!=400:raise ValueError('Wrong workload/profile manifest totals')
    source_manifest_hashes={}
    for path in paths:
        for name in ['foundation_v1_dataset_manifest.json','foundation_v1_provenance.json']:
            sidecar=path.parent/name
            if sidecar.exists():
                source_manifest_hashes[str(sidecar)]=sha256(sidecar)
                if name=='foundation_v1_provenance.json' and not fixture:validate_master_provenance(unpack(sidecar.read_text()))
            elif not fixture:raise ValueError('Canonical final export source manifest missing: '+str(sidecar))
    arrays={k:np.asarray(v) for k,v in tensors.items()};train=np.array([m['split']=='TRAIN' for m in meta]);norm=fit_normalization(arrays['jobs'],arrays['globals'],arrays['mask'],arrays['tracking_logs'],train)
    arrays['jobs']=((arrays['jobs']-np.array(norm['job_mean']))/np.array(norm['job_std']))*arrays['mask'][...,None]
    arrays['globals']=(arrays['globals']-np.array(norm['global_mean']))/np.array(norm['global_std'])
    bundle={k:torch.as_tensor(v,dtype=torch.bool if k=='mask' else torch.float64 if k in ('geometry','evidence_counts') else torch.float32) for k,v in arrays.items()}
    bundle.update(metadata=meta,normalization=norm,schema_version='FOUNDATION_CACHE_V2',fixture=fixture)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);torch.save(bundle,out/'cache.pt')
    manifest=dict(schema_version='FOUNDATION_CACHE_V2',fixture=fixture,warning='TEST FIXTURE ONLY; NOT SCIENTIFIC DATA' if fixture else None,rows=len(meta),partitions=dict(counts),tasks=len(task_counts),workloads=len({m['Workload_ID'] for m in meta}),max_j=max_j,canonical_master_phaseA_sha256='TEST_FIXTURE_ONLY' if fixture else MASTER_PHASE_A_SHA256,geometry_schema='FOUNDATION_DOMAIN_COORDINATES_V1',normalization=norm,input_hashes={str(f):sha256(f) for f in paths},source_manifest_hashes=source_manifest_hashes,profile_manifest_hash=sha256(profiles_path),workload_manifest_hash=sha256(workloads_path),cache_sha256=sha256(out/'cache.pt'))
    write_json(out/'manifest.json',manifest);return manifest


def load_cache(path):
    p=Path(path);manifest=json.loads((p/'manifest.json').read_text())
    if sha256(p/'cache.pt')!=manifest['cache_sha256']:raise ValueError('Cache hash mismatch')
    data=torch.load(p/'cache.pt',map_location='cpu',weights_only=False)
    if data['schema_version']!='FOUNDATION_CACHE_V2':raise ValueError('Wrong cache schema')
    data['cache_hash']=manifest['cache_sha256'];data['manifest']=manifest;return data


def partition_indices(cache,partition,unlock=False):
    if partition=='LOCKED_SYNTHETIC_TEST' and not unlock:raise ValueError('LOCKED evaluation requires --unlock-locked-test')
    if partition not in PARTITIONS:raise ValueError('Unknown partition')
    return torch.tensor([i for i,m in enumerate(cache['metadata']) if m['split']==partition],dtype=torch.long)


def batch(cache,indices,device='cpu'):
    return {('mask' if k=='mask' else k):cache[k][indices].to(device) for k in TENSOR_KEYS}


def sampler_weights(cache,indices):
    """Vectorized V3 base-group normalized 50/25/25 TRAIN-only mixture."""
    import pandas as pd
    indices=np.asarray(indices);meta=[cache['metadata'][int(i)] for i in indices]
    if any(m['split']!='TRAIN' for m in meta):raise ValueError('Sampler cannot access non-TRAIN labels')
    p=cache['p90'][indices].numpy();q=cache['qos'][indices].numpy();mask=cache['mask'][indices].numpy()
    rows=pd.DataFrame(dict(group=[m['Task_ID']+'|'+m['Base_Geometry_ID'] for m in meta],task=[m['Task_ID'] for m in meta],feasible=(p<=TRACKING_LIMIT)&((q<=QOS_LIMIT)|~mask).all(-1),tracking=(p>=P90_BOUNDARY[0])&(p<=P90_BOUNDARY[1]),qos=((q>=QOS_BOUNDARY[0])&(q<=QOS_BOUNDARY[1])&mask).any(-1)))
    g=rows.groupby('group',sort=False).agg(task=('task','first'),size=('task','size'),feasible=('feasible','mean'),tracking=('tracking','mean'),qos=('qos','mean'))
    natural=np.full(len(g),1/len(g));counts=g['task'].value_counts();context=np.asarray([1/(len(counts)*counts[t]) for t in g['task']]);priority=(1+3*g.feasible.to_numpy())*(1+g.tracking.to_numpy())*(1+2*g.qos.to_numpy());priority/=priority.sum()
    probabilities=pd.Series((.5*natural+.25*context+.25*priority)/g['size'].to_numpy(),index=g.index)
    weights=rows['group'].map(probabilities).to_numpy();return torch.as_tensor(weights/weights.sum(),dtype=torch.double)


def validate_task_sources(rows,task):
    primary=[r for r in rows if r['simulation_seed']==20]
    if len(primary)!=230 or len({r['Base_Geometry_ID'] for r in primary})!=230:raise ValueError('Task primary geometry quota/uniqueness differs: '+task)
    phase=[r for r in primary if r['Adaptive_Wave']==0]
    if Counter(r['Candidate_Source'] for r in phase)!=PHASE_A_QUOTAS:
        raise ValueError('Phase-A candidate-source quotas differ: '+task)
    for wave in range(1,5):
        selected=[r for r in primary if r['Adaptive_Wave']==wave]
        if len(selected)!=20 or len({r['Base_Geometry_ID'] for r in selected})!=20:
            raise ValueError('Adaptive unique geometry quota differs: '+task+' wave '+str(wave))
        if any(r['Candidate_Source'] not in ADAPTIVE_SOURCES for r in selected):
            raise ValueError('Invalid adaptive source/fallback family: '+task)
    if any(r['Candidate_Source']!='REPEAT' or r['Adaptive_Wave']!=5 for r in rows if r['simulation_seed']!=20):
        raise ValueError('Invalid repeat source/wave: '+task)


def validate_master_provenance(provenance):
    actual=provenance.get('release',{}).get('phaseA_sha256')
    if actual!=MASTER_PHASE_A_SHA256:
        raise ValueError('Canonical master Phase-A SHA mismatch: expected='+MASTER_PHASE_A_SHA256+' observed='+str(actual))
