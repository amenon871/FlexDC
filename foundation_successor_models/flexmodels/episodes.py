"""Shared, label-free context selection and geometry-disjoint query episodes."""
import hashlib,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .data import batch,write_json

K_VALUES=(0,1,2,4,8,16,32,64)


def task_pools(cache,partition):
    pools=defaultdict(list)
    for i,m in enumerate(cache['metadata']):
        if m['split']==partition and m['simulation_seed']==20:pools[m['Task_ID']].append(i)
    return dict(pools)


def diverse_order(cache,indices,rng):
    x=cache['geometry'][indices].numpy();n=len(x);first=int(rng.integers(n));order=[];dist=np.full(n,np.inf)
    for _ in range(n):
        order.append(first);dist=np.minimum(dist,np.square(x-x[first]).sum(-1));dist[order]=-1;first=int(np.argmax(dist))
    return np.asarray(indices)[order].tolist()


def episode(cache,pool,k,q,rng,mode='random',task=None):
    if k not in K_VALUES:raise ValueError('Unsupported K')
    if len(pool)<k+q:raise ValueError('Not enough distinct seed20 geometries for K+Q')
    perm=rng.permutation(pool);queries=perm[:q].tolist();remaining=perm[q:].tolist()
    order=diverse_order(cache,remaining,rng) if mode in ('diverse','nested') else remaining
    contexts=order[:k]
    if mode!='nested':rng.shuffle(contexts)
    out=dict(Task_ID=task or cache['metadata'][pool[0]]['Task_ID'],K=k,context=contexts,query=queries,selection=mode)
    validate_episode(cache,out);return out


def validate_episode(cache,record,partition=None):
    c=record['context'];q=record['query'];k=record['K']
    if k not in K_VALUES or len(c)!=k or len(c)!=len(set(c)) or len(q)!=len(set(q)):raise ValueError('Invalid episode cardinality')
    selected=[cache['metadata'][i] for i in c+q]
    if any(m['Task_ID']!=record['Task_ID'] or m['simulation_seed']!=20 or (partition and m['split']!=partition) for m in selected):raise ValueError('Episode task/partition/primary-realization leakage')
    cg={cache['metadata'][i]['Base_Geometry_ID'] for i in c};qg={cache['metadata'][i]['Base_Geometry_ID'] for i in q}
    if cg&qg or len(cg)!=len(c) or len(qg)!=len(q):raise ValueError('Context/query Base_Geometry leakage')


def make_bank(cache,partition='DEVELOPMENT',seed=73,repeats=10,q=32,mode='random',ks=K_VALUES):
    records=[]
    for task,pool in sorted(task_pools(cache,partition).items()):
        for rep in range(repeats):
            # Same query set and input ordering across K, and across all methods.
            key=int.from_bytes(hashlib.sha256(f'{seed}:{task}:{rep}:{mode}'.encode()).digest()[:8],'little')
            for k in ks:
                rec=episode(cache,pool,k,q,np.random.default_rng(key),mode,task);rec['replicate']=rep;records.append(rec)
    return dict(schema='FOUNDATION_EPISODES_V1',cache_hash=cache['cache_hash'],partition=partition,seed=seed,query_count=q,repeats=repeats,selection=mode,episodes=records)


def load_bank(path,cache,partition):
    b=json.loads(Path(path).read_text())
    if b.get('schema')!='FOUNDATION_EPISODES_V1':raise ValueError('Episode bank schema mismatch')
    if not b.get('episodes'):raise ValueError('Empty DEVELOPMENT episode bank')
    if b['cache_hash']!=cache['cache_hash'] or b['partition']!=partition:raise ValueError('Episode bank cache/partition mismatch')
    for rec in b['episodes']:validate_episode(cache,rec,partition)
    return b


def training_records(cache,rng,k,q,count,pools=None,diverse_fraction=.5):
    pools=pools or task_pools(cache,'TRAIN');names=list(pools)
    return [episode(cache,pools[t],k,q,rng,'diverse' if rng.random()<diverse_fraction else 'random',t) for t in rng.choice(names,count)]


def episode_batches(cache,records,device):
    if len({r['K'] for r in records})!=1:raise ValueError('Group microbatches by K')
    return batch(cache,torch.tensor([r['context'] for r in records],dtype=torch.long),device),batch(cache,torch.tensor([r['query'] for r in records],dtype=torch.long),device)


def encode(prior,inputs):
    shape=inputs['mask'].shape[:-1];j=inputs['mask'].shape[-1]
    with torch.no_grad():
        o=prior(inputs['globals'].reshape(-1,inputs['globals'].shape[-1]),inputs['jobs'].reshape(-1,j,inputs['jobs'].shape[-1]),inputs['mask'].reshape(-1,j),return_latents=True)
    return {k:v.reshape(*shape,*v.shape[1:]) for k,v in o.items()}


def repeat_bank(cache,partition='DEVELOPMENT'):
    """Dedicated SAME-GEOMETRY seed20 -> seed21/22 diagnostic, never training."""
    groups=defaultdict(dict)
    for i,m in enumerate(cache['metadata']):
        if m['split']==partition:groups[(m['Task_ID'],m['Base_Geometry_ID'])][m['simulation_seed']]=i
    records=[dict(Task_ID=t,K=1,context=[g[20]],query=[g[21],g[22]],selection='REPEAT_SCENARIO_DIAGNOSTIC') for (t,_),g in groups.items() if set(g)=={20,21,22}]
    return dict(schema='FOUNDATION_REPEAT_DIAGNOSTIC_V1',cache_hash=cache['cache_hash'],partition=partition,episodes=records)


def repeat_noise_report(cache,partition,epsilon,output):
    from .objective import targets
    from .data import batch
    import pandas as pd
    records=[]
    for r in repeat_bank(cache,partition)['episodes']:
        ix=r['context']+r['query'];y=targets(batch(cache,torch.tensor(ix)),epsilon);m=cache['metadata'][ix[0]];j=m['J'];variance=y[:,:j+2].var(0,unbiased=True)
        records.append(dict(Task_ID=m['Task_ID'],Base_Geometry_ID=m['Base_Geometry_ID'],hardware_class=m['hardware_class'],J=j,seeds='20,21,22',mean_log_variance=float(variance[0]),p90_log_variance=float(variance[1]),qos_logit_variance_mean=float(variance[2:].mean())))
    frame=pd.DataFrame(records);out=Path(output);out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'repeat_realization_noise.csv',index=False)
    write_json(out/'repeat_noise_summary.json',dict(diagnostic_only=True,groups=len(records),partition=partition,qos_logit_epsilon=epsilon,noise_floor_not_automatically_fitted=True,caveat='Three seed realizations per geometry are descriptive evidence, not certified calibration. This does not alter GP noise or training.',mean_variances={k:float(frame[k].mean()) for k in ['mean_log_variance','p90_log_variance','qos_logit_variance_mean']} if len(frame) else {}))
    return frame


def freeze_bank(path,cache,config,resuming=False):
    """Run-local immutable validation identity; never re-read external bank on resume."""
    from .data import sha256
    path=Path(path)
    if resuming and not path.is_file():raise ValueError('Resume DEVELOPMENT bank missing: '+str(path))
    if not path.exists():
        bank=load_bank(config['development_bank'],cache,'DEVELOPMENT') if config.get('development_bank') else make_bank(cache,repeats=config['dev_episodes_per_task'],q=config['query_count'],mode=config['dev_selection'],seed=config['dev_seed'])
        temp=path.with_name(path.name+'.tmp')
        with temp.open('w',encoding='utf8',newline='\n') as f:
            json.dump(bank,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        if os.name!='nt':
            fd=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
    bank=load_bank(path,cache,'DEVELOPMENT')
    return bank,dict(development_bank_path=str(path.resolve()),development_bank_sha256=sha256(path),development_bank_schema=bank['schema'])
