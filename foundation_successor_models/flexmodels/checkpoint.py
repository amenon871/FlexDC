"""Atomic checkpoints with full optimizer, RNG, sampler and schedule state."""
import copy,os,random
from pathlib import Path
import numpy as np
import torch
from .baseline import FoundationBaseline

SCHEMA='FOUNDATION_SUCCESSOR_CHECKPOINT_V1'


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def read_checkpoint(path,map_location='cpu'):
    value=torch.load(path,map_location=map_location,weights_only=False)
    if value.get('schema')!=SCHEMA:raise ValueError('Not a Foundation successor checkpoint (historical V3 weights are forbidden)')
    required=['model_kind','model','optimizer','scheduler','state','normalization','model_config','training_config','config_hash','cache_hash','seed','rng','sampler_state','episode_rng_state','baseline_prior_hash']
    if any(k not in value for k in required):raise ValueError('Incomplete Foundation training checkpoint')
    return value


def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    if path.name=='latest.pt' and path.exists():
        try:read_checkpoint(path);os.replace(path,path.with_name('latest_previous.pt'))
        except Exception:pass
    os.replace(temp,path)
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def resolve_resume(value,run_dir,logger=print):
    if not value:return None
    if value!='auto':return Path(value)
    existing=False
    for name in ['latest.pt','latest_previous.pt']:
        p=Path(run_dir)/name
        if p.exists():
            existing=True
            try:read_checkpoint(p);return p
            except Exception as e:logger('Resume candidate unusable: '+str(p)+' '+str(e))
    if existing:raise ValueError('No valid latest/previous checkpoint remains')
    logger('RESUME auto: no checkpoint exists; starting a fresh run');return None


def load_baseline(path,cache):
    ck=read_checkpoint(path)
    if ck['model_kind']!='baseline':raise ValueError('A selected Foundation Baseline prior is required')
    if ck['cache_hash']!=cache['cache_hash'] or ck['normalization']!=cache['normalization']:raise ValueError('Baseline prior/cache normalization mismatch')
    prior=FoundationBaseline(ck['model_config']);prior.load_state_dict(ck['model']);prior.requires_grad_(False).eval();return prior,ck


def make_model(kind,config,prior_path,cache):
    if kind=='baseline':
        if prior_path:raise ValueError('Baseline must start randomly, never from a historical checkpoint')
        return FoundationBaseline(config['model'])
    if not prior_path:raise ValueError('Few-shot models require --baseline-checkpoint')
    prior,_=load_baseline(prior_path,cache)
    if kind=='meta_v4':
        from .meta_v4 import MetaV4
        return MetaV4(prior,config['model'])
    if kind=='flexpfn':
        from .flexpfn import FlexPFN
        return FlexPFN(prior,config['model'])
    raise ValueError('Unknown model: '+kind)
