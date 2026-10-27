import copy,importlib.util,sys
from pathlib import Path
import pytest
import torch
import yaml

PACKAGE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PACKAGE/'fixtures'))
# Audit-only references in an unpacked supervisor ZIP; never production imports.
for reference in [PACKAGE.parent/'reference_only/foundation_contract',PACKAGE.parent/'reference_only/base_collector']:
    if reference.is_dir():sys.path.insert(0,str(reference))
from make_fixture import make_fixture
from flexmodels.data import prepare,load_cache

@pytest.fixture(scope='session')
def fixture_dir():
    path=PACKAGE/'fixtures/tiny';make_fixture(path);return path

@pytest.fixture(scope='session')
def cache_path(fixture_dir):
    torch.set_num_threads(1)
    path=PACKAGE/'.validation_runs/test_cache';prepare(fixture_dir/'observations.csv.gz',path,fixture_dir/'profiles.json',fixture_dir/'workloads.json',True);return path

@pytest.fixture(scope='session')
def cache(cache_path):return load_cache(cache_path)

@pytest.fixture
def tiny_configs():
    out={}
    for kind in ['baseline','meta_v4','flexpfn']:
        c=yaml.safe_load((PACKAGE/'configs'/(kind+'.yaml')).read_text());c.update(cpu_threads=1,log_every_steps=2,checkpoint_every_steps=2,eval_batch_size=512)
        if kind=='baseline':c.update(batch_size=256,max_epochs=2,warmup_epochs=1);c['model'].update(st_dim_hidden=32,global_projection_dim=16,linear_dim_hidden=32,qos_projection_dim=16)
        else:
            c.update(max_steps=4,steps_per_virtual_epoch=2,warmup_steps=2,microbatch_episodes=2,gradient_accumulation=1,query_count=4,eval_every_steps=2,eval_episode_batch_size=2,dev_episodes_per_task=1)
            if kind=='meta_v4':c['model']['latent_dim']=4
            else:c['model'].update(d_model=32,heads=4,layers=1)
        out[kind]=c
    return out

@pytest.fixture
def prior(tiny_configs):
    from flexmodels.baseline import FoundationBaseline
    torch.manual_seed(19);return FoundationBaseline(tiny_configs['baseline']['model']).eval()
