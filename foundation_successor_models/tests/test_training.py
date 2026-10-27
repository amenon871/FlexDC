import copy,hashlib,json,random,signal
from pathlib import Path
import numpy as np
import pytest
import torch
from flexmodels import training
from flexmodels.checkpoint import read_checkpoint,restore_rng,resolve_resume,atomic_save,load_baseline
from flexmodels.data import partition_indices,sha256

@pytest.fixture(scope='module')
def trained_prior(cache_path,request,tmp_path_factory):
    # This fixture uses a separately scoped config to stay entirely synthetic.
    import yaml
    from conftest import PACKAGE
    cfg=yaml.safe_load((PACKAGE/'configs/baseline.yaml').read_text());cfg.update(cpu_threads=1,batch_size=512,max_epochs=1,warmup_epochs=1,log_every_steps=2,checkpoint_every_steps=2,eval_batch_size=1024);cfg['model'].update(st_dim_hidden=32,global_projection_dim=16,linear_dim_hidden=32,qos_projection_dim=16)
    out=tmp_path_factory.mktemp('unit_prior');training.run_training('baseline',cfg,cache_path,out,seed=9,precision='fp32',device_name='cpu');return out/'best_feasibility.pt'


def equal_training_state(left,right):
    for k in left['model']:assert torch.equal(left['model'][k],right['model'][k]),k
    assert left['scheduler']==right['scheduler']
    assert left['optimizer']['param_groups']==right['optimizer']['param_groups']
    for key,values in left['optimizer']['state'].items():
        for name,value in values.items():
            other=right['optimizer']['state'][key][name]
            if torch.is_tensor(value):assert torch.equal(value,other),(key,name)
            else:assert value==other
    restore_rng(left['rng']);expected=(random.random(),np.random.random(),torch.rand(3));restore_rng(right['rng']);actual=(random.random(),np.random.random(),torch.rand(3));assert expected[0:2]==actual[0:2];assert torch.equal(expected[2],actual[2])

@pytest.mark.parametrize('kind',['baseline','meta_v4','flexpfn'])
def test_resume_matches_uninterrupted_all_parameters_optimizer_lr_rng(cache_path,tiny_configs,trained_prior,tmp_path,kind):
    cfg=tiny_configs[kind];prior=None if kind=='baseline' else trained_prior
    training.run_training(kind,cfg,cache_path,tmp_path/'full',baseline_checkpoint=prior,precision='fp32',device_name='cpu')
    training.run_training(kind,cfg,cache_path,tmp_path/'split',baseline_checkpoint=prior,precision='fp32',device_name='cpu',stop_after_steps=2)
    partial=read_checkpoint(tmp_path/'split/latest.pt');assert partial['global_step']==2
    training.run_training(kind,cfg,cache_path,tmp_path/'split',baseline_checkpoint=prior,precision='fp32',device_name='cpu',resume='auto')
    equal_training_state(read_checkpoint(tmp_path/'full/final.pt'),read_checkpoint(tmp_path/'split/final.pt'))
    ck=read_checkpoint(tmp_path/'split/final.pt')
    for name in ['normalization','config_hash','cache_hash','rng','sampler_state','episode_rng_state','wandb_run_id','metadata','grad_scaler']:assert name in ck
    for role in ['latest','latest_previous','best_loss','best_objective','best_feasibility','final']+([] if kind=='baseline' else ['best_fewshot']):assert (tmp_path/'split'/(role+'.pt')).exists(),role


def test_shared_frozen_prior_and_missing_prior_rejection(cache,trained_prior,tiny_configs):
    from flexmodels.checkpoint import make_model
    a=make_model('meta_v4',tiny_configs['meta_v4'],trained_prior,cache);b=make_model('flexpfn',tiny_configs['flexpfn'],trained_prior,cache)
    for key in a.prior.state_dict():assert torch.equal(a.prior.state_dict()[key],b.prior.state_dict()[key])
    assert all(not p.requires_grad for p in a.prior.parameters())
    with pytest.raises(ValueError,match='require'):make_model('meta_v4',tiny_configs['meta_v4'],None,cache)
    with pytest.raises(ValueError,match='random'):make_model('baseline',tiny_configs['baseline'],trained_prior,cache)


def test_auto_previous_fallback_and_corruption(cache_path,tiny_configs,tmp_path):
    cfg=tiny_configs['baseline'];training.run_training('baseline',cfg,cache_path,tmp_path,stop_after_steps=3,device_name='cpu');previous=read_checkpoint(tmp_path/'latest_previous.pt')
    (tmp_path/'latest.pt').write_bytes(b'incomplete write');assert resolve_resume('auto',tmp_path).name=='latest_previous.pt'
    (tmp_path/'latest_previous.pt').write_bytes(b'incomplete too')
    with pytest.raises(ValueError,match='No valid'):resolve_resume('auto',tmp_path)
    assert not list(tmp_path.glob('*.tmp'))


def test_walltime_and_real_signal_safe_checkpoint(cache_path,tiny_configs,tmp_path,monkeypatch):
    cfg=tiny_configs['baseline'];r=training.run_training('baseline',cfg,cache_path,tmp_path/'wall',max_runtime_hours=0,exit_before_minutes=0,device_name='cpu');assert r['status']=='WALLTIME_CHECKPOINT' and r['global_step']==1
    actual=torch.optim.AdamW.step;calls=[]
    def interrupt_after_step(opt,*args,**kwargs):
        answer=actual(opt,*args,**kwargs)
        if not calls:calls.append(1);signal.raise_signal(signal.SIGTERM)
        return answer
    monkeypatch.setattr(torch.optim.AdamW,'step',interrupt_after_step)
    r=training.run_training('baseline',cfg,cache_path,tmp_path/'signal',device_name='cpu');assert r['status']=='SIGTERM' and r['global_step']==1 and (tmp_path/'signal/latest.pt').exists()


def test_trainers_never_request_locked_and_resume_rejects_changes(cache_path,tiny_configs,tmp_path,monkeypatch):
    original=training.partition_indices;seen=[]
    def guard(cache,part,*args,**kwargs):
        assert part!='LOCKED_SYNTHETIC_TEST';seen.append(part);return original(cache,part,*args,**kwargs)
    monkeypatch.setattr(training,'partition_indices',guard);cfg=tiny_configs['baseline'];training.run_training('baseline',cfg,cache_path,tmp_path,stop_after_steps=1,device_name='cpu');assert set(seen)=={'TRAIN','DEVELOPMENT'}
    altered=copy.deepcopy(cfg);altered['lr']=1e-4
    with pytest.raises(ValueError,match='identity'):training.run_training('baseline',altered,cache_path,tmp_path,resume='auto',device_name='cpu')


def test_lr_exact_schedule_and_historical_checkpoint_rejected(tmp_path):
    assert training.learning_rate(0,250,5,3e-4,1e-6)==3e-4/5
    assert training.learning_rate(249,250,5,3e-4,1e-6)==1e-6
    p=tmp_path/'old.pt';torch.save(dict(format_version=3,model_version='flexdc_behavior_v3'),p)
    with pytest.raises(ValueError,match='historical V3'):read_checkpoint(p)


def test_wandb_unavailable_keeps_local_paths(monkeypatch,tmp_path):
    import sys
    monkeypatch.setitem(sys.modules,'wandb',None);messages=[]
    assert training.wandb_start('online',tmp_path,{}, {},None,messages.append) is None and messages
