import copy,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flexmodels.baseline import FoundationBaseline,ModelConfig
from flexmodels.meta_v4 import MetaV4
from flexmodels.flexpfn import FlexPFN
from flexmodels.data import batch
from flexmodels.episodes import episode,episode_batches,task_pools,encode
from flexmodels.objective import behavior_loss,reconstruct,CONSTANTS


def make_episode(cache,k=8):
    pool=list(task_pools(cache,'TRAIN').values())[0];r=episode(cache,pool,k,4,np.random.default_rng(7));return episode_batches(cache,[r],'cpu')


def model_for(kind,prior,config):return MetaV4(prior,config['meta_v4']['model']) if kind=='meta_v4' else FlexPFN(prior,config['flexpfn']['model'])


def test_baseline_padding_backward_and_mixed_j(cache,prior):
    ids=[next(i for i,m in enumerate(cache['metadata']) if m['J']==j) for j in [2,4,6]];b=batch(cache,torch.tensor(ids));jobs=b['jobs'].clone().requires_grad_();jobs.data[~b['mask']]=300
    o=prior(b['globals'],jobs,b['mask'],True);assert o['qos_probabilities'].shape==(3,6)
    (o['tracking_logs'].sum()+o['qos_probabilities'].sum()).backward();assert torch.count_nonzero(jobs.grad[~b['mask']])==0
    assert not o['qos_probabilities'][~b['mask']].any()
    for i in range(3):
        j=int(b['mask'][i].sum());alone=prior(b['globals'][i:i+1],jobs[i:i+1,:j],b['mask'][i:i+1,:j])
        torch.testing.assert_close(alone['tracking_logs'][0],o['tracking_logs'][i],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(alone['qos_probabilities'][0],o['qos_probabilities'][i,:j],atol=2e-6,rtol=2e-6)


def test_baseline_permutation(cache,prior):
    b=batch(cache,torch.tensor([250,350]));perm=torch.tensor([5,2,0,4,3,1]);a=prior(b['globals'],b['jobs'],b['mask']);z=prior(b['globals'],b['jobs'][:,perm],b['mask'][:,perm])
    torch.testing.assert_close(a['tracking_logs'],z['tracking_logs'],atol=2e-6,rtol=2e-6);torch.testing.assert_close(a['qos_probabilities'][:,perm],z['qos_probabilities'],atol=2e-6,rtol=2e-6)

@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
@pytest.mark.parametrize('k',[0,1,2,4,8,16,32,64])
def test_successor_variable_k_finite_and_zero_exact(cache,prior,tiny_configs,kind,k):
    c,q=make_episode(cache,k);m=model_for(kind,prior,tiny_configs).eval();o=m(c,q)
    if k==0:
        ref=encode(prior,q);assert torch.equal(o['tracking_logs'],ref['tracking_logs']);assert torch.equal(o['qos_probabilities'],ref['qos_probabilities']);assert o['variance'] is None
    else:
        assert torch.isfinite(o['tracking_logs']).all() and torch.isfinite(o['qos_probabilities']).all() and torch.isfinite(o['variance']).all();assert (o['variance']>0).all()
        (o['tracking_logs'].sum()+o['qos_probabilities'].sum()+o['variance'].sum()*.01).backward()
        assert all(p.grad is None for p in prior.parameters());assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
@pytest.mark.parametrize('j',[2,4,6])
def test_successor_context_job_permutation_and_query_independence(cache,prior,tiny_configs,kind,j):
    pool=next(v for v in task_pools(cache,'TRAIN').values() if cache['metadata'][v[0]]['J']==j);rec=episode(cache,pool,4,3,np.random.default_rng(5));c,q=episode_batches(cache,[rec],'cpu');m=model_for(kind,prior,tiny_configs).eval();a=m(c,q)
    tolerance=5e-5 if kind=='meta_v4' else 5e-6  # float32 Cholesky conditioning roundoff
    order=torch.tensor([2,0,3,1]);cp={k:v[:,order] for k,v in c.items()};b=m(cp,q)
    for key in ['tracking_logs','qos_probabilities','variance']:torch.testing.assert_close(a[key],b[key],atol=tolerance,rtol=tolerance)
    alone=m(c,{k:v[:,:1] for k,v in q.items()})
    for key in ['tracking_logs','qos_probabilities','variance']:torch.testing.assert_close(a[key][:,:1],alone[key],atol=tolerance,rtol=tolerance)
    perm=torch.arange(6).flip(0)
    def permute(b):
        z=dict(b)
        for key in ['jobs','mask','qos']:z[key]=b[key][:,:,perm]
        return z
    z=m(permute(c),permute(q));torch.testing.assert_close(a['tracking_logs'],z['tracking_logs'],atol=tolerance,rtol=tolerance);torch.testing.assert_close(a['qos_probabilities'][:,:,perm],z['qos_probabilities'],atol=tolerance,rtol=tolerance)

@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
def test_successor_padding_inert_and_zero_grad(cache,prior,tiny_configs,kind):
    c,q=make_episode(cache,4);m=model_for(kind,prior,tiny_configs).eval();a=m(c,q);d=dict(c);d['jobs']=c['jobs'].clone();d['jobs'][~c['mask']]=1000;b=m(d,q)
    torch.testing.assert_close(a['tracking_logs'],b['tracking_logs']);torch.testing.assert_close(a['qos_probabilities'],b['qos_probabilities'])

@pytest.mark.parametrize('function',['smooth','biased','boundary','noisy'])
def test_gp_synthetic_residuals_and_local_variance(prior,tiny_configs,function):
    torch.manual_seed(7);m=MetaV4(prior,tiny_configs['meta_v4']['model']);x=torch.linspace(-1,1,16).reshape(1,1,16,1).expand(-1,-1,-1,4)
    y=torch.sin(x[...,0]*2) if function=='smooth' else torch.ones(1,1,16)*2 if function=='biased' else (x[...,0]>0).float() if function=='boundary' else torch.sin(x[...,0])+torch.randn(1,1,16)*.03
    query=torch.cat([x,x[:,:,:1]+20],-2);mean,var,latent=m.posterior(x,query,y,torch.tensor([0]));assert torch.isfinite(mean).all() and (var>0).all();assert latent[0,0,0]<latent[0,0,-1]
    if function!='boundary':assert (mean[...,:16]-y).abs().mean()<.2
    (mean.sum()+var.sum()).backward();assert torch.isfinite(m.raw_length.grad).all()


def test_gp_duplicate_k64_and_jitter_fallback(cache,prior,tiny_configs,monkeypatch):
    m=MetaV4(prior,tiny_configs['meta_v4']['model']);x=torch.zeros(1,1,64,4);y=torch.ones(1,1,64);real=torch.linalg.cholesky_ex;calls=[]
    def fail_once(a):
        calls.append(1)
        if len(calls)==1:return torch.zeros_like(a),torch.ones(a.shape[:-2],dtype=torch.int32)
        return real(a)
    monkeypatch.setattr(torch.linalg,'cholesky_ex',fail_once);mu,var,_=m.posterior(x,x[:,:,:2],y,torch.tensor([0]));assert len(calls)==2 and m.last_jitter>m.config['jitter'];assert torch.isfinite(mu).all() and torch.isfinite(var).all()


def test_v3_exact_random_architecture_features_loss_objective(prior,cache,tiny_configs):
    root=Path(r'C:\Users\Achuthan Menon\Desktop\Research Work\ARGOS Hybrid\condor_set_transformer_sweep_v3_behavior_v3_best_feasibility_artifacts')
    # External reference is read-only and optional for reviewers outside this laptop;
    # packaged reference_only sources provide the portable equivalent.
    if not root.exists():root=Path(__file__).resolve().parents[2]/'reference_only/v3_original'
    assert root.exists(),'V3 reference source required for the parity audit; see reference_only in ZIP'
    def load(name):
        spec=importlib.util.spec_from_file_location(name,root/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
    old=load('data_center_model_flexdc_behavior_v3');util=load('am_flexdc_behavior_training_utilities_v3')
    cfg=dict(tiny_configs['baseline']['model']);cfg['dim_dc_features']=12;original=old.DataCenterBehaviorModel(old.FlexDCBehaviorModelConfig(**cfg));ported=FoundationBaseline(cfg);ported.load_state_dict(original.state_dict())
    b=batch(cache,torch.tensor([0,250,500]));a=original(b['globals'][:,:12],b['jobs'],b['mask']);z=ported(b['globals'][:,:12],b['jobs'],b['mask'],True)
    for k in ['tracking_logs','qos_probabilities','token_features','context_features']:torch.testing.assert_close(a[k],z[k],rtol=0,atol=0)
    from flexmodels.features import job_features,global_features
    mix=np.array([[180,650,20,50,4,1],[190,665,25,60,4,1]],float);w=np.array([.4,.6]);np.testing.assert_array_equal(job_features(mix,w,.6,250),util.build_token_features_raw(mix,w,utilization=.6,server_count=250))
    g=global_features(.4,.1,250,.6,2,'P100',.7,.2);oldg=util.build_global_features_raw(dict(Pbar_kw_per_server=.4,R_kw_per_server=.1,server_count=250,utilization=.6,workload_mix_size=2,Pbar_ratio=.7,R_ratio=.2));np.testing.assert_array_equal(g[:12],oldg)
    loss,_=behavior_loss(z,b,cache['normalization'],tiny_configs['baseline']['loss']);truth=dict(tracking_logs=b['tracking_logs'],raw_p90_tracking=b['p90'],mask=b['mask'],qos_probabilities=b['qos']);oldloss,_=util.behavior_loss(a,truth,SimpleNamespace(log_mean_tracking_std=cache['normalization']['target_std'][0],log_p90_tracking_std=cache['normalization']['target_std'][1]));torch.testing.assert_close(loss,oldloss,rtol=0,atol=0)
    new=reconstruct(z,b);previous=util.reconstruct_flexdc_outputs(tracking_logs=a['tracking_logs'],qos_probabilities=a['qos_probabilities'],mask=b['mask'],simulator_power_cost=b['power_cost'],r_actual_watts=b['r_actual_watts'],tracking_price_coefficient=b['price_coefficient'],hour_seconds=b['hour_seconds'],constants=util.FlexDCBehaviorConstants())
    torch.testing.assert_close(new['objective'],previous['objective'],rtol=0,atol=0)
