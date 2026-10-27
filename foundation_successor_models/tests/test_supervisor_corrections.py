"""Supervisor adversarial regressions; synthetic fixture only, no FlexDC."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flexmodels import training
from flexmodels.data import (prepare,load_cache,batch,validate_task_sources,
                            validate_master_provenance,MASTER_PHASE_A_SHA256,PHASE_A_QUOTAS)
from flexmodels.episodes import make_bank,episode,episode_batches,task_pools
from flexmodels.checkpoint import read_checkpoint
from flexmodels.features import canonical_coordinates


@pytest.fixture
def selected_prior(cache_path,tiny_configs,tmp_path):
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path/'prior',device_name='cpu',stop_after_steps=3)
    return tmp_path/'prior/best_feasibility.pt'


@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
def test_external_bank_mutation_ignored_and_local_mutation_rejected(cache_path,cache,tiny_configs,selected_prior,tmp_path,kind):
    bank=make_bank(cache,repeats=1,q=4);external=tmp_path/'external.json';external.write_text(json.dumps(bank))
    cfg=copy.deepcopy(tiny_configs[kind]);cfg['development_bank']=str(external)
    run=tmp_path/kind
    training.run_training(kind,cfg,cache_path,run,baseline_checkpoint=selected_prior,device_name='cpu',stop_after_steps=1)
    first=read_checkpoint(run/'latest.pt');local=run/'development_episode_bank.json';before=local.read_bytes()
    assert first['development_bank_sha256'] and first['development_bank_schema']=='FOUNDATION_EPISODES_V1'
    external.write_text('CORRUPTED EXTERNAL FILE; MUST NEVER BE RELOADED')
    training.run_training(kind,cfg,cache_path,run,baseline_checkpoint=selected_prior,device_name='cpu',resume='auto',stop_after_steps=1)
    assert local.read_bytes()==before
    local.write_bytes(before+b' ')
    with pytest.raises(ValueError,match='DEVELOPMENT bank SHA'):
        training.run_training(kind,cfg,cache_path,run,baseline_checkpoint=selected_prior,device_name='cpu',resume='auto')


def test_source_mutation_refuses_resume_and_audit_noise_is_excluded(cache_path,tiny_configs,tmp_path,monkeypatch):
    # Isolated source-tree clone: the real package is never mutated by this test.
    original=training.PACKAGE;source=tmp_path/'source';source.mkdir()
    for directory in ['flexmodels','configs']:
        shutil.copytree(original/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__'))
    for path in original.glob('*.py'):shutil.copyfile(path,source/path.name)
    monkeypatch.setattr(training,'PACKAGE',source)
    run=tmp_path/'run';cfg=tiny_configs['baseline']
    training.run_training('baseline',cfg,cache_path,run,stop_after_steps=1,device_name='cpu')
    before=training.package_source_identity()[0]
    for directory in ['dist','audit','.validation_runs','__pycache__']:
        (source/directory).mkdir(exist_ok=True);(source/directory/'noise.py').write_text('volatile audit output')
    assert training.package_source_identity()[0]==before
    target=source/'flexmodels/features.py';target.write_text(target.read_text()+'\n# mutation after checkpoint\n')
    with pytest.raises(ValueError,match='source SHA mismatch: checkpoint=.* current='):
        training.run_training('baseline',cfg,cache_path,run,resume='auto',device_name='cpu')


@pytest.mark.parametrize('kind',['baseline','meta_v4','flexpfn'])
def test_partial_exit_no_final_then_legitimate_completion(cache_path,tiny_configs,selected_prior,tmp_path,kind):
    run=tmp_path/kind;prior=None if kind=='baseline' else selected_prior
    args=dict(baseline_checkpoint=prior,device_name='cpu')
    partial=training.run_training(kind,tiny_configs[kind],cache_path,run,stop_after_steps=1,**args)
    assert partial['exit_reason']=='smoke_limit' and (run/'latest.pt').exists() and not (run/'final.pt').exists()
    assert read_checkpoint(run/'latest.pt')['exit_reason']=='smoke_limit'
    complete=training.run_training(kind,tiny_configs[kind],cache_path,run,resume='auto',**args)
    assert complete['exit_reason']=='completed_max_epochs' and (run/'final.pt').exists()
    assert read_checkpoint(run/'final.pt')['exit_reason']=='completed_max_epochs'


def test_walltime_guard_prevents_slow_validation(cache_path,tiny_configs,tmp_path,monkeypatch):
    called=[]
    def too_slow(*a,**kw):called.append(True);raise AssertionError('Must not begin expensive validation')
    monkeypatch.setattr(training,'collect_evaluation',too_slow)
    cfg=copy.deepcopy(tiny_configs['baseline']);cfg['first_validation_min_seconds']=10000
    result=training.run_training('baseline',cfg,cache_path,tmp_path,max_runtime_hours=.01,exit_before_minutes=0,device_name='cpu')
    assert not called and result['exit_reason']=='walltime_exit'
    assert (tmp_path/'latest.pt').exists() and not (tmp_path/'final.pt').exists()
    assert training.validation_budget_seconds({'checkpoint_margin_seconds':10},{'last_full_validation_seconds':100},0)==135


def test_routine_validation_compact_but_explicit_evaluation_detailed(cache_path,cache,tiny_configs,tmp_path):
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,device_name='cpu')
    for name in ['history.csv','development_metrics.json','development_k_curve.csv']:assert (tmp_path/name).exists()
    for name in ['prediction_summary.csv.gz','per_task_job_mae.csv','Task_ID_breakdown.csv']:assert not (tmp_path/name).exists()
    from flexmodels.metrics import collect_evaluation,save_evaluation
    from flexmodels.checkpoint import make_model
    model=make_model('baseline',tiny_configs['baseline'],None,cache)
    rows,frame=collect_evaluation(model,cache,indices=torch.tensor([720,721]))
    save_evaluation(tmp_path/'explicit',rows,frame)
    assert (tmp_path/'explicit/prediction_summary.csv.gz').exists()


def test_benchmark_is_partial_and_default_cadence():
    import yaml
    for name,steps in [('baseline',200),('meta_v4',500),('flexpfn',500)]:
        cfg=yaml.safe_load((training.PACKAGE/'configs'/(name+'.yaml')).read_text())
        assert cfg['checkpoint_every_steps']==steps


@pytest.mark.parametrize('j',[2,3,4,5,6])
def test_geometry_matches_foundation_domain_coordinates_and_distances(j):
    from am_flexdc_foundation_v1_common import Domain,weight_bounds
    lower,upper=weight_bounds(j,1000)[2:]
    rng=np.random.default_rng(71+j);mix=np.tile([200,700,20,50,4,1],(j,1))
    old=Domain(SimpleNamespace(job_type_count=j,effective_weight_lower=lower,effective_weight_upper=upper,r_over_p_max=.6,r_lower=.01,p_lower=.18,p_upper=.7,pr_upper=.84))
    vectors=[]
    for _ in range(100):
        p=rng.uniform(.18,.7);r=rng.uniform(.01,min(.6*p,.84-p))
        delta=rng.uniform(-.1/j,.1/j,j);w=np.full(j,1/j)+delta-delta.mean()
        vectors.append([p,r,*w])
    a=canonical_coordinates(vectors,mix,1000);b=old.coordinates(vectors)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(a-a[0],axis=1),np.linalg.norm(b-b[0],axis=1),rtol=0,atol=1e-14)
    assert a.shape==(100,j+1)


@pytest.fixture(scope='module')
def odd_j_cache(tmp_path_factory):
    from make_fixture import make_fixture
    base=tmp_path_factory.mktemp('j3j5');make_fixture(base/'fixture',job_counts=(3,5))
    prepare(base/'fixture/observations.csv.gz',base/'cache',base/'fixture/profiles.json',base/'fixture/workloads.json',True)
    return load_cache(base/'cache')


@pytest.mark.parametrize('j',[3,5])
def test_odd_j_baseline_forward_backward_padding_permutation(odd_j_cache,prior,j):
    index=next(i for i,m in enumerate(odd_j_cache['metadata']) if m['J']==j)
    b=batch(odd_j_cache,torch.tensor([index]));jobs=b['jobs'].clone().requires_grad_()
    a=prior(b['globals'],jobs,b['mask']);order=torch.arange(5).flip(0)
    z=prior(b['globals'],jobs[:,order],b['mask'][:,order])
    torch.testing.assert_close(a['tracking_logs'],z['tracking_logs'],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(a['qos_probabilities'][:,order],z['qos_probabilities'],atol=2e-6,rtol=2e-6)
    a['tracking_logs'].sum().backward()
    assert torch.isfinite(jobs.grad).all() and not jobs.grad[~b['mask']].any()
    alone=prior(b['globals'],jobs[:,:j],b['mask'][:,:j])
    torch.testing.assert_close(a['tracking_logs'],alone['tracking_logs'],atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
@pytest.mark.parametrize('j',[3,5])
def test_odd_j_successor_episode_prediction(odd_j_cache,prior,tiny_configs,kind,j):
    from flexmodels.meta_v4 import MetaV4
    from flexmodels.flexpfn import FlexPFN
    pool=next(v for v in task_pools(odd_j_cache,'TRAIN').values() if odd_j_cache['metadata'][v[0]]['J']==j)
    record=episode(odd_j_cache,pool,4,3,np.random.default_rng(33))
    c,q=episode_batches(odd_j_cache,[record],'cpu')
    model=(MetaV4 if kind=='meta_v4' else FlexPFN)(prior,tiny_configs[kind]['model']).eval()
    out=model(c,q)
    assert torch.isfinite(out['tracking_logs']).all() and torch.isfinite(out['variance']).all()
    assert out['qos_probabilities'].shape[-1]==5 and not out['qos_probabilities'][~q['mask']].any()


def test_cache_evidence_counts_preserved_with_zero_padding(cache):
    assert cache['evidence_counts'].shape==cache['qos'].shape==cache['mask'].shape
    assert torch.all(cache['evidence_counts'][cache['mask']]==20)
    assert not cache['evidence_counts'][~cache['mask']].any()
    assert cache['manifest']['canonical_master_phaseA_sha256']=='TEST_FIXTURE_ONLY'


def source_rows():
    rows=[]
    for source,count in PHASE_A_QUOTAS.items():
        rows.extend(dict(Candidate_Source=source,Adaptive_Wave=0,simulation_seed=20,Base_Geometry_ID=f'{source}_{i}') for i in range(count))
    for wave in range(1,5):
        # All-GLOBAL_ESCAPE is a permitted fallback; do not enforce nominal source quotas.
        rows.extend(dict(Candidate_Source='GLOBAL_ESCAPE',Adaptive_Wave=wave,simulation_seed=20,Base_Geometry_ID=f'W{wave}_{i}') for i in range(20))
    rows.extend(dict(Candidate_Source='REPEAT',Adaptive_Wave=5,simulation_seed=seed,Base_Geometry_ID=rows[i]['Base_Geometry_ID']) for seed in (21,22) for i in range(10))
    return rows


def test_full_task_source_quotas_and_allowed_fallbacks():validate_task_sources(source_rows(),'TEST_ONLY')


@pytest.mark.parametrize('change',['phase_quota','wave_duplicate','bad_family','bad_repeat','cross_wave_duplicate'])
def test_full_task_source_gate_rejects_corruption(change):
    rows=source_rows()
    if change=='phase_quota':rows[0]['Candidate_Source']='STRUCTURED_PR'
    elif change=='wave_duplicate':rows[151]['Base_Geometry_ID']=rows[150]['Base_Geometry_ID']
    elif change=='bad_family':rows[150]['Candidate_Source']='UNREVIEWED'
    elif change=='bad_repeat':rows[-1]['Candidate_Source']='GLOBAL_ESCAPE'
    else:rows[170]['Base_Geometry_ID']=rows[150]['Base_Geometry_ID']
    with pytest.raises(ValueError):validate_task_sources(rows,'TEST_ONLY')


def test_master_provenance_is_mandatory_and_exact():
    validate_master_provenance({'release':{'phaseA_sha256':MASTER_PHASE_A_SHA256}})
    for p in [{},{'release':{'phaseA_sha256':'TEST_FIXTURE_ONLY'}},{'release':{'phaseA_sha256':'wrong'}}]:
        with pytest.raises(ValueError,match='master Phase-A SHA mismatch'):validate_master_provenance(p)


def test_benchmark_limit_never_marks_final(cache_path,tiny_configs,tmp_path):
    result=training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,device_name='cpu',stop_after_steps=2,benchmark=True)
    assert result['exit_reason']=='benchmark_limit'
    assert (tmp_path/'latest.pt').exists() and not (tmp_path/'final.pt').exists()
