import copy,importlib.util,sys
from pathlib import Path
import pytest,torch
from flexmodels.data import load_cache
from flexmodels.episodes import repeat_bank,repeat_noise_report,validate_episode,training_records,task_pools
from flexmodels import training
from flexmodels.checkpoint import read_checkpoint,atomic_save


def test_repeats_kept_and_diagnostics_not_ordinary_episodes(cache,tmp_path):
    b=repeat_bank(cache);assert len(b['episodes'])==60
    with pytest.raises(ValueError,match='realization'):validate_episode(cache,b['episodes'][0],'DEVELOPMENT')
    frame=repeat_noise_report(cache,'DEVELOPMENT',1e-4,tmp_path);assert len(frame)==60
    assert frame.p90_log_variance.ge(0).all() and (tmp_path/'repeat_noise_summary.json').exists()


def test_cannot_compare_different_frozen_prior(cache_path,tiny_configs,tmp_path):
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path/'base',stop_after_steps=3,device_name='cpu')
    prior=tmp_path/'base/best_feasibility.pt'
    training.run_training('meta_v4',tiny_configs['meta_v4'],cache_path,tmp_path/'meta',baseline_checkpoint=prior,stop_after_steps=1,device_name='cpu')
    changed=read_checkpoint(prior);changed['checkpoint_role']='different_identity';other=tmp_path/'other.pt';atomic_save(other,changed)
    import evaluate
    with pytest.raises(ValueError,match='same frozen'):evaluate.load_model('meta_v4',tmp_path/'meta/latest.pt',load_cache(cache_path),other)


def test_actual_resume_recovers_from_corrupt_latest(cache_path,tiny_configs,tmp_path):
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,stop_after_steps=3,device_name='cpu');(tmp_path/'latest.pt').write_bytes(b'broken')
    result=training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,resume='auto',device_name='cpu');assert result['status']=='COMPLETE';assert read_checkpoint(tmp_path/'latest.pt')['global_step']==6

@pytest.mark.parametrize('kind',['meta_v4','flexpfn'])
def test_all_fewshot_trainers_exclude_locked(cache_path,tiny_configs,tmp_path,monkeypatch,kind):
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path/'base',stop_after_steps=3,device_name='cpu');prior=tmp_path/'base/best_feasibility.pt'
    original=training.partition_indices;seen=[]
    def safe(cache,part,*args,**kwargs):
        assert part!='LOCKED_SYNTHETIC_TEST';seen.append(part);return original(cache,part,*args,**kwargs)
    monkeypatch.setattr(training,'partition_indices',safe);training.run_training(kind,tiny_configs[kind],cache_path,tmp_path/kind,baseline_checkpoint=prior,stop_after_steps=1,device_name='cpu');assert set(seen)=={'TRAIN','DEVELOPMENT'}



def test_metrics_confusion_rates_and_uncertainty_fields(cache,prior,tiny_configs,tmp_path):
    import numpy as np,pandas as pd
    from flexmodels.metrics import summarize,collect_evaluation,save_evaluation
    from flexmodels.episodes import make_bank
    from flexmodels.meta_v4 import MetaV4
    frame=pd.DataFrame(dict(true_p90=[.3,.4,.5],pred_p90=[.3,.2,.5],true_max_pj=[.1,.01,.01],pred_max_pj=[.1,.01,.01],true_mean=[.1,.2,.3],pred_mean=[.1,.2,.3],true_objective=[10,20,30],pred_objective=[10,20,30]))
    x=summarize(frame);assert x['feasible_precision']==.5 and x['feasible_recall']==1 and x['feasible_f1']==2/3 and x['false_feasible_rate']==.5 and x['objective_r2']==1
    bank=make_bank(cache,repeats=1,q=2,ks=[1]);m=MetaV4(prior,tiny_configs['meta_v4']['model']);rows,frame=collect_evaluation(m,cache,records=bank['episodes'],episode_batch_size=2)
    assert len(rows)==1
    for c in ['nll','coverage_50','coverage_90','coverage_95','interval_width_50','interval_width_90','interval_width_95']:assert np.isfinite(rows[0][c])
    assert rows[0]['coverage_50']<=rows[0]['coverage_90']<=rows[0]['coverage_95']
    save_evaluation(tmp_path,rows,frame,'LOCKED_SYNTHETIC_TEST');assert (tmp_path/'locked_metrics.json').exists() and not (tmp_path/'development_metrics.json').exists()


def test_same_bank_is_consumed_and_locked_cli_refuses(cache_path,tiny_configs,tmp_path):
    import evaluate
    from flexmodels.data import load_cache,write_json
    from flexmodels.episodes import make_bank
    cache=load_cache(cache_path);bank=make_bank(cache,repeats=1,q=4);path=tmp_path/'shared.json';write_json(path,bank)
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path/'base',stop_after_steps=3,device_name='cpu');prior=tmp_path/'base/best_feasibility.pt'
    for kind in ['meta_v4','flexpfn']:
        cfg=copy.deepcopy(tiny_configs[kind]);cfg['development_bank']=str(path);training.run_training(kind,cfg,cache_path,tmp_path/kind,baseline_checkpoint=prior,stop_after_steps=1,device_name='cpu')
        import json
        assert json.loads((tmp_path/kind/'development_episode_bank.json').read_text())==bank
    with pytest.raises(ValueError,match='unlock'):evaluate.main(['--cache',str(cache_path),'--partition','locked','--output',str(tmp_path/'locked')])



def test_early_stopped_resume_is_zero_update(cache_path,tiny_configs,tmp_path):
    cfg=copy.deepcopy(tiny_configs['baseline']);cfg.update(max_epochs=3,batch_size=512,lr=0.,min_lr=0.);cfg['early_stopping']['patience']=1
    first=training.run_training('baseline',cfg,cache_path,tmp_path,device_name='cpu');assert first['status']=='EARLY_STOPPED' and first['global_step']==4
    second=training.run_training('baseline',cfg,cache_path,tmp_path,resume='auto',device_name='cpu');assert second['optimizer_steps_this_invocation']==0 and second['global_step']==4


def test_authoritative_cost_contract():
    import json
    from flexmodels.objective import CONSTANTS
    root=Path(__file__).resolve().parents[2];p=root/'foundation_v1_artifacts/canonical_contract.json'
    if not p.exists():p=root/'reference_only/foundation_contract/foundation_v1_artifacts/canonical_contract.json'
    expected=json.loads(p.read_text())['cost']
    assert all(CONSTANTS[k]==expected[k] for k in CONSTANTS)



def test_invalid_profile_cannot_hide_in_normalization(fixture_dir,tmp_path):
    import json
    from flexmodels.data import prepare
    profiles=json.loads((fixture_dir/'profiles.json').read_text());profiles['synthetic_profiles'][0]['min_time_seconds']=float('nan');path=tmp_path/'profiles.json';path.write_text(json.dumps(profiles))
    with pytest.raises(ValueError,match='physical profile'):prepare(fixture_dir/'observations.csv.gz',tmp_path/'cache',path,fixture_dir/'workloads.json',True)



def test_training_metadata_without_git_checkout(cache_path,tiny_configs,tmp_path,monkeypatch):
    import subprocess,json
    def no_git(*args,**kwargs):raise subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(training.subprocess,'check_output',no_git)
    result=training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,stop_after_steps=1,device_name='cpu')
    meta=json.loads((tmp_path/'environment.json').read_text());assert result['global_step']==1 and meta['git_SHA'] is None and meta['package_source_hash'] and meta['snapshot_base_SHA']



def test_standalone_eval_freezes_same_k0_numerical_path(cache_path,tiny_configs,tmp_path):
    import copy,evaluate
    from flexmodels.data import load_cache
    from flexmodels.episodes import make_bank,episode_batches,encode
    from flexmodels.meta_v4 import MetaV4
    from flexmodels.flexpfn import FlexPFN
    training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path,stop_after_steps=3,device_name='cpu')
    cache=load_cache(cache_path);baseline,_=evaluate.load_model('baseline',tmp_path/'best_feasibility.pt',cache);assert all(not p.requires_grad for p in baseline.parameters())
    records=make_bank(cache,repeats=1,q=4,ks=[0])['episodes'];c,q=episode_batches(cache,records,'cpu');original=torch.get_num_threads();torch.set_num_threads(4)
    try:
        expected=encode(baseline,q)
        for model in [MetaV4(copy.deepcopy(baseline),tiny_configs['meta_v4']['model']),FlexPFN(copy.deepcopy(baseline),tiny_configs['flexpfn']['model'])]:
            actual=model.eval()(c,q)
            assert torch.equal(expected['tracking_logs'],actual['tracking_logs']) and torch.equal(expected['qos_probabilities'],actual['qos_probabilities'])
    finally:torch.set_num_threads(original)
