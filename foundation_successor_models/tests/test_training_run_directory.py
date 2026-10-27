"""A fresh invocation must never overwrite any existing run state."""
import hashlib
from pathlib import Path
import pytest
from flexmodels import training


@pytest.mark.parametrize('kind',['baseline','meta_v4','flexpfn'])
@pytest.mark.parametrize('name',['latest.pt','latest_previous.pt','final.pt','best_loss.pt','best_objective.pt','best_feasibility.pt','best_fewshot.pt','history.csv','training.log','run_config.yaml','run_config.json','development_episode_bank.json','development_metrics.json','development_k_curve.csv','unrelated.txt'])
def test_nonempty_directory_refused_before_any_write(tmp_path,kind,name):
    path=tmp_path/'run';path.mkdir();file=path/name;file.write_bytes(b'PRESERVE EXACTLY')
    with pytest.raises(ValueError,match='Use --resume auto.*new --run-dir'):
        training.run_training(kind,{},'UNUSED_CACHE',path)
    assert file.read_bytes()==b'PRESERVE EXACTLY' and list(path.iterdir())==[file]


def test_new_empty_and_different_directories(tmp_path):
    fresh=tmp_path/'missing';assert training.prepare_run_directory(fresh)==fresh and not list(fresh.iterdir())
    empty=tmp_path/'empty';empty.mkdir();assert training.prepare_run_directory(empty)==empty
    (fresh/'history.csv').write_text('previous state')
    different=tmp_path/'different';assert training.prepare_run_directory(different)==different


@pytest.mark.parametrize('kind',['baseline','meta_v4','flexpfn'])
def test_existing_actual_run_requires_explicit_resume(cache_path,tiny_configs,tmp_path,kind):
    prior=None
    if kind!='baseline':
        training.run_training('baseline',tiny_configs['baseline'],cache_path,tmp_path/'prior',stop_after_steps=3,device_name='cpu')
        prior=tmp_path/'prior/best_feasibility.pt'
    path=tmp_path/kind;kw=dict(baseline_checkpoint=prior,device_name='cpu',stop_after_steps=1)
    first=training.run_training(kind,tiny_configs[kind],cache_path,path,**kw)
    checkpoint=hashlib.sha256((path/'latest.pt').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Run directory already contains'):
        training.run_training(kind,tiny_configs[kind],cache_path,path,**kw)
    assert hashlib.sha256((path/'latest.pt').read_bytes()).hexdigest()==checkpoint
    second=training.run_training(kind,tiny_configs[kind],cache_path,path,resume='auto',**kw)
    assert second['global_step']==first['global_step']+1
