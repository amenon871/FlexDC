"""Complete tiny pipeline, actual CLI UX, and resume; NEVER real Foundation data."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import yaml

PACKAGE=Path(__file__).resolve().parent


def main(output=None):
    output=Path(output or PACKAGE/'.validation_runs/smoke');output.mkdir(parents=True,exist_ok=True);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    log=output/'console.txt'
    def call(script,args):
        command=[sys.executable,'-u','-B',str(PACKAGE/script),*map(str,args)]
        with log.open('a',encoding='utf8') as stream:
            stream.write('\nCOMMAND '+ ' '.join(command)+'\n');stream.flush();subprocess.run(command,cwd=PACKAGE.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    fixture=PACKAGE/'fixtures/tiny';call('fixtures/make_fixture.py',['--output',fixture]);cache=output/'cache';call('prepare_cache.py',['--input',fixture/'observations.csv.gz','--profiles',fixture/'profiles.json','--workloads',fixture/'workloads.json','--output',cache,'--fixture'])
    configs={}
    for kind in ['baseline','meta_v4','flexpfn']:
        cfg=yaml.safe_load((PACKAGE/'configs'/(kind+'.yaml')).read_text());cfg.update(cpu_threads=1,log_every_steps=1,checkpoint_every_steps=1,eval_batch_size=512)
        if kind=='baseline':cfg.update(batch_size=256,max_epochs=2,warmup_epochs=1);cfg['model'].update(st_dim_hidden=32,global_projection_dim=16,linear_dim_hidden=32,qos_projection_dim=16)
        else:
            cfg.update(max_steps=4,steps_per_virtual_epoch=2,warmup_steps=2,microbatch_episodes=2,gradient_accumulation=1,query_count=4,eval_every_steps=2,eval_episode_batch_size=2,dev_episodes_per_task=1)
            if kind=='meta_v4':cfg['model']['latent_dim']=4
            else:cfg['model'].update(d_model=32,layers=1,heads=4)
        path=output/(kind+'.yaml');path.write_text(yaml.safe_dump(cfg));configs[kind]=path
    baseline=output/'baseline';call('train_baseline.py',['--config',configs['baseline'],'--cache',cache,'--run-dir',baseline,'--stop-after-steps',3,'--device','cpu'])
    prior=output/'selected_baseline.pt';shutil.copyfile(baseline/'best_feasibility.pt',prior);frozen_hash=hashlib.sha256(prior.read_bytes()).hexdigest()
    bank=output/'dev_bank.json';call('evaluate.py',['--cache',cache,'--build-dev-bank',bank,'--episodes-per-task',1,'--query-count',4])
    for kind in ['meta_v4','flexpfn']:call('train_'+kind+'.py',['--config',configs[kind],'--cache',cache,'--run-dir',output/kind,'--baseline-checkpoint',prior,'--dev-bank',bank,'--stop-after-steps',2,'--device','cpu'])
    for kind in ['baseline','meta_v4','flexpfn']:
        assert (output/kind/'latest.pt').exists() and not (output/kind/'final.pt').exists(),'Partial runs must not produce final.pt'
    call('evaluate.py',['--cache',cache,'--model','all','--baseline-checkpoint',prior,'--meta-checkpoint',output/'meta_v4/latest.pt','--flexpfn-checkpoint',output/'flexpfn/latest.pt','--bank',bank,'--output',output/'comparison_before_resume'])
    for kind in ['baseline','meta_v4','flexpfn']:
        args=['--config',configs[kind],'--cache',cache,'--run-dir',output/kind,'--resume','auto','--device','cpu']
        if kind!='baseline':args+=['--baseline-checkpoint',prior,'--dev-bank',bank]
        call('train_'+kind+'.py',args)
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==frozen_hash
    call('evaluate.py',['--cache',cache,'--model','all','--baseline-checkpoint',prior,'--meta-checkpoint',output/'meta_v4/latest.pt','--flexpfn-checkpoint',output/'flexpfn/latest.pt','--bank',bank,'--output',output/'comparison_after_resume'])
    import pandas as pd,numpy as np
    reference=pd.read_csv(output/'comparison_after_resume/baseline/prediction_summary.csv.gz').query('K==0')
    columns=['pred_mean','pred_p90','pred_max_pj','pred_objective']
    for kind in ['meta_v4','flexpfn']:
        actual=pd.read_csv(output/'comparison_after_resume'/kind/'prediction_summary.csv.gz').query('K==0')
        assert np.array_equal(reference[columns].to_numpy(),actual[columns].to_numpy()),'Saved K=0 predictions must match the same frozen Baseline exactly'
    import torch
    from flexmodels.training import package_source_identity
    from flexmodels.data import load_cache
    source_hash=package_source_identity()[0];bank_hashes={}
    for kind in configs:
        checkpoint=torch.load(output/kind/'final.pt',map_location='cpu',weights_only=False)
        assert checkpoint['package_source_sha256']==source_hash
        assert checkpoint['exit_reason']=='completed_max_epochs'
        if kind!='baseline':
            bank_hashes[kind]=hashlib.sha256((output/kind/'development_episode_bank.json').read_bytes()).hexdigest()
            assert checkpoint['development_bank_sha256']==bank_hashes[kind]
    assert bank_hashes['meta_v4']==bank_hashes['flexpfn']
    cached=load_cache(cache);assert cached['evidence_counts'].shape==cached['mask'].shape
    call('train_baseline.py',['--config',configs['baseline'],'--cache',cache,'--run-dir',output/'walltime_guard','--max-runtime-hours',0,'--exit-before-minutes',0,'--device','cpu'])
    wall=json.loads((output/'walltime_guard/run_result.json').read_text());assert wall['exit_reason']=='walltime_exit'
    assert (output/'walltime_guard/latest.pt').exists() and not (output/'walltime_guard/final.pt').exists()
    result=dict(package_source_sha256=source_hash,development_bank_hashes=bank_hashes,partial_final_semantics='PASS',walltime_behavior='PASS',evidence_counts_preserved=True,K0_saved_predictions_exact=True,status='PASS',test_fixture_only=True,scientific_training=False,baseline_prior_hash=frozen_hash,steps={k:json.loads((output/k/'run_result.json').read_text())['global_step'] for k in configs},methods=['baseline','meta_v4','flexpfn'],K=[0,1,2,4,8,16,32,64]);(output/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();main(a.output)
