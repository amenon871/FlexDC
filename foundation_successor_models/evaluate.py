"""One evaluation entry; every method can consume the exact same fixed bank."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from flexmodels.data import load_cache,partition_indices,write_json,sha256
from flexmodels.episodes import make_bank,load_bank,repeat_bank,repeat_noise_report
from flexmodels.checkpoint import read_checkpoint,make_model
from flexmodels.metrics import collect_evaluation,save_evaluation


def load_model(kind,path,cache,prior=None):
    ck=read_checkpoint(path)
    if ck['cache_hash']!=cache['cache_hash'] or ck['model_kind']!=kind:raise ValueError('Evaluation checkpoint kind/cache mismatch')
    prior_path=prior or ck.get('baseline_checkpoint')
    if kind!='baseline' and (not prior_path or sha256(prior_path)!=ck['baseline_prior_hash']):raise ValueError('Comparison must use the exact same frozen Foundation prior')
    model=make_model(kind,ck['training_config'],prior_path if kind!='baseline' else None,cache);model.load_state_dict(ck['model']);return model.requires_grad_(False).eval(),ck


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--model',choices=['baseline','meta_v4','flexpfn','all'],default='baseline');p.add_argument('--checkpoint');p.add_argument('--baseline-checkpoint');p.add_argument('--meta-checkpoint');p.add_argument('--flexpfn-checkpoint');p.add_argument('--bank');p.add_argument('--build-dev-bank');p.add_argument('--partition',choices=['development','locked'],default='development');p.add_argument('--unlock-locked-test',action='store_true');p.add_argument('--output');p.add_argument('--episodes-per-task',type=int,default=10);p.add_argument('--query-count',type=int,default=32);p.add_argument('--seed',type=int,default=73);p.add_argument('--selection',choices=['random','diverse','nested'],default='random');p.add_argument('--repeat-analysis',action='store_true');p.add_argument('--repeat-assessment',action='store_true');p.add_argument('--qos-logit-epsilon',type=float,default=1e-4);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--device',default='cpu');a=p.parse_args(argv)
    torch.set_num_threads(a.cpu_threads)
    cache=load_cache(a.cache);part='DEVELOPMENT' if a.partition=='development' else 'LOCKED_SYNTHETIC_TEST';ix=partition_indices(cache,part,a.unlock_locked_test)
    if a.build_dev_bank:
        bank=make_bank(cache,part,a.seed,a.episodes_per_task,a.query_count,a.selection);write_json(a.build_dev_bank,bank);print(f"Frozen {part} bank: {len(bank['episodes'])} episodes -> {a.build_dev_bank}");return bank
    if not a.output:p.error('--output is required for evaluation')
    if a.repeat_analysis:repeat_noise_report(cache,part,a.qos_logit_epsilon,a.output)
    if a.repeat_analysis and not a.repeat_assessment and not a.checkpoint and a.model!='all':return
    if a.repeat_assessment and a.bank:p.error('Repeat diagnostic cannot masquerade as an ordinary episode bank')
    bank=repeat_bank(cache,part) if a.repeat_assessment else load_bank(a.bank,cache,part) if a.bank else None
    if a.model!='baseline' and bank is None:p.error('Few-shot/three-method comparison requires an explicit shared --bank')
    checkpoints=dict(baseline=a.baseline_checkpoint,meta_v4=a.meta_checkpoint,flexpfn=a.flexpfn_checkpoint) if a.model=='all' else {a.model:a.checkpoint}
    comparison=[]
    for kind,path in checkpoints.items():
        if not path:p.error('Missing checkpoint for '+kind)
        model,ck=load_model(kind,path,cache,a.baseline_checkpoint if kind!='baseline' else None);model.to(a.device)
        rows,frame=collect_evaluation(model,cache,indices=None if bank else ix,records=bank['episodes'] if bank else None,device=a.device,loss_config=ck['training_config']['loss'])
        out=Path(a.output)/kind if a.model=='all' else Path(a.output);save_evaluation(out,rows,frame,part);write_json(out/'evaluation_identity.json',dict(partition=part,cache_hash=cache['cache_hash'],checkpoint=str(path),episode_bank=a.bank,locked_unlocked=a.unlock_locked_test,diagnostic_only_same_geometry_repeat=a.repeat_assessment))
        comparison.extend(dict(method=kind,**row) for row in rows)
    if a.model=='all':pd.DataFrame(comparison).to_csv(Path(a.output)/'three_method_comparison.csv',index=False)
    table=pd.DataFrame(comparison);columns=[x for x in ['method','K','task_macro_feasible_f1','false_feasible_rate','p90_mae','max_pj_mae','objective_mae','objective_r2'] if x in table]
    print(table[columns].to_string(index=False));return comparison

if __name__=='__main__':main()
