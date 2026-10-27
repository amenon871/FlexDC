"""Physical, feasibility, Task-macro and transformed uncertainty evaluation."""
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from .objective import reconstruct,targets,TRACKING_LIMIT,QOS_LIMIT,P90_BOUNDARY,QOS_BOUNDARY
from .data import batch,write_json
from .episodes import episode_batches


def regression(y,p,prefix):
    y=np.asarray(y);p=np.asarray(p);mse=np.mean((y-p)**2);den=np.sum((y-y.mean())**2)
    return {prefix+'_mae':float(np.mean(np.abs(y-p))),prefix+'_r2':None if den<=1e-20 else float(1-np.sum((y-p)**2)/den)}


def summarize(a):
    true=(a['true_p90']<=TRACKING_LIMIT)&(a['true_max_pj']<=QOS_LIMIT);pred=(a['pred_p90']<=TRACKING_LIMIT)&(a['pred_max_pj']<=QOS_LIMIT)
    tp=int(np.sum(true&pred));fp=int(np.sum(~true&pred));fn=int(np.sum(true&~pred));tn=int(np.sum(~true&~pred))
    div=lambda x,y:float(x/y) if y else 0.
    precision=div(tp,tp+fp);recall=div(tp,tp+fn)
    out=dict(rows=len(true),feasible_precision=precision,feasible_recall=recall,feasible_f1=div(2*tp,2*tp+fp+fn),false_feasible_rate=div(fp,fp+tn),false_infeasible_rate=div(fn,fn+tp),false_discovery_rate=div(fp,fp+tp))
    recalls=([recall] if tp+fn else [])+([div(tn,tn+fp)] if tn+fp else []);out['balanced_accuracy']=float(np.mean(recalls))
    for label,name in [('mean','mean_tracking'),('p90','p90'),('max_pj','max_pj'),('objective','objective')]:out.update(regression(a['true_'+label],a['pred_'+label],name))
    if np.std(a['true_objective'])>1e-12 and np.std(a['pred_objective'])>1e-12:out['objective_spearman']=float(spearmanr(a['true_objective'],a['pred_objective']).statistic)
    else:out['objective_spearman']=None
    return out


def collect_evaluation(model,cache,indices=None,records=None,device='cpu',batch_size=2048,episode_batch_size=16,loss_config=None):
    model.eval();parts=defaultdict(list);all_ids=[];uncertainty=defaultdict(list);losses=defaultdict(list)
    chunks=[]
    if records is not None:
        byk=defaultdict(list)
        for r in records:byk[r['K']].append(r)
        for k,recs in sorted(byk.items()):
            chunks.extend((k,recs[i:i+episode_batch_size]) for i in range(0,len(recs),episode_batch_size))
    else:chunks=[(None,indices[i:i+batch_size]) for i in range(0,len(indices),batch_size)]
    with torch.no_grad():
        for k,items in chunks:
            if records is None:
                q=batch(cache,items,device);out=model(q['globals'],q['jobs'],q['mask']);ids=items.tolist();ks=[-1]*len(ids)
            else:
                c,q=episode_batches(cache,items,device);ids=[i for r in items for i in r['query']];ks=[k]*len(ids)
                if hasattr(model,'prior'):out=model(c,q)
                else:
                    from .episodes import encode
                    out=encode(model,q)
            if loss_config is not None:
                from .objective import behavior_loss,predictive_loss
                loss,_=predictive_loss(out,q,loss_config,model.config['qos_logit_epsilon']) if out.get('uncertainty_available',False) else behavior_loss(out,q,cache['normalization'],loss_config)
                losses[-1 if k is None else k].append((float(loss),len(ids)))
            decoded=reconstruct(out,q);flat=lambda x:x.detach().float().cpu().reshape(-1).numpy()
            for label,value in [('pred_mean',decoded['mean_tracking']),('pred_p90',decoded['p90']),('pred_max_pj',decoded['pj'].masked_fill(~q['mask'],-1).max(-1).values),('pred_objective',decoded['objective']),('true_mean',q['mean_tracking']),('true_p90',q['p90']),('true_max_pj',q['qos'].masked_fill(~q['mask'],-1).max(-1).values),('true_objective',q['objective'])]:parts[label].append(flat(value))
            pj=decoded['pj'].detach().float().cpu().reshape(-1,q['mask'].shape[-1]).numpy();qt=q['qos'].detach().cpu().reshape(pj.shape).numpy();mask=q['mask'].cpu().reshape(pj.shape).numpy()
            for j in range(pj.shape[-1]):
                parts['pj_error_'+str(j)].append(np.where(mask[:,j],np.abs(pj[:,j]-qt[:,j]),np.nan))
            parts['K'].append(np.asarray(ks));all_ids.extend(ids)
            if out.get('uncertainty_available',False):
                eps=model.config['qos_logit_epsilon'];truth=targets(q,eps);mu=out['transformed_mean'];v=out['variance'].clamp_min(1e-8);valid=torch.cat([torch.ones_like(q['mask'][...,:2]),q['mask']],-1)
                err=(truth-mu).abs();std=v.sqrt();nll=.5*((truth-mu).square()/v+v.log()+math.log(2*math.pi))
                uncertainty['K'].extend([k]*int(valid.sum()));uncertainty['nll'].extend(nll[valid].cpu().tolist());uncertainty['error'].extend(err[valid].cpu().tolist());uncertainty['std'].extend(std[valid].cpu().tolist())
                for probability,z in [(50,.6744897501960817),(90,1.6448536269514722),(95,1.959963984540054)]:
                    uncertainty['coverage_'+str(probability)].extend((err[valid]<=z*std[valid]).cpu().tolist());uncertainty['interval_width_'+str(probability)].extend((2*z*std[valid]).cpu().tolist())
    frame=pd.DataFrame({k:np.concatenate(v) for k,v in parts.items()})
    for field in ['Task_ID','Workload_ID','hardware_class','J','N','U','Candidate_Source']:frame[field]=[cache['metadata'][i][field] for i in all_ids]
    numeric=[c for c in frame.columns if c.startswith(('pred_','true_'))]
    if not np.isfinite(frame[numeric].to_numpy()).all():raise ValueError('Nonfinite evaluation prediction')
    rows=[]
    for k,group in frame.groupby('K',sort=True):
        result=summarize(group);task=[summarize(g) for _,g in group.groupby('Task_ID')]
        for metric in ['feasible_f1','false_feasible_rate','false_infeasible_rate','p90_mae','max_pj_mae','objective_mae']:result['task_macro_'+metric]=float(np.mean([t[metric] for t in task]))
        for c in group.columns:
            if c.startswith('pj_error_'):result[c+'_mae']=float(group[c].mean()) if group[c].notna().any() else None
        result['K']=int(k)
        if losses[k]:result['dev_loss']=sum(v*n for v,n in losses[k])/sum(n for _,n in losses[k])
        if uncertainty:
            sel=np.asarray(uncertainty['K'])==k
            if sel.any():
                for metric in ['nll','coverage_50','coverage_90','coverage_95','interval_width_50','interval_width_90','interval_width_95']:result[metric]=float(np.mean(np.asarray(uncertainty[metric])[sel]))
                x=np.asarray(uncertainty['std'])[sel];y=np.asarray(uncertainty['error'])[sel];result['uncertainty_error_spearman']=float(spearmanr(x,y).statistic) if min(np.std(x),np.std(y))>1e-12 else None
        rows.append(result)
    return rows,frame


def save_evaluation(output,rows,frame,partition='DEVELOPMENT',write_detailed_predictions=True):
    from pathlib import Path
    out=Path(output);out.mkdir(parents=True,exist_ok=True);prefix='locked' if partition=='LOCKED_SYNTHETIC_TEST' else 'development'
    write_json(out/(prefix+'_metrics.json'),rows)
    pd.DataFrame(rows).to_csv(out/(prefix+'_k_curve.csv'),index=False)
    if not write_detailed_predictions:return
    for column,label in [('hardware_class','hardware'),('J','J'),('N','N'),('U','U'),('Candidate_Source','candidate_source'),('Workload_ID','workload'),('Task_ID','Task_ID')]:
        records=[]
        for (k,name),g in frame.groupby(['K',column]):records.append(dict(K=int(k),group=name,**summarize(g)))
        pd.DataFrame(records).to_csv(out/(label+'_breakdown.csv'),index=False)
    boundaries=[]
    for k,g in frame.groupby('K'):
        for name,mask in [('p90_0.20_0.40',g.true_p90.between(*P90_BOUNDARY)),('max_Pj_0.05_0.15',g.true_max_pj.between(*QOS_BOUNDARY))]:
            if mask.any():boundaries.append(dict(K=int(k),boundary=name,**summarize(g[mask])))
    pd.DataFrame(boundaries).to_csv(out/'boundary_metrics.csv',index=False)
    jobcols=[c for c in frame if c.startswith('pj_error_')]
    frame.groupby(['K','Task_ID'])[jobcols].mean().stack().rename('Pj_MAE').reset_index().rename(columns={'level_2':'canonical_job_position'}).to_csv(out/'per_task_job_mae.csv',index=False)
    frame.to_csv(out/'prediction_summary.csv.gz',index=False,compression='gzip')
