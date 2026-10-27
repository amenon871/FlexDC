"""One ordinary PyTorch training loop for rows or shared few-shot episodes."""
import argparse,contextlib,csv,hashlib,json,logging,math,os,random,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from .data import load_cache,partition_indices,batch,sampler_weights,write_json,sha256
from .episodes import K_VALUES,task_pools,training_records,episode_batches,make_bank,load_bank,freeze_bank
from .objective import behavior_loss,predictive_loss
from .checkpoint import SCHEMA,make_model,atomic_save,resolve_resume,read_checkpoint,rng_state,restore_rng
from .metrics import collect_evaluation,save_evaluation

PACKAGE=Path(__file__).resolve().parents[1]

class Tee:
    def __init__(self,console,file):self.console=console;self.file=file
    def write(self,text):self.console.write(text);self.file.write(text)
    def flush(self):self.console.flush();self.file.flush()
    def isatty(self):return False


def config_hash(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def package_source_identity(package=None):
    package=Path(package or PACKAGE)
    names=sorted([*package.glob('*.py'),*package.glob('flexmodels/*.py'),*package.glob('configs/*.yaml')])
    files={p.relative_to(package).as_posix():sha256(p) for p in names}
    return config_hash(files),files


def validation_budget_seconds(config,state,exit_before_minutes):
    previous=state.get('last_full_validation_seconds')
    estimate=1.25*previous if previous is not None else config.get('first_validation_min_seconds',300)
    return max(exit_before_minutes*60,estimate+config.get('checkpoint_margin_seconds',30))


def learning_rate(position,total,warmup,base,minimum):
    if warmup and position<warmup:return base*(position+1)/warmup
    progress=min(max((position-warmup)/max(total-warmup-1,1),0),1)
    return minimum+(base-minimum)*.5*(1+math.cos(math.pi*progress))


def environment():
    import scipy
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def wandb_start(mode,run_dir,config,metadata,run_id,log):
    if mode=='disabled':return None
    try:
        import wandb
        return wandb.init(project='FlexDC-Foundation-Successors',name=Path(run_dir).name,dir=str(run_dir),mode=mode,id=run_id,resume='allow',config=dict(config=config,metadata=metadata))
    except Exception as error:log('W&B unavailable; continuing with local outputs: '+type(error).__name__);return None


def prepare_run_directory(run_dir,resume=None):
    path=Path(run_dir)
    if path.exists() and (not path.is_dir() or (not resume and any(path.iterdir()))):
        raise ValueError('Run directory already contains training state or is nonempty. Use --resume auto to continue the existing run, or choose a new --run-dir for a fresh experiment.')
    path.mkdir(parents=True,exist_ok=True)
    return path


def run_training(kind,config,cache_path,run_dir,seed=101,baseline_checkpoint=None,resume=None,wandb_mode='disabled',precision='auto',device_name='auto',max_runtime_hours=None,exit_before_minutes=10,stop_after_steps=None,benchmark=False):
    invocation_start=time.perf_counter();run_dir=prepare_run_directory(run_dir,resume)
    logfile=(run_dir/'training.log').open('a',encoding='utf8');stream=Tee(sys.stdout,logfile)
    def log(message):print(message,file=stream,flush=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but not available')
    use_bf16=(device.type=='cuda' and torch.cuda.is_bf16_supported()) if precision=='auto' else precision=='bf16'
    if use_bf16 and device.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('Requested bf16 is unsupported on this GPU')
    torch.set_num_threads(config.get('cpu_threads',4))
    cache=load_cache(cache_path);train=partition_indices(cache,'TRAIN');dev=partition_indices(cache,'DEVELOPMENT')
    if not len(train) or not len(dev):raise ValueError('TRAIN and DEVELOPMENT are both required')
    model=make_model(kind,config,baseline_checkpoint,cache).to(device)
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8)
    row_training=kind=='baseline';steps_epoch=math.ceil(len(train)/config['batch_size']) if row_training else config['steps_per_virtual_epoch']
    total=config['max_epochs']*steps_epoch if row_training else config['max_steps'];warmup=config['warmup_epochs'] if row_training else config['warmup_steps']
    eval_every=steps_epoch if row_training else config['eval_every_steps']
    episode_rng=np.random.default_rng(seed+917);pools=task_pools(cache,'TRAIN') if not row_training else None
    weights=sampler_weights(cache,train.numpy()) if row_training else None;state=dict(global_step=0,history=[],best_values={},best_steps={},last_improvement_step=0,early_stop=False)
    try:source_sha=subprocess.check_output(['git','-C',str(PACKAGE.parent),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):source_sha=None
    base_file=PACKAGE/'audit/BASE_STATE.json'
    source_base=json.loads(base_file.read_text()).get('base_HEAD') if base_file.exists() else None
    package_sha,sources=package_source_identity()
    metadata=dict(git_SHA=source_sha,snapshot_base_SHA=source_base,package_source_hash=package_sha,package_source_sha256=package_sha,package_source_files=sources,cache_hash=cache['cache_hash'],foundation_input_hashes=cache['manifest']['input_hashes'],foundation_source_manifest_hashes=cache['manifest'].get('source_manifest_hashes',{}),config_hash=config_hash(config),training_seed=seed,sampler_seed=seed,baseline_prior_hash=sha256(baseline_checkpoint) if baseline_checkpoint else None,environment=environment(),precision='bf16' if use_bf16 else 'fp32')
    checkpoint_path=resolve_resume(resume,run_dir,log);previous=None
    if checkpoint_path:
        previous=read_checkpoint(checkpoint_path)
        if previous['model_kind']!=kind or previous['cache_hash']!=cache['cache_hash'] or previous['config_hash']!=metadata['config_hash'] or previous['seed']!=seed or previous['baseline_prior_hash']!=metadata['baseline_prior_hash']:raise ValueError('Resume kind/cache/config/seed/prior identity mismatch')
        old_source=previous.get('package_source_sha256')
        if old_source!=package_sha:raise ValueError('Resume package source SHA mismatch: checkpoint='+str(old_source)+' current='+package_sha)
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);state=previous['state'];episode_rng.bit_generator.state=previous['episode_rng_state'];restore_rng(previous['rng'])
        log(f"RESUME checkpoint={checkpoint_path} epoch={state['global_step']//steps_epoch} step={state['global_step']} lr={optimizer.param_groups[0]['lr']:.6g} best_f1={state['best_values'].get('feasibility')}")
    if not row_training:
        local_bank=run_dir/'development_episode_bank.json'
        if previous and (not local_bank.is_file() or sha256(local_bank)!=previous.get('development_bank_sha256')):raise ValueError('Resume DEVELOPMENT bank SHA mismatch')
        bank,bank_identity=freeze_bank(run_dir/'development_episode_bank.json',cache,config,resuming=previous is not None)
        if previous and any(previous.get(k)!=v for k,v in bank_identity.items()):raise ValueError('Resume DEVELOPMENT bank SHA/path/schema mismatch')
        metadata.update(bank_identity)
    else:
        bank=None;metadata.update(development_bank_path=None,development_bank_sha256=None,development_bank_schema=None)
    write_json(run_dir/'environment.json',metadata);(run_dir/'git_commit.txt').write_text((source_sha or 'NO_GIT_METADATA; snapshot base='+str(source_base))+'\n');(run_dir/'dataset_manifest_hash.txt').write_text(cache['cache_hash']+'\n');(run_dir/'run_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    wandb=wandb_start(wandb_mode,run_dir,config,metadata,previous.get('wandb_run_id') if previous else None,log)
    # W&B initialization must not perturb the restored training RNG.
    if previous:restore_rng(previous['rng'])
    interrupted={'stop':False,'reason':None};handlers={}
    def stop(sig,frame):interrupted.update(stop=True,reason=signal.Signals(sig).name)
    for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,stop)
    exit_reason=None

    labels={'baseline':'Foundation Baseline','meta_v4':'Meta V4','flexpfn':'FlexPFN'}
    log('='*65);log(f"FlexDC {labels[kind]} Training | Run={run_dir.name}")
    log(f"Device={device} GPU={metadata['environment']['gpu']} Precision={metadata['precision']}")
    log(f"Train rows={len(train):,} Development rows={len(dev):,} LOCKED rows excluded from training/selection")
    log(f"Optimizer=AdamW LR={config['lr']} Steps={total} Batch/episodes={config['batch_size'] if row_training else config['microbatch_episodes']} Checkpoints={run_dir}")
    if cache['fixture']:log('TEST FIXTURE ONLY â€” NOT SCIENTIFIC DATA')
    log('='*65)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    bar=tqdm(total=total,initial=state['global_step'],file=stream,ascii=True,dynamic_ncols=True,unit='step',mininterval=.2)
    observed_steps=0;epoch_order=None;order_epoch=None;running_loss=None;window_steps=0;step_times=[];cpu_times=dict(data_wait=0.,forward=0.,backward=0.,optimizer=0.);cuda_events=[];process_start=time.process_time()
    def payload(role):
        return dict(schema=SCHEMA,exit_reason=exit_reason or 'running',package_source_sha256=package_sha,**{k:metadata[k] for k in ['development_bank_path','development_bank_sha256','development_bank_schema']},checkpoint_role=role,model_kind=kind,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=dict(position=state['global_step'],total=total,warmup=warmup,steps_per_epoch=steps_epoch),grad_scaler=None,state=state,epoch=state['global_step']//steps_epoch,global_step=state['global_step'],normalization=cache['normalization'],model_config=config['model'],training_config=config,config_hash=metadata['config_hash'],cache_hash=cache['cache_hash'],baseline_prior_hash=metadata['baseline_prior_hash'],baseline_checkpoint=str(baseline_checkpoint) if baseline_checkpoint else None,seed=seed,rng=rng_state(),sampler_state=dict(base_seed=seed,epoch=state['global_step']//steps_epoch,next_batch=state['global_step']%steps_epoch),episode_rng_state=episode_rng.bit_generator.state,wandb_run_id=getattr(wandb,'id',None),metadata=metadata)
    def save(role):atomic_save(run_dir/(role+'.pt'),payload(role));log('Checkpoint: '+role+'.pt saved at step '+str(state['global_step']))
    compute_start=time.perf_counter()
    try:
        while state['global_step']<total and not state['early_stop']:
            step_start=time.perf_counter();step=state['global_step'];epoch=step//steps_epoch
            position=epoch if row_training else step;units=config['max_epochs'] if row_training else total
            lr=learning_rate(position,units,warmup,config['lr'],config['min_lr'])
            for group in optimizer.param_groups:group['lr']=lr
            model.train();optimizer.zero_grad(set_to_none=True);accum=1 if row_training else config['gradient_accumulation'];step_loss=torch.zeros((),device=device);components={};mean_k=0.
            for micro in range(accum):
                data_start=time.perf_counter()
                if row_training:
                    if order_epoch!=epoch:
                        gen=torch.Generator().manual_seed(seed+epoch);epoch_order=torch.multinomial(weights,len(train),replacement=True,generator=gen);order_epoch=epoch
                    offset=(step%steps_epoch)*config['batch_size'];ix=train[epoch_order[offset:offset+config['batch_size']]];q=batch(cache,ix,device)
                else:
                    k=int(episode_rng.choice(K_VALUES[1:]));mean_k+=k/accum
                    recs=training_records(cache,episode_rng,k,config['query_count'],config['microbatch_episodes'],pools,config['diverse_fraction']);c,q=episode_batches(cache,recs,device)
                cpu_times['data_wait']+=time.perf_counter()-data_start
                markers=None
                if benchmark and device.type=='cuda':
                    markers=[torch.cuda.Event(enable_timing=True) for _ in range(4)];markers[0].record()
                t=time.perf_counter()
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
                    out=model(q['globals'],q['jobs'],q['mask']) if row_training else model(c,q)
                    loss,components=behavior_loss(out,q,cache['normalization'],config['loss']) if row_training else predictive_loss(out,q,config['loss'],config['model']['qos_logit_epsilon'])
                cpu_times['forward']+=time.perf_counter()-t
                if markers:markers[1].record()
                t=time.perf_counter();(loss/accum).backward();cpu_times['backward']+=time.perf_counter()-t
                if markers:markers[2].record()
                step_loss+=loss.detach().float()/accum
                if markers:cuda_events.append(markers)
            t=time.perf_counter();gradient_norm=torch.nn.utils.clip_grad_norm_(params,config['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step();cpu_times['optimizer']+=time.perf_counter()-t
            if markers:markers[3].record()
            if cuda_events and accum>1:
                # Optimizer event belongs only to the last microbatch; earlier
                # ones end at backward, not at an unrecorded CUDA event.
                for marker in cuda_events[-accum:-1]:marker[3].record()
            running_loss=step_loss if running_loss is None else running_loss+step_loss;window_steps+=1
            state['global_step']+=1;observed_steps+=1;bar.update(1);bar.set_description(f"{'Epoch' if row_training else 'V-Epoch'} {epoch+1}/{config['max_epochs'] if row_training else math.ceil(total/steps_epoch)} | Step {state['global_step']}")
            if state['global_step']%config['log_every_steps']==0 or observed_steps==1:
                bar.set_postfix(loss=f'{float(step_loss):.5g}',lr=f'{lr:.3g}',Kmean=f'{mean_k:.1f}',mem=f'{torch.cuda.memory_allocated(device)/1e9:.2f}G' if device.type=='cuda' else 'CPU',**{name:f'{float(value.detach()):.4g}' for name,value in components.items() if name in ['mean','p90','qos','nll']})
            step_times.append(time.perf_counter()-step_start)
            elapsed=time.perf_counter()-invocation_start
            if interrupted['stop']:exit_reason='manual_interrupt' if interrupted['reason']=='SIGINT' else 'signal_exit'
            elif max_runtime_hours is not None and elapsed>=max_runtime_hours*3600-exit_before_minutes*60:exit_reason='walltime_exit'
            elif stop_after_steps is not None and observed_steps>=stop_after_steps:exit_reason='benchmark_limit' if benchmark else 'smoke_limit'
            at_validation=exit_reason in (None,'smoke_limit') and not benchmark and (state['global_step']%eval_every==0 or state['global_step']==total)
            if at_validation and max_runtime_hours is not None and max_runtime_hours*3600-(time.perf_counter()-invocation_start)<validation_budget_seconds(config,state,exit_before_minutes):
                exit_reason='walltime_exit';at_validation=False
                log('Walltime guard: checkpointing before DEVELOPMENT validation')
            if at_validation:
                validation_start=time.perf_counter()
                rows,frame=collect_evaluation(model,cache,indices=dev if row_training else None,records=None if row_training else bank['episodes'],device=device,batch_size=config['eval_batch_size'],episode_batch_size=config.get('eval_episode_batch_size',8),loss_config=config['loss'])
                state['last_full_validation_seconds']=time.perf_counter()-validation_start
                save_evaluation(run_dir,rows,frame,write_detailed_predictions=False);positive=[r for r in rows if r['K']!=0] if not row_training else rows
                dev_loss=float(np.mean([r['dev_loss'] for r in positive]));f1=float(np.mean([r['task_macro_feasible_f1'] for r in positive]));false=float(np.mean([r['task_macro_false_feasible_rate'] for r in positive]));obj=float(np.mean([r['task_macro_objective_mae'] for r in positive]));r2s=[r['objective_r2'] for r in positive if r['objective_r2'] is not None];r2=float(np.mean(r2s)) if r2s else -1e30
                selection_f1=rows[0]['feasible_f1'] if row_training else f1
                current=dict(loss=-dev_loss,objective=r2,feasibility=selection_f1,fewshot=(f1,-false,-obj));improved=[]
                for role,score in current.items():
                    if role=='fewshot' and row_training:continue
                    prior=state['best_values'].get(role)
                    margin=config['early_stopping']['min_delta'] if role=='loss' and row_training else 0.
                    if prior is None or (score>prior+margin if role!='fewshot' else score>prior):
                        state['best_values'][role]=score;state['best_steps'][role]=state['global_step'];improved.append(role)
                key='loss' if row_training else 'fewshot'
                # Early-stop min_delta is independent of the checkpoint ranking.
                monitor=current[key];old_monitor=state.get('early_stop_monitor')
                threshold=config['early_stopping']['min_delta'] if row_training else 1e-5
                significant=old_monitor is None or (monitor>old_monitor+threshold if row_training else monitor[0]>old_monitor[0]+threshold or (monitor[0]==old_monitor[0] and monitor>old_monitor))
                if significant:state['last_improvement_step']=state['global_step'];state['early_stop_monitor']=monitor
                history=dict(epoch=epoch+1,virtual_epoch=state['global_step']/steps_epoch,global_step=state['global_step'],wall_seconds=time.perf_counter()-invocation_start,train_loss=float(running_loss/window_steps),dev_loss=dev_loss,learning_rate=lr,gradient_norm=float(gradient_norm),p90_mae=float(np.mean([r['p90_mae'] for r in positive])),max_pj_mae=float(np.mean([r['max_pj_mae'] for r in positive])),feasible_precision=float(np.mean([r['feasible_precision'] for r in positive])),feasible_recall=float(np.mean([r['feasible_recall'] for r in positive])),feasible_f1=f1,false_feasible_rate=false,objective_r2=None if not r2s else r2,gpu_memory_allocated=torch.cuda.memory_allocated(device) if device.type=='cuda' else 0)
                state['history'].append(history);pd.DataFrame(state['history']).to_csv(run_dir/'history.csv',index=False);running_loss=None;window_steps=0
                log(f"Validation epoch={epoch+1} step={state['global_step']} dev_loss={dev_loss:.6g} p90_MAE={history['p90_mae']:.5f} max_Pj_MAE={history['max_pj_mae']:.5f} feasible_F1_macro={f1:.4f} false_feasible={false:.4f} objective_R2={r2:.4f} lr={lr:.3g}")
                if not row_training:
                    log('DEVELOPMENT K-CURVE | K F1 FalseFeas p90_MAE maxPj_MAE')
                    for row in rows:log(f"K={row['K']:2d} F1={row['task_macro_feasible_f1']:.4f} FalseFeas={row['false_feasible_rate']:.4f} p90_MAE={row['p90_mae']:.4f} maxPj_MAE={row['max_pj_mae']:.4f}")
                for role in improved:save('best_'+role)
                if wandb:
                    try:wandb.log(dict(history,**{'K'+str(r['K'])+'/F1':r['task_macro_feasible_f1'] for r in rows},**{'checkpoint/best_'+role:state['global_step'] for role in improved}),step=state['global_step'])
                    except Exception:log('W&B log unavailable; local history remains authoritative')
                patience=config['early_stopping']['patience']*steps_epoch if row_training else config['early_stopping_steps']
                if state['global_step']-state['last_improvement_step']>=patience:state['early_stop']=True
            if exit_reason is None and not benchmark and state['global_step']%config['checkpoint_every_steps']==0:save('latest')
            elapsed=time.perf_counter()-invocation_start
            deadline=max_runtime_hours is not None and elapsed>=max_runtime_hours*3600-exit_before_minutes*60
            if deadline:exit_reason='walltime_exit'
            if exit_reason is not None or state['early_stop']:break
        if benchmark and device.type=='cuda':torch.cuda.synchronize(device)
        compute_wall=time.perf_counter()-compute_start
        if exit_reason is None:exit_reason='benchmark_limit' if benchmark else 'early_stopped' if state['early_stop'] else 'completed_max_epochs'
        metadata['exit_reason']=exit_reason
        write_json(run_dir/'environment.json',metadata)
        bar.close();save('latest')
        if exit_reason in ('completed_max_epochs','early_stopped'):save('final')
        if not (run_dir/'history.csv').exists():pd.DataFrame(columns=['epoch','global_step','train_loss','dev_loss','learning_rate']).to_csv(run_dir/'history.csv',index=False)
        measured=dict(model=kind,optimizer_steps_this_invocation=observed_steps,global_step=state['global_step'],exit_reason=exit_reason,status={'walltime_exit':'WALLTIME_CHECKPOINT','smoke_limit':'REQUESTED_STEP_LIMIT','benchmark_limit':'BENCHMARK_LIMIT','signal_exit':interrupted['reason'],'manual_interrupt':interrupted['reason'] or 'SIGINT'}.get(exit_reason,'EARLY_STOPPED' if state['early_stop'] else 'COMPLETE'),wall_seconds=time.perf_counter()-invocation_start,fixture=cache['fixture'])
        if benchmark:
            if device.type=='cuda':torch.cuda.synchronize(device)
            compute=compute_wall;measured.update(steps_per_second=observed_steps/max(compute,1e-12),compute_wall_seconds=compute,query_predictions_per_second=(config['batch_size'] if row_training else config['microbatch_episodes']*config['query_count']*config['gradient_accumulation'])*observed_steps/max(compute,1e-12),projected_training_compute_hours=total*compute/max(observed_steps,1)/3600,validation_and_checkpoint_overhead_excluded=True,peak_allocated_vram=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,peak_reserved_vram=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,cpu_utilization_fraction=(time.process_time()-process_start)/max(compute,1e-9)/max(os.cpu_count() or 1,1),gpu_utilization=None,timing_cpu_seconds=cpu_times)
            if cuda_events:
                measured['timing_gpu_seconds']={name:sum(e[a].elapsed_time(e[b]) for e in cuda_events)/1000 for name,a,b in [('forward',0,1),('backward',1,2)]}
                measured['timing_gpu_seconds']['optimizer']=sum(e[2].elapsed_time(e[3]) for e in cuda_events[accum-1::accum])/1000
            write_json(run_dir/'benchmark.json',measured)
        write_json(run_dir/'run_result.json',measured);log('Training exit: '+str(measured));return measured
    except BaseException as error:
        exit_reason='manual_interrupt' if isinstance(error,KeyboardInterrupt) else 'error'
        write_json(run_dir/'run_result.json',dict(exit_reason=exit_reason,error=type(error).__name__+': '+str(error),global_step=state['global_step']))
        raise
    finally:
        bar.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if wandb:
            try:wandb.finish()
            except Exception:pass
        logfile.close()


def cli(kind,argv=None,benchmark=False):
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(PACKAGE/'configs'/('baseline.yaml' if kind=='baseline' else kind+'.yaml')));p.add_argument('--cache',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--baseline-checkpoint');p.add_argument('--dev-bank');p.add_argument('--seed',type=int,default=101);p.add_argument('--resume');p.add_argument('--wandb',choices=['online','offline','disabled'],default='disabled');p.add_argument('--precision',choices=['auto','bf16','fp32'],default='auto');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--lr',type=float);p.add_argument('--max-runtime-hours',type=float);p.add_argument('--exit-before-minutes',type=float,default=10);p.add_argument('--stop-after-steps',type=int);a=p.parse_args(argv)
    config=yaml.safe_load(Path(a.config).read_text())
    if a.lr is not None:config['lr']=a.lr
    if a.dev_bank is not None:config['development_bank']=a.dev_bank
    return run_training(kind,config,a.cache,a.run_dir,a.seed,a.baseline_checkpoint,a.resume,a.wandb,a.precision,a.device,a.max_runtime_hours,a.exit_before_minutes,a.stop_after_steps,benchmark)
