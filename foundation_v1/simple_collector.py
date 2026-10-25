"""Fail-forward collection of exact plan rows. Receipts are primary state."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict,deque
from pathlib import Path
from am_flexdc_foundation_v1_common import ROOT,write_csv,write_json,read_json,iter_csv,file_hash
from foundation_v1 import collector_receipts as receipts
from foundation_v1.resilient import worker_env,decode_returncode,interrupts
from peacsim.foundation_v1_cleanup import parent_changed

MAX_ATTEMPTS=3
REPAIR_CYCLES=2
CHUNK_ROWS=25


class StorageEmergency(RuntimeError):pass


def check_storage(root,threshold_gib):
    free=shutil.disk_usage(root).free
    if free < threshold_gib*1024**3:
        raise StorageEmergency('STORAGE_EMERGENCY: free='+str(free)+' bytes; threshold_gib='+str(threshold_gib))
    return free



def ensure_writable(area):
    """Probe the shared output filesystem only after an I/O error, never per row."""
    probe=Path(area)/('.write_probe_'+str(os.getpid()))
    try:
        with probe.open('wb') as stream:stream.write(b'probe');stream.flush();os.fsync(stream.fileno())
        probe.unlink()
    except OSError as error:raise RuntimeError('OUTPUT_FILESYSTEM_UNWRITABLE: '+str(error)) from error


def cache_write(path,body):
    """Derived progress/attempt metadata cannot veto durable scientific rows."""
    try:write_json(path,body,False)
    except OSError as error:
        ensure_writable(Path(path).parent)
        print('LOCAL_METADATA_DEBT '+str(path)+': '+str(error),flush=True)


def command(rows,plan,label):
    r=rows[0]
    return [sys.executable,'-u','-B','am_data_extraction_wizard.py',
        '--gradient-config',str(ROOT/'configs/foundation_v1/gradient_cost_rsr.ini'),
        '--experiment-config',str(ROOT/r['experiment_config']),
        '--cluster-config',str(ROOT/'configs/cluster/cluster.ini'),
        '--job-config',str(ROOT/r['workload_config']),'--policy-name','AQA',
        '--output-dir',label,'--sweep-plan-file',str(plan),
        '--plan-chunk-index','0','--plan-num-chunks','1',
        '--node-count-control','true','--r-over-p-max','0.6']


def collect(plan,area,max_workers,storage_gib=10,launch=None,chunk_rows=CHUNK_ROWS,stop_after=None):
    area=Path(area);area.mkdir(parents=True,exist_ok=True)
    launch=launch or subprocess.Popen
    rows=list(iter_csv(plan));by_id={r['plan_row_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate original Plan_Row_ID in immutable plan')
    recovered=receipts.completed(area,by_id);done=set(recovered); initial=len(done)
    if recovered:
        from am_flexdc_foundation_v1_common import atomic_bytes
        atomic_bytes(area/'events'/'recovered.jsonl', ''.join(json.dumps({'id':k,'time':v['timestamp']})+'\n' for k,v in recovered.items()).encode(),False)
    del recovered
    groups=defaultdict(list)
    for row in rows:
        if row['plan_row_id'] not in done:groups[row['Task_ID']].append(row)
    pending=deque([{'rows':rs[i:i+chunk_rows],'attempt':1} for rs in groups.values() for i in range(0,len(rs),chunk_rows)])
    work=area/'workers';work.mkdir(exist_ok=True)
    run=str(time.time_ns());active=[];quarantine={};attempt_count=0;retry_count=0;interrupted=False
    started=time.monotonic();last=started;slot_seconds=0.;peak=0;finished_at=None;gaps=[];interrupt_at=None
    def save(status):
        report={'schema':'FOUNDATION_V1_COLLECTOR_V2','status':status,'plan':str(plan),
            'planned':len(rows),'completed':len(done),'running':sum(len(w['rows']) for w in active),
            'workers_active':len(active),'worker_pids':[w['record']['pid'] for w in active],'max_workers':max_workers,'pending':len(by_id)-len(done),
            'technical_retries':retry_count,'worker_attempts':attempt_count,'quarantined':len(quarantine),
            'quarantined_ids':sorted(quarantine) if status!='COLLECTING' else [],'wall_seconds':time.monotonic()-started,
            'worker_slot_seconds':slot_seconds,'peak_workers':peak,'replacement_gaps':gaps[-128:],
            'new_completions':len(done)-initial,'updated_at':time.time()}
        cache_write(area/'progress.json',report);return report
    def stop(sig,frame):
        nonlocal interrupted,interrupt_at
        interrupted=True
        if interrupt_at is None:interrupt_at=time.monotonic()
        cache_write(area/'interrupt.json',{'signal':sig,'time':time.time()})
    (area/'interrupt.json').unlink(missing_ok=True)
    try:
        with interrupts(stop):
            while pending or active:
                current=time.monotonic();slot_seconds+=(current-last)*len(active);last=current
                if parent_changed(os.environ.get('FOUNDATION_V1_CAMPAIGN_PARENT_PID')):stop(signal.SIGTERM,None)
                check_storage(area,storage_gib)
                check_storage(ROOT/'src/peacsim/output/optimization',storage_gib)
                while pending and len(active)<max_workers and not interrupted:
                    item=pending.popleft();todo=[r for r in item['rows'] if r['plan_row_id'] not in done]
                    if not todo:continue
                    slot=next(i for i in range(max_workers) if i not in {w['slot'] for w in active})
                    key=run+'_'+str(attempt_count);path=work/(key+'.csv')
                    try:write_csv(path,todo)
                    except OSError as error:
                        ensure_writable(area);attempt_count+=1
                        print('LOCAL_PLAN_TRANSPORT_RETRY '+str(path)+': '+str(error),flush=True)
                        for row in todo:
                            if item['attempt']<MAX_ATTEMPTS:pending.appendleft({'rows':[row],'attempt':item['attempt']+1})
                            else:quarantine[row['plan_row_id']]={'error':str(error),'attempt':item['attempt']}
                        continue
                    scope_label=area.parent.parent.name if area.parent.name=='collector_v2' else 'validation'
                    label='foundation_v1_simple/'+scope_label+'/'+area.parent.name+'/'+area.name+'/'+key
                    cmd=command(todo,path,label); log=work/(key+'.log');stream=None
                    try:stream=log.open('w',encoding='utf8')
                    except OSError as error:
                        ensure_writable(area)
                        # One locked log filename is not a scientific failure.
                        stream=(area/(key+'.fallback.log')).open('w',encoding='utf8')
                    env=worker_env(os.environ);env.update(FOUNDATION_V1_SIMPLE_AREA=str(area.resolve()),
                        FOUNDATION_V1_SIMPLE_WORKER='slot_'+str(slot),FOUNDATION_V1_PARENT_PID=str(os.getpid()),
                        FOUNDATION_V1_INTERRUPTION_STOP=str((area/'interrupt.json').resolve()),
                        PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH',''))
                    env.pop('FOUNDATION_V1_STORAGE_STOP',None)
                    stream.write('FOUNDATION_V2 '+json.dumps({'command':cmd,'environment':{k:env[k] for k in worker_env({})}})+'\n');stream.flush()
                    try:proc=launch(cmd,cwd=ROOT/'src/peacsim',env=env,stdout=stream,stderr=subprocess.STDOUT)
                    except OSError as error:
                        stream.write(str(error))
                        class Failed:
                            def poll(self):return 127
                        proc=Failed()
                    record={'key':key,'rows':[r['plan_row_id'] for r in todo],'attempt':item['attempt'],
                            'command':cmd,'log':str(log),'started':time.time(),'pid':getattr(proc,'pid',None)}
                    attempt_count+=1;retry_count+=item['attempt']>1;active.append(dict(item,slot=slot,rows=todo,proc=proc,stream=stream,record=record))
                    if finished_at is not None:gaps.append(time.monotonic()-finished_at);finished_at=None
                    peak=max(peak,len(active));save('COLLECTING')
                if not active:break
                time.sleep(.1)
                for item in list(active):
                    code=item['proc'].poll()
                    if code is None:
                        if interrupted and time.monotonic()-interrupt_at>30:
                            try:item['proc'].terminate()
                            except (OSError,AttributeError):pass
                            if time.monotonic()-interrupt_at>35:
                                try:item['proc'].kill()
                                except (OSError,AttributeError):pass
                        continue
                    item['stream'].close();active.remove(item);finished_at=time.monotonic()
                    remaining=[]
                    for row in item['rows']:
                        try:
                            body=receipts.load(area,row['plan_row_id']);done.add(row['plan_row_id'])
                            events=area/'events';events.mkdir(exist_ok=True)
                            with (events/'parent.jsonl').open('a',encoding='utf8') as event_stream:
                                event_stream.write(json.dumps({'id':body['Plan_Row_ID'],'time':body['timestamp']})+'\n');event_stream.flush();os.fsync(event_stream.fileno())
                        except (ValueError,OSError,TypeError,KeyError):remaining.append(row)
                    record=item['record'];record.update(decode_returncode(code),ended=time.time(),remaining=[r['plan_row_id'] for r in remaining])
                    cache_write(work/(record['key']+'.attempt.json'),record)
                    # Failure splits remaining chunk into row-local retries, so a
                    # bad first row cannot starve the rest of its original chunk.
                    for row in remaining:
                        if item['attempt']<MAX_ATTEMPTS and not interrupted:
                            pending.appendleft({'rows':[row],'attempt':item['attempt']+1})
                        else:quarantine[row['plan_row_id']]=record
                    save('COLLECTING')
                    if stop_after is not None and len(done)-initial>=stop_after:stop(signal.SIGTERM,None)
                if interrupted:
                    # Workers finish their current row, then see interrupt file.
                    if not active:break
            status='INTERRUPTED' if interrupted else 'COLLECTED_WITH_QUARANTINE' if quarantine else 'COLLECTED'
            report=save(status);cache_write(area/'quarantine.json',quarantine)
            return report
    except BaseException:
        stop(signal.SIGTERM,None);save('INTERRUPTED_OR_GLOBAL_STOP');raise
    finally:
        # Bound shutdown; no completion is inferred from a dead process.
        deadline=time.monotonic()+30
        for item in active:
            remaining=max(0,deadline-time.monotonic())
            try:item['proc'].wait(timeout=remaining)
            except (subprocess.TimeoutExpired,AttributeError):
                try:item['proc'].terminate();item['proc'].wait(timeout=5)
                except (OSError,AttributeError,subprocess.TimeoutExpired):
                    try:item['proc'].kill()
                    except (OSError,AttributeError):pass
            item['stream'].close()
