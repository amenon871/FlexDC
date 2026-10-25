"""Foundation-only local resilient queue. No scientific acquisition logic.

Receipts, not exit codes or CSV presence, establish terminal scientific work.
A separate queue-local ledger has one writer; the campaign ingests all attempts.
"""
import csv
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections import deque,Counter
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path

from am_flexdc_foundation_v1_common import contexts,read_json,write_json,file_hash
from foundation_v1 import audit
from peacsim.foundation_v1_cleanup import (CleanupRequired,recover_output,TRANSIENT_FILES,verify_compact,parent_changed)

THREAD_DEFAULTS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','PYTHONFAULTHANDLER')}


def now():return datetime.now(timezone.utc).isoformat()


def worker_env(base):
    result=dict(base)
    for key,value in THREAD_DEFAULTS.items():result.setdefault(key,value)
    return result


def decode_returncode(code):
    number=-code if code<0 else None
    try:name=signal.Signals(number).name if number else None
    except ValueError:name={9:'SIGKILL',11:'SIGSEGV',15:'SIGTERM',2:'SIGINT'}.get(number,'SIGNAL_'+str(number))
    return {'return_code':code,'signal_number':number,'signal_name':name}


class CircuitBreaker:
    """5 DISTINCT logical workers, >=20% of last50 completions within300 seconds."""
    def __init__(self):self.events=deque(maxlen=50)
    def add(self,logical,signature,timestamp):
        self.events.append((timestamp,logical,signature))
        while self.events and timestamp-self.events[0][0]>300:self.events.popleft()
        if not signature:return False
        independent={key for _,key,s in self.events if s==signature}
        return len(independent)>=5 and len(independent)/len(self.events)>=.20


@contextmanager
def interrupts(callback):
    previous={}
    for sig in (signal.SIGINT,signal.SIGTERM):
        try:previous[sig]=signal.signal(sig,callback)
        except ValueError:pass # unit tests may use a non-main thread
    try:yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def call_combo(command,cwd):
    """Forward allocation shutdown to the queue; wait for its bounded drain."""
    proc=subprocess.Popen(command,cwd=cwd,env=dict(os.environ,FOUNDATION_V1_CAMPAIGN_PARENT_PID=str(os.getpid())))
    received=[]
    def stop(sig,frame):
        received.append(sig)
        if proc.poll() is None:
            try:proc.send_signal(sig)
            except (OSError,ValueError):proc.terminate()
    with interrupts(stop):
        code=proc.wait()
    return code


def read_rows(path):
    with Path(path).open(newline='',encoding='utf8') as stream:return list(csv.DictReader(stream))


def write_retry_plan(path,rows):
    with Path(path).open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())


class ReceiptBackend:
    def __init__(self,api,log_root):
        self.api=api;self.tasks={t['Task_ID']:t for t in contexts()}
        database=log_root/'terminal_receipts.sqlite'
        if database.exists():
            try:audit.verify_database(database)
            except (ValueError,sqlite3.Error) as error:raise ValueError('GLOBAL_FATAL_INVARIANT: queue ledger corruption: '+str(error)) from error
        self.db=audit.connection(database)
        self.done=audit.terminal_ids(self.db)
    def collect(self,item):
        pointer=item.get('output_receipt')
        if pointer and Path(pointer).exists():
            output=Path(read_json(pointer)['output']).resolve()
            base=(self.api.PEACSIM_DIR/'output/optimization'/Path(item['output_label']).parent).resolve()
            if output.parent!=base or not output.name.startswith(Path(item['output_label']).name+'_'):
                raise ValueError('GLOBAL_FATAL_INVARIANT: worker output pointer escapes expected identity')
        else:output=self.api.latest_plan_output(item['output_label'])
        if output is None:return None
        for path in (output/'foundation_evidence').glob('F1_*.json'):
            if path.stem not in item['row_map']:continue
            try:
                body=read_json(path);row=item['row_map'][path.stem]
                observation=audit.observation(row,body,self.tasks[row['Task_ID']])
            except (ValueError,KeyError,TypeError,OSError) as error:
                if any(word in str(error).lower() for word in ('mismatch','conflicting','differs')):
                    raise ValueError('GLOBAL_FATAL_INVARIANT: '+str(error)) from error
                continue # corrupt/missing receipt is not scientific completion
            audit.store_execution(self.db,observation,{'receipt':str(path),'receipt_sha256':file_hash(path)})
            self.done.add(path.stem)
        self.db.commit()
        try:recover_output(output)
        except CleanupRequired:raise
        except (ValueError,KeyError,OSError) as error:
            if any((output/n).exists() for n in TRANSIENT_FILES) and any((output/'foundation_evidence').glob('F1_*.json')):
                raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: terminal evidence with unresolved raw artifacts: '+str(error)) from error
            # Local malformed unreceipted state is preserved for diagnosis. A
            # fresh retry directory can run only remaining nonterminal rows.
        return output
    def terminal(self,ids):return set(ids)&self.done
    def close(self):self.db.close()


def run(args,invocations,chunk_root,log_root,api,backend=None,launch=None,clock=None,sleep=None):
    """Execute one attempt's bounded queue. Injection points are for synthetic tests."""
    clock=clock or time.monotonic;sleep=sleep or time.sleep
    log_root.mkdir(parents=True,exist_ok=True);chunk_root.mkdir(parents=True,exist_ok=True)
    max_attempts=getattr(args,'foundation_max_technical_attempts',3)
    if max_attempts<1:raise ValueError('technical attempt count must be positive')
    backend=backend or ReceiptBackend(api,log_root)
    launch=launch or subprocess.Popen
    cleanup_stop=log_root/'foundation_cleanup_stop.json'
    interrupt_stop=log_root/'foundation_interruption_stop.json'
    integrity_stop=log_root/'foundation_global_integrity_stop.json'
    systemic_stop=log_root/'foundation_systemic_technical_stop.json'
    summary_path=log_root/'scheduler_summary.json'
    manifest=api.REPO_ROOT/f'dataset_plan_launch_manifest_{args.policy_name}_{chunk_root.name}.csv'
    env=worker_env(api.build_env());env['FOUNDATION_V1_STORAGE_STOP']=str(cleanup_stop.resolve())
    env['FOUNDATION_V1_INTERRUPTION_STOP']=str(interrupt_stop.resolve())
    if summary_path.exists():
        previous=log_root/'invocation_history';previous.mkdir(exist_ok=True)
        write_json(previous/(str(time.time_ns())+'.json'),read_json(summary_path),False)
    records=[];quarantine=[];running=[];pending=deque();breaker=CircuitBreaker()
    start=clock();slot_seconds=0.;last=start;peak=0;interrupt_at=None;fatal=None
    # A new campaign attempt has a new queue identity. Direct same-identity
    # resume is reconstructed from attempt receipts, never a cached COMPLETE.
    attempt_root=log_root/'attempt_receipts';attempt_root.mkdir(exist_ok=True)
    historical=[read_json(p) for p in sorted(attempt_root.glob('*.json'))]
    logicals={}
    for i,item in enumerate(invocations):
        rows=read_rows(item['chunk_file']);row_map={r['plan_row_id']:r for r in rows}
        if set(row_map)!=set(item['expected_plan_ids']):raise ValueError('GLOBAL_FATAL_INVARIANT: logical chunk plan IDs differ')
        key=str(i);work=dict(item,logical=key,row_map=row_map,attempt=0,original_rows=rows)
        for record in historical:
            if record['logical_chunk']!=key:continue
            old=dict(work,output_label=record['output_label'],output_receipt=record.get('output_receipt'))
            backend.collect(old)
            work['attempt']=max(work['attempt'],record['attempt'])
        logicals[key]=work
        remaining=set(row_map)-backend.terminal(row_map)
        if remaining and work['attempt']>=max_attempts:
            quarantine.append({'logical_chunk':key,'remaining_row_ids':sorted(remaining),'status':'TECHNICAL_RETRY_EXHAUSTED'})
        elif remaining:pending.append(work)
    records.extend(historical)
    all_ids={r for w in logicals.values() for r in w['row_map']}
    terminal_total=len(backend.terminal(all_ids))
    def persist(status):
        result={'status':status,'attempt_count':len(records),'attempt_receipts':str(attempt_root),'quarantine':quarantine,'max_technical_attempts':max_attempts,
            'max_workers':args.plan_max_workers,'worker_environment':{k:env[k] for k in THREAD_DEFAULTS},
            'worker_slot_seconds':slot_seconds,'wall_seconds':max(0,clock()-start),'peak_workers':peak,
            'terminal_rows':terminal_total,
            'updated_at':now(),'host':socket.gethostname()}
        write_json(summary_path,result,False)
        return result
    def record_attempt(record,finished=False):
        write_json(attempt_root/('logical_'+record['logical_chunk']+'_attempt'+str(record['attempt'])+'.json'),record,False)
        if finished:
            # Append one compact physical-attempt row; never rewrite a dataset-wide manifest.
            columns=('logical_chunk','attempt','output_label','status','return_code','signal_number','signal_name','log_file','started_at','ended_at','wall_seconds','terminal_rows','remaining_row_ids','output_dir','command')
            exists=manifest.exists()
            with manifest.open('a',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
                if not exists:writer.writeheader()
                writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in record.items()});stream.flush();os.fsync(stream.fileno())
    def request_stop(sig,frame):
        nonlocal interrupt_at
        if interrupt_at is None:interrupt_at=clock()
        write_json(interrupt_stop,{'status':'INTERRUPTED','signal':decode_returncode(-sig),'time':now()},False)
    def start_one(work):
        nonlocal peak
        completed=backend.terminal(work['row_map'])
        remaining=[r for r in work['original_rows'] if r['plan_row_id'] not in completed]
        if not remaining:return
        work=dict(work,attempt=work['attempt']+1)
        label=work['output_label']+'__attempt'+str(work['attempt']).zfill(2)
        plan=chunk_root/('logical_'+work['logical']+'_attempt'+str(work['attempt'])+'.csv')
        write_retry_plan(plan,remaining)
        command=list(work['command']);command[command.index('--sweep-plan-file')+1]=str(plan)
        command[command.index('--output-dir')+1]=label
        logfile=log_root/('logical_'+work['logical']+'_attempt'+str(work['attempt'])+'.log')
        log=logfile.open('w',encoding='utf8')
        log.write('FOUNDATION_LAUNCH '+json.dumps({'logical_chunk':work['logical'],'attempt':work['attempt'],
            'environment':{k:env[k] for k in THREAD_DEFAULTS},'host':socket.gethostname(),'time':now()})+'\n');log.flush()
        output_receipt=log_root/('output_'+work['logical']+'_attempt'+str(work['attempt'])+'.json')
        attempt_env=dict(env,FOUNDATION_V1_WORKER_OUTPUT_RECEIPT=str(output_receipt.resolve()),FOUNDATION_V1_PARENT_PID=str(os.getpid()))
        kwargs={'cwd':api.PEACSIM_DIR,'env':attempt_env,'stdout':log,'stderr':subprocess.STDOUT,'text':True}
        if os.name!='nt':kwargs['start_new_session']=True
        try:proc=launch(command,**kwargs)
        except OSError as error:
            class Dead:
                def poll(self):return 127
            proc=Dead();log.write('LAUNCH_ERROR '+str(error)+'\n');log.flush()
        record={'logical_chunk':work['logical'],'attempt':work['attempt'],'output_label':label,'command':command,
            'log_file':str(logfile),'output_receipt':str(output_receipt),'expected_plan_ids':[r['plan_row_id'] for r in remaining],
            'status':'RUNNING','started_at':now(),'host':socket.gethostname(),'plan_sha256':file_hash(plan)}
        records.append(record);record_attempt(record)
        running.append(dict(work,output_receipt=str(output_receipt),output_label=label,proc=proc,log=log,record=record,started=clock(),plan=plan))
        peak=max(peak,len(running));persist('RUNNING')
    try:
        if cleanup_stop.exists():
            raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: resume through campaign after cleanup recovery')
        # Interrupt/systemic/integrity markers belong to the OLD invocation;
        # retain history but archive stale marker so fresh resume can drain.
        for marker in (interrupt_stop,systemic_stop):
            if marker.exists():os.replace(marker,marker.with_suffix('.previous_'+str(time.time_ns())+'.json'))
        if integrity_stop.exists():raise ValueError('GLOBAL_FATAL_INVARIANT: unresolved integrity marker')
        with interrupts(request_stop):
            while pending or running:
                current=clock();slot_seconds+=(current-last)*len(running);last=current
                campaign_parent=os.environ.get('FOUNDATION_V1_CAMPAIGN_PARENT_PID')
                if parent_changed(campaign_parent) and interrupt_at is None:
                    request_stop(signal.SIGTERM,None)
                if cleanup_stop.exists():fatal='TRANSIENT_CLEANUP_REQUIRED'
                if integrity_stop.exists():fatal='GLOBAL_FATAL_INVARIANT'
                while pending and len(running)<args.plan_max_workers and not fatal and interrupt_at is None:
                    start_one(pending.popleft())
                if not running:break
                sleep(.2)
                finished=[]
                for item in list(running):
                    code=item['proc'].poll()
                    if code is None:
                        if interrupt_at is not None and clock()-interrupt_at>=30:
                            try:item['proc'].terminate()
                            except (OSError,AttributeError):pass
                            if clock()-interrupt_at>=35:
                                try:item['proc'].kill()
                                except (OSError,AttributeError):pass
                        continue
                    item['log'].close();finished.append(item)
                    record=item['record'];record.update(decode_returncode(code),ended_at=now(),wall_seconds=clock()-item['started'])
                    before=backend.terminal(item['row_map'])
                    try:
                        if file_hash(item['plan'])!=record['plan_sha256']:raise ValueError('GLOBAL_FATAL_INVARIANT: launched plan changed')
                        output=backend.collect(item)
                    except CleanupRequired as error:
                        fatal='TRANSIENT_CLEANUP_REQUIRED';record['error']=str(error)
                        write_json(cleanup_stop,{'status':fatal,'error':str(error),'time':now()},False);output=None
                    except (ValueError,OSError,KeyError,TypeError) as error:
                        record['error']=str(error);output=None
                        if 'GLOBAL_FATAL_INVARIANT' in str(error):
                            fatal='GLOBAL_FATAL_INVARIANT'
                            write_json(integrity_stop,{'status':fatal,'error':str(error),'time':now()},False)
                    terminal=backend.terminal(item['row_map']);terminal_total+=len(terminal-before);remaining=set(item['row_map'])-terminal
                    record.update(terminal_rows=len(terminal),remaining_row_ids=sorted(remaining),output_dir=str(output) if output else '')
                    if fatal:record['status']=fatal
                    elif interrupt_at is not None:record['status']='INTERRUPTED'
                    elif not remaining:record['status']='SUCCESS' if code==0 else 'SCIENTIFIC_TERMINAL'
                    else:
                        record['status']='TECHNICAL_RETRY_REQUIRED'
                        signature=record['signal_name'] or ('EXIT_'+str(code) if code else 'INCOMPLETE_RECEIPTS')
                        record['technical_signature']=signature
                        if breaker.add(item['logical'],signature,clock()):
                            fatal='SYSTEMIC_TECHNICAL_FAILURE'
                            write_json(systemic_stop,{'status':fatal,'signature':signature,'window_seconds':300,'events':list(breaker.events)},False)
                        if item['attempt']<max_attempts:
                            retry=dict(logicals[item['logical']],attempt=item['attempt']);pending.append(retry)
                        else:
                            quarantine.append({'logical_chunk':item['logical'],'remaining_row_ids':sorted(remaining),
                                'status':'TECHNICAL_RETRY_EXHAUSTED','attempts':item['attempt']})
                    if record['status']=='SCIENTIFIC_TERMINAL' and code and not fatal:
                        signature=record['signal_name'] or 'EXIT_'+str(code)
                        record['technical_signature']=signature
                        if breaker.add(item['logical'],signature,clock()):
                            fatal='SYSTEMIC_TECHNICAL_FAILURE'
                            write_json(systemic_stop,{'status':fatal,'signature':signature,'events':list(breaker.events)},False)
                    elif record['status']=='SUCCESS':breaker.add(item['logical'],None,clock())
                    print('Foundation worker '+item['logical']+' attempt '+str(item['attempt'])+': '+record['status']+' terminal='+str(len(terminal))+' remaining='+str(len(remaining)),flush=True)
                for item in finished:
                    record_attempt(item['record'],finished=True);running.remove(item)
                if finished:persist(fatal or ('INTERRUPTED' if interrupt_at is not None else 'RUNNING'))
                if (fatal or interrupt_at is not None) and not running:break
        status=fatal or ('INTERRUPTED' if interrupt_at is not None else 'STAGE_COMPLETE_WITH_TECHNICAL_SHORTFALL' if quarantine or pending else 'VALID_COMPLETE')
        result=persist(status)
        if status!='VALID_COMPLETE':raise SystemExit(status+': '+str(summary_path))
        return result
    finally:
        # Parent exceptions must not leave unbounded orphan workers collecting.
        for item in running:
            try:item['proc'].terminate()
            except (OSError,AttributeError):pass
            item['log'].close()
        backend.close()
