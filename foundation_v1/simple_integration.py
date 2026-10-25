"""Real isolated validation of the production simple collector, never the corpus."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,contexts,iter_csv,write_csv,write_json,read_json,file_hash
from foundation_v1 import simple_collector as c,simple_campaign as campaign,collector_receipts as receipts
from foundation_v1.full_campaign import verify_release,MASTER_SHA256
from peacsim.foundation_v1_cleanup import TRANSIENT_FILES

BASE=ARTIFACTS/'simple_collector_validation'


def test_plan():
    tasks=contexts();chosen=[next(t for t in tasks if t['hardware_class']=='A100' and t['domain']['job_type_count']==2 and t['domain']['server_count']==250 and t['domain']['utilization']==.6),
        next(t for t in tasks if t['hardware_class']=='A100' and t['domain']['job_type_count']==4 and t['domain']['server_count']==1000 and t['domain']['utilization']==.8),
        next(t for t in tasks if t['hardware_class']=='P100' and t['domain']['job_type_count']==6 and t['domain']['server_count']==1000 and t['domain']['utilization']==.8)]
    # Include a real immutable identity-rounding-edge row's context.
    edge='F1_e72484ba822d2dc2993b7f85a945f814'
    pool={t['Task_ID']:[] for t in tasks}
    edge_row=None
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if len(pool[row['Task_ID']])<8:pool[row['Task_ID']].append(row)
        if row['plan_row_id']==edge:edge_row=row
    task=next(t for t in tasks if t['Task_ID']==edge_row['Task_ID']);chosen.append(task)
    if edge not in {r['plan_row_id'] for r in pool[task['Task_ID']]}:pool[task['Task_ID']][-1]=edge_row
    for task in tasks:
        if task['Task_ID'] not in {t['Task_ID'] for t in chosen}:chosen.append(task)
        if len(chosen)==32:break
    return chosen,pool


def run(case,tag,rows_per_context=8):
    release=verify_release();chosen,pool=test_plan()
    area=BASE/'runtime'/(case+'_'+tag)/'phaseA'
    if area.exists():raise FileExistsError('Do not silently rerun isolated tests: '+str(area))
    plan=BASE/'runtime'/(case+'_'+tag)/'test_plan.csv.gz'
    rows=[pool[t['Task_ID']][0] for t in chosen[:3]] if case=='integrations' else [r for t in chosen for r in pool[t['Task_ID']][:rows_per_context]]
    edge='F1_e72484ba822d2dc2993b7f85a945f814'
    if case!='integrations' and edge not in {r['plan_row_id'] for r in rows}:
        edge_row=next(r for values in pool.values() for r in values if r['plan_row_id']==edge)
        index=next(i for i,r in enumerate(rows) if r['Task_ID']==edge_row['Task_ID']);rows[index]=edge_row
    write_csv(plan,rows);real=subprocess.Popen;injected=[];cleanup_injected=[]
    def launch(cmd,**kwargs):
        if case=='fault' and not injected:
            injected.append(True)
            class Failed:
                def poll(self):return -11
                def terminate(self):pass
            kwargs['stdout'].write('TEST_ONLY_SAFE_BOUNDARY_SIGSEGV\n');return Failed()
        if case=='fault' and not cleanup_injected:
            cleanup_injected.append(True)
            # Actual wizard/simulator; only unlink is mocked for one physical
            # worker. Completion remains durable; following rows get fresh dirs.
            wrapper=("import runpy,sys; from peacsim import foundation_v1_cleanup as c; "
                "c._unlink=lambda path: (_ for _ in ()).throw(PermissionError('TEST_ONLY_UNLINK_DEBT')); "
                "sys.argv=['am_data_extraction_wizard.py']+sys.argv[1:]; "
                "runpy.run_path('am_data_extraction_wizard.py',run_name='__main__')")
            cmd=[cmd[0],'-u','-B','-c',wrapper,*cmd[4:]]
        return real(cmd,**kwargs)
    start=time.monotonic();workers=2 if case=='integrations' else 8
    result=c.collect(plan,area,workers,0,launch,chunk_rows=1 if case=='integrations' else 8)
    elapsed=time.monotonic()-start
    assert result['completed']==len(rows),result
    debt_outputs=set()
    for row in rows:
        body=receipts.load(area,row['plan_row_id']);output=Path(body['result_file']).parent
        if any((output/name).exists() for name in TRANSIENT_FILES):debt_outputs.add(str(output))
    if case=='fault':assert len(debt_outputs)==1
    else:assert not debt_outputs
    if case=='fault':
        # Real one-row forensic discrepancy after ALL unrelated collection, then
        # real repair of only that same immutable row, never a changed geometry.
        key=rows[-1]['plan_row_id'];body=receipts.load(area,key);body['Base_Geometry_ID']='TEST_ONLY_BAD_RECEIPT_GEOMETRY';body.pop('checksum')
        from am_flexdc_foundation_v1_common import digest
        body['checksum']=digest(body);write_json(receipts.receipt_path(area,key),body,False)
        first=campaign.certify(plan,area,BASE/'runtime'/(case+'_'+tag)/'derived.sqlite')
        assert set(first['QUARANTINED'])=={key} and first['PASS']==len(rows)-1
        receipts.archive(area,key,first['QUARANTINED'][key])
        repair=c.collect(plan,area,workers,0,launch,chunk_rows=8)
        assert repair['new_completions']==1
    audit=campaign.certify(plan,area,BASE/'runtime'/(case+'_'+tag)/'derived.sqlite')
    assert audit['status']=='PASS',audit
    completed=receipts.completed(area);outputs=set()
    for key in completed:
        body=receipts.load(area,key);outputs.add(Path(body['result_file']).parent)
        assert body['evidence']['runtime_provenance']['reviewed_source_identity']==release['uncommitted_source_sha256']
    debts=[]
    for output in outputs:
        present=[name for name in TRANSIENT_FILES if (output/name).exists()]
        if present:debts.append({'output':str(output),'files':present})
    assert not debts, 'Stage-boundary best-effort cleanup should settle the injected debt'
    # Retry only local cleanup debt once the injected permission problem is gone.
    if debts:
        from peacsim.foundation_v1_cleanup import finalize_row
        with patch.dict(os.environ,{'FOUNDATION_V1_SIMPLE_AREA':str(area)}):
            for debt in debts:
                output=Path(debt['output']);keys=[p.stem for p in (output/'foundation_evidence').glob('F1_*.json')]
                for key in keys:finalize_row(output,key)
        assert all(not (output/name).exists() for output in outputs for name in TRANSIENT_FILES)
    def forbidden(*args,**kwargs):raise AssertionError('Completed test row resimulated')
    again=c.collect(plan,area,workers,0,forbidden)
    assert again['new_completions']==0
    attempts=[read_json(p) for p in (area/'workers').glob('*.attempt.json')]
    measured={'status':'PASS','case':case,'test_only':True,'real_rows':len(rows)+(1 if case=='fault' else 0),
        'original_rows':len(rows),'workers':workers,'logical_chunks':3 if case=='integrations' else 32,
        'collection_seconds':elapsed,'rows_per_second':len(rows)/elapsed,
        'effective_worker_seconds_per_row':result['worker_slot_seconds']/len(rows),
        'slot_utilization':result['worker_slot_seconds']/(elapsed*workers),
        'peak_workers':result['peak_workers'],'replacement_gaps':result['replacement_gaps'],
        'attempts':len(attempts),'sigsegv_attempts':sum(a.get('return_code')==-11 for a in attempts),
        'cleanup_debts':len(debt_outputs),'audit_repair_rows':1 if case=='fault' else 0,
        'audit':'PASS','resume_zero_simulations':True,'source_identity':release['uncommitted_source_sha256'],
        'master_sha256':file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')}
    assert measured['master_sha256']==MASTER_SHA256
    write_json(BASE/(case+'_'+tag+'_real_results.json'),measured,False)
    print(json.dumps(measured,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True,choices=['integrations','soak','fault']);parser.add_argument('--tag',required=True);parser.add_argument('--rows-per-context',type=int,choices=[4,8],default=8);args=parser.parse_args();run(args.case,args.tag,args.rows_per_context)
