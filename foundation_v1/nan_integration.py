"""Explicit isolated TEST FlexDC runs for the SCC NaN regression; never a campaign.

No observation is ingested into Foundation production ledgers. The six-row mini
plan is a view of immutable existing rows. Canonical configs/duration are kept.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,contexts,iter_csv,write_csv,
    write_json,read_json,file_hash,make_row,row_geometry)
from foundation_v1.campaign import commands
from foundation_v1.audit import observation
from peacsim.foundation_v1_cleanup import TRANSIENT_FILES
from foundation_v1.full_campaign import verify_release,MASTER_SHA256

OUT=ARTIFACTS/'scc_nan_validation'
SCRATCH=ROOT/'.flexdc_plan_chunks/foundation_nan_test_only_20261023'


def selected_rows():
    tasks=contexts()
    a=next(t for t in tasks if t['Workload_ID']=='F1_DEVELOPMENT_A100_J2_039' and t['domain']['server_count']==250 and t['domain']['utilization']==.6)
    b=next(t for t in tasks if t['Workload_ID']==a['Workload_ID'] and t['domain']['server_count']==1000 and t['domain']['utilization']==.8)
    c=next(t for t in tasks if t['hardware_class']=='P100' and t['domain']['job_type_count']==6 and t['domain']['server_count']==1000 and t['domain']['utilization']==.8)
    chosen=[a,b,c]
    for j in (3,4,5):
        chosen.append(next(t for t in tasks if t['domain']['job_type_count']==j and t['hardware_class']==('A100' if j==4 else 'P100') and t['domain']['server_count']==250 and t['domain']['utilization']==.6))
    wanted={t['Task_ID'] for t in chosen};found={}
    for row in iter_csv(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'):
        if row['Task_ID'] in wanted and row['Task_ID'] not in found:found[row['Task_ID']]=row
        if len(found)==len(wanted):break
    assert set(found)==wanted
    return [found[t['Task_ID']] for t in chosen],{t['Task_ID']:t for t in chosen}


def strict_load(path):
    def reject(token):raise ValueError('Non-strict JSON token '+token+' in '+str(path))
    return json.loads(path.read_text(encoding='utf8'),parse_constant=reject)


def run_case(name,rows,tasks,workers):
    SCRATCH.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
    plan=SCRATCH/(name+'.csv.gz');write_csv(plan,rows)
    run_id='foundation_nan_test_only_20261023_'+name
    manifest=ROOT/('dataset_plan_launch_manifest_AQA_'+run_id+'.csv')
    if manifest.exists():raise FileExistsError('Refusing to rerun completed/partial TEST case automatically: '+str(manifest))
    command=commands(plan,workers,run_id,'TRAIN_DEV')
    command[command.index('--output-prefix')+1]='foundation_nan_test_only_20261023_'+name
    command[command.index('--plan-num-chunks-per-context')+1]='1'
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1')
    log=OUT/(name+'_real_combo.log')
    print('REAL TEST START '+name+' rows='+str(len(rows))+' workers='+str(workers),flush=True)
    with log.open('w',encoding='utf8') as stream:
        result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Real TEST failed; no automatic retry: '+str(log))
    with manifest.open(newline='') as stream:launch=list(csv.DictReader(stream))
    assert all(r['status']=='completed' and int(r['return_code'])==0 for r in launch)
    outputs={};summaries=[]
    for launch_row in launch:
        output=Path(launch_row['output_dir']);assert output.is_dir()
        assert not any((output/name).exists() for name in TRANSIENT_FILES)
        for receipt_path in (output/'foundation_evidence').glob('F1_*.json'):
            receipt=strict_load(receipt_path);outputs[receipt['Plan_Row_ID']]=(receipt,output)
    assert set(outputs)=={r['plan_row_id'] for r in rows}
    for row in rows:
        receipt,output=outputs[row['plan_row_id']]
        obs=observation(row,receipt,tasks[row['Task_ID']])
        cleanup=strict_load(output/'foundation_cleanup'/(row['plan_row_id']+'.json'))
        assert cleanup['cleanup_status']=='CLEANUP_COMPLETE'
        expected_parent=row['replicate_of_plan_row_id']
        if expected_parent:
            assert receipt['results']['Replicate_Of_Plan_Row_ID']==expected_parent
            assert receipt['diagnostics']['Replicate_Of_Plan_Row_ID']==expected_parent
        else:
            assert receipt['results']['Replicate_Of_Plan_Row_ID']==''
            assert receipt['diagnostics']['Replicate_Of_Plan_Row_ID']==''
        summaries.append({'Plan_Row_ID':row['plan_row_id'],'Task_ID':row['Task_ID'],
            'test_only':True,'J':receipt['applied']['J'],'N':receipt['applied']['N'],'U':receipt['applied']['U'],
            'seed':receipt['applied']['seed'],'hardware':tasks[row['Task_ID']]['hardware_class'],
            'parent':expected_parent,'p90':obs['p90'],'mean_tracking':obs['mean_tracking'],'Pj':obs['Pj'],
            'objective':obs['objective'],'observation_status':obs['observation_status'],
            'worker_returncode':0,'strict_json':'PASS','cleanup':'PASS','output':str(output),
            'reviewed_source_identity':receipt['runtime_provenance']['reviewed_source_identity']})
    stop=ROOT/'.flexdc_worker_logs'/('plan_'+run_id)/'foundation_storage_stop.json'
    assert not stop.exists()
    report={'status':'PASS','label':'TEST_ONLY_NOT_FOUNDATION_CORPUS','rows':summaries,
        'workers':workers,'real_simulator_rows':len(rows),'global_storage_stop':False,'failed_process':False,
        'campaign_ledger_used':False,'plan_sha256':file_hash(plan),'command':command}
    write_json(OUT/(name+'_real_results.json'),report,False)
    print('REAL TEST PASS '+name,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['individuals','mini','repeat','mini-final','repeat-final'],required=True)
    a=p.parse_args();verify_release();rows,tasks=selected_rows()
    if a.case=='individuals':
        for i in range(3):run_case('individual_'+chr(65+i),[rows[i]],tasks,1)
    elif a.case in ('mini','mini-final'):run_case('mini_parallel_final' if a.case=='mini-final' else 'mini_parallel',rows,tasks,3)
    else:
        base=rows[0];repeat=make_row(tasks[base['Task_ID']],row_geometry(base),base['Candidate_Source'],int(base['Adaptive_Wave']),simulation_seed=21,base=base['plan_row_id'])
        run_case('repeat_seed21_final' if a.case=='repeat-final' else 'repeat_seed21',[repeat],tasks,1)
    assert file_hash(ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz')==MASTER_SHA256


if __name__=='__main__':main()
