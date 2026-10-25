from tests.test_foundation_v1_legacy_entry_fixture import invoke as legacy_stage_entry
"""SCC missing-provenance regression; no simulator is used by this unit file."""
import ast
import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from am_flexdc_foundation_v1_common import ROOT, contexts, read_json
from tests.test_foundation_v1 import fake_receipt
from peacsim.foundation_v1_evidence import write_evidence


def provenance(row):
    tree = ast.parse((ROOT / 'src/peacsim/am_data_extraction_wizard.py').read_text(encoding='utf8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_provenance_values')
    columns = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PLAN_PROVENANCE_COLUMNS' for t in n.targets))
    namespace = {}
    exec(compile(ast.Module(body=[columns, function], type_ignores=[]), '<real provenance function>', 'exec'), namespace)
    return dict(zip(namespace['PLAN_PROVENANCE_COLUMNS'], namespace['plan_provenance_values'](row)))


def evidence_fixture(output, monkeypatch, changes=None, result_changes=None, diagnostic_changes=None):
    task = contexts()[0]
    row, receipt = fake_receipt(task)
    row.update(design_version='FOUNDATION_V1', is_replicate=False, replicate_of_plan_row_id=np.nan)
    row.update(changes or {})
    values = provenance(row)
    receipt['results'].update(values)
    receipt['diagnostics'].update(values)
    receipt['results'].update(result_changes or {})
    receipt['diagnostics'].update(diagnostic_changes or {})
    applied = receipt['applied']; j = applied['J']
    table = pd.DataFrame({'job_type_id': list(range(j)), 'arrival_time': [10.]*j, 'end_time': [20.]*j})
    simulator = SimpleNamespace(_job_table=table, _runtime_policy=SimpleNamespace(P=applied['P_watts'], R=applied['R_watts'], weights=dict(enumerate(applied['weights'])), _policy_parameters={'node_count_control': True}))
    experiment = SimpleNamespace(server_count=applied['N'], utilization=applied['U'], random_seed=row['simulation_seed'], simulation_duration=3600, idle_power=120., iso_file_path=str(ROOT/'data/1-17-2025_traditional.csv'), iso_signal_start_hour=16, iso_signal_granularity=2, normalize_iso_signal=True, randomize_iso_start=False)
    job = SimpleNamespace(job_type_count=j, all_min_execution_time={i:1. for i in range(j)}, all_jobs={i:str(i) for i in range(j)}, workload_mix_name=str(ROOT/task['domain']['workload_config']))
    monkeypatch.setattr('peacsim.foundation_v1_evidence.runtime_provenance', lambda: {'execution_git_SHA':'SYNTHETIC_SCC_REGRESSION'})
    for name, key in [('grid_search_results.csv','results'), ('grid_search_diagnostics.csv','diagnostics')]:
        with (output/name).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(receipt[key]);writer.writerow(receipt[key].values())
    return row, receipt, values, simulator, experiment, job


def emit(output, fixture):
    row, receipt, _, simulator, experiment, job = fixture
    write_evidence(output,row,simulator,experiment,job,[row['Pbar_kw_per_server'],row['R_kw_per_server']],list(receipt['results'].values()),list(receipt['diagnostics'].values()))
    path=output/'foundation_evidence'/(row['plan_row_id']+'.json')
    text=path.read_text(encoding='utf8')
    def reject(token): raise AssertionError('Non-strict JSON token: '+token)
    body=json.loads(text,parse_constant=reject)
    return body


def test_exact_scc_nonrepeat_nan_parent_regression(tmp_path,monkeypatch):
    fixture=evidence_fixture(tmp_path,monkeypatch)
    body=emit(tmp_path,fixture)
    assert fixture[2]['Replicate_Of_Plan_Row_ID']==''
    assert body['results']['Replicate_Of_Plan_Row_ID']==''
    assert body['diagnostics']['Replicate_Of_Plan_Row_ID']==''
    assert body['plan_metadata']['replicate_of_plan_row_id'] in ('',None)


from peacsim.foundation_v1_schema import (PROVENANCE_KEYS,OPTIONAL_PROVENANCE,REQUIRED_PROVENANCE,
    normalize_metadata,json_safe,InvalidFoundationObservation)


@pytest.mark.parametrize('column',sorted(OPTIONAL_PROVENANCE))
@pytest.mark.parametrize('missing',[None,np.nan,pd.NA,pd.NaT,'','nan','none','   '])
def test_every_optional_provenance_missing_value_is_safe(tmp_path,monkeypatch,column,missing):
    fixture=evidence_fixture(tmp_path,monkeypatch,{PROVENANCE_KEYS[column]:missing})
    body=emit(tmp_path,fixture)
    assert fixture[2][column]==''
    assert body['results'][column]=='' and body['diagnostics'][column]==''
    with (tmp_path/'grid_search_results.csv').open(newline='') as stream:
        assert next(csv.DictReader(stream))[column]==''


@pytest.mark.parametrize('column',sorted(REQUIRED_PROVENANCE))
@pytest.mark.parametrize('missing',[None,np.nan,pd.NA,'','nan'])
def test_required_provenance_never_becomes_blank(column,missing):
    row,_=fake_receipt(contexts()[0])
    row[PROVENANCE_KEYS[column]]=missing
    with pytest.raises(ValueError,match=column): normalize_metadata(row)


def test_repeat_parent_survives_both_csvs_and_receipt(tmp_path,monkeypatch):
    parent='F1_0123456789abcdef0123456789abcdef'
    fixture=evidence_fixture(tmp_path,monkeypatch,{'is_replicate':True,'replicate_of_plan_row_id':parent,'simulation_seed':21})
    body=emit(tmp_path,fixture)
    assert body['plan_metadata']['replicate_of_plan_row_id']==parent
    assert body['results']['Replicate_Of_Plan_Row_ID']==parent
    assert body['diagnostics']['Replicate_Of_Plan_Row_ID']==parent
    assert body['plan_metadata']['is_replicate'] is True
    for name in ('grid_search_results.csv','grid_search_diagnostics.csv'):
        with (tmp_path/name).open(newline='') as stream:
            assert next(csv.DictReader(stream))['Replicate_Of_Plan_Row_ID']==parent


@pytest.mark.parametrize('field',['Mtrack_Error_MeanAbs_Normalized','Simulator_RSR_Total_Cost'])
@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_required_result_nonfinite_has_named_error(tmp_path,monkeypatch,field,value):
    fixture=evidence_fixture(tmp_path,monkeypatch,result_changes={field:value})
    with pytest.raises(InvalidFoundationObservation,match='results.'+field): emit(tmp_path,fixture)
    assert not (tmp_path/'foundation_evidence').exists()


@pytest.mark.parametrize('field',['Ctrack_Epsilon_90th','Diagnostic_FullPaperObjective_Cost'])
@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_required_diagnostic_nonfinite_has_named_error(tmp_path,monkeypatch,field,value):
    fixture=evidence_fixture(tmp_path,monkeypatch,diagnostic_changes={field:value})
    with pytest.raises(InvalidFoundationObservation,match='diagnostics.'+field): emit(tmp_path,fixture)


def test_numpy_nested_optional_metadata_is_strict_json():
    safe=json_safe({'a':np.int64(3),'b':(np.float64(.5),pd.NA,np.nan,np.inf,pd.NaT),
        'nested':{'x':np.bool_(True)}},'metadata',optional=True)
    assert safe=={'a':3,'b':[.5,None,None,None,None],'nested':{'x':True}}
    json.dumps(safe,allow_nan=False)


def test_defensive_writer_catches_optional_raw_record_nan(tmp_path,monkeypatch):
    fixture=list(evidence_fixture(tmp_path,monkeypatch))
    fixture[1]['results']['Replicate_Of_Plan_Row_ID']=np.nan
    fixture[1]['diagnostics']['Replicate_Of_Plan_Row_ID']=pd.NA
    body=emit(tmp_path,fixture)
    assert body['results']['Replicate_Of_Plan_Row_ID']==''
    assert body['diagnostics']['Replicate_Of_Plan_Row_ID']==''


def test_required_qos_nonfinite_names_vector(tmp_path,monkeypatch):
    fixture=evidence_fixture(tmp_path,monkeypatch,result_changes={'QoS_Delay_Probabilities':'[NaN] '})
    with pytest.raises(InvalidFoundationObservation,match='QoS_Delay_Probabilities'):emit(tmp_path,fixture)


@pytest.mark.parametrize('key',['Pbar_kw_per_server','R_kw_per_server','server_count','utilization','job_type_count'])
def test_required_scientific_metadata_not_normalized_to_missing(key):
    row,_=fake_receipt(contexts()[0]);row[key]=np.nan
    with pytest.raises(ValueError,match=key):normalize_metadata(row)


def test_resume_exact_scc_csv_without_receipt_uses_new_attempt(tmp_path,monkeypatch):
    from foundation_v1 import audit,campaign,scope
    from peacsim import foundation_v1_cleanup as cleanup
    from tests.test_foundation_v1_patch import prepare_fake_campaign
    from am_flexdc_foundation_v1_common import write_json
    import sqlite3
    task=contexts()[0];row,receipt=fake_receipt(task)
    plan,db=prepare_fake_campaign(task,tmp_path,monkeypatch,row)
    old=scope.raw_root('TRAIN_DEV')/'foundation_v1_old_SCC_attempt'
    old.mkdir(parents=True)
    cleanup.begin_row(old,{**row,'design_version':'FOUNDATION_V1'})
    for name in cleanup.TRANSIENT_FILES:(old/name).write_text('PRESERVED FAILED ATTEMPT DEBUG')
    for name,key in [('grid_search_results.csv','results'),('grid_search_diagnostics.csv','diagnostics')]:
        record={**receipt[key],'Replicate_Of_Plan_Row_ID':float('nan')}
        with (old/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=record);writer.writeheader();writer.writerow(record)
    stale=tmp_path/'logs/old_attempt/foundation_storage_stop.json'
    write_json(stale,{'status':'FOUNDATION_WORKER_STOP','old_attempt':True})
    conn=audit.connection(db)
    with conn:conn.execute('INSERT INTO attempts VALUES(?,?,?)',('old_SCC_attempt0001','TECHNICAL_RETRY_REQUIRED','{}'))
    conn.close()
    assert audit.ingest(plan,[old],db)['counts']['EXECUTION_COMPLETE']==0
    assert campaign.pending_rows(plan,db)[0]['plan_row_id']==row['plan_row_id']
    launches=[]
    def execute(command,**kwargs):
        # Simulate the new successful attempt, not a cached old CSV-only result.
        launches.append(command)
        assert command[command.index('--plan-run-id')+1].endswith('attempt0002')
        output=scope.raw_root('TRAIN_DEV')/'foundation_v1_new_attempt'
        output.mkdir(parents=True)
        write_json(output/'foundation_evidence'/(row['plan_row_id']+'.json'),receipt)
        assert stale.exists()  # historical marker does not need destructive removal
        return 0
    monkeypatch.setattr(campaign,'call_combo',execute)
    monkeypatch.setattr('sys.argv',['foundation','--execute','--resume','--database',str(db),'--task',task['Task_ID']])
    result=legacy_stage_entry()
    assert result['status']=='VALID_COMPLETE' and result['simulation_calls_this_invocation']==1
    assert len(launches)==1 and stale.exists()
    assert (old/'power_trace.csv').exists()  # incomplete old attempt debugging kept
    assert not (old/'foundation_evidence').exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM observations').fetchone()[0]==1
        assert conn.execute('SELECT count(*) FROM terminal_executions').fetchone()[0]==1
        source=json.loads(conn.execute('SELECT source FROM observations').fetchone()[0])
        assert 'foundation_v1_new_attempt' in source['receipt']
    assert campaign.pending_rows(plan,db)==[]
    assert legacy_stage_entry()['simulation_calls_this_invocation']==0
    assert len(launches)==1


def test_defensive_writer_rejects_required_metadata_even_if_boundary_bypassed(tmp_path,monkeypatch):
    fixture=list(evidence_fixture(tmp_path,monkeypatch))
    fixture[0]['Pbar_kw_per_server']=np.nan
    with pytest.raises(ValueError,match='plan_metadata.Pbar_kw_per_server'):emit(tmp_path,fixture)


def test_defensive_writer_rejects_required_csv_provenance_missing(tmp_path,monkeypatch):
    fixture=list(evidence_fixture(tmp_path,monkeypatch))
    fixture[1]['results']['Context_ID']='nan'
    with pytest.raises(ValueError,match='results.Context_ID'):emit(tmp_path,fixture)


def test_repeat_group_identity_is_preserved():
    from am_flexdc_foundation_v1_common import make_row,row_geometry
    task=contexts()[0];base,_=fake_receipt(task)
    repeat=make_row(task,row_geometry(base),base['Candidate_Source'],simulation_seed=21,base=base['plan_row_id'])
    assert repeat['Base_Geometry_ID']==base['Base_Geometry_ID']
    assert repeat['plan_row_id']!=base['plan_row_id']
    assert provenance(repeat)['Replicate_Of_Plan_Row_ID']==base['plan_row_id']
