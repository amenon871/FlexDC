import copy
import json
from pathlib import Path

import pytest

from am_flexdc_foundation_v1_common import digest,file_hash,write_json
from foundation_v1 import operational_release as op,stage_certification as cert
from tests.test_foundation_v1_final_fixes import RELEASE,certified_stage


def approved_transition(tmp_path,old=RELEASE['uncommitted_source_sha256'],new='PORTABLE_SOURCE',master=cert.MASTER):
    release_path=tmp_path/'foundation_v1_artifacts/operational_patch_release_test.json'
    write_json(release_path,{'new_source_identity':new},False)
    release={'uncommitted_source_sha256':new,'phaseA_sha256':cert.MASTER,
        'operational_patch_release_sha256':file_hash(release_path),
        'operational_release_chain':[str(release_path)]}
    body={'schema':'FOUNDATION_OPERATIONAL_SOURCE_TRANSITION_V2','old_source_identity':old,
        'new_source_identity':new,'master_phaseA_sha256':master,'scientific_plan_unchanged':master==cert.MASTER,
        'operational_patch_release_sha256':file_hash(release_path)}
    body['body_sha256']=digest(body)
    path=tmp_path/'foundation_v1_runs/provenance/source_transition_TEST.json';write_json(path,body,False)
    return release,path


def validate(stage,release=RELEASE,transitions=None,root=None):
    _,plan,area,database,_,_=stage
    return cert.validate(plan,area,database,'TRAIN_DEV','phaseA',3,release,transitions,root)


def remap(values,device_index,new_device):
    result=copy.deepcopy(values)
    for key,value in result.items():
        if isinstance(value,list):value[device_index]=str(new_device)
    return result


def test_exact_source_and_device_pass(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch)
    result,reason=validate(stage)
    assert reason is None and result['validation_diagnostics']['source_continuity']['status']=='EXACT_SOURCE_IDENTITY'
    assert result['validation_diagnostics']['filesystem_device_remaps']==[]


def test_arbitrary_source_mismatch_without_transition_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch)
    result,reason=validate(stage,{**RELEASE,'uncommitted_source_sha256':'ARBITRARY'},[],tmp_path)
    assert result is None and 'without approved transition' in reason


def test_approved_transition_preserves_old_certificate(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);release,path=approved_transition(tmp_path)
    certificate=stage[2]/'stage_certification.json';before=certificate.read_bytes()
    result,reason=validate(stage,release,[path],tmp_path)
    assert reason is None and result['validation_diagnostics']['source_continuity']['status']=='APPROVED_OPERATIONAL_SOURCE_CONTINUITY'
    assert certificate.read_bytes()==before


def test_approved_multihop_transition_chain_passes(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);old=RELEASE['uncommitted_source_sha256'];middle='BOUNDARY_SOURCE';new='PORTABLE_SOURCE'
    release_paths=[];receipts=[]
    for index,(previous,current) in enumerate(((old,middle),(middle,new)),1):
        release_path=tmp_path/f'foundation_v1_artifacts/release_{index}.json';write_json(release_path,{'new_source_identity':current},False);release_paths.append(str(release_path))
        body={'schema':'FOUNDATION_OPERATIONAL_SOURCE_TRANSITION_V2','old_source_identity':previous,
            'new_source_identity':current,'master_phaseA_sha256':cert.MASTER,'scientific_plan_unchanged':True,
            'operational_patch_release_sha256':file_hash(release_path)}
        body['body_sha256']=digest(body);path=tmp_path/f'foundation_v1_runs/provenance/source_transition_{index}.json';write_json(path,body,False);receipts.append(path)
    release={'uncommitted_source_sha256':new,'phaseA_sha256':cert.MASTER,'operational_release_chain':release_paths}
    result,reason=validate(stage,release,receipts,tmp_path)
    continuity=result['validation_diagnostics']['source_continuity']
    assert reason is None and continuity['status']=='APPROVED_OPERATIONAL_SOURCE_CONTINUITY'
    assert continuity['transition_receipts']==[str(path) for path in receipts]


def test_transition_with_changed_scientific_identity_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);release,path=approved_transition(tmp_path,master='CHANGED_MASTER')
    result,reason=validate(stage,release,[path],tmp_path)
    assert result is None and ('changed scientific identity' in reason or 'invalid/tampered' in reason)


def test_tampered_transition_receipt_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);release,path=approved_transition(tmp_path)
    body=json.loads(path.read_text());body['new_source_identity']='TAMPERED';write_json(path,body,False)
    result,reason=validate(stage,release,[path],tmp_path)
    assert result is None and 'invalid/tampered' in reason


def test_st_dev_only_remap_passes_with_diagnostic(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);area=stage[2]
    journals=remap(cert.journal_watermarks(area),1,987654)
    buckets=remap(cert.receipt_buckets(area),0,987654)
    monkeypatch.setattr(cert,'journal_watermarks',lambda unused:journals)
    monkeypatch.setattr(cert,'receipt_buckets',lambda unused:buckets)
    result,reason=validate(stage)
    diagnostics=result['validation_diagnostics']['filesystem_device_remaps']
    assert reason is None and diagnostics and all(d['status']=='FILESYSTEM_DEVICE_REMAP_ACCEPTED' for d in diagnostics)
    assert all(d['stable_watermark_fields_match'] for d in diagnostics)


@pytest.mark.parametrize('kind',['journal_stable','receipt_stable'])
def test_device_remap_with_stable_watermark_change_fails(tmp_path,monkeypatch,kind):
    stage=certified_stage(tmp_path,monkeypatch);area=stage[2]
    if kind=='journal_stable':
        values=remap(cert.journal_watermarks(area),1,987654);key=next(k for k,v in values.items() if isinstance(v,list));values[key][3]='changed-content-hash'
        monkeypatch.setattr(cert,'journal_watermarks',lambda unused:values)
    else:
        values=remap(cert.receipt_buckets(area),0,987654);key=next(iter(values));values[key][2]+=1
        monkeypatch.setattr(cert,'receipt_buckets',lambda unused:values)
    result,reason=validate(stage)
    assert result is None and 'stable watermark changed' in reason


def test_same_device_journal_hash_change_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);values=cert.journal_watermarks(stage[2])
    key=next(k for k,v in values.items() if isinstance(v,list));values[key][3]='changed-content-hash'
    monkeypatch.setattr(cert,'journal_watermarks',lambda unused:values)
    result,reason=validate(stage)
    assert result is None and 'stable watermark changed' in reason


def test_journal_content_change_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);area=stage[2]
    with next((area/'events').glob('*.jsonl')).open('a') as stream:stream.write('{"id":"NEW"}\n')
    result,reason=validate(stage)
    assert result is None and 'stable watermark changed' in reason


def test_receipt_bucket_change_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);values=cert.receipt_buckets(stage[2]);key=next(iter(values));values[key][2]+=1
    monkeypatch.setattr(cert,'receipt_buckets',lambda unused:values)
    result,reason=validate(stage)
    assert result is None and 'stable watermark changed' in reason


def test_pending_file_hash_change_fails(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);_,plan,area,database,_,_=stage
    pending=area/'events/test.pending.json';pending.write_text('{"Plan_Row_ID":"A"}')
    cert.create(plan,area,database,'TRAIN_DEV','phaseA',3,RELEASE)
    pending.write_text('{"Plan_Row_ID":"B"}')
    result,reason=validate(stage)
    assert result is None and 'stable watermark changed' in reason


def test_transition_receipt_must_match_approved_release(tmp_path,monkeypatch):
    stage=certified_stage(tmp_path,monkeypatch);release,path=approved_transition(tmp_path)
    body=json.loads(path.read_text());body['operational_patch_release_sha256']='UNAPPROVED';body['body_sha256']=digest({k:v for k,v in body.items() if k!='body_sha256'});write_json(path,body,False)
    result,reason=validate(stage,release,[path],tmp_path)
    assert result is None and 'not tied to the approved operational release' in reason
