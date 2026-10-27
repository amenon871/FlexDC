import json
from pathlib import Path
import pytest
from am_flexdc_foundation_v1_common import digest,file_hash,read_json,write_json
from foundation_v1 import operational_release as op
from foundation_v1.stage_certification import MASTER


BUNDLE_SHA='0b2d322bc3cf85115e8b94aac661aefa9f0ff00eeb2466bf886b151ff7bb7226'


def release_record(previous,files,**extra):
    body=dict(previous_source_identity=previous,master_phaseA_sha256=MASTER,
        scientific_master_changed=False,operational_patch_after_campaign_start=True,
        scientific_simulations_started=True,source_files=files,new_source_identity=digest(files),**extra)
    body['body_sha256']=digest(body)
    return body


def fixture(tmp_path):
    artifacts=tmp_path/'foundation_v1_artifacts';artifacts.mkdir()
    source=tmp_path/'source.py';source.write_text('original')
    old_files={'source.py':file_hash(source)}
    historical=dict(uncommitted_source_sha256=digest(old_files),source_files=old_files,
        phaseA_sha256=MASTER,scientific_simulations_started=False)
    write_json(artifacts/'preexecution_release.json',historical)
    source.write_text('fast resume')
    first=release_record(historical['uncommitted_source_sha256'],{'source.py':file_hash(source)},
        historical_release_sha256=file_hash(artifacts/'preexecution_release.json'))
    write_json(artifacts/op.NAME,first);first_bytes=(artifacts/op.NAME).read_bytes()
    current=op.verify(tmp_path,artifacts,historical)
    return artifacts,source,historical,current,first_bytes


def test_second_release_and_transition_chain_are_append_only(tmp_path):
    artifacts,source,historical,fast,first_bytes=fixture(tmp_path)
    first_status=tmp_path/'status_first.json'
    write_json(first_status,dict(test_fixture_only=True,source_identity=historical['uncommitted_source_sha256'],
        overall_complete=100,active={'partition':'TRAIN_DEV','stage':'phaseA'}))
    first_receipt=op.record_transition(tmp_path,fast,first_status,historical['uncommitted_source_sha256'],
        'TEST','FAST_RESUME','FIRST',old_git_commit='oldcommit',new_git_commit='fastcommit')
    source.write_text('aqa speedup')
    second=release_record(fast['uncommitted_source_sha256'],{'source.py':file_hash(source)},
        previous_operational_release_path='foundation_v1_artifacts/'+op.NAME,
        previous_operational_release_sha256=file_hash(artifacts/op.NAME),
        validation_bundle_sha256=BUNDLE_SHA,simulator_scientific_output_parity_validated=True,
        aqa_decision_parity_validated=True,non_aqa_legacy_behavior_preserved=True,
        optimization='validated AQA lossless simulator speedup')
    write_json(artifacts/op.LATEST_NAME,second)
    optimized=op.verify(tmp_path,artifacts,historical)
    assert optimized['collection_initial_source_identity']==historical['uncommitted_source_sha256']
    assert optimized['collection_previous_source_identity']==fast['uncommitted_source_sha256']
    assert optimized['uncommitted_source_sha256']==second['new_source_identity']
    assert (artifacts/op.NAME).read_bytes()==first_bytes
    second_status=tmp_path/'status_second.json'
    write_json(second_status,dict(test_fixture_only=True,source_identity=fast['uncommitted_source_sha256'],
        overall_complete=117000,active={'partition':'TRAIN_DEV','stage':'phaseA'},
        source_transition_receipts=[str(first_receipt)],git_SHA='fastcommit'))
    write_json(tmp_path/'foundation_v1_runs/simple_campaign_status.json',
        {'source_identity':fast['uncommitted_source_sha256']},False)
    with pytest.raises(ValueError,match='SOURCE_TRANSITION_REQUIRED'):op.require_transition(tmp_path,optimized)
    second_receipt=op.record_transition(tmp_path,optimized,second_status,fast['uncommitted_source_sha256'],
        'TEST','AQA_SPEEDUP','SECOND',old_git_commit='fastcommit',new_git_commit='optimizedcommit')
    chain=op.require_transition(tmp_path,optimized)
    assert chain==[str(first_receipt),str(second_receipt)]
    body=read_json(second_receipt)
    assert body['old_source_identity']==fast['uncommitted_source_sha256']
    assert body['new_source_identity']==optimized['uncommitted_source_sha256']
    assert body['previous_transition_receipts']==[str(first_receipt)]
    assert body['validation_bundle_sha256']==BUNDLE_SHA
    assert body['old_git_commit']=='fastcommit' and body['new_git_commit']=='optimizedcommit'
    assert (artifacts/op.NAME).read_bytes()==first_bytes
    with pytest.raises(FileExistsError):op.record_transition(tmp_path,optimized,second_status,
        fast['uncommitted_source_sha256'],'TEST','AQA_SPEEDUP','SECOND',
        old_git_commit='fastcommit',new_git_commit='optimizedcommit')
    with pytest.raises(ValueError,match='predecessor'):op.record_transition(tmp_path,optimized,second_status,
        historical['uncommitted_source_sha256'],'TEST','AQA_SPEEDUP','WRONG')


def test_production_transition_requires_operator_and_commit_ids(tmp_path):
    artifacts,source,historical,fast,_=fixture(tmp_path)
    status=tmp_path/'status.json'
    write_json(status,dict(source_identity=historical['uncommitted_source_sha256'],overall_complete=1,
        active={'partition':'TRAIN_DEV','stage':'phaseA'}))
    with pytest.raises(ValueError,match='requires operator'):op.record_transition(
        tmp_path,fast,status,historical['uncommitted_source_sha256'],transition_id='PRODUCTION')
