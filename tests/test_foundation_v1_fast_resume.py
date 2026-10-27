"""Synthetic receipts only. No FlexDC process, corpus, or production run is touched."""
import json
import sqlite3
import time
from pathlib import Path
import pytest
from am_flexdc_foundation_v1_common import digest, write_csv
from foundation_v1 import collector_receipts as receipts
from foundation_v1.recovery_index import RecoveryIndex, invalidate


def fixture(tmp_path, n=100):
    area=tmp_path/'train_dev'/'collector_v2'/'phaseA'
    area.mkdir(parents=True)
    keys=['F1_'+str(i).zfill(32) for i in range(n)]
    plan=tmp_path/'plan.csv'
    write_csv(plan,[{'plan_row_id':k} for k in keys])
    return area,plan,keys


def publish(area,key,journal=True,pending=False):
    body=dict(schema=receipts.SCHEMA,Plan_Row_ID=key,Base_Geometry_ID='TEST_ONLY',
              Task_ID='TEST_ONLY',Workload_ID='TEST_ONLY',context='TEST_ONLY',
              simulation_seed=20,requested={},applied={},result_file='TEST_ONLY',
              diagnostic_file='TEST_ONLY',evidence_file='TEST_ONLY',timestamp=time.time(),
              evidence={},execution_status='EXECUTION_COMPLETE')
    body['checksum']=digest(body)
    path=receipts.receipt_path(area,key)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(body))
    events=area/'events'
    events.mkdir(exist_ok=True)
    if journal:
        with (events/'slot_0.jsonl').open('a') as f:f.write(json.dumps({'id':key,'time':body['timestamp']})+'\n')
    if pending:(events/'slot_0.pending.json').write_text(json.dumps({'Plan_Row_ID':key}))
    return body


def open_count(monkeypatch):
    original=receipts.load
    calls=[]
    def counted(area,key):calls.append(key);return original(area,key)
    monkeypatch.setattr(receipts,'load',counted)
    return calls


def test_indexed_parity_and_missing_index(tmp_path):
    area,plan,keys=fixture(tmp_path,20)
    for k in keys:publish(area,k)
    index=RecoveryIndex(plan,area,keys)
    assert index.stats['recovery_mode']=='authoritative_rebuild'
    index.close()
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(receipts.completed(area,keys))==set(keys)
    assert index.stats['verified_new_receipts']==0
    index.close()
    (area/'recovery_index.sqlite').unlink()
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys) and index.stats['recovery_mode']=='authoritative_rebuild'
    index.close()


@pytest.mark.parametrize('damage',['bytes','schema','identity','quick_check','truncated','replaced'])
def test_corrupt_or_wrong_identity_and_journal_falls_back(tmp_path,damage):
    area,plan,keys=fixture(tmp_path,10)
    for key in keys:publish(area,key)
    index=RecoveryIndex(plan,area,keys)
    index.close()
    dbpath=area/'recovery_index.sqlite'
    if damage=='bytes':dbpath.write_bytes(b'corrupt SQLite')
    elif damage in ('schema','identity','quick_check'):
        with sqlite3.connect(dbpath) as db:
            if damage=='schema':db.execute('DROP TABLE completed')
            elif damage=='identity':db.execute("UPDATE metadata SET value='WRONG' WHERE key='plan_sha256'")
            else:db.execute('PRAGMA writable_schema=ON');db.execute("UPDATE sqlite_master SET rootpage=999999 WHERE name='completed'")
        db.close()
    else:
        journal=area/'events/slot_0.jsonl'
        if damage=='truncated':journal.write_text('')
        else:
            journal.unlink();journal.write_text(json.dumps({'id':keys[0],'time':time.time()})+'\n')
    index=RecoveryIndex(plan,area,keys)
    assert index.stats['recovery_mode']=='authoritative_rebuild'
    assert index.done==set(keys)
    index.close()


def test_wrong_actual_plan_hash_is_not_accepted(tmp_path):
    area,plan,keys=fixture(tmp_path,10)
    for k in keys:publish(area,k)
    index=RecoveryIndex(plan,area,keys);index.close()
    other=tmp_path/'other_plan.csv';write_csv(other,[{'plan_row_id':keys[-1]}])
    index=RecoveryIndex(other,area,[keys[-1]])
    assert index.done=={keys[-1]} and index.stats['recovery_mode']=='authoritative_rebuild'
    index.close()


@pytest.mark.parametrize('receipt_state',['missing','corrupt','valid'])
def test_new_journal_claim_always_verifies_receipt(tmp_path,receipt_state):
    area,plan,keys=fixture(tmp_path,2)
    index=RecoveryIndex(plan,area,keys);index.close()
    publish(area,keys[0])
    path=receipts.receipt_path(area,keys[0])
    if receipt_state=='missing':path.unlink()
    if receipt_state=='corrupt':path.write_text('{}')
    index=RecoveryIndex(plan,area,keys)
    assert (keys[0] in index.done)==(receipt_state=='valid')
    assert index.stats['verified_new_receipts']==1
    index.close()


def test_pending_before_receipt_and_receipt_before_journal(tmp_path):
    area,plan,keys=fixture(tmp_path,2)
    index=RecoveryIndex(plan,area,keys);index.close()
    events=area/'events'
    (events/'slot_0.pending.json').write_text(json.dumps({'Plan_Row_ID':keys[0]}))
    index=RecoveryIndex(plan,area,keys)
    assert not index.done
    index.close()
    publish(area,keys[0],journal=False,pending=True)
    index=RecoveryIndex(plan,area,keys)
    assert index.done=={keys[0]} and index.stats['pending_recovered']==1
    index.close()


def test_index_lag_90_plus_10_only_opens_tail(tmp_path,monkeypatch):
    area,plan,keys=fixture(tmp_path)
    for key in keys[:90]:publish(area,key)
    index=RecoveryIndex(plan,area,keys);index.close()
    for key in keys[90:]:publish(area,key)
    calls=open_count(monkeypatch)
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys)
    assert calls==keys[90:]
    assert index.stats['replayed_journal_events']==10
    index.close()


def test_thousands_of_historical_receipts_not_reopened(tmp_path,monkeypatch):
    area,plan,keys=fixture(tmp_path,4005)
    for key in keys[:4000]:publish(area,key)
    index=RecoveryIndex(plan,area,keys);index.close()
    for key in keys[4000:]:publish(area,key)
    calls=open_count(monkeypatch)
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys) and calls==keys[4000:]
    assert index.stats['full_historical_receipt_opens']==0
    assert index.stats['recovery_index_rows']==4000
    index.close()


def test_partial_journal_line_is_not_consumed(tmp_path):
    area,plan,keys=fixture(tmp_path,1)
    index=RecoveryIndex(plan,area,keys);index.close()
    body=publish(area,keys[0],journal=False)
    journal=area/'events/slot_0.jsonl';line=json.dumps({'id':keys[0],'time':body['timestamp']})+'\n'
    journal.write_bytes(line[:12].encode())
    index=RecoveryIndex(plan,area,keys)
    assert not index.done and index.offsets[journal.name][0]==0
    index.close()
    with journal.open('ab') as f:f.write(line[12:].encode())
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys)
    index.close()


def test_accepted_batches_update_index_without_reopening_receipts(tmp_path,monkeypatch):
    area,plan,keys=fixture(tmp_path,103)
    index=RecoveryIndex(plan,area,keys)
    for key in keys:index.accept(publish(area,key))
    calls=open_count(monkeypatch)
    index.flush(force=True)
    assert calls==[]
    assert index.db.execute('SELECT count(*) FROM completed').fetchone()[0]==103
    index.close()
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys) and calls==[]
    index.close()


def test_strong_audit_detects_deleted_cached_receipt_and_repairs_only_row(tmp_path,monkeypatch):
    from tests.test_foundation_v1_simple_collector import setup_queue,FakeLaunch
    from foundation_v1 import simple_collector as c,simple_campaign as full
    task,rows,plan,area=setup_queue(tmp_path,monkeypatch,3)
    launch=FakeLaunch(tmp_path,[task])
    c.collect(plan,area,2,0,launch,1)
    key=rows[0]['plan_row_id'];receipts.receipt_path(area,key).unlink()
    index=RecoveryIndex(plan,area,{r['plan_row_id'] for r in rows})
    assert key in index.done  # Ordinary cache is not stage certification.
    index.close()
    report=full.certify(plan,area,tmp_path/'derived.sqlite')
    assert report['MISSING']==[key] and report['PASS']==2
    invalidate(area,key)
    before=len(launch.calls)
    c.collect(plan,area,2,0,launch,1)
    assert len(launch.calls)==before+1 and launch.calls[-1][0]['plan_row_id']==key
    assert full.certify(plan,area,tmp_path/'derived.sqlite')['status']=='PASS'


def test_archive_invalidates_only_affected_index_id(tmp_path):
    area,plan,keys=fixture(tmp_path,3)
    for key in keys:publish(area,key)
    index=RecoveryIndex(plan,area,keys);index.close()
    receipts.archive(area,keys[0],{'reason':'TEST_ONLY'})
    index=RecoveryIndex(plan,area,keys)
    assert index.done==set(keys[1:])
    index.close()
