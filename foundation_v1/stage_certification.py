"""Small disposable certificates for stages that already passed strong audit."""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from am_flexdc_foundation_v1_common import digest,file_hash,read_json,write_json
from foundation_v1 import collector_receipts as receipts
from foundation_v1.recovery_index import RecoveryIndex

SCHEMA='FOUNDATION_STAGE_CERTIFICATION_V1'
MASTER='bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f'


def invalidate(area,reason=None):
    # Certificate is derived; journals remain the durable invalidation history.
    (Path(area)/'stage_certification.json').unlink(missing_ok=True)


def journal_watermarks(area):
    area=Path(area);out={}
    for path in sorted((area/'events').glob('*.jsonl')):
        out[path.name]=list(RecoveryIndex._stamp(path,path.stat().st_size))
    for path in sorted((area/'events').glob('*.pending.json')):
        out[path.name]=file_hash(path)
    return out


def receipt_buckets(area):
    # Bounded prefix directories, not one stat/open per receipt. Detect removal,
    # atomic replacement and normal publish/quarantine without historical reads.
    result={}
    for path in sorted((Path(area)/'receipts').glob('*')):
        if path.is_dir():
            s=path.stat();result[path.name]=[str(s.st_dev),str(s.st_ino),s.st_mtime_ns,s.st_ctime_ns]
    return result


def recovery_generation(area):
    path=Path(area)/'recovery_index.sqlite'
    if not path.exists():raise ValueError('recovery index missing')
    db=sqlite3.connect('file:'+path.resolve().as_posix()+'?mode=ro',uri=True)
    try:
        row=db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()
        if not row:raise ValueError('recovery index generation missing')
        return row[0]
    finally:db.close()


def ledger_marker(database,partition,stage,body=None):
    path=Path(database)
    if not path.exists():raise ValueError('audited observation database missing')
    db=sqlite3.connect('file:'+path.resolve().as_posix()+('?mode=rw' if body is not None else '?mode=ro'),uri=True)
    try:
        if body is not None:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS stage_certification_cache(partition TEXT,stage TEXT,body TEXT,PRIMARY KEY(partition,stage))')
                db.execute('INSERT OR REPLACE INTO stage_certification_cache VALUES(?,?,?)',(partition,stage,json.dumps(body,sort_keys=True)))
            return body
        row=db.execute('SELECT body FROM stage_certification_cache WHERE partition=? AND stage=?',(partition,stage)).fetchone()
        if not row:raise ValueError('stage audit database marker missing')
        return json.loads(row[0])
    finally:db.close()


def create(plan,area,database,partition,stage,expected_rows,release):
    area=Path(area);audit_path=area/'stage_audit.json';report=read_json(audit_path)
    if (report.get('status')!='PASS' or report.get('PASS')!=expected_rows or report.get('planned')!=expected_rows
        or report.get('QUARANTINED') or report.get('MISSING') or report.get('evidence_shortfalls')):
        raise ValueError('Cannot certify cache before complete PASS strong stage audit')
    token=uuid.uuid4().hex;plan_sha=file_hash(plan);audit_sha=file_hash(audit_path)
    marker=dict(certificate_id=token,plan_sha256=plan_sha,audit_sha256=audit_sha,certified_valid_rows=expected_rows)
    ledger_marker(database,partition,stage,marker)
    body=dict(certificate_schema_version=SCHEMA,partition=partition,stage=stage,
        plan_path=str(Path(plan).resolve()),plan_sha256=plan_sha,expected_rows=expected_rows,
        certified_valid_rows=expected_rows,receipt_schema=receipts.SCHEMA,
        master_phaseA_sha256=release.get('phaseA_sha256',MASTER),
        strong_audit_path=str(audit_path.resolve()),strong_audit_sha256=audit_sha,
        audit_database_path=str(Path(database).resolve()),audit_database_stage_identity=digest(marker),
        certificate_id=token,recovery_index_generation=recovery_generation(area),
        journal_watermarks=journal_watermarks(area),receipt_bucket_watermarks=receipt_buckets(area),
        collector_source_identity=release['uncommitted_source_sha256'],certified_at=time.time(),
        authority='DERIVED_REBUILDABLE_CACHE; authoritative receipts and strong audit remain primary')
    body['body_sha256']=digest(body)
    write_json(area/'stage_certification.json',body,False)
    return body


def validate(plan,area,database,partition,stage,expected_rows,release):
    try:
        area=Path(area);body=read_json(area/'stage_certification.json')
        if body.get('body_sha256')!=digest({k:v for k,v in body.items() if k!='body_sha256'}):raise ValueError('certificate checksum mismatch')
        if not isinstance(body.get('certified_at'),(int,float)) or body['certified_at']<=0:raise ValueError('certificate timestamp missing/invalid')
        required=dict(certificate_schema_version=SCHEMA,partition=partition,stage=stage,
            plan_path=str(Path(plan).resolve()),plan_sha256=file_hash(plan),expected_rows=expected_rows,
            certified_valid_rows=expected_rows,receipt_schema=receipts.SCHEMA,
            master_phaseA_sha256=release.get('phaseA_sha256',MASTER),
            audit_database_path=str(Path(database).resolve()),
            collector_source_identity=release['uncommitted_source_sha256'])
        if any(body.get(k)!=v for k,v in required.items()):raise ValueError('certificate stage/plan/science/source identity mismatch')
        audit_path=area/'stage_audit.json'
        if body['strong_audit_path']!=str(audit_path.resolve()) or file_hash(audit_path)!=body['strong_audit_sha256']:raise ValueError('strong audit artifact mismatch')
        if body['journal_watermarks']!=journal_watermarks(area):raise ValueError('new/inconsistent completion or invalidation journals')
        if body['receipt_bucket_watermarks']!=receipt_buckets(area):raise ValueError('receipt bucket changed; strong audit required')
        if body['recovery_index_generation']!=recovery_generation(area):raise ValueError('recovery generation changed')
        marker=ledger_marker(database,partition,stage)
        if marker.get('certificate_id')!=body['certificate_id'] or digest(marker)!=body['audit_database_stage_identity']:raise ValueError('stage audit database identity mismatch')
        return body,None
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as error:
        return None,str(error)
