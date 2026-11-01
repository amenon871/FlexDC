"""Small disposable certificates for stages that already passed strong audit."""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from am_flexdc_foundation_v1_common import digest,file_hash,read_json,write_json
from foundation_v1 import collector_receipts as receipts
from foundation_v1.recovery_index import RecoveryIndex
from foundation_v1.operational_release import validate_source_continuity

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


def portable_watermark_compare(expected,actual,device_index,label):
    """Accept only a mount-device remap; every stable watermark field stays exact."""
    if not isinstance(expected,dict) or not isinstance(actual,dict) or set(expected)!=set(actual):
        raise ValueError(label+' entries changed')
    diagnostics=[]
    for name in sorted(expected):
        old,new=expected[name],actual[name]
        if old==new:continue
        if (not isinstance(old,(list,tuple)) or not isinstance(new,(list,tuple)) or len(old)!=len(new)
            or device_index>=len(old) or any(old[i]!=new[i] for i in range(len(old)) if i!=device_index)):
            raise ValueError(label+' stable watermark changed: '+name)
        diagnostics.append({'status':'FILESYSTEM_DEVICE_REMAP_ACCEPTED','watermark':label,'name':name,
            'old_st_dev':old[device_index],'new_st_dev':new[device_index],
            'stable_watermark_fields_match':True})
    return diagnostics


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


def validate(plan,area,database,partition,stage,expected_rows,release,transition_receipts=None,provenance_root=None):
    try:
        area=Path(area);body=read_json(area/'stage_certification.json')
        if body.get('body_sha256')!=digest({k:v for k,v in body.items() if k!='body_sha256'}):raise ValueError('certificate checksum mismatch')
        if not isinstance(body.get('certified_at'),(int,float)) or body['certified_at']<=0:raise ValueError('certificate timestamp missing/invalid')
        required=dict(certificate_schema_version=SCHEMA,partition=partition,stage=stage,
            plan_path=str(Path(plan).resolve()),plan_sha256=file_hash(plan),expected_rows=expected_rows,
            certified_valid_rows=expected_rows,receipt_schema=receipts.SCHEMA,
            master_phaseA_sha256=release.get('phaseA_sha256',MASTER),
            audit_database_path=str(Path(database).resolve()))
        if any(body.get(k)!=v for k,v in required.items()):raise ValueError('certificate stage/plan/science identity mismatch')
        source=validate_source_continuity(provenance_root or Path(plan).resolve().parent,
            body.get('collector_source_identity'),release,transition_receipts)
        audit_path=area/'stage_audit.json'
        if body['strong_audit_path']!=str(audit_path.resolve()) or file_hash(audit_path)!=body['strong_audit_sha256']:raise ValueError('strong audit artifact mismatch')
        remaps=portable_watermark_compare(body['journal_watermarks'],journal_watermarks(area),1,'journal')
        remaps+=portable_watermark_compare(body['receipt_bucket_watermarks'],receipt_buckets(area),0,'receipt_bucket')
        if body['recovery_index_generation']!=recovery_generation(area):raise ValueError('recovery generation changed')
        marker=ledger_marker(database,partition,stage)
        if marker.get('certificate_id')!=body['certificate_id'] or digest(marker)!=body['audit_database_stage_identity']:raise ValueError('stage audit database identity mismatch')
        if source['status']=='APPROVED_OPERATIONAL_SOURCE_CONTINUITY':
            print('CERTIFICATE_SOURCE_CONTINUITY_ACCEPTED old_source='+source['certificate_source_identity']+
                ' new_source='+source['current_source_identity']+' transitions='+str(len(source['transition_receipts'])),flush=True)
        if remaps:
            pairs=sorted({str(item['old_st_dev'])+'->'+str(item['new_st_dev']) for item in remaps})
            print('FILESYSTEM_DEVICE_REMAP_ACCEPTED device_remaps='+','.join(pairs)+' entries='+str(len(remaps))+
                ' stable_watermark_fields_match=true',flush=True)
        result=dict(body);result['validation_diagnostics']={'source_continuity':source,'filesystem_device_remaps':remaps}
        return result,None
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error) as error:
        return None,str(error)
