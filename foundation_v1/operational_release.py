"""Historical source seal plus explicit, immutable mid-campaign patch provenance."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import tempfile
import uuid
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,digest,file_hash,read_json

NAME='operational_patch_release_resume_v3.json'
MASTER='bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f'


def verify(root,artifacts,historical):
    root=Path(root);artifacts=Path(artifacts);record=read_json(artifacts/NAME)
    if record.get('body_sha256')!=digest({k:v for k,v in record.items() if k!='body_sha256'}):raise ValueError('Operational release checksum mismatch')
    if record['previous_source_identity']!=historical['uncommitted_source_sha256']:raise ValueError('Historical source identity changed')
    if file_hash(artifacts/'preexecution_release.json')!=record['historical_release_sha256']:raise ValueError('Historical preexecution bytes changed')
    if record.get('scientific_simulations_started') is not True:raise ValueError('Mid-campaign release must acknowledge collection already started')
    if record['master_phaseA_sha256']!=MASTER or record['scientific_master_changed'] or not record['operational_patch_after_campaign_start']:raise ValueError('Operational release scientific/provenance contract mismatch')
    if not set(historical['source_files'])<=set(record['source_files']):raise ValueError('Previously guarded paths removed')
    if digest(record['source_files'])!=record['new_source_identity']:raise ValueError('Operational source identity checksum mismatch')
    for name,expected in record['source_files'].items():
        if any(p.startswith('.env') for p in Path(name).parts):raise ValueError('Forbidden source path')
        if file_hash(root/name)!=expected:raise ValueError('Frozen operational source/config changed: '+name)
    return {**historical,'scientific_simulations_started':True,'operational_patch_after_campaign_start':True,'source_files':record['source_files'],'uncommitted_source_sha256':record['new_source_identity'],
            'collection_initial_source_identity':record['previous_source_identity'],
            'operational_patch_release':str(artifacts/NAME),'operational_patch_release_sha256':file_hash(artifacts/NAME)}


def atomic_immutable(path,body):
    """Atomic no-overwrite publication, including a crash before final pathname."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.transition_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(body,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path)  # fails if target exists; never overwrites history
        if os.name!='nt':
            d=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(d)
            finally:os.close(d)
    finally:Path(temp).unlink(missing_ok=True)


def record_transition(root,release,status_snapshot,old_source_identity,operator=None,release_id=None,transition_id=None):
    root=Path(root);snapshot=read_json(status_snapshot)
    expected=release['collection_initial_source_identity']
    if old_source_identity!=expected:raise ValueError('Transition old source identity does not match preserved historical release')
    if snapshot.get('source_identity',old_source_identity)!=old_source_identity:raise ValueError('Captured status has a different old source identity')
    count=snapshot.get('overall_complete')
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=500000:raise ValueError('Captured completed-row count missing/invalid')
    active=snapshot.get('active')
    if count<500000 and (not isinstance(active,dict) or active.get('partition') not in ('TRAIN_DEV','TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST') or active.get('stage') not in ('phaseA','wave01','wave02','wave03','wave04','repeat')):raise ValueError('Captured active partition/stage missing/invalid')
    stamp=datetime.now(timezone.utc).isoformat();key=transition_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
    if not key.replace('_','').isalnum():raise ValueError('Invalid transition identifier')
    body=dict(schema='FOUNDATION_OPERATIONAL_SOURCE_TRANSITION_V1',deployment_timestamp=stamp,
        old_source_identity=old_source_identity,new_source_identity=release['uncommitted_source_sha256'],
        overall_completed_rows_at_transition=count,active=active,master_phaseA_sha256=MASTER,
        scientific_plan_unchanged=True,operator=operator,release_identifier=release_id,
        captured_status_sha256=file_hash(status_snapshot),count_source='operator-captured pre-update status snapshot; not a new receipt audit',
        operational_patch_release_sha256=release['operational_patch_release_sha256'],
        test_fixture_only=bool(snapshot.get('test_fixture_only',False)))
    body['body_sha256']=digest(body)
    path=root/'foundation_v1_runs/provenance'/('source_transition_'+key+'.json')
    atomic_immutable(path,body)
    return path


def require_transition(root,release):
    """Do not silently overwrite an existing campaign's old source provenance."""
    state_path=Path(root)/'foundation_v1_runs/simple_campaign_status.json'
    if not release.get('operational_patch_release') or not state_path.exists():return []
    state=read_json(state_path);old=state.get('source_identity')
    if old in (None,release['uncommitted_source_sha256']):return state.get('source_transition_receipts',[])
    for path in (Path(root)/'foundation_v1_runs/provenance').glob('source_transition_*.json'):
        try:
            body=read_json(path)
            if (body.get('body_sha256')==digest({k:v for k,v in body.items() if k!='body_sha256'})
                and body['old_source_identity']==old and body['new_source_identity']==release['uncommitted_source_sha256']
                and body['master_phaseA_sha256']==MASTER and body['operational_patch_release_sha256']==release['operational_patch_release_sha256']):return [str(path)]
        except (ValueError,OSError,KeyError,TypeError):pass
    raise ValueError('MID_CAMPAIGN_SOURCE_TRANSITION_REQUIRED: capture old status and write the explicit source-transition receipt before resume')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['verify','transition']);p.add_argument('--status-snapshot');p.add_argument('--old-source-identity');p.add_argument('--operator');p.add_argument('--release-id')
    a=p.parse_args()
    from foundation_v1.full_campaign import verify_release
    release=verify_release()
    if a.operation=='verify':print(json.dumps(dict(status='PASS',previous_source_identity=release['collection_initial_source_identity'],new_source_identity=release['uncommitted_source_sha256'],master_phaseA_sha256=MASTER)));return
    if not a.status_snapshot or not a.old_source_identity:p.error('transition requires --status-snapshot and --old-source-identity')
    from foundation_v1.campaign import CampaignLock
    with CampaignLock(ROOT/'foundation_v1_campaign.lock'):
        print(record_transition(ROOT,release,a.status_snapshot,a.old_source_identity,a.operator,a.release_id))


if __name__=='__main__':main()
