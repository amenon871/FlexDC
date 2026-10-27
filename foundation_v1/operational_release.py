"""Historical source seals plus explicit, immutable mid-campaign provenance chains."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import tempfile
import uuid
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,digest,file_hash,read_json

NAME='operational_patch_release_resume_v3.json'
LATEST_NAME='operational_patch_release_aqa_speedup_v1.json'
MASTER='bb4a1c0da7f15c53b8201766ff48a7b15d672e78a82d636b5d0159e80f47ed7f'


def _verify_source_record(root,record,prior_files,check_current=True):
    if record.get('body_sha256')!=digest({k:v for k,v in record.items() if k!='body_sha256'}):raise ValueError('Operational release checksum mismatch')
    if record.get('scientific_simulations_started') is not True:raise ValueError('Mid-campaign release must acknowledge collection already started')
    if record['master_phaseA_sha256']!=MASTER or record['scientific_master_changed'] or not record['operational_patch_after_campaign_start']:raise ValueError('Operational release scientific/provenance contract mismatch')
    if not set(prior_files)<=set(record['source_files']):raise ValueError('Previously guarded paths removed')
    if digest(record['source_files'])!=record['new_source_identity']:raise ValueError('Operational source identity checksum mismatch')
    for name,expected in record['source_files'].items():
        if any(p.startswith('.env') for p in Path(name).parts):raise ValueError('Forbidden source path')
        if check_current and file_hash(root/name)!=expected:raise ValueError('Frozen operational source/config changed: '+name)


def verify(root,artifacts,historical):
    root=Path(root);artifacts=Path(artifacts)
    first_path=artifacts/NAME;first=read_json(first_path)
    if first['previous_source_identity']!=historical['uncommitted_source_sha256']:raise ValueError('Historical source identity changed')
    if file_hash(artifacts/'preexecution_release.json')!=first['historical_release_sha256']:raise ValueError('Historical preexecution bytes changed')
    latest_path=artifacts/LATEST_NAME
    _verify_source_record(root,first,historical['source_files'],check_current=not latest_path.exists())
    current={**historical,'scientific_simulations_started':True,'operational_patch_after_campaign_start':True,
        'source_files':first['source_files'],'uncommitted_source_sha256':first['new_source_identity'],
        'collection_initial_source_identity':first['previous_source_identity'],
        'collection_previous_source_identity':first['previous_source_identity'],
        'operational_patch_release':str(first_path),'operational_patch_release_sha256':file_hash(first_path),
        'operational_release_chain':[str(first_path)]}
    if not latest_path.exists():return current
    latest=read_json(latest_path)
    if latest.get('previous_source_identity')!=current['uncommitted_source_sha256']:raise ValueError('Second operational release predecessor changed')
    if latest.get('previous_operational_release_path')!=(Path('foundation_v1_artifacts')/NAME).as_posix():raise ValueError('Second operational release path chain changed')
    if latest.get('previous_operational_release_sha256')!=file_hash(first_path):raise ValueError('First operational release bytes changed')
    if latest.get('validation_bundle_sha256')!='0b2d322bc3cf85115e8b94aac661aefa9f0ff00eeb2466bf886b151ff7bb7226':raise ValueError('AQA validation bundle identity changed')
    if latest.get('simulator_scientific_output_parity_validated') is not True or latest.get('aqa_decision_parity_validated') is not True:raise ValueError('AQA validation claims missing')
    if latest.get('non_aqa_legacy_behavior_preserved') is not True:raise ValueError('Non-AQA fallback contract missing')
    _verify_source_record(root,latest,current['source_files'])
    return {**current,'source_files':latest['source_files'],'uncommitted_source_sha256':latest['new_source_identity'],
        'collection_previous_source_identity':latest['previous_source_identity'],
        'operational_patch_release':str(latest_path),'operational_patch_release_sha256':file_hash(latest_path),
        'operational_release_chain':current['operational_release_chain']+[str(latest_path)],
        'validation_bundle_sha256':latest['validation_bundle_sha256'],
        'optimization':latest.get('optimization')}


def atomic_immutable(path,body):
    """Atomic no-overwrite publication, including a crash before final pathname."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.transition_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(body,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path)
        if os.name!='nt':
            d=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(d)
            finally:os.close(d)
    finally:Path(temp).unlink(missing_ok=True)


def _valid_transition_receipts(root):
    valid=[]
    for path in sorted((Path(root)/'foundation_v1_runs/provenance').glob('source_transition_*.json')):
        try:
            body=read_json(path)
            if (body.get('body_sha256')==digest({k:v for k,v in body.items() if k!='body_sha256'})
                and body.get('master_phaseA_sha256')==MASTER and body.get('scientific_plan_unchanged') is True):
                valid.append((path,body))
        except (ValueError,OSError,KeyError,TypeError):pass
    return valid


def transition_chain(root,initial_source_identity,target_source_identity):
    """Return an append-only, acyclic transition chain from initial to target."""
    if initial_source_identity==target_source_identity:return []
    by_old={}
    for path,body in _valid_transition_receipts(root):by_old.setdefault(body['old_source_identity'],[]).append((path,body))
    chain=[];current=initial_source_identity;seen=set()
    while current!=target_source_identity:
        if current in seen:raise ValueError('Source-transition chain contains a cycle')
        seen.add(current);choices=by_old.get(current,[])
        if len(choices)!=1:raise ValueError('SOURCE_TRANSITION_CHAIN_INCOMPLETE_OR_AMBIGUOUS at '+current)
        path,body=choices[0];chain.append(str(path));current=body['new_source_identity']
    return chain


def record_transition(root,release,status_snapshot,old_source_identity,operator=None,release_id=None,transition_id=None,old_git_commit=None,new_git_commit=None):
    root=Path(root);snapshot=read_json(status_snapshot)
    expected=release.get('collection_previous_source_identity',release['collection_initial_source_identity'])
    if old_source_identity!=expected:raise ValueError('Transition old source identity does not match reviewed predecessor release')
    if snapshot.get('source_identity',old_source_identity)!=old_source_identity:raise ValueError('Captured status has a different old source identity')
    prior_chain=transition_chain(root,release['collection_initial_source_identity'],old_source_identity)
    count=snapshot.get('overall_complete')
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=500000:raise ValueError('Captured completed-row count missing/invalid')
    active=snapshot.get('active')
    if count<500000 and (not isinstance(active,dict) or active.get('partition') not in ('TRAIN_DEV','TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST') or active.get('stage') not in ('phaseA','wave01','wave02','wave03','wave04','repeat')):raise ValueError('Captured active partition/stage missing/invalid')
    fixture=bool(snapshot.get('test_fixture_only',False))
    old_git_commit=old_git_commit or snapshot.get('git_SHA')
    new_git_commit=new_git_commit or release.get('new_git_commit')
    if not fixture and (not operator or not old_git_commit or not new_git_commit):raise ValueError('Production transition requires operator, old Git commit, and new Git commit')
    stamp=datetime.now(timezone.utc).isoformat();key=transition_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
    if not key.replace('_','').isalnum():raise ValueError('Invalid transition identifier')
    body=dict(schema='FOUNDATION_OPERATIONAL_SOURCE_TRANSITION_V2',deployment_timestamp=stamp,
        old_source_identity=old_source_identity,new_source_identity=release['uncommitted_source_sha256'],
        old_git_commit=old_git_commit,new_git_commit=new_git_commit,
        overall_completed_rows_at_transition=count,active=active,master_phaseA_sha256=MASTER,
        scientific_plan_unchanged=True,operator=operator,release_identifier=release_id,
        captured_status_sha256=file_hash(status_snapshot),count_source='operator-captured pre-update status snapshot; not a new receipt audit',
        operational_patch_release_sha256=release['operational_patch_release_sha256'],
        previous_transition_receipts=prior_chain,
        validation_bundle_sha256=release.get('validation_bundle_sha256'),
        optimization=release.get('optimization'),
        test_fixture_only=fixture)
    body['body_sha256']=digest(body)
    path=root/'foundation_v1_runs/provenance'/('source_transition_'+key+'.json')
    atomic_immutable(path,body)
    return path


def require_transition(root,release):
    """Require a direct transition and validate its complete append-only ancestry."""
    state_path=Path(root)/'foundation_v1_runs/simple_campaign_status.json'
    if not release.get('operational_patch_release') or not state_path.exists():return []
    state=read_json(state_path);old=state.get('source_identity')
    if old in (None,release['uncommitted_source_sha256']):return state.get('source_transition_receipts',[])
    direct=None
    for path,body in _valid_transition_receipts(root):
        if (body['old_source_identity']==old and body['new_source_identity']==release['uncommitted_source_sha256']
            and body.get('operational_patch_release_sha256')==release['operational_patch_release_sha256']):
            direct=str(path);break
    if direct:
        chain=transition_chain(root,release['collection_initial_source_identity'],release['uncommitted_source_sha256'])
        if not chain or chain[-1]!=direct:raise ValueError('Source-transition chain does not terminate at reviewed release')
        return chain
    raise ValueError('MID_CAMPAIGN_SOURCE_TRANSITION_REQUIRED: capture old status and write the explicit source-transition receipt before resume')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['verify','transition']);p.add_argument('--status-snapshot');p.add_argument('--old-source-identity');p.add_argument('--operator');p.add_argument('--release-id');p.add_argument('--old-git-commit');p.add_argument('--new-git-commit')
    a=p.parse_args()
    from foundation_v1.full_campaign import verify_release
    release=verify_release()
    if a.operation=='verify':
        print(json.dumps(dict(status='PASS',initial_source_identity=release['collection_initial_source_identity'],previous_source_identity=release.get('collection_previous_source_identity'),new_source_identity=release['uncommitted_source_sha256'],master_phaseA_sha256=MASTER,release_chain=release.get('operational_release_chain',[]))));return
    if not a.status_snapshot or not a.old_source_identity:p.error('transition requires --status-snapshot and --old-source-identity')
    from foundation_v1.campaign import CampaignLock
    with CampaignLock(ROOT/'foundation_v1_campaign.lock'):
        print(record_transition(ROOT,release,a.status_snapshot,a.old_source_identity,a.operator,a.release_id,old_git_commit=a.old_git_commit,new_git_commit=a.new_git_commit))


if __name__=='__main__':main()
