"""One new immutable review ZIP; no commit, deployment or scientific execution."""
from pathlib import Path,PurePosixPath
import difflib,hashlib,json,re,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'foundation_successor_models';AUD=ROOT/'supervisor_final_fix_audit';OUT=ROOT/'supervisor_bundles'
BASE='0190fe60d53cddb6afbaffad9a67b82e50ccf0de'
ALLOWED={'foundation_v1/collector_receipts.py','foundation_v1/simple_campaign.py','foundation_v1/simple_collector.py','foundation_v1/full_campaign.py'}
SKIP={'.git','dist','.validation_runs','.pytest_cache','__pycache__','wandb','cache','.cache','runs','venv','.venv'}
def digest(b):return hashlib.sha256(b).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
def safe(p):return not any(n in SKIP or n.lower().startswith('.env') for n in p.parts)
def category(n):
    if n.startswith('reference_only/'):return 'read-only historical reference'
    if n.startswith('patches/'):return 'previous-review versus current diff'
    if '/tests/' in n or n.startswith('tests/'):return 'synthetic/unit regression tests'
    if '/fixtures/' in n:return 'TEST FIXTURE ONLY; NOT SCIENTIFIC DATA'
    if '/audit/' in n or n.startswith('supervisor_final_fix_audit/'):return 'audit/test/benchmark evidence and tooling'
    return 'source/config/provenance/documentation'
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8',newline='\n')


def main():
    assert git('rev-parse','HEAD').decode().strip()==BASE
    assert git('branch','--show-current').decode().strip()=='foundation-successor-models-prep'
    assert not git('diff','--cached','--name-only').strip()
    assert set(git('diff','--name-only').decode().splitlines())==ALLOWED
    from foundation_v1.full_campaign import verify_release
    release=verify_release()
    assert (ROOT/'foundation_v1_artifacts/preexecution_release.json').read_bytes()==git('show',BASE+':foundation_v1_artifacts/preexecution_release.json')
    old=OUT/'flexdc_successor_training_and_fast_resume_supervisor_0190fe6.zip'
    assert digest(old.read_bytes())=='34c92f12061527a083db9a65cfab80127676ac65a3b4e81cea467d57a31b3d21'
    totals=json.loads((AUD/'VALIDATION_TOTALS.json').read_text());assert totals['status']=='PASS'
    write(AUD/'FINAL_GIT_STATE.txt','\n'.join('$ git '+cmd+'\n'+git(*cmd.split()).decode() for cmd in ['status --short','branch --show-current','rev-parse HEAD','diff --name-only','diff --stat','diff --cached --name-only']))
    (AUD/'FINAL_GIT_DIFF.patch').write_bytes(git('diff','--no-ext-diff','--binary'))
    files={}
    def add(name,b):
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and safe(p),name
        assert len(b)<8*1024*1024,('unexpected large review member',name)
        assert name not in files,name
        if p.suffix!='.gz':assert not re.search(rb'(?:sk-proj-|wandb_v1_)[A-Za-z0-9_-]{20,}',b),'Credential-pattern review required; no values printed'
        files[name]=b
    def tree(path):
        for p in sorted(path.rglob('*')):
            rel=p.relative_to(ROOT)
            if p.is_file() and safe(rel) and p.suffix not in ('.pt','.pth','.pyc','.zip','.sqlite','.db'):add(rel.as_posix(),p.read_bytes())
    inventory=[]
    for p in sorted(PKG.rglob('*')):
        rel=p.relative_to(PKG)
        if p.is_file() and safe(rel) and p.suffix not in ('.pt','.pth','.pyc','.zip') and rel.as_posix()!='audit/FILE_INVENTORY.json':
            b=p.read_bytes();inventory.append(dict(relative_path=rel.as_posix(),size_bytes=len(b),sha256=digest(b),purpose=category('foundation_successor_models/'+rel.as_posix())))
    write(PKG/'audit/FILE_INVENTORY.json',json.dumps(dict(schema='SUCCESSOR_FINAL_REVIEW_INVENTORY_V1',base_HEAD=BASE,files=inventory,self_hash='covered by master ZIP manifest/archive SHA'),indent=2)+'\n')
    tree(PKG);tree(AUD)
    collector_sources=ALLOWED|{'foundation_v1/recovery_index.py','foundation_v1/stage_certification.py','foundation_v1/operational_release.py','tests/test_foundation_v1_fast_resume.py','tests/test_foundation_v1_final_fixes.py','foundation_v1_artifacts/preexecution_release.json','foundation_v1_artifacts/operational_patch_release_resume_v3.json'}
    for name in sorted(collector_sources):add(name,(ROOT/name).read_bytes())
    add('foundation_v1_artifacts/fast_resume_validation/benchmark_recovery.py',(ROOT/'foundation_v1_artifacts/fast_resume_validation/benchmark_recovery.py').read_bytes())
    with zipfile.ZipFile(old) as z:
        previous={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
        for name,b in previous.items():
            if name.startswith('reference_only/'):add(name,b)
    for name in ['foundation_v1/full_campaign.py','foundation_v1/simple_campaign.py']:
        archive='reference_only/final_fix_base/'+name
        add(archive,git('show',BASE+':'+name))
    def previous_bytes(name):
        if name in previous:return previous[name]
        for prefix in ['reference_only/foundation_contract/','reference_only/base_collector/']:
            if prefix+name in previous:return previous[prefix+name]
        return None
    differences=[];patches={'training_final_correction.patch':'','collector_final_correction.patch':''}
    for name in sorted(files):
        is_training=name.startswith('foundation_successor_models/')
        if not is_training and name not in collector_sources:continue
        old_data=previous_bytes(name);new=files[name]
        if old_data==new:continue
        differences.append(dict(path=name,change='ADDED' if old_data is None else 'MODIFIED',previous_sha256=digest(old_data) if old_data is not None else None,current_sha256=digest(new)))
        if is_training and ('/audit/' in name or Path(name).suffix not in ('.py','.yaml','.md','.txt')):continue
        if not is_training or Path(name).suffix in ('.py','.yaml','.md','.txt'):
            key='training_final_correction.patch' if is_training else 'collector_final_correction.patch'
            patches[key]+=''.join(difflib.unified_diff((old_data or b'').decode('utf8').splitlines(keepends=True),new.decode('utf8').splitlines(keepends=True),fromfile='a/'+name if old_data is not None else '/dev/null',tofile='b/'+name))
    for name,text in patches.items():add('patches/'+name,text.encode())
    add('patches/changes_against_previous_supervisor.json',(json.dumps(dict(previous_zip_sha256=digest(old.read_bytes()),changes=differences),indent=2)+'\n').encode())
    add('run_supervisor_tests.py',(AUD/'run_supervisor_tests.py').read_bytes())
    readme=f'''# Final four-fix supervisor review

Local uncommitted overlay on base {BASE}, branch foundation-successor-models-prep. No SCC access/deployment, real FlexDC, real Foundation training, commit, push or merge. Start with supervisor_final_fix_audit/FINAL_FIX_AUDIT.md.

## Four fixes and where to inspect

1. Fresh training safety: foundation_successor_models/flexmodels/training.py prepare_run_directory runs before any log or state write. All three trainers share it. Tests: test_training_run_directory.py.
2. Stage skipping: foundation_v1/stage_certification.py; run_full/certify integration in simple_campaign.py; invalidation hook/generation in recovery_index.py. Certificate path: <partition>/collector_v2/<stage>/stage_certification.json. No historical receipt opens on a valid certified-stage skip; strong audit remains mandatory for missing/changed cache identity.
3. Status: simple_campaign._status_event/live_status; timestamp-ordered valid_complete projection including invalidation and later repair. Legacy cache rebuilt.
4. Provenance: historical preexecution_release.json is byte-identical to base; operational_patch_release_resume_v3.json is a separate immutable review record acknowledging collection already started. operational_release.py validates it and supports an explicit future immutable source-transition receipt. Actual deployment fields are null; no production transition receipt exists from this task.

See STAGE_CERTIFICATE_DESIGN.md for identity/watermark checks and immutable-receipt assumptions. Direct out-of-band corruption must trigger a strong audit; the cheap cache is not continuous bit-rot monitoring. First missing certificates need one full certification. Final export checks remain intact.

## One-command successor tests after unzip

Use a Python environment with foundation_successor_models/requirements.txt plus pytest and scikit-learn (the latter only for read-only V3 reference tests). From this extracted archive root:

```
python -B run_supervisor_tests.py
```

The conftest automatically adds the audit-only Foundation reference directories. No PYTHONPATH adjustment is needed. A clean temporary unpack with PYTHONPATH unset passed all {totals['unpacked_training']['passed']} tests. Production training code does not import the reference copies.

## Collector tests

Collector files/tests are a portable overlay on a base {BASE} checkout containing its original scientific inputs. This is intentionally not a full repository/data clone. Use the pinned Foundation Python3.10.x/NumPy2.2.6/pandas2.3.3/SciPy1.15.3 environment and editable simulator installation. Exact full suite:

```
python -B -m pytest tests/test_foundation_v1.py tests/test_foundation_v1_cleanup.py tests/test_foundation_v1_fast_resume.py tests/test_foundation_v1_final_fixes.py tests/test_foundation_v1_full_campaign.py tests/test_foundation_v1_legacy_entry_fixture.py tests/test_foundation_v1_nan.py tests/test_foundation_v1_patch.py tests/test_foundation_v1_resilience.py tests/test_foundation_v1_simple_collector.py -q
```

All are synthetic/fake-simulator tests. Saved logs: {totals['training']['passed']} training and {totals['collector']['passed']} collector tests passed, with no failures/skips/pytest warnings. The recorded collector wall time includes a large outlier in the existing real-planners/fake-simulator state-machine test; see TIMING_NOTE.md. No new scientific workload was executed and no SCC performance claim is made.

## Audits, patches and future deployment

All current artifacts are under supervisor_final_fix_audit/. Earlier training audit documents remain historical evidence; the final audit is authoritative for this revision. patches/ contains diffs against the previous supervisor ZIP and a change manifest. Model architecture/objective files remain byte-identical. Previous ZIPs were not overwritten.

MID_CAMPAIGN_PROVENANCE.md gives exact future verify/transition/resume commands. Do not deploy before supervisor approval or while the current SCC allocation is running. Source guard identity: {release['uncommitted_source_sha256']}. Scientific master SHA: {release['phaseA_sha256']}.

SUPERVISOR_MANIFEST.json lists each payload member's path, size, SHA and category. Its own bytes are directly verified during ZIP construction and sealed by the archive SHA. Read-only references are marked 0444. No .git, .env content, actual receipts/data/results, virtual environments, runtime ledgers, checkpoints, caches or old ZIPs are included.
'''
    add('SUPERVISOR_README.md',readme.encode())
    manifest=dict(schema='FINAL_SUPERVISOR_REVIEW_V1',base_HEAD=BASE,branch='foundation-successor-models-prep',four_fixes='PASS',source_identity=release['uncommitted_source_sha256'],scientific_master_sha256=release['phaseA_sha256'],files=[dict(relative_path=n,size_bytes=len(b),sha256=digest(b),category=category(n)) for n,b in sorted(files.items())],self_hash='Manifest bytes are checked directly; archive SHA covers manifest. Self hash intentionally excluded.')
    add('SUPERVISOR_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode())
    archive=OUT/'flexdc_final_training_and_resume_fixes_supervisor_0190fe6.zip';assert not archive.exists(),'Never overwrite previous review ZIPs'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,b in sorted(files.items()):
            if name.startswith('reference_only/'):
                info=zipfile.ZipInfo(name);info.external_attr=0o444<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
            else:z.writestr(name,b)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
        for name,b in files.items():assert digest(z.read(name))==digest(b),name
        for row in manifest['files']:assert digest(z.read(row['relative_path']))==row['sha256']
    result=dict(relative_path=archive.relative_to(ROOT).as_posix(),absolute_path=str(archive),size_bytes=archive.stat().st_size,file_count=len(files),sha256=digest(archive.read_bytes()),zip_integrity_verified=True,all_member_hashes_verified=True,reference_files=sum(n.startswith('reference_only/') for n in files))
    write(archive.with_suffix('.zip.sha256'),result['sha256']+'  '+archive.name+'\n');write(archive.with_suffix('.zip.json'),json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
