"""Package approved-scope local corrections for review; never stages or commits."""
from pathlib import Path,PurePosixPath
import difflib,hashlib,json,re,subprocess,zipfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'foundation_successor_models'
OUT=ROOT/'supervisor_bundles'
AUDIT=ROOT/'supervisor_patch_audit'
COL=ROOT/'foundation_v1_artifacts/fast_resume_validation'
BASE='0190fe60d53cddb6afbaffad9a67b82e50ccf0de'
SKIP={'.git','dist','.validation_runs','.pytest_cache','__pycache__','wandb','.cache','cache','runs','venv','.venv'}
CHANGED={'foundation_v1/collector_receipts.py','foundation_v1/simple_collector.py',
         'foundation_v1/simple_campaign.py','foundation_v1_artifacts/preexecution_release.json'}
def digest(data):return hashlib.sha256(data).hexdigest()
def safe(path):
    return not any(p.lower().startswith('.env') or p in SKIP for p in path.parts)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def write(path,body):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body,encoding='utf8',newline='\n')
def purpose(name):
    if name.startswith('reference_only/'):return 'read-only historical reference'
    if '/tests/' in name or name.startswith('tests/'):return 'unit/synthetic regression tests'
    if name.startswith('patches/'):return 'review diff or change manifest'
    if '/audit/' in name or name.startswith('supervisor_patch_audit/') or name.startswith('foundation_v1_artifacts/fast_resume_validation/'):return 'audit/validation evidence and review tooling'
    if '/fixtures/' in name:return 'TEST FIXTURE ONLY; NOT SCIENTIFIC DATA'
    return 'source/config/documentation'


def main():
    assert git('rev-parse','HEAD').decode().strip()==BASE
    assert git('branch','--show-current').decode().strip()=='foundation-successor-models-prep'
    assert not git('diff','--cached','--name-only').strip(),'Staging is forbidden'
    changed=set(git('diff','--name-only').decode().splitlines())
    assert changed==CHANGED,changed
    isolation=json.loads((AUDIT/'TRACKED_FILE_ISOLATION.json').read_text());assert isolation['status']=='PASS'
    oldzip=PACKAGE/'dist/foundation_successor_models_review_bundle_0190fe6.zip'
    assert digest(oldzip.read_bytes())=='a2444cf18576f0b3801042e4c77a876434ab6006138b04e1938779e52c0e9afe'
    files={}
    def add(name,data):
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and safe(p),name
        assert len(data)<8*1024*1024,('unexpected large file',name)
        assert name not in files,('duplicate member',name)
        if p.suffix not in ('.gz',):
            assert not re.search(rb'(?:sk-proj-|wandb_v1_)[A-Za-z0-9_-]{20,}',data),'Credential-pattern review required; no values printed'
        files[name]=data
    def add_tree(path):
        for p in sorted(path.rglob('*')):
            rel=p.relative_to(ROOT)
            if p.is_file() and safe(rel) and p.suffix not in ('.pt','.pth','.pyc','.zip','.sqlite','.db'):
                add(rel.as_posix(),p.read_bytes())
    OUT.mkdir(exist_ok=True)
    write(OUT/'README.md','Review archives only. No source commit, real training, simulation, or SCC deployment occurred.\n')
    write(AUDIT/'FINAL_GIT_STATE.txt','\n'.join('$ git '+cmd+'\n'+git(*cmd.split()).decode() for cmd in ['branch --show-current','rev-parse HEAD','status --short','diff --name-only','diff --stat','diff --cached --name-only']))
    # Training inventory describes source and evidence, excluding its own hash.
    inventory=[]
    for p in sorted(PACKAGE.rglob('*')):
        rel=p.relative_to(PACKAGE)
        if p.is_file() and safe(rel) and p.suffix not in ('.pt','.pth','.pyc','.zip') and rel.as_posix()!='audit/FILE_INVENTORY.json':
            data=p.read_bytes();inventory.append(dict(relative_path=rel.as_posix(),size_bytes=len(data),sha256=digest(data),purpose=purpose('foundation_successor_models/'+rel.as_posix())))
    write(PACKAGE/'audit/FILE_INVENTORY.json',json.dumps(dict(schema='SUCCESSOR_REVIEW_INVENTORY_V2',base_HEAD=BASE,files=inventory,self_hash_covered_by='master ZIP manifest and archive SHA'),indent=2)+'\n')
    add_tree(PACKAGE);add_tree(AUDIT);add_tree(COL)
    for name in sorted(CHANGED|{'foundation_v1/recovery_index.py','tests/test_foundation_v1_fast_resume.py'}):add(name,(ROOT/name).read_bytes())
    # Reuse only the previously reviewed read-only references, byte for byte.
    with zipfile.ZipFile(oldzip) as z:
        for name in z.namelist():
            if name.startswith('reference_only/') and not name.endswith('/'):add(name,z.read(name))
        previous={name:z.read(name) for name in z.namelist() if name.startswith('foundation_successor_models/') and not name.endswith('/')}
    for name in ['foundation_v1/collector_receipts.py','foundation_v1/simple_collector.py','foundation_v1/simple_campaign.py',
                 'foundation_v1/release.py','am_generate_flexdc_exact_sweep_plan.py','requirements-foundation-v1.txt',
                 'tests/test_foundation_v1.py','tests/test_foundation_v1_cleanup.py','tests/test_foundation_v1_simple_collector.py']:
        archive='reference_only/base_collector/'+name
        add(archive,git('show',BASE+':'+name))
    collector_diff=git('diff','--no-ext-diff','--binary','--',*sorted(CHANGED)).decode('utf8')
    for name in ['foundation_v1/recovery_index.py','tests/test_foundation_v1_fast_resume.py']:
        collector_diff+=''.join(difflib.unified_diff([],files[name].decode('utf8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+name))
    add('patches/collector_fast_resume.patch',collector_diff.encode())
    corrections=[];training_diff=''
    for name,data in sorted(files.items()):
        if not name.startswith('foundation_successor_models/'):continue
        old=previous.get(name)
        if old==data:continue
        corrections.append(dict(path=name,change='ADDED' if old is None else 'MODIFIED',old_sha256=digest(old) if old is not None else None,new_sha256=digest(data),new_size_bytes=len(data)))
        if Path(name).suffix in ('.py','.yaml','.txt') and '/audit/' not in name or name=='foundation_successor_models/README.md':
            training_diff+=''.join(difflib.unified_diff((old or b'').decode('utf8').splitlines(keepends=True),data.decode('utf8').splitlines(keepends=True),fromfile='a/'+name if old is not None else '/dev/null',tofile='b/'+name))
    add('patches/training_package_corrections.patch',training_diff.encode())
    add('patches/training_changes_against_previous_zip.json',(json.dumps(dict(previous_zip_sha256=digest(oldzip.read_bytes()),changes=corrections),indent=2)+'\n').encode())
    readme=(AUDIT/'SUPERVISOR_README.md').read_text(encoding='utf8')+'\nCollector tests/patches are a portable overlay on base checkout '+BASE+' with its original scientific inputs. The ZIP intentionally is not a full repository or dataset clone. Use the pinned Foundation numerical environment for collector tests; training fixture tests use the package requirements and included V3 reference copies.\n'
    add('SUPERVISOR_README.md',readme.encode())
    manifest=dict(schema='FLEXDC_SUPERVISOR_REVIEW_V1',base_HEAD=BASE,branch='foundation-successor-models-prep',no_commit=True,no_push=True,no_merge=True,no_scc_deployment=True,simulator_calls=0,real_training=False,
                  master_phaseA_sha256=isolation['master_phaseA_sha256_after'],files=[dict(relative_path=n,size_bytes=len(b),sha256=digest(b),category=purpose(n)) for n,b in sorted(files.items())],
                  self_hash='Manifest bytes are directly checked after ZIP creation; archive SHA seals the manifest. Self-referential member hash is intentionally excluded.')
    add('SUPERVISOR_MANIFEST.json',(json.dumps(manifest,indent=2)+'\n').encode())
    archive=OUT/'flexdc_successor_training_and_fast_resume_supervisor_0190fe6.zip'
    assert not archive.exists(),'Do not overwrite an existing supervisor ZIP'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            if name.startswith('reference_only/'):
                entry=zipfile.ZipInfo(name);entry.external_attr=0o444<<16;entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,data)
            else:z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)
        for name,data in files.items():assert digest(z.read(name))==digest(data),name
        for row in manifest['files']:assert digest(z.read(row['relative_path']))==row['sha256']
    result=dict(relative_path=archive.relative_to(ROOT).as_posix(),absolute_path=str(archive),size_bytes=archive.stat().st_size,file_count=len(files),sha256=digest(archive.read_bytes()),zip_integrity_verified=True,all_member_hashes_verified=True,reference_files=sum(n.startswith('reference_only/') for n in files))
    write(archive.with_suffix('.zip.sha256'),result['sha256']+'  '+archive.name+'\n')
    write(archive.with_suffix('.zip.json'),json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
