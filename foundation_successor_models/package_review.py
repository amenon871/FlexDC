"""Read-only reference copies belong in the ZIP only. Never stage or commit."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path,PureWindowsPath

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
SKIP={'.git','dist','.validation_runs','.pytest_cache','__pycache__','wandb','cache','.cache','runs','training_runs'}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(path):return not any(x.lower().startswith('.env') for x in path.parts) and not any(x in SKIP for x in path.parts)


def main():
    base=json.loads((PACKAGE/'audit/BASE_STATE.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines();status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    if head!=base['base_HEAD'] or branch!='foundation-successor-models-prep' or changed or staged:raise ValueError('Isolation/branch/HEAD/staging gate failed; do not package')
    original=json.loads((PACKAGE/'audit/ORIGINAL_FILE_HASHES.json').read_text())
    for name,expected in original.items():
        if not safe(Path(name)):raise ValueError('Forbidden original hash path')
        if file_hash(ROOT/name)!=expected:raise ValueError('Pre-existing file changed: '+name)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    if any(not x.startswith('foundation_successor_models/') for x in untracked):raise ValueError('Unexpected new file outside isolated package')
    reference=json.loads((PACKAGE/'audit/REFERENCE_FILES.json').read_text());refs={}
    for record in reference:
        path=Path(record['path'])
        if not safe(path):raise ValueError('Forbidden reference path')
        try:relative=path.relative_to(ROOT).as_posix();archive='reference_only/foundation_contract/'+relative
        except ValueError:archive='reference_only/v3_original/'+path.name
        if not path.exists() and 'archive_path' in record:path=ROOT/record['archive_path']
        if file_hash(path)!=record['sha256']:raise ValueError('Read-only reference changed: '+str(path))
        data=path.read_bytes()
        # Never package an accidentally embedded real credential. No match value
        # is printed. Actual .env paths are rejected before any read.
        if re.search(rb'(?:sk-proj-|wandb_v1_)[A-Za-z0-9_-]{20,}',data):raise ValueError('Potential credential in reference; human redaction review required: '+str(path))
        if archive in refs:raise ValueError('Reference archive name collision')
        record['archive_path']=archive;refs[archive]=data
    (PACKAGE/'audit/REFERENCE_FILES.json').write_text(json.dumps(reference,indent=2)+'\n')
    isolation=dict(status='PASS',base_HEAD=head,branch=branch,pre_existing_hashes_verified=len(original),git_status_short=status,git_diff_name_only=changed,git_staged_name_only=staged,no_commit=True,no_push=True,no_merge=True,reference_hashes_verified=len(reference))
    (PACKAGE/'audit/ISOLATION_RESULT.json').write_text(json.dumps(isolation,indent=2)+'\n')
    inventory=[];files={}
    for path in sorted(PACKAGE.rglob('*')):
        rel=path.relative_to(PACKAGE)
        if not path.is_file() or not safe(rel) or path.suffix in ('.pt','.pth','.pyc','.zip') or rel.as_posix()=='audit/FILE_INVENTORY.json':continue
        data=path.read_bytes()
        if len(data)>8*1024*1024:raise ValueError('Unexpected large review file: '+str(rel))
        purpose='configuration' if rel.parts[0]=='configs' else 'validation and audit' if rel.parts[0]=='audit' else 'CPU correctness tests' if rel.parts[0]=='tests' else 'TEST FIXTURE ONLY; NOT SCIENTIFIC DATA' if rel.parts[0]=='fixtures' else 'documentation' if path.suffix=='.md' else 'training/evaluation source or dependencies'
        inventory.append(dict(relative_path=rel.as_posix(),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),purpose=purpose));files['foundation_successor_models/'+rel.as_posix()]=data
    info=dict(schema='SUCCESSOR_REVIEW_INVENTORY_V1',base_HEAD=head,files=inventory,exclusions=['dist (no recursive ZIP)','runtime caches/checkpoints/W&B/pytest/pycache','any .env file or directory','FILE_INVENTORY.json self-hash (covered instead by the ZIP top-level manifest)'])
    ip=PACKAGE/'audit/FILE_INVENTORY.json';ip.write_text(json.dumps(info,indent=2)+'\n');files['foundation_successor_models/audit/FILE_INVENTORY.json']=ip.read_bytes();files.update(refs)
    files['REFERENCE_HASHES.sha256']=''.join(hashlib.sha256(b).hexdigest()+'  '+n+'\n' for n,b in sorted(refs.items())).encode()
    files['FILE_HASHES.sha256']=''.join(hashlib.sha256(b).hexdigest()+'  '+n+'\n' for n,b in sorted(files.items())).encode()
    out=PACKAGE/'dist';out.mkdir(exist_ok=True);zip_path=out/('foundation_successor_models_review_bundle_'+head[:7]+'.zip')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            if name.startswith('reference_only/'):
                entry=zipfile.ZipInfo(name);entry.external_attr=0o444<<16;entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,data)
            else:z.writestr(name,data)
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC validation failed')
        for line in z.read('FILE_HASHES.sha256').decode().splitlines():
            expected,name=line.split('  ',1)
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('ZIP member hash mismatch: '+name)
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
    result=dict(zip_path=str(zip_path),size_bytes=zip_path.stat().st_size,sha256=file_hash(zip_path),file_count=len(files),reference_files=len(refs),integrity='PASS')
    zip_path.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+zip_path.name+'\n');zip_path.with_suffix('.zip.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
