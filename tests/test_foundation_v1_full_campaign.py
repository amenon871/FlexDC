"""End-to-end mocked collection: no real FlexDC or scientific outcomes."""
import copy
import json
from argparse import Namespace
from pathlib import Path
import pytest

from foundation_v1 import campaign,full_campaign as full,scope
from am_flexdc_foundation_v1_common import read_json,write_json

class Backend:
    def __init__(self,tmp):
        self.root=tmp;self.progress={};self.generated=[];self.submitted=[];self.commands=[];self.exports=[];self.exported=set();self.fail_once=None;self.shortfall=None
        for p in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST'):self.plan(p,'phaseA')
    def path(self,stage,part,task_id=None):return self.root/'plans'/f'{part}_{stage}.csv.gz'
    def plan(self,part,stage):
        path=self.path(stage,part);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'MOCK_COORDINATES_NOT_SCIENTIFIC')
        write_json(path.with_suffix('.receipt.json'),{'synthetic':True})
    def expected(self,part,stage):return full.counts_for(part)['tasks']*(150 if stage=='phaseA' else 20)
    def seed(self,part,stage,done=None):
        self.plan(part,stage);self.progress[part,stage]=self.expected(part,stage) if done is None else done
    def inspect(self,part,stage):
        if not self.path(stage,part).exists():return {'status':'NOT_GENERATED'}
        n=self.progress.get((part,stage),0)
        return {'status':'COMPLETE' if n==self.expected(part,stage) else 'TECHNICAL_RETRY_REQUIRED' if n else 'PENDING'}
    def validate(self,path,part,stage):return {'status':'VALID_PLAN','rows':self.expected(part,stage),'plan_sha256':'MOCK'}
    def generate(self,database,part,wave_number=None,repeat=False,unlock=False):
        stage='repeat' if repeat else f'wave{wave_number:02d}';prior=full.STAGES[full.STAGES.index(stage)-1]
        assert self.progress.get((part,prior))==self.expected(part,prior)
        self.generated.append((part,stage));self.plan(part,stage);return self.path(stage,part)
    def execute(self,a,part):
        stage=a.stage;n=self.expected(part,stage);done=self.progress.get((part,stage),0);key=(part,stage)
        cmd=campaign.commands(self.path(stage,part),a.max_workers,'MOCK_ONLY_'+part+'_'+stage,part);self.commands.append(cmd)
        assert a._campaign_lock_held and a.execute and a.resume
        if key==self.shortfall:
            return {'status':'STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL','scientifically_complete':False,
                'counts':{'planned':n,'VALID_COMPLETE':n-1,'INSUFFICIENT_JOB_TYPE_EVIDENCE':1}}
        if key==self.fail_once:
            half=(n-done)//2;self.progress[key]=done+half;self.submitted.append((key,half));self.fail_once=None
            raise RuntimeError('SYNTHETIC TECHNICAL_RETRY_REQUIRED')
        missing=n-done
        if missing:self.submitted.append((key,missing))
        self.progress[key]=n
        return {'status':'VALID_COMPLETE','scientifically_complete':True,'counts':{'planned':n,'VALID_COMPLETE':n,'EXECUTION_COMPLETE':n,'missing':0,'INSUFFICIENT_JOB_TYPE_EVIDENCE':0},
            'simulation_calls_this_invocation':missing,'simulation_rows_submitted_this_invocation':missing}
    def export(self,part,unlock):
        assert all(self.progress.get((part,s))==self.expected(part,s) for s in full.STAGES)
        if part not in self.exported:self.exports.append(part);self.exported.add(part)
        return {'status':'COMPLETE'}
    def metrics(self,part):
        assert all(self.progress.get((part,s))==self.expected(part,s) for s in full.STAGES)
        c=full.counts_for(part)
        return {'valid_rows':c['total'],'repeat_rows':c['repeat'],'evidence_shortfall_rows':0,'technical_retry_row_submissions':0}

@pytest.fixture
def backend(tmp_path,monkeypatch):
    b=Backend(tmp_path)
    monkeypatch.setattr(full,'ROOT',tmp_path)
    monkeypatch.setattr(full,'verify_release',lambda:{'uncommitted_source_sha256':'MOCK_RELEASE'})
    monkeypatch.setattr(full,'inspect_stage',b.inspect);monkeypatch.setattr(full,'validate_plan',b.validate)
    monkeypatch.setattr(full,'export_scope',b.export);monkeypatch.setattr(full,'ledger_metrics',b.metrics)
    monkeypatch.setattr(scope,'plan_path',b.path)
    monkeypatch.setattr(scope,'run_root',lambda p:tmp_path/'runs'/scope.group(p))
    monkeypatch.setattr(scope,'database_path',lambda p:tmp_path/'runs'/scope.group(p)/'MOCK.sqlite')
    monkeypatch.setattr(scope,'assert_database_scope',lambda *args:None)
    monkeypatch.setattr(campaign,'execute_scope',b.execute)
    monkeypatch.setattr(full.adaptive,'generate_stage',b.generate)
    monkeypatch.setattr(full.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_GIT_SHA')
    monkeypatch.setattr(full.subprocess,'call',lambda *a,**k:pytest.fail('Real subprocess forbidden'))
    return b

def args(workers=96,execute=True,unlock=True):
    return Namespace(max_workers=workers,execute=execute,resume=True,unlock_locked_test=unlock,
        stage=None,full_campaign=True,database=None,task=None,chunk_index=0,chunk_count=1)

@pytest.mark.parametrize('workers',[32,96])
def test_full_happy_path_order_workers_counts_and_exports(backend,workers):
    state=full.run_full(args(workers),'ALL')
    assert state['status']=='FOUNDATION_V1_COMPLETE' and state['valid_rows']==500000
    assert state['simulation_calls_this_invocation']==500000
    assert [k for k,n in backend.submitted]==[(p,s) for p in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') for s in full.STAGES]
    assert backend.generated==[(p,s) for p in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') for s in full.STAGES[1:]]
    assert len(backend.commands)==12
    assert all(c[c.index('--plan-max-workers')+1]==str(workers) for c in backend.commands)
    assert backend.exports==['TRAIN_DEV','LOCKED_SYNTHETIC_TEST']
    assert state['expected']['TRAIN_DEV']['total']==450000 and state['expected']['LOCKED_SYNTHETIC_TEST']['total']==50000

@pytest.mark.parametrize('stage',['phaseA','wave02','wave03'])
def test_resume_from_partial_stage_skips_terminal_rows(backend,stage):
    for old in full.STAGES[:full.STAGES.index(stage)]:backend.seed('TRAIN_DEV',old)
    n=backend.expected('TRAIN_DEV',stage);backend.seed('TRAIN_DEV',stage,n//2)
    state=full.run_full(args(),'ALL')
    assert state['status']=='FOUNDATION_V1_COMPLETE'
    assert backend.submitted[0]==(('TRAIN_DEV',stage),n-n//2)
    assert not any(p=='TRAIN_DEV' and s in full.STAGES[:full.STAGES.index(stage)] for ((p,s),_) in backend.submitted)
    assert ('TRAIN_DEV',stage) not in backend.generated

def test_resume_after_train_dev_then_locked_wave1(backend):
    for stage in full.STAGES:backend.seed('TRAIN_DEV',stage)
    backend.seed('LOCKED_SYNTHETIC_TEST','phaseA');backend.seed('LOCKED_SYNTHETIC_TEST','wave01',100)
    state=full.run_full(args(),'ALL')
    assert state['status']=='FOUNDATION_V1_COMPLETE'
    assert all(k[0]=='LOCKED_SYNTHETIC_TEST' for k,n in backend.submitted)
    assert backend.submitted[0][0]==('LOCKED_SYNTHETIC_TEST','wave01')

def test_evidence_shortfall_stops_before_next_generation(backend):
    backend.shortfall=('TRAIN_DEV','phaseA')
    with pytest.raises(RuntimeError,match='SHORTFALL'):full.run_full(args(),'ALL')
    assert not backend.generated and not backend.exports
    state=read_json(backend.root/'foundation_v1_runs/full_campaign_status__ALL.json')
    assert state['status']=='EVIDENCE_SHORTFALL_REQUIRES_POLICY'

def test_technical_failure_then_same_command_resumes_and_advances(backend):
    backend.fail_once=('TRAIN_DEV','wave02')
    with pytest.raises(RuntimeError,match='TECHNICAL_RETRY'):full.run_full(args(),'ALL')
    before=list(backend.submitted)
    result=full.run_full(args(),'ALL')
    assert result['status']=='FOUNDATION_V1_COMPLETE'
    assert sum(n for k,n in backend.submitted)==500000
    assert not any(k==('TRAIN_DEV','phaseA') for k,n in backend.submitted[len(before):])
    assert backend.generated.count(('TRAIN_DEV','wave02'))==1

def test_locked_full_execution_requires_unlock(backend):
    with pytest.raises(ValueError,match='sealed synthetic test partition'):full.run_full(args(unlock=False),'ALL')
    assert not backend.commands

def test_fully_complete_rerun_and_deleted_status_cache_make_zero_calls(backend):
    full.run_full(args(),'ALL');before=list(backend.submitted)
    status=backend.root/'foundation_v1_runs/full_campaign_status__ALL.json'
    status.write_text('broken interrupted status cache',encoding='utf8')
    result=full.run_full(args(),'ALL')
    assert result['status']=='FOUNDATION_V1_COMPLETE' and result['simulation_calls_this_invocation']==0
    assert result['simulation_rows_submitted_this_invocation']==0 and backend.submitted==before
    assert len(backend.exports)==2

def test_fake_complete_status_does_not_override_authoritative_progress(backend):
    folder=backend.root/'foundation_v1_runs';folder.mkdir()
    write_json(folder/'full_campaign_status__ALL.json',{'status':'FOUNDATION_V1_COMPLETE','valid_rows':500000})
    result=full.run_full(args(),'ALL')
    assert result['simulation_calls_this_invocation']==500000

def test_full_dry_run_has_all_steps_no_plans_or_workers(backend):
    before=list(backend.root.rglob('*'))
    result=full.run_full(args(execute=False,unlock=False),'ALL')
    assert [(r['partition'],r['stage']) for r in result['steps']]==[(p,s) for p in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST') for s in (*full.STAGES,'export')]
    assert sum(r.get('rows',0) for r in result['steps'])==500000
    assert [r['plan_status'] for r in result['steps'] if r['stage']=='phaseA']==['EXISTING_PLAN']*2
    assert all(r['plan_status']=='WILL_GENERATE_AFTER_PREVIOUS_STAGE' for r in result['steps'] if r['stage'] not in ('phaseA','export'))
    assert list(backend.root.rglob('*'))==before and not backend.generated and not backend.commands

def test_no_stage_flag_with_full_mode_and_no_chunk_overrides(monkeypatch):
    monkeypatch.setattr('sys.argv',['foundation','--full-campaign','--stage','wave01'])
    with pytest.raises(SystemExit):campaign.main()
    monkeypatch.setattr('sys.argv',['foundation','--full-campaign','--chunk-count','2'])
    with pytest.raises(SystemExit):campaign.main()

def test_environment_mismatch_fails_before_execution(monkeypatch):
    from foundation_v1.environment import versions
    original=full.versions
    monkeypatch.setattr('foundation_v1.environment.importlib.metadata.version',lambda name:'WRONG')
    with pytest.raises(ValueError,match='numerical stack mismatch'):versions(require=True)


def test_contract_failure_cannot_advance_even_if_counts_look_complete(backend,monkeypatch):
    original=backend.execute
    def execute(a,part):
        r=original(a,part);r['whole_scope_progress']={'failures':[{'reason':'Applied workload mismatch'}]};return r
    monkeypatch.setattr(campaign,'execute_scope',execute)
    with pytest.raises(ValueError,match='input/output disagreement'):full.run_full(args(),'ALL')
    assert not backend.generated


def test_approved_scientific_functions_and_master_bytes_unchanged():
    import ast,hashlib
    from am_flexdc_foundation_v1_common import ROOT,file_hash,read_json
    baseline=read_json(ROOT/'configs/foundation_v1/approved_scientific_function_hashes.json')
    cache={}
    for key,expected in baseline.items():
        path,fn=key.split('::')
        if path not in cache:cache[path]={n.name:n for n in ast.parse((ROOT/path).read_text(encoding='utf8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        assert hashlib.sha256(ast.dump(cache[path][fn],include_attributes=False).encode()).hexdigest()==expected,key
    assert file_hash(ROOT/'foundation_v1_artifacts/foundation_v1_phaseA_plan.csv.gz')==full.MASTER_SHA256


def test_real_stage_plan_quota_and_identity_checks():
    # The full300000-row existing-plan gate is independently run in the precommit
    # audit; this unit test checks the known stage-count algebra without workers.
    assert full.counts_for('TRAIN_DEV')=={'tasks':1800,'phaseA':270000,'adaptive':144000,'repeat':36000,'total':450000}
    assert full.counts_for('LOCKED_SYNTHETIC_TEST')=={'tasks':200,'phaseA':30000,'adaptive':16000,'repeat':4000,'total':50000}


def test_storage_failure_preserves_explicit_full_campaign_stop_state(backend,monkeypatch):
    from peacsim.foundation_v1_cleanup import CleanupRequired
    def stop(*args,**kwargs):
        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: FAKE file lock; scientific row remains terminal')
    monkeypatch.setattr(campaign,'execute_scope',stop)
    with pytest.raises(CleanupRequired):full.run_full(args(),'ALL')
    state=read_json(backend.root/'foundation_v1_runs/full_campaign_status__ALL.json')
    assert state['status']=='TRANSIENT_CLEANUP_REQUIRED'
    assert not backend.generated and not backend.submitted
