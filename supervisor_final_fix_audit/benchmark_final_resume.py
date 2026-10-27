"""Synthetic multi-stage parent benchmark; never launches FlexDC."""
import contextlib,io,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from types import SimpleNamespace
from pytest import MonkeyPatch
from tests.test_foundation_v1_final_fixes import multi_stage_fixture
from foundation_v1 import simple_campaign as full,collector_receipts as receipts


def main():
    with tempfile.TemporaryDirectory(prefix='foundation_certified_resume_') as tmp, MonkeyPatch.context() as m:
        with contextlib.redirect_stdout(io.StringIO()):multi_stage_fixture(Path(tmp),m,n=50)
        loads=[];stages=[];audits=[];actual_load=receipts.load;actual_stage=full.execute_stage;actual_audit=full.certify
        started=time.perf_counter();active_started=[]
        def load(area,key):loads.append(Path(area).name);return actual_load(area,key)
        def stage(a,p):stages.append(a.stage);active_started.append(time.perf_counter()-started);return actual_stage(a,p)
        def audit(plan,area,database):audits.append(Path(area).name);return actual_audit(plan,area,database)
        m.setattr(receipts,'load',load);m.setattr(full,'execute_stage',stage);m.setattr(full,'certify',audit)
        with contextlib.redirect_stdout(io.StringIO()):state=full.run_full(SimpleNamespace(max_workers=4,execute=True,resume=True,unlock_locked_test=False,storage_emergency_gib=0),'TRAIN_DEV')
        result=dict(test_fixture_only=True,simulator_calls=0,previous_certified_stages=2,previous_stage_rows=100,
            certificates_validated=state['resume_telemetry']['stages_skipped'],certificate_checks=state['resume_telemetry']['certificates_checked'],
            stages_skipped=state['resume_telemetry']['stages_skipped'],previous_stage_historical_receipt_opens=sum(s in ('phaseA','wave01') for s in loads),
            previous_stage_strong_audit_calls=sum(s in ('phaseA','wave01') for s in audits),
            active_stage_receipt_opens=loads.count('wave02'),active_stage_strong_audit_calls=audits.count('wave02'),
            startup_to_active_stage_seconds=active_started[0],total_fixture_resume_seconds=time.perf_counter()-started,
            stage_execution_order=stages,scc_gpfs_target='UNVERIFIED until reviewed deployment; local timing is not an SCC measurement')
        assert result['previous_stage_historical_receipt_opens']==result['previous_stage_strong_audit_calls']==0
        assert stages==['wave02'] and result['stages_skipped']==2
    path=Path(__file__).with_name('MULTI_STAGE_RESUME_PERFORMANCE.json');path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
