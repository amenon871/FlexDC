"""Local synthetic receipt/index benchmark; never invokes a simulator or campaign."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tests.test_foundation_v1_fast_resume import fixture,publish
from foundation_v1.recovery_index import RecoveryIndex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical',type=int,default=5000)
    p.add_argument('--delta',type=int,default=10)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('RECOVERY_BENCHMARK.json'))
    a=p.parse_args();results=[]
    with tempfile.TemporaryDirectory(prefix='foundation_recovery_benchmark_') as temporary:
        base=Path(temporary);area,plan,keys=fixture(base,a.historical+a.delta)
        for key in keys[:a.historical]:publish(area,key)
        for name in ['authoritative_full_rebuild','indexed_delta_resume','indexed_no_delta_resume']:
            if name=='indexed_delta_resume':
                for key in keys[a.historical:]:publish(area,key)
            started=time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):index=RecoveryIndex(plan,area,keys)
            seconds=time.perf_counter()-started
            results.append(dict(scenario=name,completed=len(index.done),delta=a.delta if name=='indexed_delta_resume' else 0,
                elapsed_seconds=seconds,receipt_opens=index.stats['full_historical_receipt_opens']+index.stats['verified_new_receipts'],
                historical_receipt_opens=index.stats['full_historical_receipt_opens'],journal_lines_processed=index.stats['replayed_journal_events'],
                sqlite_rows_loaded=index.stats['recovery_index_rows']))
            index.close()
        # Measure ONLY batched derived-index update overhead. The supplied bodies
        # stand for receipts already validated by the parent, outside this timer.
        overhead_root=base/'overhead';overhead_root.mkdir()
        update_area,update_plan,update_keys=fixture(overhead_root,1000)
        with contextlib.redirect_stdout(io.StringIO()):index=RecoveryIndex(update_plan,update_area,update_keys)
        bodies=[dict(Plan_Row_ID=k,timestamp=time.time()) for k in update_keys]
        started=time.perf_counter();control=set()
        for body in bodies:control.add(body['Plan_Row_ID'])
        control_seconds=time.perf_counter()-started
        batches=[]
        for start in range(0,len(bodies),100):
            started=time.perf_counter()
            for body in bodies[start:start+100]:index.accept(body)
            index.flush(force=True);batches.append(time.perf_counter()-started)
        index.close()
        update=dict(rows=len(bodies),batch_size=100,batches=len(batches),set_only_control_seconds=control_seconds,
                    total_index_accept_and_flush_seconds=sum(batches),mean_batch_seconds=sum(batches)/len(batches),
                    max_batch_seconds=max(batches),seconds_per_accepted_row=sum(batches)/len(bodies),
                    receipt_persistence_and_simulator_cost_excluded=True,
                    claim='Derived-index microbenchmark only, not measured SCC collection throughput.')
    result=dict(test_fixture_only=True,simulator_calls=0,platform=sys.platform,scenarios=results,
                batched_update_overhead=update,scc_target='SCC GPFS <5-minute target remains to be verified after deployment.')
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
