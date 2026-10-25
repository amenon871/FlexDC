from tests.test_foundation_v1_legacy_entry_fixture import invoke as legacy_stage_entry
"""Per-row storage/recovery tests. All executions and scientific labels are fake."""
import ast
import copy
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import pytest
from peacsim import foundation_v1_cleanup as cleanup
from foundation_v1 import audit, campaign, scope, storage
from am_flexdc_foundation_v1_common import ROOT, contexts, write_json, write_csv, read_json
from tests.test_foundation_v1 import fake_receipt
from tests.test_foundation_v1_patch import insufficient, prepare_fake_campaign


@pytest.fixture
def task():
    return copy.deepcopy(contexts()[0])


def persist(output, row, receipt, append=False):
    output.mkdir(parents=True, exist_ok=True)
    for name, key in [('grid_search_results.csv', 'results'), ('grid_search_diagnostics.csv', 'diagnostics')]:
        existed = (output / name).exists()
        with (output / name).open('a' if append else 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(receipt[key]))
            if not append or not existed:
                writer.writeheader()
            writer.writerow(receipt[key])
    cleanup.sync_compact_csvs(output)
    write_json(output / 'foundation_evidence' / (row['plan_row_id'] + '.json'), receipt)


def transients(output):
    output.mkdir(parents=True, exist_ok=True)
    for name in cleanup.TRANSIENT_FILES:
        (output / name).write_bytes(b'SYNTHETIC TRANSIENT' * 7)


def begin(output, row):
    cleanup.begin_row(output, {**row, 'design_version': 'FOUNDATION_V1'})


def test_success_preserves_compact_and_recovery_files(task, tmp_path):
    row, receipt = fake_receipt(task)
    begin(tmp_path, row)
    transients(tmp_path)
    for name in ['worker.log', 'chunk.json', 'ledger.sqlite', 'receipt.json', 'plan.csv.gz',
                 'qos_summary.csv', 'tracking_error_90th.csv', 'cost_power_rsr.csv', 'QoSDegAvg.csv']:
        (tmp_path / name).write_text('KEEP')
    persist(tmp_path, row, receipt)
    result = cleanup.finalize_row(tmp_path, row['plan_row_id'])
    assert result['execution_status'] == 'EXECUTION_COMPLETE'
    assert result['cleanup_status'] == 'CLEANUP_COMPLETE'
    assert result['transient_cleanup_files_deleted'] == len(cleanup.TRANSIENT_FILES)
    assert result['transient_cleanup_bytes_deleted'] > 0
    cleanup.assert_absent(tmp_path)
    assert not (tmp_path / 'power_trace.csv').exists()
    assert (tmp_path / 'ledger.sqlite').read_text(encoding='utf8') == 'KEEP'
    assert (tmp_path / 'worker.log').exists()
    assert (tmp_path / 'grid_search_results.csv').exists()
    assert (tmp_path / 'grid_search_diagnostics.csv').exists()
    assert (tmp_path / 'foundation_evidence' / (row['plan_row_id'] + '.json')).exists()
    # Repeated cleanup/recovery cannot repeat deletion telemetry.
    cleanup.recover_output(tmp_path)
    assert cleanup.telemetry([tmp_path])['transient_cleanup_files_deleted'] == len(cleanup.TRANSIENT_FILES)


def test_real_worker_loop_cleans_before_next_mock_simulator(task, tmp_path):
    # Compile the ACTUAL exact-plan loop, replacing only simulator extraction.
    source = ast.parse((ROOT / 'src/peacsim/am_data_extraction_wizard.py').read_text(encoding='utf8'))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'grid_search_plan_rows')
    first, receipt = fake_receipt(task)
    second_task = copy.deepcopy(task)
    second_task['Task_ID'] += '_ROW_TWO_SYNTHETIC'
    second, other = fake_receipt(second_task)
    rows = [{**r, 'design_version': 'FOUNDATION_V1', 'weights': list(r['weights'])} for r in [first, second]]
    receipts = {first['plan_row_id']: receipt, second['plan_row_id']: other}
    calls = []
    def objective(params, *args, **kwargs):
        metadata = kwargs['plan_row_metadata']
        # Mandatory assertion happens BEFORE row 2's mocked FlexDC creates files.
        cleanup.assert_absent(tmp_path)
        begin(tmp_path, metadata)
        calls.append(metadata['plan_row_id'])
        transients(tmp_path)
        persist(tmp_path, metadata, receipts[metadata['plan_row_id']], append=True)
        cleanup.finalize_row(tmp_path, metadata['plan_row_id'])
        return 1.0
    namespace = {'os': os, 'objective_function': objective,
        'initialize_grid_search_output_files': lambda *args: None,
        'set_experiment_utilization': lambda *args: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual worker loop>', 'exec'), namespace)
    namespace['grid_search_plan_rows'](None, SimpleNamespace(runtime_policy_name='AQA'), SimpleNamespace(),
        None, SimpleNamespace(job_type_count=receipt['applied']['J']), str(tmp_path), rows, {})
    assert calls == [first['plan_row_id'], second['plan_row_id']]
    cleanup.assert_absent(tmp_path)
    assert cleanup.telemetry([tmp_path])['transient_cleanup_files_deleted'] == 2 * len(cleanup.TRANSIENT_FILES)


def test_insufficient_evidence_cleans_and_remains_terminal(task, tmp_path):
    row, receipt = fake_receipt(task)
    receipt = insufficient(receipt)
    begin(tmp_path, row)
    transients(tmp_path)
    persist(tmp_path, row, receipt)
    cleanup.finalize_row(tmp_path, row['plan_row_id'])
    cleanup.assert_absent(tmp_path)
    plan = tmp_path / 'p.csv.gz'; write_csv(plan, [row])
    db = tmp_path / 'ledger.sqlite'
    report = audit.ingest(plan, [tmp_path], db)
    assert report['status'] == 'STAGE_COMPLETE_WITH_EVIDENCE_SHORTFALL'
    assert campaign.pending_rows(plan, db) == []
    assert audit.read_task_observations(db, task['Task_ID']) == []
    assert read_json(tmp_path / 'foundation_evidence' / (row['plan_row_id'] + '.json'))['evidence_counts'][0]['observations'] == 0


@pytest.mark.parametrize('failure', ['missing_evidence', 'missing_diagnostics', 'bad_checksum', 'wrong_csv', 'duplicate_csv'])
def test_incomplete_extraction_never_deletes(task, tmp_path, failure):
    row, receipt = fake_receipt(task)
    begin(tmp_path, row); transients(tmp_path); persist(tmp_path, row, receipt)
    evidence = tmp_path / 'foundation_evidence' / (row['plan_row_id'] + '.json')
    if failure == 'missing_evidence': evidence.unlink()
    elif failure == 'missing_diagnostics': (tmp_path / 'grid_search_diagnostics.csv').unlink()
    elif failure == 'bad_checksum':
        receipt['body_sha256'] = 'wrong'; evidence.write_text(json.dumps(receipt))
    elif failure == 'wrong_csv':
        p = tmp_path / 'grid_search_results.csv'; p.write_text(p.read_text(encoding='utf8').replace(row['plan_row_id'], 'F1_WRONG'))
    else:
        with (tmp_path / 'grid_search_results.csv').open('a', newline='') as stream:
            csv.writer(stream).writerow(receipt['results'].values())
    with pytest.raises((ValueError, OSError)):
        cleanup.finalize_row(tmp_path, row['plan_row_id'])
    assert all((tmp_path / name).exists() for name in cleanup.TRANSIENT_FILES)
    assert cleanup._current(tmp_path)['execution_status'] == 'RUNNING'


def test_cleanup_failure_bounded_terminal_and_cleanup_only_resume(task, tmp_path, monkeypatch):
    row, receipt = fake_receipt(task)
    plan, db = prepare_fake_campaign(task, tmp_path, monkeypatch, row)
    out = scope.raw_root('TRAIN_DEV') / 'foundation_v1_fake'
    begin(out, row); transients(out); persist(out, row, receipt)
    original = cleanup._unlink; attempts = []
    def denied(path):
        if path.name == 'power_trace.csv':
            attempts.append(str(path)); raise PermissionError('SYNTHETIC SCC FILE LOCK')
        original(path)
    monkeypatch.setattr(cleanup, '_unlink', denied)
    monkeypatch.setattr(cleanup.time, 'sleep', lambda _: None)
    stop = tmp_path / 'storage_stop.json'
    monkeypatch.setenv(cleanup.STOP_ENV, str(stop))
    with pytest.raises(cleanup.CleanupRequired, match='SYNTHETIC SCC FILE LOCK'):
        cleanup.finalize_row(out, row['plan_row_id'])
    assert len(attempts) == 3
    state = cleanup._current(out)
    assert state['execution_status'] == 'EXECUTION_COMPLETE'
    assert state['cleanup_status'] == 'CLEANUP_REQUIRED'
    assert state['transient_cleanup_retries'] == 2
    assert state['errors'][-1]['path'] == str((out / 'power_trace.csv').resolve())
    assert stop.exists()
    with pytest.raises(cleanup.CleanupRequired): begin(out, row)
    audit.ingest(plan, [out], db)
    assert campaign.pending_rows(plan, db) == []
    with pytest.raises(cleanup.CleanupRequired): storage.gate([out], tmp_path, retry=False)
    # Same real campaign resume path: terminal ingest BEFORE cleanup, no simulator
    # subprocess even while cleanup still fails, nor after the filesystem recovers.
    monkeypatch.setattr(campaign, 'call_combo', lambda *a, **k: pytest.fail('Completed row was resimulated'))
    monkeypatch.setattr('sys.argv', ['foundation', '--execute', '--resume', '--database', str(db), '--task', task['Task_ID']])
    with pytest.raises(cleanup.CleanupRequired): legacy_stage_entry()
    assert campaign.pending_rows(plan, db) == []
    monkeypatch.setattr(cleanup, '_unlink', original)
    monkeypatch.delenv(cleanup.STOP_ENV)
    result = legacy_stage_entry()
    assert result['status'] == 'VALID_COMPLETE'
    assert result['simulation_calls_this_invocation'] == 0
    assert cleanup._current(out)['cleanup_status'] == 'CLEANUP_COMPLETE'
    cleanup.assert_absent(out)


def test_crash_after_receipt_before_cleanup_is_recovered(task, tmp_path):
    row, receipt = fake_receipt(task)
    begin(tmp_path, row); transients(tmp_path); persist(tmp_path, row, receipt)
    assert cleanup._current(tmp_path)['execution_status'] == 'RUNNING'
    with pytest.raises(cleanup.CleanupRequired): storage.gate([tmp_path], tmp_path, retry=False)
    assert cleanup.telemetry([tmp_path])['transient_cleanup_pending'] == 1
    cleanup.recover_output(tmp_path)
    assert cleanup._current(tmp_path)['execution_status'] == 'EXECUTION_COMPLETE'
    cleanup.assert_absent(tmp_path)


def test_true_simulator_crash_preserves_debug_files(task, tmp_path):
    row, _ = fake_receipt(task)
    begin(tmp_path, row); transients(tmp_path)
    cleanup.recover_output(tmp_path)
    assert all((tmp_path / name).exists() for name in cleanup.TRANSIENT_FILES)
    assert cleanup._current(tmp_path)['execution_status'] == 'RUNNING'
    with pytest.raises(cleanup.CleanupRequired): begin(tmp_path, row)


def test_non_foundation_behavior_unchanged(task, tmp_path):
    row, _ = fake_receipt(task); transients(tmp_path)
    cleanup.begin_row(tmp_path, {**row, 'design_version': 'HISTORICAL'})
    assert all((tmp_path / name).exists() for name in cleanup.TRANSIENT_FILES)
    assert not (tmp_path / 'foundation_cleanup').exists()
    # Whole historical wizard AST equality is also required by the original suite.


def test_combo_validation_requires_compact_not_transients(task, tmp_path, monkeypatch):
    import am_run_flexdc_dataset_parallel_combo as combo
    row, receipt = fake_receipt(task)
    begin(tmp_path, row); transients(tmp_path); persist(tmp_path, row, receipt)
    cleanup.finalize_row(tmp_path, row['plan_row_id'])
    monkeypatch.setattr(combo, 'latest_plan_output', lambda _: tmp_path)
    assert combo.validate_plan_worker_output('fake', [row['plan_row_id']])['rows'] == 1


def test_parallel_transient_peak_tracks_current_workers_not_completed_rows(task, tmp_path):
    workers = 4; barrier = threading.Barrier(workers); counts = []; lock = threading.Lock()
    def worker(index):
        output = tmp_path / str(index)
        for iteration in range(5):
            t = copy.deepcopy(task); t['Task_ID'] += f'_FAKE_{index}_{iteration}'
            row, receipt = fake_receipt(t)
            cleanup.assert_absent(output); begin(output, row); transients(output)
            barrier.wait()
            with lock: counts.append(sum((tmp_path / str(i) / 'power_trace.csv').exists() for i in range(workers)))
            barrier.wait()
            persist(output, row, receipt, append=True); cleanup.finalize_row(output, row['plan_row_id'])
            barrier.wait()
            assert not (output / 'power_trace.csv').exists()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    assert max(counts) == workers  # 20 completions, only 4 current power traces.
    assert all(not (tmp_path / str(i) / 'power_trace.csv').exists() for i in range(workers))
    assert cleanup.telemetry([tmp_path / str(i) for i in range(workers)])['transient_cleanup_files_deleted'] == 20 * len(cleanup.TRANSIENT_FILES)


def test_hook_order_and_foundation_stop_barriers():
    source = (ROOT / 'src/peacsim/am_data_extraction_wizard.py').read_text(encoding='utf8')
    objective = source[source.index('def objective_function('):source.index('def generate_weight_grid(')]
    assert objective.index('begin_row(') < objective.index('simulator.run()')
    assert objective.index('writer.writerow(raw_results)') < objective.index('writer.writerow(diagnostic_results)')
    assert objective.index('writer.writerow(diagnostic_results)') < objective.index('sync_compact_csvs(') < objective.index('write_evidence(') < objective.index('finalize_row(')
    combo = (ROOT / 'am_run_flexdc_dataset_parallel_combo.py').read_text(encoding='utf8')
    assert 'from foundation_v1.resilient import run' in combo
    assert 'FOUNDATION_WORKER_STOP' not in combo


def test_unowned_files_not_deleted_and_existing_evidence_not_overwritten(task, tmp_path):
    row, receipt = fake_receipt(task); transients(tmp_path)
    with pytest.raises(cleanup.CleanupRequired): begin(tmp_path, row)
    assert (tmp_path / 'power_trace.csv').exists()
    # Reset only synthetic files explicitly, then prove prepare refuses old compact evidence.
    for name in cleanup.TRANSIENT_FILES: (tmp_path / name).unlink()
    begin(tmp_path, row); persist(tmp_path, row, receipt); cleanup.finalize_row(tmp_path, row['plan_row_id'])
    with pytest.raises(ValueError, match='resume through the Foundation campaign'): cleanup.prepare_worker(tmp_path)


def test_allowlist_cannot_escape_output(tmp_path):
    with pytest.raises(ValueError): cleanup._path(tmp_path, '../grid_search_results.csv')
    with pytest.raises(ValueError): cleanup._path(tmp_path, 'foundation_evidence/x.json')


def test_pool_stops_pending_foundation_chunks_on_worker_failure(tmp_path,monkeypatch):
    # The historical test's generic-process global-stop expectation was the SCC
    # regression. Keep the storage invariant, but inject a TRUE cleanup failure.
    from tests.test_foundation_v1_resilience import queue_fixture
    from foundation_v1 import resilient
    fixture=queue_fixture(tmp_path,monkeypatch,cleanup_failure=True)
    with pytest.raises(SystemExit,match='TRANSIENT_CLEANUP_REQUIRED'):fixture.run()
    assert (fixture.logs/'foundation_cleanup_stop.json').exists()
    assert len(fixture.launched)<=2


def test_after_cleanup_only_recovery_next_row_can_begin(task, tmp_path, monkeypatch):
    row, receipt = fake_receipt(task)
    begin(tmp_path, row); transients(tmp_path); persist(tmp_path, row, receipt)
    original = cleanup._unlink
    monkeypatch.setattr(cleanup, '_unlink', lambda _: (_ for _ in ()).throw(PermissionError('FAKE LOCK')))
    monkeypatch.setattr(cleanup.time, 'sleep', lambda _: None)
    with pytest.raises(cleanup.CleanupRequired): cleanup.finalize_row(tmp_path, row['plan_row_id'])
    monkeypatch.setattr(cleanup, '_unlink', original)
    cleanup.recover_output(tmp_path)
    next_task = copy.deepcopy(task); next_task['Task_ID'] += '_NEXT_SYNTHETIC'
    next_row, _ = fake_receipt(next_task)
    begin(tmp_path, next_row)
    cleanup.assert_absent(tmp_path)
    assert cleanup._current(tmp_path)['Plan_Row_ID'] == next_row['plan_row_id']
