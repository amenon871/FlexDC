"""Foundation-only per-row storage finalization, independent of scientific status.

Only explicit filenames in the worker output directory may be removed. No simulator
is imported or invoked here. Interrupted finalization is recovered before scheduling.
"""
import csv
import hashlib
import json
import os
import stat
import time
from pathlib import Path

TRANSIENT_FILES = (
    'power_trace.csv', 'job_table.csv', 'PRtable.csv', 'base_weights.csv',
    'waiting_jobs_by_job_type.csv', 'probabilistic_results.csv',
)
STOP_ENV = 'FOUNDATION_V1_STORAGE_STOP'
MAX_ATTEMPTS = 3


class CleanupRequired(RuntimeError):
    """Scientific execution is terminal; only filesystem cleanup may be retried."""


def _atomic(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf8', newline='\n') as stream:
        json.dump(body, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _sync_directory(path):
    # Windows has no portable directory fsync; file flush + atomic replace remain.
    if os.name != 'nt':
        fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _area(output):
    return Path(output) / 'foundation_cleanup'


def _state_path(output, row_id):
    if not row_id.startswith('F1_') or not row_id.replace('_', '').isalnum():
        raise ValueError('Invalid Foundation Plan_Row_ID')
    return _area(output) / (row_id + '.json')


def _save(output, state):
    _atomic(_state_path(output, state['Plan_Row_ID']), state)
    _atomic(_area(output) / 'current.json', {'Plan_Row_ID': state['Plan_Row_ID']})


def _current(output):
    pointer = _area(output) / 'current.json'
    if not pointer.exists():
        return None
    return _read(_state_path(output, _read(pointer)['Plan_Row_ID']))


def _path(output, name):
    if name not in TRANSIENT_FILES:
        raise ValueError('Not on Foundation transient allowlist: ' + name)
    root = Path(output).resolve()
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise OSError('Unsafe transient path: ' + str(path))
    return path


def assert_absent(output):
    for name in TRANSIENT_FILES:
        path = _path(output, name)
        if path.exists():
            raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: previous transient still exists: ' + str(path))


def check_stop():
    signal = os.environ.get(STOP_ENV)
    if signal and Path(signal).exists():
        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: another Foundation worker stopped; ' + str(signal))


def prepare_worker(output):
    """Before CSV initialization: never overwrite compact evidence on chunk reuse."""
    check_stop()
    recover_output(output)
    if any((Path(output) / 'foundation_evidence').glob('F1_*.json')):
        raise ValueError('Foundation compact evidence already exists: resume through the Foundation campaign, not by overwriting a worker chunk')


def begin_row(output, metadata):
    if metadata.get('design_version') != 'FOUNDATION_V1':
        return
    check_stop()
    previous = _current(output)
    if previous and previous['cleanup_status'] != 'CLEANUP_COMPLETE':
        # A true failed attempt retains debugging files; campaign resume creates a
        # fresh worker directory. Never overwrite it with the next candidate.
        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: prior row not finalized: ' + previous['Plan_Row_ID'])
    assert_absent(output)
    row_id = str(metadata['plan_row_id'])
    _save(output, {'schema': 'FOUNDATION_V1_CLEANUP', 'Plan_Row_ID': row_id,
        'execution_status': 'RUNNING', 'cleanup_status': 'AWAITING_COMPACT_OUTPUTS',
        'transient_cleanup_files_deleted': 0, 'transient_cleanup_bytes_deleted': 0,
        'transient_cleanup_failures': 0, 'transient_cleanup_retries': 0,
        'errors': [], 'deleted': []})


def sync_compact_csvs(output):
    for name in ('grid_search_results.csv', 'grid_search_diagnostics.csv'):
        with (Path(output) / name).open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())


def verify_compact(output, row_id):
    """Durability barrier: receipt checksum AND exact matching CSV rows, then fsync.

    CSVs are compact per-worker chunks, not power traces or campaign-wide scans.
    The scientific ingest layer independently verifies requested/applied semantics.
    """
    path = Path(output) / 'foundation_evidence' / (row_id + '.json')
    with path.open('r+b') as stream:
        body = json.load(stream)
        os.fsync(stream.fileno())
    unsigned = dict(body)
    expected = unsigned.pop('body_sha256')
    encoded = json.dumps(unsigned, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    if hashlib.sha256(encoded).hexdigest() != expected or body['Plan_Row_ID'] != row_id:
        raise ValueError('Foundation evidence identity/checksum mismatch: ' + str(path))
    for filename, field in (('grid_search_results.csv', 'results'), ('grid_search_diagnostics.csv', 'diagnostics')):
        csv_path = Path(output) / filename
        with csv_path.open('r+', newline='', encoding='utf8') as stream:
            records = [r for r in csv.DictReader(stream) if r.get('Plan_Row_ID') == row_id]
            if len(records) != 1:
                raise ValueError('Required compact row missing/duplicated: ' + str(csv_path) + ' ' + row_id)
            wanted = {k: '' if v is None else str(v) for k, v in body[field].items()}
            if records[0] != wanted:
                raise ValueError('Compact CSV/evidence disagreement: ' + str(csv_path))
            stream.flush()
            os.fsync(stream.fileno())
    _sync_directory(path.parent)
    _sync_directory(Path(output))
    return body


def _unlink(path):
    path.unlink()


def finalize_row(output, row_id):
    # Never clean incomplete/invalid compact extraction. Verification happens
    # BEFORE changing scientific execution state or touching a transient file.
    verify_compact(output, row_id)
    state_path = _state_path(output, row_id)
    state = _read(state_path) if state_path.exists() else {
        'schema': 'FOUNDATION_V1_CLEANUP', 'Plan_Row_ID': row_id,
        'transient_cleanup_files_deleted': 0, 'transient_cleanup_bytes_deleted': 0,
        'transient_cleanup_failures': 0, 'transient_cleanup_retries': 0,
        'errors': [], 'deleted': []}
    state.update(execution_status='EXECUTION_COMPLETE', cleanup_status='CLEANUP_REQUIRED')
    _save(output, state)
    errors = []
    for name in TRANSIENT_FILES:
        for attempt in range(MAX_ATTEMPTS):
            path = Path(output).resolve() / name
            try:
                path = _path(output, name)
                try:
                    info = path.stat()
                except FileNotFoundError:
                    break
                if not stat.S_ISREG(info.st_mode):
                    raise OSError('Transient is not a regular file: ' + str(path))
                _unlink(path)
                if path.exists():
                    raise OSError('Transient still exists after unlink: ' + str(path))
                _sync_directory(path.parent)
                state['transient_cleanup_files_deleted'] += 1
                state['transient_cleanup_bytes_deleted'] += info.st_size
                state['deleted'].append({'file': name, 'bytes': info.st_size})
                _save(output, state)
                break
            except OSError as error:
                state['transient_cleanup_failures'] += 1
                detail = {'path': str(path), 'error': str(error), 'attempt': attempt + 1}
                state['errors'].append(detail)
                _save(output, state)
                if attempt + 1 == MAX_ATTEMPTS:
                    errors.append(detail)
                else:
                    state['transient_cleanup_retries'] += 1
                    _save(output, state)
                    time.sleep(0.05 * (attempt + 1))
    if errors:
        state['cleanup_status'] = 'CLEANUP_REQUIRED'
        _save(output, state)
        signal = os.environ.get(STOP_ENV)
        if signal:
            _atomic(signal, {'status': 'TRANSIENT_CLEANUP_REQUIRED', 'output': str(Path(output).resolve()),
                'Plan_Row_ID': row_id, 'errors': errors})
        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: scientific row remains EXECUTION_COMPLETE; no FlexDC rerun; ' + json.dumps(errors))
    assert_absent(output)
    state['cleanup_status'] = 'CLEANUP_COMPLETE'
    _save(output, state)
    print('Foundation cleanup ' + row_id + ': CLEANUP_COMPLETE; files=' + str(state['transient_cleanup_files_deleted']) + '; bytes=' + str(state['transient_cleanup_bytes_deleted']), flush=True)
    return state


def recover_output(output):
    state = _current(output)
    if not state:
        return
    row_id = state['Plan_Row_ID']
    receipt = Path(output) / 'foundation_evidence' / (row_id + '.json')
    if not receipt.exists():
        # True execution failure: preserve debugging files, remain retryable.
        return
    # Handles crash after atomic evidence write but before cleanup finalization.
    if state['cleanup_status'] != 'CLEANUP_COMPLETE':
        finalize_row(output, row_id)
    else:
        assert_absent(output)


def recover_outputs(outputs):
    for output in outputs:
        recover_output(output)


def telemetry(outputs):
    result = {name: 0 for name in ('transient_cleanup_files_deleted', 'transient_cleanup_bytes_deleted',
        'transient_cleanup_failures', 'transient_cleanup_retries')}
    result['transient_cleanup_pending'] = 0
    for output in outputs:
        for path in _area(output).glob('F1_*.json'):
            state = _read(path)
            for name in tuple(result):
                if name in state:
                    result[name] += state[name]
            result['transient_cleanup_pending'] += (Path(output) / 'foundation_evidence' / (state['Plan_Row_ID'] + '.json')).exists() and state['cleanup_status'] != 'CLEANUP_COMPLETE'
    return result
