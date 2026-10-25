"""Storage-only gate; never schedules or retries scientific simulations."""
from am_flexdc_foundation_v1_common import write_json
from peacsim.foundation_v1_cleanup import recover_output, telemetry, CleanupRequired, _current, assert_absent, TRANSIENT_FILES


def gate(outputs, directory, retry):
    try:
        local_errors=[]
        if retry:
            for output in outputs:
                try:recover_output(output)
                except (ValueError,KeyError,OSError) as error:
                    has_receipt=any((output/'foundation_evidence').glob('F1_*.json'))
                    if has_receipt and any((output/n).exists() for n in TRANSIENT_FILES):
                        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: terminal receipt and unresolved raw files: '+str(error)) from error
                    local_errors.append({'output':str(output),'classification':'TECHNICAL_RETRY_REQUIRED','error':str(error)})
        pending = []
        for output in outputs:
            try:state = _current(output)
            except (ValueError,KeyError,OSError):continue
            if state and (output / 'foundation_evidence' / (state['Plan_Row_ID'] + '.json')).exists() and state['cleanup_status'] != 'CLEANUP_COMPLETE':
                pending.append({'output': str(output), 'Plan_Row_ID': state['Plan_Row_ID'], 'errors': state['errors']})
            elif state and state['cleanup_status'] == 'CLEANUP_COMPLETE':
                assert_absent(output)
        if pending:
            raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: terminal scientific rows must not rerun: ' + str(pending))
    except (OSError, ValueError, CleanupRequired) as error:
        write_json(directory / 'transient_cleanup_status.json', {
            'status': 'TRANSIENT_CLEANUP_REQUIRED', 'error': str(error), **telemetry(outputs)}, False)
        raise CleanupRequired('TRANSIENT_CLEANUP_REQUIRED: ' + str(error)) from error
    report = {'status': 'CLEANUP_COMPLETE', **telemetry(outputs)}
    write_json(directory / 'transient_cleanup_status.json', report, False)
    return report
