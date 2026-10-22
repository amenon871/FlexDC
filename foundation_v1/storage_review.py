"""Reproduce storage-patch pre-execution audits. NEVER launches a simulator."""
import json
import os
import subprocess
import sys
from pathlib import Path
from am_flexdc_foundation_v1_common import ROOT, ARTIFACTS, file_hash, read_json, write_json, iter_csv, write_csv
from foundation_v1.full_campaign import MASTER_SHA256

BASE = 'dd4320ebbd8213570cd630a360cd37a515a69cae'
OUT = ARTIFACTS / 'storage_safety_validation'


def run(command, cwd=ROOT, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, encoding='utf8', errors='strict')
    if result.returncode:
        raise RuntimeError('Validation-only command failed: ' + repr(command) + '\n' + result.stdout + '\n' + result.stderr)
    return result.stdout


def validate():
    OUT.mkdir(exist_ok=True)
    old = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':foundation_v1_artifacts/preexecution_release.json']))
    names = [name for name in old['source_files'] if name.startswith(('configs/', 'data/', 'foundation_v1_artifacts/'))]
    identities = {name: file_hash(ROOT / name) for name in names}
    assert all(identities[n] == old['source_files'][n] for n in names)
    master = ARTIFACTS / 'foundation_v1_phaseA_plan.csv.gz'
    assert file_hash(master) == MASTER_SHA256
    profiles = read_json(ARTIFACTS / 'profiles.json')
    workloads = read_json(ARTIFACTS / 'workloads.json')
    contexts = read_json(ARTIFACTS / 'contexts.json')
    assert len(profiles['synthetic_profiles']) == 400 and len(workloads) == 500 and len(contexts) == 2000
    # Count existing master rows only; never regenerate coordinates or identities.
    assert sum(1 for _ in iter_csv(master)) == 300000
    scientific = {'status': 'PASS', 'baseline': BASE, 'master_before': MASTER_SHA256,
        'master_after': file_hash(master), 'unchanged_scientific_files': identities,
        'profiles': 400, 'workloads': 500, 'contexts': 2000, 'static_rows': 300000,
        'expected_adaptive_rows': 160000, 'expected_repeat_rows': 40000, 'expected_total': 500000,
        'real_FlexDC_calls': 0}
    write_json(OUT / 'scientific_identity.json', scientific, False)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf8')
    full = []
    for workers in (32, 96):
        command = [sys.executable, '-B', str(ROOT / 'am_run_flexdc_foundation_campaign_v1.py'),
            '--full-campaign', '--partition', 'ALL', '--unlock-locked-test', '--max-workers', str(workers), '--dry-run']
        output = json.loads(run(command, env=environment))
        assert output['expected_total_rows'] == 500000 and len(output['steps']) == 14
        assert output['simulation_calls_this_invocation'] == 0 and not output['adaptive_coordinates_generated']
        for step in output['steps']:
            if step['stage'] != 'export':
                c = step['command_template']; assert c[c.index('--plan-max-workers') + 1] == str(workers)
        write_json(OUT / ('full_dry_run_workers' + str(workers) + '.json'), output, False)
        full.append({'workers': workers, 'steps': 14, 'rows': 500000, 'status': 'PASS'})
    # Tiny metadata-only child view of EXISTING master rows. No new coordinates.
    from itertools import islice
    scratch = ROOT / '.flexdc_plan_chunks/storage_safety_validation'
    scratch.mkdir(parents=True, exist_ok=True)
    rows = list(islice(iter_csv(master), 150))
    subset = scratch / 'existing_rows.csv.gz'; write_csv(subset, rows)
    import am_run_flexdc_dataset_parallel_combo as combo
    from foundation_v1.campaign import commands
    command = commands(subset, 32, 'foundation_v1_storage_validation_dry_only', 'TRAIN_DEV')
    # Avoid nested log labels for this command-preparation-only validation.
    command[command.index('--output-prefix') + 1] = 'foundation_v1_storage_validation'
    command.append('--plan-dry-run')
    combo_log = run(command, env=environment)
    (OUT / 'combo_log.txt').write_text(combo_log, encoding='utf8')
    manifest = ROOT / 'dataset_plan_launch_manifest_AQA_foundation_v1_storage_validation_dry_only.csv'
    import pandas as pd
    launch = pd.read_csv(manifest)
    assert len(launch) == 6 and set(launch['status']) == {'dry_run'}
    args = combo.build_parser().parse_args(command[4:])
    invocations, _, _ = combo.prepare_plan_invocations(args)
    wizard_logs = []
    for item in (invocations[0], invocations[-1]):
        wizard = item['command'] + ['--validate-sweep-plan-only']
        env = combo.build_env(); env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf8')
        log = run(wizard, cwd=combo.PEACSIM_DIR, env=env)
        assert 'simulation was not run' in log
        wizard_logs.append({'chunk': item['chunk_index'], 'status': 'PASS', 'rows': item['expected_rows']})
        (OUT / ('wizard_' + str(item['chunk_index']) + '_log.txt')).write_text(log, encoding='utf8')
    report = {'status': 'PASS', 'full_campaign': full, 'combo_dry_run_chunks': len(launch),
        'wizard_validation_only': wizard_logs, 'real_FlexDC_calls': 0,
        'master_before': MASTER_SHA256, 'master_after': file_hash(master)}
    write_json(OUT / 'cli_and_identity_gates.json', report, False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    validate()
