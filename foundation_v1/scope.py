"""Operational partition isolation; scientific IDs and coordinates are untouched."""
from pathlib import Path
from am_flexdc_foundation_v1_common import ROOT,ARTIFACTS,contexts,iter_csv,write_csv,write_json,file_hash

SCOPES={'TRAIN':('TRAIN',),'DEVELOPMENT':('DEVELOPMENT',),
        'TRAIN_DEV':('TRAIN','DEVELOPMENT'),
        'LOCKED_SYNTHETIC_TEST':('LOCKED_SYNTHETIC_TEST',),
        'ALL':('TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST')}
CHOICES=(*SCOPES,'TRAIN+DEVELOPMENT')
LOCK_MESSAGE='LOCKED_SYNTHETIC_TEST is a sealed synthetic test partition; use --unlock-locked-test to explicitly authorize isolated precomputation. Its outcomes must never be used for model fitting or architecture/hyperparameter selection.'

def normalize(scope=None):
    scope=scope or 'TRAIN_DEV'
    if scope=='TRAIN+DEVELOPMENT':scope='TRAIN_DEV'
    if scope not in SCOPES:raise ValueError('Unknown Foundation partition scope: '+scope)
    return scope

def require_unlock(scope,unlock=False):
    if 'LOCKED_SYNTHETIC_TEST' in SCOPES[normalize(scope)] and not unlock:raise ValueError(LOCK_MESSAGE)

def selected_tasks(scope=None,task_id=None):
    selected=[t for t in contexts() if t['split'] in SCOPES[normalize(scope)] and (task_id is None or t['Task_ID']==task_id)]
    if not selected:raise ValueError('Unknown task or task outside selected partition scope')
    return selected

def group(scope):
    scope=normalize(scope)
    if scope=='ALL':raise ValueError('ALL must dispatch sequentially to physically separate TRAIN_DEV and LOCKED_SYNTHETIC_TEST runs')
    return 'locked_synthetic_test' if scope=='LOCKED_SYNTHETIC_TEST' else 'train_dev'

def run_root(scope):return ROOT/'foundation_v1_runs'/group(scope)
def database_path(scope):return run_root(scope)/'observations.sqlite'
def raw_root(scope):return ROOT/'src/peacsim/output/optimization/foundation_v1_runs'/group(scope)

def plan_path(stage,scope=None,task_id=None):
    suffix=normalize(scope)+(('__'+task_id) if task_id else '')
    return ARTIFACTS/f'foundation_v1_{stage}__{suffix}.csv.gz'

def make_phaseA_view(scope=None):
    scope=normalize(scope);master=ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'
    if scope=='ALL':return master  # Existing master is already the ALL metadata view.
    path=plan_path('phaseA',scope)
    count=[0]
    def rows():
        for row in iter_csv(master):
            if row['data_split'] in SCOPES[scope]:
                count[0]+=1;yield row
    write_csv(path,rows())
    write_json(path.with_suffix('.receipt.json'),{'scope':scope,'rows':count[0],
        'master_sha256':file_hash(master),'sha256':file_hash(path),'identity_policy':'EXACT_MASTER_ROWS_UNCHANGED',
        'status':'PLANNED_NOT_RUN'})
    return path

def assert_database_scope(path,scope):
    # Metadata-only SQL checks avoid reading held-out outcome bodies in train/dev.
    import sqlite3
    path=Path(path)
    if not path.exists():return
    allowed={t['Task_ID'] for t in selected_tasks('LOCKED_SYNTHETIC_TEST' if group(scope)=='locked_synthetic_test' else 'TRAIN_DEV')}
    db=sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro',uri=True)
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ('observations','terminal_executions'):
            if table in tables and any(t[0] not in allowed for t in db.execute('SELECT DISTINCT task_id FROM '+table)):
                raise ValueError('Partition contamination: database contains tasks outside '+group(scope))
    finally:db.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partition',choices=CHOICES,default='TRAIN_DEV');p.add_argument('--all-views',action='store_true');a=p.parse_args()
    for s in SCOPES if a.all_views else [normalize(a.partition)]:print(make_phaseA_view(s))
