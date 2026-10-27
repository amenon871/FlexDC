"""User-controlled real optimizer benchmark; not a scientific quality claim."""
import argparse,time
from pathlib import Path
from flexmodels.training import cli,PACKAGE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['baseline','meta_v4','flexpfn'],required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--run-dir',default=str(PACKAGE/'.validation_runs'/('benchmark_'+str(time.time_ns()))));a,remaining=p.parse_known_args()
    if a.steps<1:p.error('--steps must be positive')
    cli(a.model,remaining+['--run-dir',a.run_dir,'--stop-after-steps',str(a.steps)],benchmark=True)
