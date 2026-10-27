"""Prepare only; never train or launch FlexDC."""
import argparse
from pathlib import Path
from flexmodels.data import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--profiles',default=str(Path(__file__).resolve().parents[1]/'foundation_v1_artifacts/profiles.json'));p.add_argument('--workloads',default=str(Path(__file__).resolve().parents[1]/'foundation_v1_artifacts/workloads.json'));p.add_argument('--fixture',action='store_true');a=p.parse_args()
    print(prepare(a.input,a.output,a.profiles,a.workloads,a.fixture))
