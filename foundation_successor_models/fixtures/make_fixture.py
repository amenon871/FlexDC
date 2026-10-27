"""TEST FIXTURE ONLY. No simulator output or scientific claim."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np


def make_fixture(output,job_counts=(2,4,6)):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(741);profiles=[];workloads=[];rows=[]
    for split in ['TRAIN','DEVELOPMENT','LOCKED_SYNTHETIC_TEST']:
        for hardware in ['P100','A100']:
            for j in job_counts:
                wid=f'F1_{split}_{hardware}_J{j}_{len(workloads):03d}';n=250 if j==2 else 1000;u=.6 if hardware=='P100' else .8;ids=[]
                for job in range(j):
                    pid=f'SYN_{split}_{hardware}_{j}_{job}';ids.append(pid)
                    profiles.append(dict(profile_id=pid,split=split,hardware_class=hardware,min_job_power_watts=180+job*10,max_job_power_watts=650+job*15,min_time_seconds=20+job*5,max_time_seconds=50+job*10,job_size=1))
                workloads.append(dict(Workload_ID=wid,split=split,hardware_class=hardware,J=j,Profile_IDs=ids,QoS_values=[4.]*j))
                task=f'{wid}__N{n}__U{round(u*100):03d}'
                for geom in range(100):
                    p=.30+.25*rng.random();r=.02+.1*rng.random();delta=rng.uniform(-.1/j,.1/j,j);weights=np.full(j,1/j)+delta-delta.mean();bid=[p,r,*weights];base=hashlib.sha256((task+str(geom)).encode()).hexdigest()
                    for seed in ([20,21,22] if geom<10 else [20]):
                        mean=.05+.2*(p-.30)+.02*np.sin(geom)+.001*(seed-20);p90=.12+.38*(p-.3)/.25+.02*np.cos(geom)+.002*(seed-20)
                        pj=np.clip(.02+.13*(.55-p)/.25+.01*np.arange(j)+.002*(seed-20),0,1);rw=round(r*n*1000);pw=round(p*n*1000);price=.1/(3600*1000);power=(pw-rw)*price*3600
                        objective=power+rw*price*3600*mean+np.logaddexp(0,10*(p90-.3))+20*np.logaddexp(0,2*(pj-.1)).sum()
                        row=dict(Plan_Row_ID='F1_'+hashlib.md5((base+str(seed)).encode()).hexdigest(),Base_Geometry_ID=base,Task_ID=task,Workload_ID=wid,split=split,hardware_class=hardware,Profile_IDs=ids,J=j,N=n,U=u,x=bid,simulation_seed=seed,Candidate_Source=['SOBOL_GLOBAL','BOUNDARY_TRACKING','LOCAL_FEEDBACK'][geom%3] if seed==20 else 'REPEAT',Adaptive_Wave=0 if seed==20 else 5,mean_tracking=mean,p90=p90,Pj=pj.tolist(),objective=objective,evidence_counts=[20]*j,input_status='LEGAL_INPUT',observation_status='VALID_OBSERVATION',evidence_status='SUFFICIENT_JOB_TYPE_EVIDENCE',raw_results=dict(Simulator_Power_Cost=power,R_actual_watts=rw,Mtrack_Price_Coefficient_piE=price,Mtrack_Hour_Seconds=3600,Pbar_ratio=round((pw/n/1000)/(((650+15*(j-1)/2)/1000)*u+.12*(1-u)),6),R_ratio=round((rw/n/1000)/((((650+15*(j-1)/2)/1000)-.12)/2),6)))
                        rows.append(row)
    for name,data in [('profiles.json',dict(real_anchors=[],synthetic_profiles=profiles)),('workloads.json',workloads)]: (out/name).write_text(json.dumps(data,indent=2)+'\n')
    with gzip.open(out/'observations.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    (out/'README.md').write_text('# TEST FIXTURE ONLY\n\nNOT SCIENTIFIC DATA. 18 tasks / 2160 rows. P100+A100, J2/4/6, N250/1000, U.6/.8, all partitions, 100 unique geometries and 10 paired repeats per task.\n')
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(Path(__file__).parent/'tiny'));a=p.parse_args();print(make_fixture(a.output))
