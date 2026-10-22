"""Measured-anchor convex latent profiles; partitions freeze before compositions."""
import argparse
import math
import numpy as np

from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,SPLITS,parser,file_hash,digest,
    largest_remainder,sobol,write_csv,write_json,read_json)

FIELDS=('min_job_power_watts','max_job_power_watts','min_time_seconds','max_time_seconds')
SOURCES=[('P100','training-like','configs/workload/W1-train-qos4444.ini'),
         ('P100','inference-like','configs/workload/W2-short-qos5555.ini'),
         ('A100','inference-like','configs/foundation_v1/measured_moe_anchors.ini')]


def latent(p):
    a,b,c,d=(p[k] for k in FIELDS)
    if not (b>a>0 and d>c>0 and int(p['job_size'])==p['job_size']>=1): raise ValueError('Invalid physical profile')
    return np.log([a,b/a,c,d/c])


def physical(z):
    a,b,c,d=np.exp(z)
    return dict(zip(FIELDS,map(float,[a,a*b,c,c*d])))


def anchors():
    output=[]
    for hardware,kind,path in SOURCES:
        config=parser(ROOT/path)
        for name in config.sections():
            p={k:config.getfloat(name,k) for k in FIELDS}
            p.update(profile_id='REAL_'+hardware+'_'+name,split='TRAIN',hardware_class=hardware,
                     workload_class=kind,job_size=config.getint(name,'job_size',fallback=1),
                     generation_mode='MEASURED_REAL_ANCHOR',source_path=path,source_section=name,
                     source_sha256=file_hash(ROOT/path),hardware_provenance=hardware)
            p['latent']=latent(p).tolist()
            # The v1 real-anchor identity included the former attribution token.
            # Preserve that frozen identity; attribution is now confirmed A100.
            identity=dict(p)
            if hardware=='A100':identity['hardware_provenance']='BELIEVED_A100_PENDING_CAN'
            p['profile_hash']=digest(identity);output.append(p)
    assert len(output)==13
    return output


def generate():
    real=anchors();profiles=[]
    for split,names in zip(SPLITS,((90,90,120),(15,15,20),(15,15,20))):
        for (hardware,kind,_),total in zip(SOURCES,names):
            groups={}
            for a in real:
                if (a['hardware_class'],a['workload_class'])==(hardware,kind): groups.setdefault(a['job_size'],[]).append(a)
            # Current anchors all have size1. Refuse implicit job-size allocation on future changes.
            if len(groups)!=1: raise ValueError('New job-size subgroup requires explicit count allocation')
            size,group=next(iter(groups.items()))
            if len(group)<2: raise ValueError('Convex synthetic variation needs at least two compatible anchors')
            counts=largest_remainder(total,[.5,.3,.2]);k=0
            for mode,count in zip(('ANCHORED_LOCAL','PAIR_INTERPOLATION','GLOBAL_CONVEX_SOBOL'),counts):
                points=sobol(count,len(group)+2,'profiles',split,hardware,kind,mode)
                for i,u in enumerate(points):
                    w=-np.log(np.clip(u[:len(group)],1e-12,1-1e-12));w/=w.sum()
                    fraction=None
                    if mode=='ANCHORED_LOCAL':
                        fraction=.05+.25*u[-1];w*=fraction;w[i%len(group)]+=1-fraction
                    elif mode=='PAIR_INTERPOLATION':
                        fraction=.2+.6*u[-1];a=i%len(group);b=(a+1+i//len(group)%(len(group)-1))%len(group)
                        w=np.zeros(len(group));w[a]=1-fraction;w[b]=fraction
                    z=w@np.array([a['latent'] for a in group])
                    p=dict(profile_id=f'SYN_{split}_{hardware}_{kind}_{k:03d}',split=split,
                           hardware_class=hardware,workload_class=kind,job_size=size,
                           generation_mode=mode,anchor_ids=[a['profile_id'] for a in group],
                           anchor_coefficients=w.tolist(),interpolation_fraction=fraction,
                           latent=z.tolist(),generation_namespace=['profiles',split,hardware,kind,mode],
                           sobol_index=i,**physical(z))
                    latent(p);p['profile_hash']=digest(p);profiles.append(p);k+=1
    assert len(profiles)==400
    signatures=[tuple(round(p[f],10) for f in FIELDS)+(p['job_size'],) for p in profiles]
    if len(set(signatures))!=400: raise ValueError('Duplicate physical synthetic profile across partitions')
    return real,profiles


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    real,synthetic=generate()
    write_json(ARTIFACTS/'profiles.json',{'real_anchors':real,'synthetic_profiles':synthetic})
    # Fixed schema; real anchors are separate rows with synthetic count explicit.
    rows=[]
    for p in synthetic:
        rows.append({**{k:p[k] for k in ('profile_id','split','hardware_class','workload_class','generation_mode','anchor_ids','anchor_coefficients','latent',*FIELDS,'job_size','generation_namespace','sobol_index','profile_hash')},'interpolation_fraction':p['interpolation_fraction']})
    write_csv(ARTIFACTS/'foundation_v1_profile_manifest.csv',rows)
    write_json(ARTIFACTS/'profile_partition_freeze.json',{'profiles_sha256':file_hash(ARTIFACTS/'profiles.json'),'synthetic_count':400,'split_counts':[300,50,50],'master_seed':20261021})
    print('Frozen400 synthetic profiles:300 TRAIN/50 DEVELOPMENT/50 LOCKED_SYNTHETIC_TEST;13 real training anchors')
