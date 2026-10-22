"""Single Foundation geometry, immutable I/O, seed and row identity contract."""
from __future__ import annotations

import configparser
import csv
import gzip
import hashlib
import io
import json
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import helmert
from scipy.stats import qmc

import am_generate_flexdc_exact_sweep_plan as exact

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'foundation_v1_artifacts'
MASTER = 20261021
SPLITS = ('TRAIN', 'DEVELOPMENT', 'LOCKED_SYNTHETIC_TEST')
VERSION = 'FOUNDATION_V1'
PHASE_A = dict(JOINT_SOBOL_INTERIOR=48, STRUCTURED_PR=30, PHYSICAL_PR_BOUNDARY=24,
               WEIGHT_SIMPLEX_BOUNDARY=24, CROSS_EXTREME=16, EQUAL_WEIGHT_REFERENCE=8)
WAVE_QUOTA = dict(ADAPTIVE_LOCAL=6, ADAPTIVE_BOUNDARY=6, BRIDGE=4, GLOBAL_ESCAPE=4)
RADII = (.12, .08, .05, .03)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed(*parts):
    return int(digest([VERSION, MASTER, *parts])[:8], 16)


def file_hash(path):
    p = Path(path)
    if any(s.startswith('.env') for s in p.parts):
        raise ValueError('Environment files are prohibited')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content, immutable=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        if path.read_bytes() != content: raise ValueError(f'Immutable artifact differs: {path}')
        return
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        f.write(content); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def write_json(path, value, immutable=True):
    atomic_bytes(path, (canonical(value)+'\n').encode(), immutable)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def iter_csv(path):
    p = Path(path)
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p, 'rt', encoding='utf8', newline='') as f:
        yield from csv.DictReader(f)


def read_csv(path):return list(iter_csv(path))


def write_csv(path, rows, columns=None):
    rows = iter(rows); first = next(rows, None)
    if columns is None:
        if first is None: raise ValueError('Empty schema must be explicit')
        columns = list(first)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    # mtime=0 and empty filename make compressed plans byte-reproducible.
    with tmp.open('wb') as raw:
        compressed = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) if path.suffix == '.gz' else None
        f = io.TextIOWrapper(compressed or raw, encoding='utf8', newline='', write_through=True)
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in (() if first is None else (first,)):
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict, tuple)) else v for k,v in row.items()})
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict, tuple)) else v for k,v in row.items()})
        f.flush()
        if compressed: f.detach(); compressed.close()
        else: f.detach()
        raw.flush(); os.fsync(raw.fileno())
    if path.exists():
        if file_hash(path) != file_hash(tmp):
            tmp.unlink(); raise ValueError(f'Immutable CSV differs: {path}')
        tmp.unlink()
    else: os.replace(tmp, path)


def largest_remainder(total, fractions):
    raw = np.array(fractions, float) * total
    n = np.floor(raw).astype(int)
    for i in sorted(range(len(n)), key=lambda i: (-(raw[i]-n[i]),i))[:total-int(n.sum())]: n[i]+=1
    assert int(n.sum()) == total
    return n.tolist()


def sobol(n, dim, *namespace):
    return qmc.Sobol(dim, scramble=True, seed=seed(*namespace)).random_base2(int(np.ceil(np.log2(max(n,2)))))[:n]


def parser(path):
    c = configparser.ConfigParser(interpolation=None)
    if not c.read(path): raise FileNotFoundError(path)
    return c


def weight_bounds(j, n):
    return exact.resolve_configured_weight_bounds(job_type_count=j, server_count=n,
        lower_factor_of_equal=.6, upper_factor_of_equal=1.8, explicit_lower=None, explicit_upper=None)


@lru_cache(None)
def contrasts(j):return helmert(j)


class Domain:
    def __init__(self, context):
        self.context = exact.Context(**context) if isinstance(context, dict) else context
        c=self.context; self.j=c.job_type_count
        self.lower=c.effective_weight_lower; self.upper=c.effective_weight_upper
        assert c.r_over_p_max == .6 and c.r_lower == .01

    def validate(self, x):
        x=np.asarray(x,float); c=self.context
        if x.shape != (self.j+2,) or not np.isfinite(x).all(): raise ValueError('Illegal shape/nonfinite candidate')
        if not exact.legal_pr(c,*x[:2]): raise ValueError('Illegal coupled P/R')
        exact.validate_weight_vector(x[2:],self.j,self.lower,self.upper)
        if abs(sum(x[2:])-1)>1e-9: raise ValueError('Non-unit bounded simplex')
        return x

    def weights(self,u):
        # Sequential conditional simplex map; no clip-and-renormalize.
        w=[]; remaining=1.
        order=np.argsort(u[self.j:2*self.j])
        for k in range(self.j-1):
            left=self.j-k-1
            lo=max(self.lower,remaining-left*self.upper)
            hi=min(self.upper,remaining-left*self.lower)
            v=lo+(hi-lo)*u[k]; w.append(v); remaining-=v
        w.append(remaining)
        result=np.empty(self.j); result[order]=w
        exact.validate_weight_vector(result,self.j,self.lower,self.upper)
        return result

    def decode(self,u):
        c=self.context
        p=c.p_lower+(c.p_upper-c.p_lower)*float(u[0])
        r=c.r_lower+(exact.rmax(c,p)-c.r_lower)*float(u[1])
        return self.validate([p,r,*self.weights(np.asarray(u[2:]))])

    def global_pool(self,n,*namespace):
        return [self.decode(u) for u in sobol(n,2+2*self.j,*namespace)]

    def coordinates(self,x):
        a=np.atleast_2d(np.asarray(x,float)); c=self.context
        p=(a[:,0]-c.p_lower)/(c.p_upper-c.p_lower)
        r=(a[:,1]-c.r_lower)/(np.minimum(.6*a[:,0],c.pr_upper-a[:,0])-c.r_lower)
        w=(a[:,2:]-self.lower)/(self.upper-self.lower)
        return np.column_stack([p/np.sqrt(3),r/np.sqrt(3),w@contrasts(self.j).T/np.sqrt(3*self.j)])

    def distance(self,a,b):
        return float(np.linalg.norm(self.coordinates(a)-self.coordinates(b)))

    def interpolate(self,a,b,t): return self.validate((1-t)*np.asarray(a)+t*np.asarray(b))

    def local(self,anchor,radius,n,*namespace):
        out=[]; c=self.context
        for u in sobol(n*16,2+2*self.j,*namespace):
            # Shrink convex moves until true normalized displacement <=radius.
            target=self.decode(u); t=1.
            for _ in range(24):
                x=self.interpolate(anchor,target,t)
                if self.distance(anchor,x)<=radius+1e-12: break
                t*=.5
            if self.distance(anchor,x)>radius+1e-12: continue
            out.append(x)
            if len(out)==n: break
        return out


def novel(domain,x,used,tolerance=1e-7):
    domain.validate(x)
    return not used or bool(np.all(np.linalg.norm(domain.coordinates(used)-domain.coordinates(x),axis=1)>tolerance))


def maximin(domain,pool,used,count):
    if not pool: return []
    a=np.asarray(pool); z=domain.coordinates(a)
    distances=np.full(len(a),np.inf)
    for old in used: distances=np.minimum(distances,np.linalg.norm(z-domain.coordinates(old),axis=1))
    out=[]
    for _ in range(count):
        i=int(np.argmax(distances))
        if distances[i]<=1e-7: break
        out.append(a[i]); distances=np.minimum(distances,np.linalg.norm(z-z[i],axis=1)); distances[i]=-np.inf
    return out


def violations(p90,pj):
    a=np.array([p90,*pj],float)
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a[1:]>1): raise ValueError('Invalid behaviors')
    v=np.maximum(a/np.array([.3]+[.1]*len(pj))-1,0)
    return float(v.max()),float(v.sum())


def make_row(task,x,family,wave=0,provenance=None,simulation_seed=20,base=None):
    d=Domain(task['domain']); x=d.validate(x).tolist()
    geometry=digest([VERSION,task['Task_ID'],[round(v,12) for v in x]])
    row_id='F1_'+digest([VERSION,task['Task_ID'],geometry,simulation_seed])[:32]
    c=d.context
    row={
      'Foundation_Row_ID':row_id,'Plan_Row_ID':row_id,'plan_row_id':row_id,
      'Base_Geometry_ID':geometry,'Task_ID':task['Task_ID'],'Workload_ID':task['Workload_ID'],
      'Context_ID':task['Task_ID'],'context_id':task['Task_ID'],'Profile_IDs':task['Profile_IDs'],
      'data_split':task['split'],'hardware_class':task['hardware_class'],'job_type_count':d.j,
      'QoS_pattern':task['QoS_pattern'],'workload_config':c.workload_config,'experiment_config':c.experiment_config,
      'server_count':c.server_count,'utilization':c.utilization,'Pbar_kw_per_server':x[0],
      'R_kw_per_server':x[1],'weights':x[2:],'simulation_seed':simulation_seed,
      'Candidate_Source':family,'Adaptive_Wave':wave,'design_version':VERSION,
      'pr_stratum':family,'weight_family':family,'configured_r_over_p_max':.6,
      'configured_weight_lower':d.lower,'configured_weight_upper':d.upper,
      'base_plan_row_id':base or row_id,'is_replicate':simulation_seed!=20,
      'replicate_of_plan_row_id':base or '', 'Provenance':provenance or {},
      'Cost_SHA256':task['Cost_SHA256'],'Experiment_SHA256':task['Experiment_SHA256'],
      'Workload_SHA256':task['Workload_SHA256'],'ISO_SHA256':task['ISO_SHA256'],
      'Policy_Cluster_SHA256':task['Policy_Cluster_SHA256'],'FlexDC_Git_SHA':task['FlexDC_Git_SHA'],
    }
    return row


def row_geometry(row):
    w=json.loads(row['weights']) if isinstance(row['weights'],str) else row['weights']
    return np.array([float(row['Pbar_kw_per_server']),float(row['R_kw_per_server']),*w])


def contexts(): return read_json(ARTIFACTS/'contexts.json')
