"""V3 engineered features, plus global hardware identity; TRAIN-only statistics."""
import numpy as np
from scipy.linalg import helmert

JOB_FEATURES=['pmin_kw','pmax_kw','power_range_kw','log10_tmin','log10_tmax','runtime_ratio','qos_threshold','job_size','qos_headroom','weight','weight_relative_to_equal','allocated_nodes_per_1000','queue_pressure']
GLOBAL_FEATURES=['Pbar_kw_per_server','R_kw_per_server','Pbar_minus_R','Pbar_plus_R','R_over_Pbar','Pbar_ratio','R_ratio','utilization','server_count_per_1000','log10_server_count','inverse_job_count','job_count_per_8','hardware_P100','hardware_A100']


def job_features(mix,weights,utilization,server_count):
    mix=np.asarray(mix,dtype=np.float64);w=np.asarray(weights,dtype=np.float64);j=len(w)
    if mix.shape!=(j,6):raise ValueError('Physical job feature shape must be [J,6]')
    pmin,pmax,tmin,tmax,qos,size=mix.T;ratio=tmax/np.maximum(tmin,1e-9)
    return np.column_stack([pmin/1000,pmax/1000,(pmax-pmin)/1000,np.log10(np.maximum(tmin,1e-9)),np.log10(np.maximum(tmax,1e-9)),ratio,qos,size,qos/np.maximum(ratio,1e-9),w,w*j,w*server_count/1000,utilization/np.maximum(j*w,1e-6)])


def global_features(p,r,n,u,j,hardware,p_ratio,r_ratio):
    if hardware not in ('P100','A100'):raise ValueError('Unknown hardware class: '+str(hardware))
    return np.array([p,r,p-r,p+r,r/max(p,1e-9),p_ratio,r_ratio,u,n/1000,np.log10(max(n,1)),1/j,j/8,hardware=='P100',hardware=='A100'],dtype=np.float64)


def guarded_stats(x):
    mean=np.mean(x,axis=0,dtype=np.float64);std=np.std(x,axis=0,dtype=np.float64)
    std[(~np.isfinite(std))|(std<1e-3)]=1.
    return mean,std


def fit_normalization(jobs,globals_,mask,logs,train):
    if not np.any(train):raise ValueError('TRAIN rows required for normalization')
    jm,js=guarded_stats(jobs[train][mask[train]]);gm,gs=guarded_stats(globals_[train]);ts=np.std(logs[train],axis=0,dtype=np.float64)
    ts[(~np.isfinite(ts))|(ts<1e-8)]=1.
    return dict(schema_version='FOUNDATION_FEATURES_V1',job_names=JOB_FEATURES,global_names=GLOBAL_FEATURES,job_mean=jm.tolist(),job_std=js.tolist(),global_mean=gm.tolist(),global_std=gs.tolist(),target_std=ts.tolist(),fit_partition='TRAIN',fit_rows=int(np.sum(train)))


def validate_domain(x,mix,n):
    x=np.asarray(x,dtype=np.float64);mix=np.asarray(mix);j=len(mix)
    if len(x)!=j+2 or not np.isfinite(x).all():raise ValueError('Candidate length/nonfinite input')
    p,r=x[:2];w=x[2:];pmin=mix[:,0].min()/1000;pmax=mix[:,1].max()/1000
    legacy=max(.1/j,1/n);lower=max(legacy,.6/j);upper=min(4/j,1-(j-1)*legacy,1.8/j,1-(j-1)*lower)
    tol=1e-9
    if p<.9*pmin-tol or p>pmax+tol or r<.01-tol or r>.6*p+tol or p+r>1.2*pmax+tol or p-r<-tol:raise ValueError('Illegal workload-specific P/R domain')
    if abs(w.sum()-1)>tol or np.any(w<lower-tol) or np.any(w>upper+tol):raise ValueError('Illegal dynamic bounded-simplex weights')
    return pmin,pmax,lower,upper


def canonical_coordinates(x,mix,n):
    """Exact Foundation Domain.coordinates metric; independent P/R/weight blocks."""
    a=np.atleast_2d(np.asarray(x,dtype=np.float64));j=len(mix)
    pmin,pmax,lower,upper=validate_domain(a[0],mix,n)
    for row in a[1:]:validate_domain(row,mix,n)
    p=(a[:,0]-.9*pmin)/(pmax-.9*pmin)
    r=(a[:,1]-.01)/(np.minimum(.6*a[:,0],1.2*pmax-a[:,0])-.01)
    w=(a[:,2:]-lower)/(upper-lower)
    return np.column_stack([p/np.sqrt(3),r/np.sqrt(3),w@helmert(j).T/np.sqrt(3*j)])
