"""Joint150-point exact designs, not Cartesian P/R-by-weight sweeps."""
import argparse
from functools import lru_cache
import numpy as np

import am_generate_flexdc_exact_sweep_plan as exact
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,PHASE_A,RADII,WAVE_QUOTA,Domain,
    contexts,novel,make_row,seed,sobol,write_csv,write_json,file_hash,read_json)


@lru_cache(None)
def weight_bank(j,lower,upper):
    entries=exact.build_weight_bank(job_type_count=j,lower=lower,upper=upper,total_count=24,
        lattice_step=.05,temperatures=[.5,1.,2.,4.],alphas=[.3,1.,3.,10.],
        candidate_seed=seed('weight-bank',j),draws_per_setting=64,sobol_draws=512)
    return [np.array(x.weights) for x in entries]


def corners(d):
    c=d.context
    choices=[(c.p_lower,c.r_lower),(c.p_lower,exact.rmax(c,c.p_lower)),
        (c.p_upper,c.r_lower),(c.p_upper,exact.rmax(c,c.p_upper)),
        (c.pr_upper/(1+.6),.6*c.pr_upper/(1+.6))]
    return [p for p in choices if exact.legal_pr(c,*p)]


def static_candidates(task):
    d=Domain(task['domain']);c=d.context
    bank=weight_bank(d.j,d.lower,d.upper);output=[];used=[]
    def add(family,x,detail):
        if not novel(d,x,used):raise ValueError(f'Duplicate static geometry {task["Task_ID"]} {family}')
        used.append(d.validate(x));output.append((family,np.asarray(x),detail))
    for i,x in enumerate(d.global_pool(48,'phaseA',task['Task_ID'],'joint')):
        add('JOINT_SOBOL_INTERIOR',x,{'sobol_index':i})
    for i,p in enumerate(np.linspace(.05,.95,6)):
        for k,r in enumerate(np.linspace(.05,.95,5)):
            base=d.decode(np.r_[p,r,np.full(2*d.j,.5)])
            x=np.r_[base[:2],bank[(i*5+k)%len(bank)]]
            add('STRUCTURED_PR',x,{'P_level':i,'R_level':k,'weight_bank_index':(i*5+k)%len(bank)})
    edge=corners(d)
    generated=exact.physical_boundary_points(c,24-len(edge),set(exact.pr_key(*v) for v in edge))
    prs=[(*v,'corner') for v in edge]+[(v['Pbar_kw_per_server'],v['R_kw_per_server'],v['pr_stratum_detail']) for v in generated]
    for i,(p,r,kind) in enumerate(prs):add('PHYSICAL_PR_BOUNDARY',[p,r,*bank[(i+3)%len(bank)]],{'physical_edge':kind})
    boundary=[]
    for i in range(max(24,2*d.j)):
        u=np.r_[np.array([(i>>k)&1 for k in range(d.j)]),np.roll(np.linspace(.05,.95,d.j),i%d.j)]
        boundary.append(d.weights(u))
    for i in range(24):
        base=d.decode(np.r_[.2+.6*(i+.5)/24,.2+.6*((i*7)%24+.5)/24,np.full(2*d.j,.5)])
        add('WEIGHT_SIMPLEX_BOUNDARY',np.r_[base[:2],boundary[i]],{'weight_face':i})
    for i in range(16):
        p=(.02,.98,.05,.95)[i%4];r=(.02+.003*(i//8),.98-.003*(i//8))[(i//4)%2]
        base=d.decode(np.r_[p,r,np.full(2*d.j,.5)])
        add('CROSS_EXTREME',np.r_[base[:2],boundary[(i+9)%len(boundary)]],{'PR_fraction':[p,r],'weight_face':(i+9)%len(boundary)})
    for i in range(8):
        base=d.decode(np.r_[.08+.84*(i+.5)/8,(.15,.5,.85)[i%3],np.full(2*d.j,.5)])
        add('EQUAL_WEIGHT_REFERENCE',np.r_[base[:2],np.full(d.j,1/d.j)],{'equal_weight':True})
    assert len(output)==150
    return output


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    tasks=contexts(); assert len(tasks)==2000
    preview=[];counts={k:0 for k in PHASE_A}
    def rows():
        for number,task in enumerate(tasks):
            for family,x,detail in static_candidates(task):
                r=make_row(task,x,family,provenance=detail);counts[family]+=1
                if len(preview)<150:preview.append(r)
                yield r
            if (number+1)%100==0:print(f'PhaseA planning:{number+1}/2000 tasks;zero simulations',flush=True)
    path=ARTIFACTS/'foundation_v1_phaseA_plan.csv.gz'
    write_csv(path,rows())
    assert sum(counts.values())==300000
    write_csv(ARTIFACTS/'foundation_v1_phaseA_preview.csv',preview)
    write_json(ARTIFACTS/'phaseA_plan_seal.json',{'plan':path.name,'sha256':file_hash(path),'rows':300000,'task_count':2000,'source_counts':counts,'status':'PLANNED_NOT_RUN'})
    write_json(ARTIFACTS/'foundation_v1_adaptive_protocol.json',{
      'version':'FOUNDATION_V1','master_seed':20261021,'waves':[{'wave':i+1,'radius':r,'unique_per_task':20,'global_rows':40000,'quota':WAVE_QUOTA} for i,r in enumerate(RADII)],
      'required_input_counts_per_task':[150,170,190,210],'fallback':{'BRIDGE':['ADAPTIVE_BOUNDARY','ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_BOUNDARY':['ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_LOCAL':['GLOBAL_ESCAPE']},
      'geometry':'conditional P/R plus J-1 Helmert bounded-simplex contrasts;three RMS blocks',
      'global_pool_size':4096,'novelty_tolerance':1e-7,'repeat_base_count':10,'repeat_seeds':[21,22],
      'primary_seed':20,'expected_adaptive_rows':160000,'expected_repeat_rows':40000,'expected_final_rows':500000,
      'acquisition_uses':'audited actual current-task results only;no learned predictor',
      'waves_status':'NOT_GENERATED_REQUIRES_AUDITED_PREVIOUS_RESULTS','splits':'every task uses its own partition only'})
    print('Frozen300000 exact Phase-A rows;adaptive/repeat coordinates NOT generated')
