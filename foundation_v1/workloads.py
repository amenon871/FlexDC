"""Deterministic partition-safe composition diversity and canonical contexts."""
import argparse
import configparser
import io
from collections import Counter,defaultdict
from dataclasses import asdict

import numpy as np

import am_generate_flexdc_exact_sweep_plan as exact
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,SPLITS,seed,digest,canonical,
    file_hash,parser,read_json,write_json,write_csv,atomic_bytes,largest_remainder)
from foundation_v1.profiles import FIELDS


def descriptors(profiles,real):
    z=np.array([p['latent'] for p in profiles]);a=np.array([p['latent'] for p in real])
    lo=a.min(axis=0);scale=np.maximum(a.max(axis=0)-lo,1e-8)
    normalized=(z-lo)/scale
    return np.r_[normalized.mean(axis=0),normalized.std(axis=0)]


def generate(data):
    real=data['real_anchors'];synthetic=data['synthetic_profiles'];output=[]
    usage=Counter();bottleneck=Counter();qos_usage=Counter();compositions=set();history=defaultdict(list)
    for split,j_totals,qos_counts in zip(SPLITS,([60,80,120,80,60],[8,10,14,10,8],[8,10,14,10,8]),([100,60,140,100],[12,8,18,12],[12,8,18,12])):
        p_counts=np.floor(np.array(j_totals)*.6).astype(int)
        remaining=int(sum(j_totals)*.6)-int(sum(p_counts))
        for index in sorted(range(5),key=lambda i:(-(j_totals[i]*.6-p_counts[i]),i))[:remaining]:p_counts[index]+=1
        slots={h:[] for h in ('P100','A100')}
        for j,total,pcount in zip(range(2,7),j_totals,p_counts):
            slots['P100'] += [j]*int(pcount);slots['A100'] += [j]*(total-int(pcount))
        all_slots=[]
        for hardware,js in slots.items():
            rng=np.random.default_rng(seed('workload-allocation',split,hardware))
            rng.shuffle(js);grounded_count=(48 if hardware=='P100' else 32) if split=='TRAIN' else 0
            grounded=set(rng.permutation(len(js))[:grounded_count])
            if hardware=='P100':
                counts=largest_remainder(len(js),[.25,.25,.5])
                classes=['training-only']*counts[0]+['inference-only']*counts[1]+['mixed']*counts[2]
                rng.shuffle(classes)
            else:classes=['inference-only']*len(js)
            all_slots += [(hardware,j,classes[i],i in grounded) for i,j in enumerate(js)]
        patterns=sum(([name]*n for name,n in zip(('ALL_5','ALL_4','BALANCED_MIXED','BOTTLENECK'),qos_counts)),[])
        np.random.default_rng(seed('qos-assignment',split)).shuffle(patterns)
        for number,((hardware,j,kind,grounded),pattern) in enumerate(zip(all_slots,patterns)):
            wid=f'F1_{split}_{hardware}_J{j}_{number:03d}'
            pool=[p for p in synthetic if p['split']==split and p['hardware_class']==hardware]
            anchors=[p for p in real if p['hardware_class']==hardware]
            target=('HOMOGENEOUS','POWER_HETEROGENEOUS','RUNTIME_HETEROGENEOUS','ELASTICITY_HETEROGENEOUS','LIKELY_BOTTLENECK')[number%5]
            candidates=[]
            for attempt in range(80):
                rng=np.random.default_rng(seed('composition',wid,attempt))
                types=(['training-like']*(j//2+int(number%2 and j%2))+['inference-like']*(j-j//2-int(number%2 and j%2))) if kind=='mixed' else [kind.replace('-only','-like')]*j
                rng.shuffle(types);selected=[];nreal=(j+1)//2 if grounded else 0
                for position,typ in enumerate(types):
                    eligible=[p for p in (anchors if position<nreal else pool) if p['workload_class']==typ and p not in selected]
                    if not eligible: raise ValueError('Insufficient compatible distinct profiles')
                    weights=np.array([1/(1+usage[p['profile_id']])**2 for p in eligible]);weights/=weights.sum()
                    selected.append(eligible[int(rng.choice(len(eligible),p=weights))])
                key=tuple(sorted(p['profile_id'] for p in selected))
                if key in compositions: continue
                desc=descriptors(selected,anchors)
                old=history[(split,hardware,kind)]
                distance=min((np.linalg.norm(desc-x) for x in old),default=1.)
                hetero=desc[4:]
                target_score={'HOMOGENEOUS':-float(hetero.sum()),'POWER_HETEROGENEOUS':float(hetero[:2].sum()),'RUNTIME_HETEROGENEOUS':float(hetero[2:].sum()),'ELASTICITY_HETEROGENEOUS':float(hetero[[1,3]].sum()),'LIKELY_BOTTLENECK':float(np.max(hetero))}[target]
                load=np.mean([usage[p['profile_id']] for p in selected])
                candidates.append((distance+.35*target_score-.04*load,attempt,selected,desc,key))
            if not candidates:raise ValueError('No unique legal composition')
            _,_,selected,desc,key=max(candidates,key=lambda x:(x[0],-x[1]))
            compositions.add(key);history[(split,hardware,kind)].append(desc)
            ids=[p['profile_id'] for p in selected]
            if pattern=='ALL_5':qos=[5.]*j
            elif pattern=='ALL_4':qos=[4.]*j
            elif pattern=='BALANCED_MIXED':
                alternatives=[]
                for shift in range(4):
                    values=[(3.5,4.,4.5,5.)[(i+shift)%4] for i in range(j)]
                    cost=sum(qos_usage[(pid,value)] for pid,value in zip(ids,values))
                    alternatives.append((cost,shift,values))
                qos=min(alternatives)[2]
            else:
                receiver=min(range(j),key=lambda i:(bottleneck[ids[i]],bottleneck[(j,i)],seed('bottleneck',wid,ids[i])))
                qos=[4.5 if (i+number)%2 else 5. for i in range(j)];qos[receiver]=3. if number%2 else 3.5
                bottleneck[ids[receiver]]+=1;bottleneck[(j,receiver)]+=1
            for pid,q in zip(ids,qos):usage[pid]+=1;qos_usage[(pid,q)]+=1
            c=configparser.ConfigParser(interpolation=None)
            for p,q in zip(selected,qos):c[p['profile_id']]={**{k:format(p[k],'.17g') for k in FIELDS},'qos_constraint':str(q),'job_size':str(p['job_size'])}
            buffer=io.StringIO();c.write(buffer,space_around_delimiters=True)
            path=f'configs/workload/foundation_v1/{wid}.ini';content=buffer.getvalue().encode()
            atomic_bytes(ROOT/path,content)
            output.append({'Workload_ID':wid,'split':split,'hardware_class':hardware,'workload_class':kind,
                'J':j,'Profile_IDs':ids,'grounding':'ANCHOR_GROUNDED' if grounded else 'SYNTHETIC_ONLY',
                'real_profile_slots':nreal,'QoS_pattern':pattern,'QoS_values':qos,'job_sizes':[p['job_size'] for p in selected],
                'heterogeneity_target':target,'heterogeneity_descriptor':desc.tolist(),'INI_path':path,'INI_SHA256':file_hash(ROOT/path)})
    assert len(output)==500
    return output,usage


def generate_contexts(workloads):
    # Source canonical configs are verified, never edited.
    costpath=ROOT/exact.DEFAULT_SA_CONSTANTS_CONFIG
    constants=exact.read_cost_constants(costpath,ROOT)
    expected=[1.,10.,.3,20.,2.,.1,'RSR']
    if [constants.psi1,constants.psi2,constants.tracking_error_constraint,constants.beta,constants.rho,constants.qos_constraint,constants.program_type]!=expected:raise ValueError('Canonical cost contract changed')
    gradient=ROOT/'configs/foundation_v1/gradient_cost_rsr.ini'
    exact.write_canonical_gradient_config(base_path=ROOT/'configs/gradient_descent/gradient_descent.ini',output_path=gradient,constants=constants,overwrite=False)
    templates=[];exps=[]
    for n in (250,1000):
        original=ROOT/f'configs/experiment/new_iso/traditional_signal/generated_server_counts/exp_traditional_iso16_servers_{n}.ini'
        original_config=parser(original)
        if original_config.getboolean('iso','randomize_iso_start'):raise ValueError('Canonical grid is randomized')
        for u in (.6,.8):
            c=parser(original);c.set('system','utilization',str(u));c.set('system','random_seed','20')
            f=ROOT/f'configs/foundation_v1/experiments/exp_N{n}_U{int(u*100):03d}.ini'
            s=io.StringIO();c.write(s);atomic_bytes(f,s.getvalue().encode())
            if dict(c['iso'])!=dict(original_config['iso']):raise ValueError('ISO inheritance mismatch')
            for k,v in original_config['system'].items():
                if k not in ('utilization','random_seed') and c.get('system',k)!=v:raise ValueError('System inheritance mismatch')
            templates.append({'N':n,'U':u,'source':str(original.relative_to(ROOT)).replace('\\','/'),'source_sha256':file_hash(original),'generated':str(f.relative_to(ROOT)).replace('\\','/'),'sha256':file_hash(f),'idle_watts':c.getfloat('system','idle_watts'),'ISO':dict(c['iso'])})
            exps.append(str(f))
    import subprocess
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    tasks=[]
    # Matching template U is used, rather than cross-product duplicate contexts.
    for w in workloads:
        for t in templates:
            generated=exact.build_contexts(repo_root=ROOT,workload_configs=[w['INI_path']],workload_manifest={},experiment_configs=[t['generated']],utilizations=[t['U']],pbar_lower_factor=.9,pbar_upper_factor=1.,pr_upper_factor=1.2,r_lower=.01,r_over_p_max=.6,weight_lower_factor_of_equal=.6,weight_upper_factor_of_equal=1.8,configured_weight_lower=None,configured_weight_upper=None)[0]
            iso=ROOT/'src/peacsim'/generated.iso_file_path
            # Interpret the canonical path relative to the established worker CWD.
            tasks.append({'Task_ID':generated.context_id,'Workload_ID':w['Workload_ID'],'split':w['split'],
                'hardware_class':w['hardware_class'],'Profile_IDs':w['Profile_IDs'],'QoS_pattern':w['QoS_pattern'],
                'domain':asdict(generated),'Cost_SHA256':file_hash(costpath),'Experiment_SHA256':t['sha256'],
                'Workload_SHA256':w['INI_SHA256'],'ISO_SHA256':file_hash(iso),'Policy_Cluster_SHA256':file_hash(ROOT/'configs/cluster/cluster.ini'),
                'FlexDC_Git_SHA':commit,'status':'PLANNED','gradient_config':'configs/foundation_v1/gradient_cost_rsr.ini'})
    write_json(ARTIFACTS/'canonical_contract.json',{'cost':asdict(constants),'cost_sha256':file_hash(costpath),'gradient_sha256':file_hash(gradient),'templates':templates,'policy':'peacsim.aqa_runtimepolicy.AQARuntimePolicy','node_count_control':True,'canonical_idle_power_intentionally_preserved':True})
    return tasks


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    frozen=read_json(ARTIFACTS/'profile_partition_freeze.json')
    if file_hash(ARTIFACTS/'profiles.json')!=frozen['profiles_sha256']:raise ValueError('Profile split changed before composition')
    workloads,usage=generate(read_json(ARTIFACTS/'profiles.json'))
    write_json(ARTIFACTS/'workloads.json',workloads);write_csv(ARTIFACTS/'foundation_v1_workload_manifest.csv',workloads)
    tasks=generate_contexts(workloads)
    write_json(ARTIFACTS/'contexts.json',tasks)
    write_csv(ARTIFACTS/'foundation_v1_context_manifest.csv',tasks)
    write_json(ARTIFACTS/'profile_reuse_counts.json',dict(sorted(usage.items())))
    print('Generated500 workload INIs and2000 contexts; real FlexDC parsers used; zero simulations')
