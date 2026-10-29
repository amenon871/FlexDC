"""Four measured-only waves and grouped repeats; no surrogate imports."""
import argparse
import time
from collections import Counter
import numpy as np

from am_flexdc_foundation_v1_common import (ARTIFACTS,Domain,RADII,WAVE_QUOTA,contexts,
    novel,maximin,violations,make_row,row_geometry,read_json,write_json,write_csv,file_hash,sobol)

PRODUCTION_PROTOCOL={'initial':150,'wave_quotas':(4,6,6,4),'radii':tuple(RADII),'global_pool':4096,
    'repeat_bases':10,'repeat_seeds':(21,22),'name':'FOUNDATION_V1_PRODUCTION'}


def exact_global_pool(domain,n,*namespace):
    """Vectorized, bit-exact Domain.global_pool without changing Domain science."""
    u=sobol(n,2+2*domain.j,*namespace);c=domain.context
    p=c.p_lower+(c.p_upper-c.p_lower)*u[:,0]
    r=c.r_lower+(np.minimum(.6*p,c.pr_upper-p)-c.r_lower)*u[:,1]
    result=np.empty((n,domain.j+2),dtype=float);result[:,0]=p;result[:,1]=r
    remaining=np.ones(n);order=np.argsort(u[:,2+domain.j:2+2*domain.j],axis=1);rows=np.arange(n)
    for k in range(domain.j-1):
        left=domain.j-k-1
        lo=np.maximum(domain.lower,remaining-left*domain.upper)
        hi=np.minimum(domain.upper,remaining-left*domain.lower)
        value=lo+(hi-lo)*u[:,2+k];result[rows,2+order[:,k]]=value;remaining-=value
    result[rows,2+order[:,-1]]=remaining
    if (not np.isfinite(result).all() or np.any(result[:,2:]<domain.lower-1e-12)
        or np.any(result[:,2:]>domain.upper+1e-12) or np.any(np.abs(result[:,2:].sum(axis=1)-1)>1e-9)
        or np.any(result[:,0]<c.p_lower) or np.any(result[:,0]>c.p_upper)
        or np.any(result[:,1]<c.r_lower) or np.any(result[:,1]>np.minimum(.6*result[:,0],c.pr_upper-result[:,0])+1e-12)):
        raise ValueError('Vectorized global pool violated the canonical domain')
    return list(result)


def rank(o): return (*violations(o['p90'],o['Pj']),o['objective'],o['Plan_Row_ID'])


def flags(o):return (o['p90']<=.3,all(p<=.1 for p in o['Pj']))


def usable(observations,j):
    if any(o.get('observation_status')!='VALID_OBSERVATION' or o.get('evidence_status')!='SUFFICIENT_JOB_TYPE_EVIDENCE' or len(o['Pj'])!=j for o in observations):raise ValueError('Unaudited, invalid or insufficient previous evidence')
    return sorted(observations,key=rank)


def boundary_pair_sets(domain,obs):
    pairs=[];bridges=[]
    z=domain.coordinates([o['x'] for o in obs])
    distances=np.linalg.norm(z[:,None,:]-z[None,:,:],axis=2)
    for i,a in enumerate(obs):
        at,aq=flags(a)
        for k,b in enumerate(obs[i+1:],i+1):
            bt,bq=flags(b)
            categories=[name for name,test in (('JOINT',(at and aq)!=(bt and bq)),('TRACKING',at!=bt),('QOS',aq!=bq)) if test]
            if categories:pairs.append((float(distances[i,k]),','.join(categories),a,b))
            # Exclusive safe sets; a joint-feasible point needs no bridge to itself.
            if at and not aq and bq and not bt:bridges.append((float(distances[i,k]),'TRACKING_TO_QOS',a,b))
            elif bt and not bq and aq and not at:bridges.append((float(distances[i,k]),'TRACKING_TO_QOS',b,a))
    key=lambda p:(p[0],p[1],p[2]['Plan_Row_ID'],p[3]['Plan_Row_ID'])
    return sorted(pairs,key=key),sorted(bridges,key=key)


def boundary_pairs(domain,obs,bridge=False):
    pairs,bridges=boundary_pair_sets(domain,obs)
    return bridges if bridge else pairs


def wave(task,observations,wave_number,pending=(),protocol=None):
    if wave_number not in (1,2,3,4):raise ValueError('Wave must be1..4')
    protocol=protocol or PRODUCTION_PROTOCOL
    d=Domain(task['domain']);obs=usable(observations,d.j)
    total_quota=sum(protocol['wave_quotas']);expected=protocol['initial']+total_quota*(wave_number-1)
    if len(obs)!=expected or len({o['Base_Geometry_ID'] for o in obs})!=expected:raise ValueError('Previous unique task count mismatch')
    used=[o['x'] for o in obs]+list(pending);out=[];fallback=[]
    radius=protocol['radii'][wave_number-1]
    def add(x,family,provenance):
        if not novel(d,x,[*used,*[a[1] for a in out]]):return False
        out.append((family,np.asarray(x),provenance));return True
    global_quota,local_quota,boundary_quota,bridge_quota=protocol['wave_quotas'];pool_size=protocol['global_pool']
    pool=exact_global_pool(d,pool_size,'global-escape',task['Task_ID'],wave_number)
    # Protect global slots before adaptive allocation.
    for x in maximin(d,pool,used,global_quota):add(x,'GLOBAL_ESCAPE',{'selection':'maximin','pool_size':pool_size})
    if len(out)!=global_quota:raise ValueError('Global escape quota unavailable')
    feasible=[o for o in obs if all(flags(o))];infeasible=[o for o in obs if not all(flags(o))]
    anchors=([min(feasible,key=lambda o:(o['objective'],o['Plan_Row_ID']))] if feasible else [])+infeasible
    anchors += obs
    selected=[]
    for a in anchors:
        if all(d.distance(a['x'],b['x'])>1e-7 for b in selected):selected.append(a)
        if len(selected)==3:break
    def local_candidates(count):
        candidate_groups=[]
        for a in selected:
            candidates=[]
            for x in d.local(a['x'],radius,max(2,count),'local',task['Task_ID'],wave_number,a['Plan_Row_ID']):
                candidates.append((x,{'parent_row_ids':[a['Plan_Row_ID']],'local_radius':radius,'normalized_parent_distance':d.distance(a['x'],x)}))
            candidate_groups.append(candidates)
        # Round-robin preserves the intended two points per anchor for6 slots.
        return [group[i] for i in range(max(map(len,candidate_groups),default=0)) for group in candidate_groups if i<len(group)]
    normal_pairs,bridge_pairs=boundary_pair_sets(d,obs)
    def interpolations(bridge):
        result=[];lambdas=(.5,.25,.75,.375)
        for dist,category,a,b in (bridge_pairs if bridge else normal_pairs):
            for t in lambdas:
                result.append((d.interpolate(a['x'],b['x'],t),{'parent_row_ids':[a['Plan_Row_ID'],b['Plan_Row_ID']],
                    'tracking_anchor_id':a['Plan_Row_ID'] if bridge else None,'qos_anchor_id':b['Plan_Row_ID'] if bridge else None,
                    'lambda':t,'anchor_outcomes':[[a['p90'],a['Pj']],[b['p90'],b['Pj']]],'normalized_parent_distance':dist,'boundary_category':category}))
            if len(result)>=128:break
        return result
    pools={'ADAPTIVE_LOCAL':local_candidates(local_quota),'ADAPTIVE_BOUNDARY':interpolations(False),'BRIDGE':interpolations(True)}
    for family,quota in (('ADAPTIVE_LOCAL',local_quota),('ADAPTIVE_BOUNDARY',boundary_quota),('BRIDGE',bridge_quota)):
        added=0
        for x,p in pools[family]:
            if add(x,family,p):added+=1
            if added==quota:break
        shortage=quota-added
        for alternative in {'BRIDGE':['ADAPTIVE_BOUNDARY','ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_BOUNDARY':['ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_LOCAL':['GLOBAL_ESCAPE']}[family]:
            if not shortage:break
            choices=pools.get(alternative,[]) if alternative!='GLOBAL_ESCAPE' else [(x,{'pool_size':pool_size,'selection':'maximin-fallback'}) for x in maximin(d,pool,[*used,*[a[1] for a in out]],shortage)]
            for x,p in choices:
                if add(x,alternative,{**p,'fallback_from':family}):shortage-=1;fallback.append({'from':family,'to':alternative})
                if not shortage:break
        if shortage:raise ValueError('Unable to fill '+str(total_quota)+' unique legal adaptive candidates')
    assert len(out)==total_quota
    rows=[make_row(task,x,family,wave_number,p) for family,x,p in out]
    return rows,{'Task_ID':task['Task_ID'],'wave':wave_number,'source_counts':dict(Counter(r['Candidate_Source'] for r in rows)),
        'fallback':fallback,'input_count':len(obs),'radius':radius,'novelty':'PASS','status':'PLANNED_NOT_RUN'}


def repeats(task,observations,protocol=None):
    protocol=protocol or PRODUCTION_PROTOCOL
    d=Domain(task['domain']);obs=usable(observations,d.j)
    expected=protocol['initial']+4*sum(protocol['wave_quotas'])
    if len(obs)!=expected or len({o['Base_Geometry_ID'] for o in obs})!=expected:raise ValueError('Repeats require '+str(expected)+' unique audited seed20 results')
    if protocol is not PRODUCTION_PROTOCOL and protocol!=PRODUCTION_PROTOCOL:
        bases=maximin(d,[o['x'] for o in obs],[],protocol['repeat_bases']);rows=[]
        for x in bases:
            o=next(o for o in obs if np.array_equal(o['x'],x))
            for s in protocol['repeat_seeds']:
                rows.append(make_row(task,o['x'],'REPEAT',5,{'parent_row_ids':[o['Plan_Row_ID']],
                    'repeat_rationale':'test_protocol_measured_maximin'},s,o['Plan_Row_ID']))
        return rows
    selected=[]
    def choose(pool,count,reason):
        for o in pool:
            if rank(o)[0]>=4 and any(rank(a)[0]<4 and a['Plan_Row_ID'] not in [b[0]['Plan_Row_ID'] for b in selected] for a in obs):continue
            if o['Plan_Row_ID'] not in [a[0]['Plan_Row_ID'] for a in selected]:selected.append((o,reason));count-=1
            if count==0:break
    def margin(o):return abs(max(np.array([o['p90'],*o['Pj']])/np.array([.3]+[.1]*d.j)-1))
    choose(sorted(obs,key=lambda o:(margin(o),*rank(o))),4,'closest_joint_constraint_boundary')
    choose(sorted([o for o in obs if all(flags(o))],key=lambda o:(margin(o),*rank(o))),2,'feasible_near_boundary')
    choose([o for o in obs if flags(o)==(True,False)],1,'tracking_safe_qos_fail')
    choose([o for o in obs if flags(o)==(False,True)],1,'qos_safe_tracking_fail')
    choose([o for o in obs if np.allclose(o['x'][2:],1/d.j,rtol=0,atol=1e-9)],2,'equal_weight_reference')
    if len(selected)<10:
        informative=[o for o in obs if rank(o)[0]<4]
        if len({o['Plan_Row_ID'] for o in informative}-{o['Plan_Row_ID'] for o,_ in selected})<10-len(selected):informative=obs
        xs=maximin(d,[o['x'] for o in informative],[o['x'] for o,_ in selected],10-len(selected))
        for x in xs:
            o=next(o for o in obs if np.array_equal(o['x'],x));selected.append((o,'measured_maximin_fallback'))
    if len(selected)!=10:raise ValueError('Repeat base selection must contain10 distinct geometries')
    rows=[]
    for o,reason in selected:
        for s in (21,22):
            r=make_row(task,o['x'],'REPEAT',5,{'parent_row_ids':[o['Plan_Row_ID']],'repeat_rationale':reason},s,o['Plan_Row_ID'])
            assert r['Base_Geometry_ID']==o['Base_Geometry_ID'];rows.append(r)
    return rows


def generate_stage(database,partition='TRAIN_DEV',wave_number=None,repeat=False,task_id=None,unlock=False,
        progress_seconds=10.,stall_seconds=300.,protocol=None):
    from foundation_v1.scope import normalize,require_unlock,selected_tasks,plan_path,assert_database_scope,run_root
    from foundation_v1.audit import (read_task_observations,verify_planning_database,
        ensure_planning_indexes,readonly_connection)
    partition=normalize(partition);require_unlock(partition,unlock)
    from foundation_v1.environment import versions
    versions(require=True)
    if partition=='ALL':
        raise ValueError('Generate TRAIN_DEV and LOCKED_SYNTHETIC_TEST separately with their isolated ledgers')
    if (wave_number is None)==(not repeat):raise ValueError('Choose exactly one wave or repeats')
    assert_database_scope(database,partition)
    index_report=ensure_planning_indexes(database);verification=verify_planning_database(database)
    seal=verification['database_content_sha256']
    protocol=protocol or PRODUCTION_PROTOCOL;wave_total=sum(protocol['wave_quotas'])
    selected=selected_tasks(partition,task_id);expected=protocol['initial']+wave_total*(4 if repeat else wave_number-1)
    output=[];reports=[];shortfalls=[];started=time.perf_counter();last_report=started
    loading_seconds=0.;candidate_seconds=0.
    db=readonly_connection(database)
    try:
        for number,task in enumerate(selected,1):
            task_started=time.perf_counter()
            obs=read_task_observations(database,task['Task_ID'],primary_only=True,db=db,verify_bodies=True)
            loaded=time.perf_counter();loading_seconds+=loaded-task_started
            usable(obs,Domain(task['domain']).j)
            unique=len({o['Base_Geometry_ID'] for o in obs})
            if len(obs)!=expected or unique!=expected:
                shortfalls.append({'Task_ID':task['Task_ID'],'required_usable_unique':expected,'usable_unique':unique})
            elif repeat:output.extend(repeats(task,obs,protocol))
            else:
                rows,report=wave(task,obs,wave_number,protocol=protocol);output.extend(rows);reports.append(report)
            now=time.perf_counter();candidate_seconds+=now-loaded;task_elapsed=now-task_started
            if task_elapsed>stall_seconds:raise RuntimeError('ADAPTIVE_PLANNING_STALL: task='+task['Task_ID']+' elapsed='+str(round(task_elapsed,3))+'s')
            if now-last_report>=progress_seconds or number==len(selected):
                elapsed=now-started;rate=number/elapsed if elapsed else 0.;eta=(len(selected)-number)/rate if rate else None
                print('ADAPTIVE PLAN '+('repeat' if repeat else f'wave{wave_number:02d}')+' '+str(number)+' / '+str(len(selected))+
                    ' tasks elapsed='+f'{elapsed:.1f}'+'s rate='+f'{rate:.1f}'+' tasks/s ETA='+f'{eta:.0f}'+'s',flush=True)
                last_report=now
    finally:db.close()
    if shortfalls:
        write_json(run_root(partition)/'adaptive_shortfall.json',{'status':'EVIDENCE_SHORTFALL_REQUIRES_POLICY',
            'partition':partition,'tasks':shortfalls,'replacement_coordinates_generated':False},False)
        raise ValueError('EVIDENCE_SHORTFALL_REQUIRES_POLICY: no replacement/adaptive coordinates generated')
    name='repeat' if repeat else f'wave{wave_number:02d}'
    path=plan_path(name,partition,task_id)
    write_started=time.perf_counter();write_csv(path,output);plan_write_seconds=time.perf_counter()-write_started
    write_json(path.with_suffix('.receipt.json'),{'scope':partition,'plan_sha256':file_hash(path),'rows':len(output),
        'audited_database_sha256':seal,'selection_reports':reports,'source_tasks':[t['Task_ID'] for t in selected],
        'source_partitions':sorted({t['split'] for t in selected}),
        'adaptive_protocol_sha256':file_hash(ARTIFACTS/'foundation_v1_adaptive_protocol.json'),'status':'PLANNED_NOT_RUN',
        'planning':{'index':index_report,'database_verification':verification,'elapsed_seconds':time.perf_counter()-started,
            'observation_query_decode_verify_seconds':loading_seconds,'candidate_generation_seconds':candidate_seconds,
            'plan_write_seconds':plan_write_seconds,'tasks':len(selected),'rows':len(output)}})
    return path


def main():
    from foundation_v1.scope import CHOICES,normalize,database_path,require_unlock
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wave',type=int,choices=range(1,5));p.add_argument('--repeats',action='store_true')
    p.add_argument('--database');p.add_argument('--task');p.add_argument('--partition',choices=CHOICES,default='TRAIN_DEV');p.add_argument('--unlock-locked-test',action='store_true')
    args=p.parse_args();partition=normalize(args.partition)
    require_unlock(partition,args.unlock_locked_test)
    if partition=='ALL':
        if args.database or args.task:p.error('ALL requires separate default ledgers; select a single scope for overrides')
        for scope in ('TRAIN_DEV','LOCKED_SYNTHETIC_TEST'):
            print(generate_stage(database_path(scope),scope,args.wave,args.repeats,unlock=args.unlock_locked_test))
    else:print(generate_stage(args.database or database_path(partition),partition,args.wave,args.repeats,args.task,args.unlock_locked_test))
