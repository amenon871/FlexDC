"""Four measured-only waves and grouped repeats; no surrogate imports."""
import argparse
from collections import Counter
import numpy as np

from am_flexdc_foundation_v1_common import (ARTIFACTS,Domain,RADII,WAVE_QUOTA,contexts,
    novel,maximin,violations,make_row,row_geometry,read_json,write_json,write_csv,file_hash)


def rank(o): return (*violations(o['p90'],o['Pj']),o['objective'],o['Plan_Row_ID'])


def flags(o):return (o['p90']<=.3,all(p<=.1 for p in o['Pj']))


def usable(observations,j):
    if any(o.get('observation_status')!='VALID_OBSERVATION' or o.get('evidence_status')!='SUFFICIENT_JOB_TYPE_EVIDENCE' or len(o['Pj'])!=j for o in observations):raise ValueError('Unaudited, invalid or insufficient previous evidence')
    return sorted(observations,key=rank)


def boundary_pairs(domain,obs,bridge=False):
    pairs=[]
    z=domain.coordinates([o['x'] for o in obs])
    distances=np.linalg.norm(z[:,None,:]-z[None,:,:],axis=2)
    for i,a in enumerate(obs):
        at,aq=flags(a)
        for k,b in enumerate(obs[i+1:],i+1):
            bt,bq=flags(b)
            if bridge:
                # Exclusive safe sets; a joint-feasible point needs no bridge to itself.
                if at and not aq and bq and not bt: category='TRACKING_TO_QOS'
                elif bt and not bq and aq and not at: a1,b1=b,a;category='TRACKING_TO_QOS';pairs.append((float(distances[i,k]),category,a1,b1));continue
                else:continue
            else:
                categories=[name for name,test in (('JOINT',(at and aq)!=(bt and bq)),('TRACKING',at!=bt),('QOS',aq!=bq)) if test]
                if not categories:continue
                category=','.join(categories)
            pairs.append((float(distances[i,k]),category,a,b))
    return sorted(pairs,key=lambda p:(p[0],p[1],p[2]['Plan_Row_ID'],p[3]['Plan_Row_ID']))


def wave(task,observations,wave_number,pending=()):
    if wave_number not in (1,2,3,4):raise ValueError('Wave must be1..4')
    d=Domain(task['domain']);obs=usable(observations,d.j)
    expected=150+20*(wave_number-1)
    if len(obs)!=expected or len({o['Base_Geometry_ID'] for o in obs})!=expected:raise ValueError('Previous unique task count mismatch')
    used=[o['x'] for o in obs]+list(pending);out=[];fallback=[]
    radius=RADII[wave_number-1]
    def add(x,family,provenance):
        if not novel(d,x,[*used,*[a[1] for a in out]]):return False
        out.append((family,np.asarray(x),provenance));return True
    pool=d.global_pool(4096,'global-escape',task['Task_ID'],wave_number)
    # Protect4 global slots before adaptive allocation.
    for x in maximin(d,pool,used,4):add(x,'GLOBAL_ESCAPE',{'selection':'maximin','pool_size':4096})
    if len(out)!=4:raise ValueError('Global escape quota unavailable')
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
    def interpolations(bridge):
        result=[];lambdas=(.5,.25,.75,.375)
        for dist,category,a,b in boundary_pairs(d,obs,bridge):
            for t in lambdas:
                result.append((d.interpolate(a['x'],b['x'],t),{'parent_row_ids':[a['Plan_Row_ID'],b['Plan_Row_ID']],
                    'tracking_anchor_id':a['Plan_Row_ID'] if bridge else None,'qos_anchor_id':b['Plan_Row_ID'] if bridge else None,
                    'lambda':t,'anchor_outcomes':[[a['p90'],a['Pj']],[b['p90'],b['Pj']]],'normalized_parent_distance':dist,'boundary_category':category}))
            if len(result)>=128:break
        return result
    pools={'ADAPTIVE_LOCAL':local_candidates(6),'ADAPTIVE_BOUNDARY':interpolations(False),'BRIDGE':interpolations(True)}
    for family,quota in (('ADAPTIVE_LOCAL',6),('ADAPTIVE_BOUNDARY',6),('BRIDGE',4)):
        added=0
        for x,p in pools[family]:
            if add(x,family,p):added+=1
            if added==quota:break
        shortage=quota-added
        for alternative in {'BRIDGE':['ADAPTIVE_BOUNDARY','ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_BOUNDARY':['ADAPTIVE_LOCAL','GLOBAL_ESCAPE'],'ADAPTIVE_LOCAL':['GLOBAL_ESCAPE']}[family]:
            if not shortage:break
            choices=pools.get(alternative,[]) if alternative!='GLOBAL_ESCAPE' else [(x,{'pool_size':4096,'selection':'maximin-fallback'}) for x in maximin(d,pool,[*used,*[a[1] for a in out]],shortage)]
            for x,p in choices:
                if add(x,alternative,{**p,'fallback_from':family}):shortage-=1;fallback.append({'from':family,'to':alternative})
                if not shortage:break
        if shortage:raise ValueError('Unable to fill20 unique legal adaptive candidates')
    assert len(out)==20
    rows=[make_row(task,x,family,wave_number,p) for family,x,p in out]
    return rows,{'Task_ID':task['Task_ID'],'wave':wave_number,'source_counts':dict(Counter(r['Candidate_Source'] for r in rows)),
        'fallback':fallback,'input_count':len(obs),'radius':radius,'novelty':'PASS','status':'PLANNED_NOT_RUN'}


def repeats(task,observations):
    d=Domain(task['domain']);obs=usable(observations,d.j)
    if len(obs)!=230 or len({o['Base_Geometry_ID'] for o in obs})!=230:raise ValueError('Repeats require230 unique audited seed20 results')
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


def generate_stage(database,partition='TRAIN_DEV',wave_number=None,repeat=False,task_id=None,unlock=False):
    from foundation_v1.scope import normalize,require_unlock,selected_tasks,plan_path,assert_database_scope,run_root
    from foundation_v1.audit import read_task_observations,verify_database
    partition=normalize(partition);require_unlock(partition,unlock)
    from foundation_v1.environment import versions
    versions(require=True)
    if partition=='ALL':
        raise ValueError('Generate TRAIN_DEV and LOCKED_SYNTHETIC_TEST separately with their isolated ledgers')
    if (wave_number is None)==(not repeat):raise ValueError('Choose exactly one wave or repeats')
    assert_database_scope(database,partition);seal=verify_database(database)
    selected=selected_tasks(partition,task_id);expected=230 if repeat else 150+20*(wave_number-1)
    inputs={};shortfalls=[]
    for task in selected:
        obs=read_task_observations(database,task['Task_ID'],primary_only=True)
        usable(obs,Domain(task['domain']).j)
        if len(obs)!=expected or len({o['Base_Geometry_ID'] for o in obs})!=expected:
            shortfalls.append({'Task_ID':task['Task_ID'],'required_usable_unique':expected,'usable_unique':len({o['Base_Geometry_ID'] for o in obs})})
        inputs[task['Task_ID']]=obs
    if shortfalls:
        write_json(run_root(partition)/'adaptive_shortfall.json',{'status':'EVIDENCE_SHORTFALL_REQUIRES_POLICY',
            'partition':partition,'tasks':shortfalls,'replacement_coordinates_generated':False},False)
        raise ValueError('EVIDENCE_SHORTFALL_REQUIRES_POLICY: no replacement/adaptive coordinates generated')
    output=[];reports=[]
    for task in selected:
        obs=inputs[task['Task_ID']]
        if repeat:output.extend(repeats(task,obs))
        else:
            rows,report=wave(task,obs,wave_number);output.extend(rows);reports.append(report)
    name='repeat' if repeat else f'wave{wave_number:02d}'
    path=plan_path(name,partition,task_id)
    write_csv(path,output)
    write_json(path.with_suffix('.receipt.json'),{'scope':partition,'plan_sha256':file_hash(path),'rows':len(output),
        'audited_database_sha256':seal,'selection_reports':reports,'source_tasks':[t['Task_ID'] for t in selected],
        'source_partitions':sorted({t['split'] for t in selected}),
        'adaptive_protocol_sha256':file_hash(ARTIFACTS/'foundation_v1_adaptive_protocol.json'),'status':'PLANNED_NOT_RUN'})
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
