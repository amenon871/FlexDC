"""Planning/contract tests only; no simulator or model is invoked."""
import copy
import json
from collections import Counter
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import am_generate_flexdc_exact_sweep_plan as exact
from am_flexdc_foundation_v1_common import (ROOT,ARTIFACTS,Domain,canonical,seed,weight_bounds,
    contexts,digest,file_hash,write_csv,read_csv,write_json,read_json,novel,maximin,violations,make_row,row_geometry)
from foundation_v1.profiles import generate,latent,FIELDS
from foundation_v1.planning import static_candidates
from foundation_v1.adaptive import wave,repeats,boundary_pairs
from foundation_v1.audit import observation,connection,store,verify_database,read_task_observations
from foundation_v1.campaign import commands,CampaignLock,pending_rows


@pytest.fixture
def task():return contexts()[0]


def resized(task,j):
    t=copy.deepcopy(task);c=t['domain'];c['job_type_count']=j
    lo,hi,lower,upper=weight_bounds(j,c['server_count'])
    c.update(configured_weight_lower=lo,configured_weight_upper=hi,effective_weight_lower=lower,effective_weight_upper=upper)
    t['Task_ID']='SYNTHETIC_TEST_J'+str(j);t['Profile_IDs']=['FAKE']*j
    return t


def fake_observations(task,landscape,n=150):
    d=Domain(task['domain']);xs=d.global_pool(n,'SYNTHETIC_TEST',task['Task_ID'])
    out=[]
    for i,x in enumerate(xs):
        row=make_row(task,x,'JOINT_SOBOL_INTERIOR');p90,pj=landscape(x,i)
        out.append({'Plan_Row_ID':row['plan_row_id'],'Base_Geometry_ID':row['Base_Geometry_ID'],'x':x.tolist(),
            'p90':p90,'Pj':pj,'objective':100+i,'observation_status':'VALID_OBSERVATION','evidence_status':'SUFFICIENT_JOB_TYPE_EVIDENCE'})
    return out


def test_profiles_deterministic_and_physical():
    a,p=generate();b,q=generate();assert canonical([a,p])==canonical([b,q]);assert len(a)==13 and len(p)==400
    assert Counter(x['split'] for x in p)=={'TRAIN':300,'DEVELOPMENT':50,'LOCKED_SYNTHETIC_TEST':50}
    assert Counter(x['hardware_class'] for x in p)=={'P100':240,'A100':160}
    lookup={x['profile_id']:x for x in a}
    for x in p:
        z=latent(x);anchors=[lookup[i] for i in x['anchor_ids']]
        assert all((v['hardware_class'],v['workload_class'],v['job_size'])==(x['hardware_class'],x['workload_class'],x['job_size']) for v in anchors)
        assert np.allclose(z,np.array(x['anchor_coefficients'])@np.array([v['latent'] for v in anchors]),rtol=0,atol=1e-12)
        assert min(x['anchor_coefficients'])>=0 and sum(x['anchor_coefficients'])==pytest.approx(1)
    assert len({tuple(round(x[f],10) for f in FIELDS) for x in p})==400


def test_workload_distributions_and_isolation():
    w=read_json(ARTIFACTS/'workloads.json');p=read_json(ARTIFACTS/'profiles.json');lookup={v['profile_id']:v for v in p['synthetic_profiles']}
    assert len(w)==500
    for split,n,js,qs in [('TRAIN',400,[60,80,120,80,60],[100,60,140,100]),('DEVELOPMENT',50,[8,10,14,10,8],[12,8,18,12]),('LOCKED_SYNTHETIC_TEST',50,[8,10,14,10,8],[12,8,18,12])]:
        subset=[x for x in w if x['split']==split]
        assert len(subset)==n
        assert Counter(x['J'] for x in subset)==dict(zip(range(2,7),js))
        assert Counter(x['QoS_pattern'] for x in subset)==dict(zip(('ALL_5','ALL_4','BALANCED_MIXED','BOTTLENECK'),qs))
        assert Counter(x['hardware_class'] for x in subset)=={'P100':int(n*.6),'A100':int(n*.4)}
        for x in subset:
            assert len(set(x['Profile_IDs']))==x['J']
            for pid in x['Profile_IDs']:
                if pid.startswith('REAL_'):assert split=='TRAIN'
                else:assert lookup[pid]['split']==split and lookup[pid]['hardware_class']==x['hardware_class']
            if x['grounding']=='ANCHOR_GROUNDED':assert x['real_profile_slots']>=(x['J']+1)//2
    grounded=[x for x in w if x['grounding']=='ANCHOR_GROUNDED']
    assert len(grounded)==80 and Counter(x['hardware_class'] for x in grounded)=={'P100':48,'A100':32}


@pytest.mark.parametrize('j',range(2,7))
def test_domain_and_full_joint_static_design(task,j):
    t=resized(task,j);d=Domain(t['domain']);lo,hi,l,u=weight_bounds(j,d.context.server_count)
    assert lo==pytest.approx(.6/j) and hi==pytest.approx(1.8/j)
    xs=static_candidates(t)
    assert Counter(f for f,_,_ in xs)=={'JOINT_SOBOL_INTERIOR':48,'STRUCTURED_PR':30,'PHYSICAL_PR_BOUNDARY':24,'WEIGHT_SIMPLEX_BOUNDARY':24,'CROSS_EXTREME':16,'EQUAL_WEIGHT_REFERENCE':8}
    for _,x,_ in xs:d.validate(x)
    assert np.array_equal(np.array([x for _,x,_ in xs]),np.array([x for _,x,_ in static_candidates(t)]))
    seen=[]
    for f,x,_ in xs:assert novel(d,x,seen);seen.append(x)
    eq=xs[-1][1];assert np.allclose(eq[2:],1/j,rtol=0,atol=1e-12)
    for pos,value in [(0,d.context.p_lower-1e-5),(0,d.context.p_upper+1e-5),(1,.009)]:
        x=eq.copy();x[pos]=value
        with pytest.raises(ValueError):d.validate(x)
    x=eq.copy();x[1]=.6*x[0]+1e-4
    with pytest.raises(ValueError):d.validate(x)
    x=eq.copy();x[2]=l-.001;x[3]+=1/j-x[2]
    with pytest.raises(ValueError):d.validate(x)
    x=eq.copy();x[2]+=1e-3
    with pytest.raises(ValueError):d.validate(x)
    x=eq.copy();x[0]=float('nan')
    with pytest.raises(ValueError):d.validate(x)


@pytest.mark.parametrize('scenario',range(6))
def test_fake_adaptive_landscapes(task,scenario):
    j=task['domain']['job_type_count']
    def landscape(x,i):
        if scenario==0:return (.2,[.05]*j) if i%2==0 else (.5,[.2]*j)
        if scenario==1:return (.2,[.4]*j) if i%2==0 else (.6,[.05]*j)
        if scenario==2:return (.31+.001*(i%7),[.101]*j)
        if scenario==3:return (10.,[.99]*j)
        if scenario==4:return (.2,[.05]*j) if i==75 else (.4,[.3]*j)
        return (.2,[.05]*j)
    obs=fake_observations(task,landscape)
    rows,report=wave(task,obs,1)
    assert len(rows)==20 and report['source_counts']['GLOBAL_ESCAPE']>=4
    d=Domain(task['domain']);used=[o['x'] for o in obs]
    for r in rows:
        x=row_geometry(r);assert novel(d,x,used);used.append(x)
        if r['Candidate_Source']=='ADAPTIVE_LOCAL':assert r['Provenance']['normalized_parent_distance']<=.12+1e-12
        if r['Candidate_Source']=='BRIDGE':
            p=r['Provenance'];a,b=[next(o for o in obs if o['Plan_Row_ID']==pid)['x'] for pid in p['parent_row_ids']]
            assert np.allclose(x,(1-p['lambda'])*np.array(a)+p['lambda']*np.array(b))
    if scenario in (2,3,5):assert report['fallback']
    assert canonical(rows)==canonical(wave(task,obs,1)[0])


def test_bridge_safe_pair_and_boundary_equality(task):
    j=task['domain']['job_type_count'];d=Domain(task['domain'])
    obs=fake_observations(task,lambda x,i:(.3,[.2]*j) if i%2==0 else (.4,[.1]*j))
    pairs=boundary_pairs(d,obs,True);assert pairs
    assert pairs[0][0]==pytest.approx(min(d.distance(a['x'],b['x']) for a in obs[::2] for b in obs[1::2]))
    assert violations(.3,[.1]*j)==(0.,0.)
    with pytest.raises(ValueError):violations(float('nan'),[.1]*j)
    assert violations(.3001,[.1]*j)[0]>0


@pytest.mark.parametrize('j',[2,3,4,5,6])
def test_adaptive_and_repeats_variable_j(task,j):
    task=resized(task,j);d=Domain(task['domain'])
    obs=fake_observations(task,lambda x,i:(.2,[.3]*j) if i%2 else (.8,[.05]*j))
    for number,radius in enumerate((.12,.08,.05,.03),1):
        rows,report=wave(task,obs,number)
        assert len(rows)==20
        for r in rows:
            obs.append({'Plan_Row_ID':r['plan_row_id'],'Base_Geometry_ID':r['Base_Geometry_ID'],'x':row_geometry(r).tolist(),'p90':.4,'Pj':[.2]*j,'objective':100.,'observation_status':'VALID_OBSERVATION','evidence_status':'SUFFICIENT_JOB_TYPE_EVIDENCE'})
    repeat=repeats(task,obs);assert len(repeat)==20
    assert Counter(r['simulation_seed'] for r in repeat)=={21:10,22:10}
    assert len({r['Base_Geometry_ID'] for r in repeat})==10
    assert all(any(o['Base_Geometry_ID']==r['Base_Geometry_ID'] for o in obs) for r in repeat)


def test_identity_compression_and_immutable_plan(task,tmp_path):
    d=Domain(task['domain']);x=d.global_pool(1,'identity')[0]
    a=make_row(task,x,'GLOBAL_ESCAPE');b=make_row(task,x,'BRIDGE',2)
    assert a['plan_row_id']==b['plan_row_id']
    p=tmp_path/'a.csv.gz';q=tmp_path/'b.csv.gz';write_csv(p,[a]);write_csv(q,[a]);assert file_hash(p)==file_hash(q)
    write_csv(p,[a]);a['simulation_seed']=21
    with pytest.raises(ValueError):write_csv(p,[a])
    assert seed('a')==seed('a') and seed('a')!=seed('b')


def test_maximin_pending_and_radius(task):
    d=Domain(task['domain']);pool=d.global_pool(128,'test-global');used=pool[:4]
    chosen=maximin(d,pool,used,4)
    for c in chosen:
        best=max(min(d.distance(x,y) for y in used) for x in pool)
        assert min(d.distance(c,y) for y in used)==pytest.approx(best);used.append(c)
    for radius in (.12,.08,.05,.03):
        for c in d.local(pool[0],radius,10,'radius',radius):assert 0<d.distance(c,pool[0])<=radius+1e-12


def test_resume_database_and_corruption(task,tmp_path):
    dbpath=tmp_path/'x.sqlite';db=connection(dbpath)
    o={'Plan_Row_ID':'F1_test','Task_ID':task['Task_ID'],'simulation_seed':20,'p90':.2,'Pj':[.1]*4}
    with db:store(db,o,{'new_foundation_only':True})
    with db:store(db,o,{'new_foundation_only':True})
    assert db.execute('select count(*) from observations').fetchone()[0]==1
    with pytest.raises(ValueError):store(db,{**o,'p90':.4},{})
    db.close();assert verify_database(dbpath)
    assert read_task_observations(dbpath,task['Task_ID'])[0]==o
    db=connection(dbpath)
    with db:db.execute("UPDATE observations SET body='{}'")
    db.close()
    with pytest.raises(ValueError):verify_database(dbpath)


def test_no_execution_by_default_and_no_hidden_simulator(task,monkeypatch,tmp_path):
    from foundation_v1 import campaign
    def forbidden(*a,**k):raise AssertionError('No process may launch')
    monkeypatch.setattr(campaign.subprocess,'call',forbidden)
    plan=tmp_path/'foundation_v1_phaseA_plan.csv.gz'
    write_csv(plan,[make_row(task,Domain(task['domain']).global_pool(1,'dry')[0],'GLOBAL_ESCAPE')])
    monkeypatch.setattr(campaign,'ARTIFACTS',tmp_path)
    monkeypatch.setattr(campaign,'stage_path',lambda *args:plan)
    monkeypatch.setattr(campaign,'read_json',lambda path:{'source_files':{},'phaseA_sha256':file_hash(plan)})
    monkeypatch.setattr('sys.argv',['foundation','--task',task['Task_ID']])
    campaign.main()
    cmd=commands(Path('x.csv'),3,'fixed')
    assert cmd[cmd.index('--plan-max-workers')+1]=='3'
    assert 'am_run_flexdc_dataset_parallel_combo.py' in ' '.join(cmd)


def fake_receipt(task):
    d=Domain(task['domain']);x=d.global_pool(1,'audit')[0];row=make_row(task,x,'JOINT_SOBOL_INTERIOR')
    r={'Plan_Row_ID':row['plan_row_id'],'Pbar_kw_per_server':x[0],'R_kw_per_server':x[1],'server_count':d.context.server_count,'utilization':d.context.utilization,'Simulation_Seed':20,'Mtrack_Error_MeanAbs_Normalized':.1,'Simulator_RSR_Total_Cost':50,'QoS_Delay_Probabilities':json.dumps([.1]*d.j)}
    r.update({f'Weight_{i}':x[i+2] for i in range(d.j)})
    diag={k:v for k,v in r.items() if k in ('Plan_Row_ID','Pbar_kw_per_server','R_kw_per_server','server_count','utilization','Simulation_Seed')}
    diag.update(Ctrack_Epsilon_90th=.3,Diagnostic_FullPaperObjective_Cost=50+np.logaddexp(0,0)+20*d.j*np.logaddexp(0,0))
    a={'N':d.context.server_count,'U':d.context.utilization,'seed':20,'J':d.j,'duration':3600,'idle_watts':120.,'policy_module':'peacsim.aqa_runtimepolicy','node_count_control':True,'weights':x[2:].tolist(),'P_watts':round(x[0]*1000*d.context.server_count),'R_watts':round(x[1]*1000*d.context.server_count),'workload_path':d.context.workload_config,'workload_sha256':task['Workload_SHA256'],'iso_sha256':task['ISO_SHA256'],'randomize_iso_start':False,'normalize_iso_signal':True,'iso_signal_start_hour':16,'iso_signal_granularity':2}
    body={'schema':'FOUNDATION_V1_EVIDENCE','Plan_Row_ID':row['plan_row_id'],'results':r,'diagnostics':diag,'applied':a,'evidence_counts':[{'job_type_id':i,'observations':1} for i in range(d.j)],'evidence_status':'SUFFICIENT_JOB_TYPE_EVIDENCE'}
    body['body_sha256']=digest(body)
    return row,body


def test_requested_applied_evidence_and_missing_qos(task):
    row,r=fake_receipt(task);o=observation(row,r,task)
    assert o['evidence_qualified_feasible'] and o['Vmax']==0
    for field,value in [('J',3),('U',.1),('seed',999),('P_watts',0),('weights',[.25]*3),('iso_signal_start_hour',15)]:
        bad=copy.deepcopy(r);bad['applied'][field]=value;bad.pop('body_sha256');bad['body_sha256']=digest(bad)
        with pytest.raises(ValueError):observation(row,bad,task)
    bad=copy.deepcopy(r);bad['results']['QoS_Delay_Probabilities']='[]';bad.pop('body_sha256');bad['body_sha256']=digest(bad)
    with pytest.raises(ValueError):observation(row,bad,task)
    bad=copy.deepcopy(r);bad['evidence_counts'][0]['observations']=0;bad['evidence_status']='INSUFFICIENT_JOB_TYPE_EVIDENCE';bad.pop('body_sha256');bad['body_sha256']=digest(bad)
    o=observation(row,bad,task);assert o['numerical_status']=='NOT_EVALUABLE_DUE_TO_INSUFFICIENT_EVIDENCE' and o['observation_status']=='INSUFFICIENT_JOB_TYPE_EVIDENCE' and not o['evidence_qualified_feasible']


def test_foundation_evidence_hook_is_observational(task,tmp_path,monkeypatch):
    from peacsim.foundation_v1_evidence import write_evidence
    monkeypatch.setattr('peacsim.foundation_v1_evidence.runtime_provenance',lambda:{'execution_git_SHA':'SYNTHETIC_TEST_ONLY'})
    row,receipt=fake_receipt(task);row['design_version']='FOUNDATION_V1';a=receipt['applied'];j=a['J']
    table=pd.DataFrame({'job_type_id':list(range(j)),'arrival_time':[10.]*j,'end_time':[20.]*j})
    before=table.copy(deep=True)
    sim=SimpleNamespace(_job_table=table,_runtime_policy=SimpleNamespace(P=a['P_watts'],R=a['R_watts'],weights=dict(enumerate(a['weights'])),_policy_parameters={'node_count_control':True}))
    experiment=SimpleNamespace(server_count=a['N'],utilization=a['U'],random_seed=20,simulation_duration=3600,idle_power=120.,iso_file_path=str(ROOT/'data/1-17-2025_traditional.csv'),iso_signal_start_hour=16,iso_signal_granularity=2,normalize_iso_signal=True,randomize_iso_start=False)
    job=SimpleNamespace(job_type_count=j,all_min_execution_time={i:1. for i in range(j)},all_jobs={i:str(i) for i in range(j)},workload_mix_name=str(ROOT/task['domain']['workload_config']))
    for name,key in [('grid_search_results.csv','results'),('grid_search_diagnostics.csv','diagnostics')]:
        with (tmp_path/name).open('w',newline='') as f:
            import csv
            writer=csv.writer(f);writer.writerow(receipt[key]);writer.writerow(receipt[key].values())
    write_evidence(tmp_path,row,sim,experiment,job,[row['Pbar_kw_per_server'],row['R_kw_per_server']],list(receipt['results'].values()),list(receipt['diagnostics'].values()))
    pd.testing.assert_frame_equal(table,before)
    result=read_json(tmp_path/'foundation_evidence'/f"{row['plan_row_id']}.json")
    assert [r['observations'] for r in result['evidence_counts']]==[1]*j
    assert result['evidence_status']=='SUFFICIENT_JOB_TYPE_EVIDENCE'
    assert max(abs(v) for v in result['power_quantization_delta_watts'])<=.5


def test_historical_worker_ast_unchanged_except_guard():
    import ast,subprocess
    path='src/peacsim/am_data_extraction_wizard.py'
    historical=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','525dc684d73ab0c6f6c479f5b54811ddf02f1221:'+path],text=True))
    current=ast.parse((ROOT/path).read_text(encoding='utf8'))
    count=0
    for node in ast.walk(current):
        if isinstance(node,ast.FunctionDef) and node.name=='objective_function':
            for statement in list(node.body):
                if isinstance(statement,ast.If) and "'FOUNDATION_V1'" in ast.unparse(statement.test):
                    node.body.remove(statement);count+=1
    assert count==1 and ast.dump(historical,include_attributes=False)==ast.dump(current,include_attributes=False)


def test_repeat_selection_does_not_fill_with_catastrophe_when_better_exists(task):
    d=Domain(task['domain']);j=d.j
    obs=fake_observations(task,lambda x,i:(.31,[.101]*j),230)
    x=np.array(obs[0]['x']);x[2:]=1/j
    row=make_row(task,x,'EQUAL_WEIGHT_REFERENCE')
    obs[0].update(x=x.tolist(),Plan_Row_ID=row['plan_row_id'],Base_Geometry_ID=row['Base_Geometry_ID'],p90=20.,Pj=[.99]*j)
    selected=repeats(task,obs)
    assert len(selected)==20 and obs[0]['Base_Geometry_ID'] not in {r['Base_Geometry_ID'] for r in selected}


def test_only_seven_canonical_cost_keys_change():
    from am_flexdc_foundation_v1_common import parser
    base=parser(ROOT/'configs/gradient_descent/gradient_descent.ini');new=parser(ROOT/'configs/foundation_v1/gradient_cost_rsr.ini')
    allowed={('calculate_gradient','psi1'),('calculate_gradient','psi2'),('calculate_gradient','tracking_error_constraint'),('calculate_gradient','qos_threshold'),('gradient_driver','beta'),('gradient_driver','rho'),('dr_program','program_type')}
    for section in set(base.sections())|set(new.sections()):
        for key in set(base.options(section) if base.has_section(section) else [])|set(new.options(section)):
            if (section,key) not in allowed:assert base.get(section,key)==new.get(section,key)


def test_disconnected_pocket_reached_by_protected_global(task):
    d=Domain(task['domain']);j=d.j
    obs=fake_observations(task,lambda x,i:(.2,[.05]*j) if i==0 else (.8,[.8]*j))
    proposed,_=wave(task,obs,1)
    target=row_geometry(next(r for r in proposed if r['Candidate_Source']=='GLOBAL_ESCAPE'))
    original=np.array(obs[0]['x'])
    other_inputs=[o['x'] for o in obs[1:]]
    non_global=[row_geometry(r) for r in proposed if r['Candidate_Source']!='GLOBAL_ESCAPE']
    radius=min(.005,min(d.distance(target,x) for x in [o['x'] for o in obs])/4,
        min(d.distance(target,x) for x in non_global)/4,
        min(d.distance(original,x) for x in other_inputs)/4)
    assert radius>0 and d.distance(target,original)>4*radius
    def second_pocket(x):return d.distance(x,target)<radius
    assert not any(second_pocket(o['x']) for o in obs)
    assert any(second_pocket(row_geometry(r)) for r in proposed if r['Candidate_Source']=='GLOBAL_ESCAPE')
    assert not any(second_pocket(row_geometry(r)) for r in proposed if r['Candidate_Source']!='GLOBAL_ESCAPE')


def test_violation_ranking_precedes_objective(task):
    from foundation_v1.adaptive import rank
    j=Domain(task['domain']).j
    obs=fake_observations(task,lambda x,i:(.31,[.101]*j) if i==0 else (.9,[.8]*j))
    obs[0]['objective']=1e6;obs[1]['objective']=-1e6
    assert rank(obs[0])<rank(obs[1])


def test_canonical_contract_and_count_algebra():
    c=read_json(ARTIFACTS/'canonical_contract.json')
    assert [c['cost'][k] for k in ('psi1','psi2','tracking_error_constraint','beta','rho','qos_constraint')]==[1,10,.3,20,2,.1]
    assert c['cost']['program_type']=='RSR' and all(t['idle_watts']==120 for t in c['templates'])
    assert all(t['ISO']['randomize_iso_start']=='False' for t in c['templates'])
    assert 500*4==2000 and 2000*(150+4*20+10*2)==500000
