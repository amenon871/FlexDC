import copy,csv,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from flexmodels.data import prepare,load_cache,partition_indices,sampler_weights
from flexmodels.features import fit_normalization,validate_domain
from flexmodels.episodes import K_VALUES,task_pools,episode,make_bank,validate_episode,diverse_order,load_bank
from flexmodels.objective import reconstruct,MEAN_FLOOR,P90_FLOOR


def test_fixture_schema_and_repeat_groups(cache):
    assert cache['fixture'] and cache['manifest']['rows']==2160
    assert {m['J'] for m in cache['metadata']}=={2,4,6} and {m['N'] for m in cache['metadata']}=={250,1000} and {m['U'] for m in cache['metadata']}=={.6,.8}
    groups=defaultdict(set)
    for m in cache['metadata']:groups[(m['Task_ID'],m['Base_Geometry_ID'])].add(m['simulation_seed'])
    assert set(map(len,groups.values()))=={1,3}
    assert {m['hardware_class'] for m in cache['metadata']}=={'P100','A100'}


def test_train_only_normalization_and_constant_guard():
    jobs=np.zeros((4,2,13));globals_=np.zeros((4,14));mask=np.ones((4,2),bool);train=np.array([True,True,False,False]);logs=np.array([[0,0],[.001,.001],[500,500],[900,900]])
    a=fit_normalization(jobs,globals_,mask,logs,train);jobs[2:]=1e8;globals_[2:]=1e9;logs[2:]=-9e10;b=fit_normalization(jobs,globals_,mask,logs,train)
    assert a==b and a['job_std']==[1.]*13 and a['global_std']==[1.]*14;assert a['target_std']==[.0005,.0005]


def corrupted(fixture_dir,tmp_path,change):
    with gzip.open(fixture_dir/'observations.csv.gz','rt',newline='') as f:rows=list(csv.DictReader(f))
    change(rows);out=tmp_path/'bad.csv.gz'
    with gzip.open(out,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return out

@pytest.mark.parametrize('field,value,message',[
 ('Task_ID','wrong','identity'),('split','DEVELOPMENT','identity'),('hardware_class','V100','identity'),('J','5','identity'),('Pj','[0.1]','Pj'),('evidence_counts','[0,20]','evidence'),('observation_status','INSUFFICIENT_JOB_TYPE_EVIDENCE','invalid'),('mean_tracking','nan','nonfinite'),('objective','100000','objective'),('simulation_seed','23','realization'),('Profile_IDs','[]','identity')])
def test_bad_export_rejected(fixture_dir,tmp_path,field,value,message):
    path=corrupted(fixture_dir,tmp_path,lambda rows:rows[0].update({field:value}))
    with pytest.raises(ValueError,match=message):prepare(path,tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',True)


def test_missing_columns_duplicate_rows_and_real_count_gate(fixture_dir,tmp_path):
    path=tmp_path/'missing.csv';path.write_text('Plan_Row_ID\nF1_x\n')
    with pytest.raises(ValueError,match='Missing Foundation columns'):prepare(path,tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',True)
    path=corrupted(fixture_dir,tmp_path,lambda rows:rows.append(dict(rows[0])))
    with pytest.raises(ValueError,match='Duplicate Plan_Row_ID'):prepare(path,tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',True)
    with pytest.raises(ValueError,match='incomplete'):prepare(fixture_dir/'observations.csv.gz',tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',False)


def test_repeat_geometry_and_parent_realization_rejected(fixture_dir,tmp_path):
    def change(rows):
        x=json.loads(rows[1]['x']);x[2:]=[.4,.6];rows[1]['x']=json.dumps(x)
    path=corrupted(fixture_dir,tmp_path,change)
    with pytest.raises(ValueError,match='different candidate'):prepare(path,tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',True)
    path=corrupted(fixture_dir,tmp_path,lambda rows:rows.pop(0))
    with pytest.raises(ValueError,match='Repeat groups'):prepare(path,tmp_path/'out',fixture_dir/'profiles.json',fixture_dir/'workloads.json',True)


def test_locked_guards_and_sampler_exclusion(cache):
    with pytest.raises(ValueError,match='unlock'):partition_indices(cache,'LOCKED_SYNTHETIC_TEST')
    assert len(partition_indices(cache,'LOCKED_SYNTHETIC_TEST',True))==720
    with pytest.raises(ValueError,match='non-TRAIN'):sampler_weights(cache,partition_indices(cache,'DEVELOPMENT').numpy())
    w=sampler_weights(cache,partition_indices(cache,'TRAIN'));assert torch.isclose(w.sum(),torch.tensor(1.,dtype=torch.double)) and (w>0).all()

@pytest.mark.parametrize('k',K_VALUES)
@pytest.mark.parametrize('mode',['random','diverse'])
def test_all_k_geometries_disjoint_seed20_and_deterministic(cache,k,mode):
    pool=list(task_pools(cache,'TRAIN').values())[0];a=episode(cache,pool,k,32,np.random.default_rng(19),mode);b=episode(cache,pool,k,32,np.random.default_rng(19),mode);assert a==b
    validate_episode(cache,a,'TRAIN');assert all(cache['metadata'][i]['simulation_seed']==20 for i in a['context']+a['query'])


def test_geometry_leak_rejected_including_repeats(cache):
    pool=list(task_pools(cache,'TRAIN').values())[0];r=episode(cache,pool,2,3,np.random.default_rng(8));r['query'][0]=r['context'][0]
    with pytest.raises(ValueError,match='Geometry'):validate_episode(cache,r)
    r=episode(cache,pool,2,3,np.random.default_rng(8));r['query'][0]=1
    with pytest.raises(ValueError,match='realization'):validate_episode(cache,r)


def test_frozen_bank_nested_prefix_and_label_free_diversity(cache,tmp_path):
    a=make_bank(cache,repeats=1,q=4,mode='nested');b=make_bank(cache,repeats=1,q=4,mode='nested');assert a==b
    assert len(a['episodes'])==6*8
    bytask=defaultdict(list)
    for r in a['episodes']:bytask[r['Task_ID']].append(r)
    for group in bytask.values():
        largest=group[-1]
        for r in group:assert r['context']==largest['context'][:r['K']] and r['query']==largest['query']
    pool=list(task_pools(cache,'TRAIN').values())[0];z=diverse_order(cache,pool,np.random.default_rng(4));changed=dict(cache,qos=cache['qos']+100,objective=cache['objective']*-1);assert z==diverse_order(changed,pool,np.random.default_rng(4))
    f=tmp_path/'bank.json';f.write_text(json.dumps(a));assert load_bank(f,cache,'DEVELOPMENT')==a
    with pytest.raises(ValueError,match='partition'):load_bank(f,cache,'TRAIN')


def test_feasibility_equality_and_objective(cache):
    mask=torch.tensor([[True,True,False,False,False,False]]);logs=torch.log(torch.tensor([[.2+MEAN_FLOOR,.3+P90_FLOOR]]));qos=torch.tensor([[.1,.1,0,0,0,0]])
    b=dict(mask=mask,r_actual_watts=torch.tensor([10000.]),price_coefficient=torch.tensor([.1/3600000]),hour_seconds=torch.tensor([3600.]),power_cost=torch.tensor([12.]))
    # Exponentiation may land one float ulp below .3; either representation is legal equality.
    a=reconstruct(dict(tracking_logs=logs,qos_probabilities=qos),b);assert a['joint_feasible'].item()
    qos[0,0]=.1001;assert not reconstruct(dict(tracking_logs=logs,qos_probabilities=qos),b)['joint_feasible'].item()

@pytest.mark.parametrize('j',[2,3,4,5,6])
def test_dynamic_domain(j):
    mix=np.tile([200,700,20,50,4,1],(j,1));x=np.array([.4,.1]+[1/j]*j);validate_domain(x,mix,1000)
    for index,value in [(0,.1),(0,.8),(1,.001),(1,.3)]:
        z=x.copy();z[index]=value
        with pytest.raises(ValueError):validate_domain(z,mix,1000)
