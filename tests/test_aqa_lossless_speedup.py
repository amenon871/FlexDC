"""Lossless AQA edge contracts; no production campaign execution."""
import pathlib,subprocess,types,importlib.util
import numpy as np
import pytest
from peacsim.simulator import Simulator
from peacsim.job_table_dao import get_job_queue
from peacsim.scheduler import adjust_weights
from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.runtime_policy_new import RuntimePolicyNew

@pytest.fixture(scope='module')
def old():
 root=pathlib.Path(__file__).resolve().parents[1]
 body=subprocess.check_output(['git','-C',str(root),'show','c50aea78071a903b20a581891d4106a9da3d2d57:src/peacsim/simulator.py'])
 module=types.ModuleType('reference_simulator');exec(compile(body,'<reviewed-baseline>','exec'),module.__dict__);return module.Simulator

def table(arrivals):
 a=np.zeros((10,len(arrivals)),dtype=float);a[0]=np.arange(len(arrivals));a[1]=np.arange(len(arrivals))%3;a[2]=arrivals;a[3:5]=-1;a[6]=arrivals;a[8]=30.;a[9]=5.;return a

def simulator(cls,a):
 s=cls.__new__(cls);s._job_table=a.copy();s._idle_status=-1;s._arrivals_sorted=bool(np.all(np.diff(a[2])>=0));return s

@pytest.mark.parametrize('arrivals',[[],[1],[0,1,2],[0,0,0],[1,1,2,3],[3,1,5,0],[20,30,40]])
@pytest.mark.parametrize('t',[0,1,2,3,10,30])
def test_queue_exact_boundaries_and_order(arrivals,t):
 a=table(arrivals)
 if len(arrivals)>2:a[3,1]=0
 expected=a[:2,np.where((a[3]==-1)&(a[2]<=t))[0]]
 end=np.searchsorted(a[2],t,side='right') if np.all(np.diff(a[2])>=0) else None
 np.testing.assert_array_equal(get_job_queue(a,t,end),expected)
 weights={0:.2,1:.3,2:.5};types=np.unique(a[1,(a[3]==-1)&(a[2]<=t)]);w={i:x if i in types else 0 for i,x in weights.items()};total=sum(w.values())
 if total:w={i:x/total for i,x in w.items()}
 assert adjust_weights(t,a,weights,expected)==w

@pytest.mark.parametrize('arrivals',[[0,0,1,2,100],[100,1,3,0,500],[10,10,10,10],[100,200,300]])
def test_waiting_timestamps_full_legacy_formula_every_cycle(old,arrivals):
 a=table(arrivals);new=simulator(Simulator,a);legacy=simulator(old,a)
 for t in range(31):
  legacy.update_realtime_qos(t);new.update_waiting_job_times(t)
  np.testing.assert_array_equal(new._job_table[6],legacy._job_table[6])
  if t==15:new._job_table[3,0]=legacy._job_table[3,0]=t
 new.update_realtime_qos(30)
 np.testing.assert_array_equal(new._job_table,legacy._job_table)

@pytest.mark.parametrize('name',['supports_deferred_realtime_qos'])
def test_only_audited_direct_capability(name):
 class Unknown:pass
 class FutureAQA(AQARuntimePolicy):pass
 from peacsim.proportional_cap_policy import ProportionalCapPolicy
 from peacsim.priority_cap_policy import PriorityCapPolicy
 from peacsim.flex_resource_policy import FlexResourcePolicy
 for cls in (Unknown,FutureAQA,RuntimePolicyNew,ProportionalCapPolicy,PriorityCapPolicy,FlexResourcePolicy):assert cls.__dict__.get(name,False) is not True
 assert AQARuntimePolicy.__dict__[name] is True

@pytest.mark.parametrize('values',[[0.,-0.,1,1.,np.int64(1),np.float64(1),float('inf'),-float('inf'),float('nan')],list(range(-1,7)),[120.,234.1234567,1500.,-0.]])
def test_exact_formatter_candidates_adversarial(values):
 a=np.array(values,dtype=object)
 assert ','.join([str(x) for x in a])==','.join(map(str,a))
 assert str(values[0]) in ','.join(map(str,a))

def test_selective_csv_exact_p90(tmp_path):
 import pandas as pd
 from peacsim.extract_power_metrics import get_tracking_error_90th
 p=tmp_path/'trace.csv';pd.DataFrame({'trackingError':[0.,-.3,1.257,.03,-.1],'other':[0,2,8,4,9]}).to_csv(p,index=False)
 a=get_tracking_error_90th(pd.read_csv(p));b=get_tracking_error_90th(pd.read_csv(p,usecols=['trackingError']));assert a.hex()==b.hex()

@pytest.mark.parametrize('dtype',[np.float64,np.float32,np.int64,object])
def test_production_formatter_exact_and_type_safe(dtype):
 s=Simulator.__new__(Simulator);s._trace_format_cache={}
 values=[0.,-0.,1.,3.,120.,234.123456789] if dtype!=np.int64 else [0,-1,1,3,120,234]
 a=np.asarray(values,dtype=dtype)
 assert s._format_trace_values(a)==','.join([str(x) for x in a])
 if dtype==np.float64:
  b=np.array([np.inf,-np.inf,np.nan,0.,-0.]);assert s._format_trace_values(b)==','.join([str(x) for x in b]);assert 0 in s._trace_format_cache and np.uint64(1<<63) in s._trace_format_cache


def test_production_formatter_cache_bounded_and_empty():
 s=Simulator.__new__(Simulator);s._trace_format_cache={}
 for n in range(8):
  a=np.arange(n*1000.,(n+1)*1000.);assert s._format_trace_values(a)==','.join([str(x) for x in a]);assert len(s._trace_format_cache)<=4096
 a=np.arange(5000.);assert s._format_trace_values(a)==','.join([str(x) for x in a]);assert len(s._trace_format_cache)<=4096
 assert s._format_trace_values(np.array([]))==''
