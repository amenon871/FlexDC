"""Synthetic loop cadence and terminal ordering, with real run() and no science collection."""
import random
from types import SimpleNamespace
import numpy as np
import pytest
from peacsim.simulator import Simulator
from peacsim.aqa_runtimepolicy import AQARuntimePolicy

@pytest.mark.parametrize('fast',[True,False])
@pytest.mark.parametrize('duration',[0,1,4])
def test_loop_only_audited_aqa_skips_both_unused_updates(tmp_path,fast,duration):
 events=[];sim=Simulator.__new__(Simulator);sim._defer_realtime_qos=fast
 sim._simulation_duration=duration;sim._time_granularity=1;sim._random_seed=20;sim._iso_signal_granularity=1;sim._cluster_power_cap_per_signal=[0.]*(duration+1)
 sim._job_table=np.zeros((10,1));sim._node_table=np.zeros((12,1));sim._output_dir=str(tmp_path)+'/'
 sim._runtime_policy=SimpleNamespace(P=1.,R=1.,weights={0:1.},execute=lambda t,p:events.append(('execute',t)))
 sim.qos_emergency=False;sim.tracking_num=0;sim.waiting_sum=0;sim.tracking_error=0.;sim.standby_power=0.
 sim.update_job_progress=lambda t:None;sim.update_cluster=lambda t:events.append(('cluster',t));sim.calc_qos_emergency=lambda t:events.append(('emergency',t));sim.calc_tracking_num=lambda p:None;sim.record_state=lambda *args:None
 sim.update_realtime_qos=lambda t:events.append(('canonical_qos',t))
 def forbidden_timestamp_only(t):raise AssertionError('Timestamp-only replacement must not execute in run()')
 sim.update_waiting_job_times=forbidden_timestamp_only;sim.error_violate=lambda:0.;sim.final_output=lambda:(1.,2.)
 import peacsim.simulator as module
 original=module.write_qos_summary_df;module.write_qos_summary_df=lambda s:([False],0.)
 py_state=random.getstate();np_state=np.random.get_state()
 try:assert sim.run()==(1.,2.)
 finally:module.write_qos_summary_df=original;random.setstate(py_state);np.random.set_state(np_state)
 updates=[t for event,t in events if event=='canonical_qos'];assert updates==([duration] if fast else list(range(duration+1)))
 for t in updates:assert events.index(('cluster',t))<events.index(('canonical_qos',t))<events.index(('emergency',t))<events.index(('execute',t))


def test_unknown_aqa_subclass_does_not_opt_in_by_inheritance():
 class Unknown(AQARuntimePolicy):pass
 assert AQARuntimePolicy.__dict__.get('supports_deferred_realtime_qos') is True
 assert Unknown.__dict__.get('supports_deferred_realtime_qos',False) is False
