import numpy as np
from peacsim.macros import *

# Columns in job_table
(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


def get_job_queue(job_table, current_time, arrived_end=None):
    # A caller may bound this view only after verifying immutable sorted arrivals.
    if arrived_end is not None:
        job_table = job_table[:, :arrived_end]
    job_queue = job_table[job_id:arrival_time, np.where
                            ((job_table[start_time] == SERVER_STATUS_IDLE)
                             & (job_table[arrival_time] <= current_time))[0]]

    return job_queue