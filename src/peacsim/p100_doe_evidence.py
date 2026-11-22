"""Opt-in evidence derived from the terminal simulator table.

The job ID, type, and arrival columns are generated before bids are applied
and are never mutated by the simulator. This module reads those columns and
the exact one-hour delay-estimator population without changing simulator state.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def arrival_digest(arrivals):
    return hashlib.sha256(
        np.asarray(arrivals, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def append_observation(simulator):
    job = simulator._job_config
    policy = simulator._runtime_policy
    duration = int(simulator._simulation_duration)
    if duration != 3600:
        raise ValueError("P100 DOE evidence requires a one-hour simulation")
    table = simulator._job_table
    arrivals = table[["job_id", "job_type_id", "arrival_time"]].to_numpy(
        dtype="<f8", copy=True
    )
    arrival_hash = arrival_digest(arrivals)
    counts = []
    for i, name in job.all_jobs.items():
        typed = table[table["job_type_id"] == i]
        arrived = typed["arrival_time"] > 0
        finished = arrived & (typed["end_time"] != -1)
        unfinished = (
            arrived
            & (typed["end_time"] == -1)
            & (typed["arrival_time"] <= duration)
            & (typed["arrival_time"] + job.all_min_execution_time[i] < duration)
        )
        counts.append(
            dict(
                job_id=i,
                name=name,
                generated=len(typed),
                completed=int(finished.sum()),
                eligible_unfinished=int(unfinished.sum()),
            )
        )
    evidence = dict(
        arrival_sha256=arrival_hash,
        arrival_rows=len(arrivals),
        job_counts=counts,
        qos=[float(value) for value in job.all_job_qos_constraints.values()],
        p_watts=int(policy.P),
        r_watts=int(policy.R),
        weights=[float(policy.weights[i]) for i in job.all_jobs],
        n=int(simulator._server_count),
        u=float(simulator._utilization),
        seed=int(simulator._random_seed),
        duration=duration,
        fast_aqa=simulator._defer_realtime_qos,
        policy_module=type(policy).__module__,
    )
    with (Path(simulator._output_dir) / "p100_doe_evidence.jsonl").open(
        "a", encoding="utf8"
    ) as stream:
        stream.write(json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n")
