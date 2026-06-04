import pandas as pd

def calculate_qos_distribution(job_table):
    """
    Calculate QoS for started jobs using the selected logic.
    """
    started_jobs = job_table[
        (job_table['end_time'] > 0) &  # Ensure jobs have ended
        (job_table['arrival_time'] <= 18000)  # Filter jobs within the simulation duration
        ]
    if started_jobs.empty:
        return []

    # Calculate QoS for each job
    qos = (started_jobs['end_time'] - started_jobs['arrival_time'] - started_jobs['min_execution_time']) / started_jobs[
        'min_execution_time']
    job_table = job_table.assign(qos=qos)
    return job_table

def write_qos_summary_df(simulator):
    """
    Checking whether QoS is violated.

    Robust version:
    - Handles cases where only a subset of job types have started/finished jobs.
    - Avoids pandas index-alignment crashes when comparing QoS percentiles to constraints.
    - Returns a full job-type-indexed violation Series plus the violation ratio.
    """
    qos_constraints = simulator._job_config.all_job_qos_constraints

    started_jobs = simulator._job_table[
        (simulator._job_table['end_time'] > simulator._idle_status) &
        (simulator._job_table['arrival_time'] < simulator._simulation_duration)
    ].copy()

    if started_jobs.empty:
        print('No started jobs')

        constraint_violations = pd.Series(
            [False] * simulator._job_type_count,
            index=range(simulator._job_type_count)
        )

        empty_qos_df = pd.DataFrame({
            'job_type_id': [],
            'qos_' + str(int(simulator.qos_violation_percentile_cutoff * 100)) + 'th': [],
            'mean_exec_slowdown': [],
            'qos_threshold': []
        })
        empty_qos_df.to_csv(simulator._output_dir + 'qos_summary.csv', index=False, header=True)

        return constraint_violations, 0.0

    started_jobs['qos_deg'] = (
        started_jobs['end_time']
        - started_jobs['arrival_time']
        - started_jobs['min_execution_time']
    ) / started_jobs['min_execution_time']

    started_jobs['exec_slowdown'] = (
        started_jobs['end_time']
        - started_jobs['start_time']
        - started_jobs['min_execution_time']
    ) / started_jobs['min_execution_time']

    str_qos_viol_percentile_cutoff = str(
        int(simulator.qos_violation_percentile_cutoff * 100)
    )

    print('QoS deg at ' + str_qos_viol_percentile_cutoff + ' by job type')

    qos_deg_at_percentile = started_jobs.groupby('job_type_id')['qos_deg'].quantile(
        simulator.qos_violation_percentile_cutoff
    )

    mean_exec_slowdown = started_jobs.groupby('job_type_id')['exec_slowdown'].mean()

    # Make constraints a job_type_id-indexed Series.
    # This is the key fix. We compare only the observed job types against
    # their matching constraints instead of comparing mismatched Series labels.
    constraints_series = pd.Series(qos_constraints, dtype=float)
    constraints_for_observed_types = constraints_series.reindex(qos_deg_at_percentile.index)

    observed_constraint_violations = qos_deg_at_percentile > constraints_for_observed_types

    # Return a full-length violation Series so downstream code always sees
    # one value per job type. Missing/unobserved job types count as no violation.
    constraint_violations = pd.Series(
        [False] * simulator._job_type_count,
        index=range(simulator._job_type_count)
    )
    constraint_violations.loc[observed_constraint_violations.index] = observed_constraint_violations.astype(bool)

    job_type_qos_violation_count = int(constraint_violations.sum())
    job_type_qos_violation_ratio = job_type_qos_violation_count / simulator._job_type_count

    print(qos_deg_at_percentile.to_string())

    missing_job_types = sorted(set(range(simulator._job_type_count)) - set(qos_deg_at_percentile.index))
    if missing_job_types:
        print(f'Job types with no started/finished jobs in QoS summary: {missing_job_types}')

    print('Mean exec slowdown by job type')
    print(mean_exec_slowdown)

    qos_df = pd.DataFrame({
        'job_type_id': qos_deg_at_percentile.index,
        'qos_' + str_qos_viol_percentile_cutoff + 'th': qos_deg_at_percentile.values,
        'mean_exec_slowdown': mean_exec_slowdown.reindex(qos_deg_at_percentile.index).values,
        'qos_threshold': constraints_for_observed_types.values
    })
    qos_df.to_csv(simulator._output_dir + 'qos_summary.csv', index=False, header=True)

    print(
        f'Count of job types with QoS violations from {simulator._job_type_count} job types: '
        f'{job_type_qos_violation_count}'
    )
    print(f'Ratio of job types with QoS violation: {job_type_qos_violation_ratio}')

    violating_job_types = [
        str(job_type_id)
        for job_type_id, violation in constraint_violations.items()
        if violation
    ]
    print('Violations:', ', '.join(violating_job_types))

    queued_jobs = simulator._job_table.loc[
        simulator._job_table['arrival_time'] <= simulator._simulation_duration
    ]

    if queued_jobs.empty:
        all_jobs_qos_violation_ratio = 0.0
    else:
        all_jobs_qos_violation_ratio = round(
            (queued_jobs['realtime_qos'].values > queued_jobs['qos_constraint'].values).mean(),
            7
        )

    print(
        f'Ratio of all jobs violating QoS, queued in the first hour '
        f'(from {len(queued_jobs)} jobs):',
        all_jobs_qos_violation_ratio
    )

    return constraint_violations, job_type_qos_violation_ratio